import numpy
import time
import math
import sys
import itertools
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from util import loadPCD, savePCD, get_cls_id_metrics, get_obj_id_metrics
import psutil
import os

import roslib
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PoseStamped
import std_msgs
import rosbag

local_range = 10
resolution = 0.1
normal_resolution = 0.3
threshold = 0.995
margin = 2 #voxel margin for clustering
mode = 'euclidean'
#mode = 'normals'
viz = 'cluster'
AREA = None

for i in range(len(sys.argv)-1):
    if sys.argv[i]=='--mode':
        mode = sys.argv[i+1]
    elif sys.argv[i]=='--threshold':
        threshold = float(sys.argv[i+1])
    elif sys.argv[i]=='--margin':
        margin = int(sys.argv[i+1])
    elif sys.argv[i]=='--viz':
        viz = sys.argv[i+1]
    elif sys.argv[i]=='--area':
        AREA = sys.argv[i+1]
    elif sys.argv[i]=='--range':
        local_range = float(sys.argv[i+1])

point_id_map = {}
accA = {}
accN = {}
point_orig_list = []
gt_obj_id = []
gt_cls_id = []
predicted_obj_id = []
predicted_cls_id = []
normals = []

count_msg = 0
clusters = {}
obj_count = 0
comp_time = []
numpy.random.seed(0)
sample_state = numpy.random.RandomState(0)
obj_color = {0:[100,100,100]}
cls_color = {0:[100,100,100]}
cls_color = numpy.random.randint(0, 256, (17, 3))
cls_color[0] = [100,100,100]

def publish_output(cloud):
	header = std_msgs.msg.Header()
	header.stamp = rospy.Time.now()
	header.frame_id = '/map'
	fields = [
		PointField('x',0,PointField.FLOAT32,1),
		PointField('y',4,PointField.FLOAT32,1),
		PointField('z',8,PointField.FLOAT32,1),
		PointField('rgb',12,PointField.INT32,1),
	]
	cloud = [[p[0],p[1],p[2],(int(p[3])<<16)|int(p[4])<<8|int(p[5])] for p in cloud]
	msg = point_cloud2.create_cloud(header,fields, cloud)
	pubOutput.publish(msg)

robot_position = None
def pose_callback(msg):
	global robot_position
	robot_position = [msg.pose.position.x, msg.pose.position.y]

def cloud_surround_callback(cloud):
    global count_msg, obj_count
#    if count_msg >= 1:
#        return
    t = time.time()
    pcd = []
    for p in point_cloud2.read_points(cloud, field_names=("x","y","z","r","g","b","o","c"), skip_nans=True):
        pcd.append(p + (0,0))

    if robot_position is None:
        centroid = numpy.median(pcd[:,:2],axis=0)
    else:
        centroid = robot_position
    pcd = numpy.array(pcd)
    local_mask = numpy.sum((pcd[:,:2]-centroid)**2, axis=1) < local_range * local_range
    pcd = pcd[local_mask, :]
    if len(pcd)==0:
        return
    pcd[:,3:6] = pcd[:,3:6] / 255.0 - 0.5
    original_pcd = pcd.copy()
    pcd[:,:2] -= centroid
    pcd[:,2] -= pcd[:,2].min()

    update_list = []
    acc = 0.0
    pcdi = [tuple(p) for p in (original_pcd[:,:3]/resolution).round().astype(int)]
    update_list = []
    for i in range(len(pcdi)):
        if not pcdi[i] in point_id_map:
            point_id_map[pcdi[i]] = len(point_orig_list)
            point_orig_list.append(original_pcd[i,:6].copy())
            gt_obj_id.append(int(original_pcd[i,6]))
            gt_cls_id.append(int(original_pcd[i,7]))
            update_list.append(pcdi[i])
            normals.append(None)
            predicted_obj_id.append(0)

    if len(update_list)>0:
        if mode=='normals':
            for u in update_list:
                idx = point_id_map[u]
                p = point_orig_list[idx][:3]
                k = tuple((p / normal_resolution).round().astype(int))
                for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
                    kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
                    if not kk in accN:
                        accA[kk] = numpy.zeros((3,3))
                        accN[kk] = 0

                    c = numpy.array(kk) * normal_resolution
                    accA[kk] += numpy.outer(p - c, p - c)
                    accN[kk] += 1
            
            for u in update_list:
                idx = point_id_map[u]
                p = point_orig_list[idx][:3]
                k = tuple((p / normal_resolution).round().astype(int))
                if accN[k] > 3:
                    cov = accA[k] / accN[k]
                    U,S,V = numpy.linalg.svd(cov)
                    normals[idx] = numpy.fabs(V[2])

        neighbor_key = []
        neighbor_probs = []
        for k in update_list:
            nk = [point_id_map[k]]
            for offset in itertools.product(range(-margin,margin+1),range(-margin,margin+1),range(-margin,margin+1)):
                if offset!=(0,0,0):
                    kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
                    if kk in point_id_map:
                        if mode=='normals':
                            ni = normals[point_id_map[k]]
                            nj = normals[point_id_map[kk]]
                            if ni is not None and nj is not None and ni.dot(nj) > threshold:
                                nk.append(point_id_map[kk])
                        elif mode=='euclidean':
                            nk.append(point_id_map[kk])
            if len(nk) > 1:
                neighbor_key.append(nk)
                neighbor_probs.append([True] * (len(nk)-1))
            else:
                predicted_obj_id[point_id_map[k]] = obj_count + 1
                clusters[obj_count + 1] = [point_id_map[k]]
                obj_count += 1

        for i in range(len(neighbor_probs)):
            if not numpy.any(neighbor_probs[i]): #isolated point
                predicted_obj_id[neighbor_key[i][0]] = obj_count + 1
                clusters[obj_count + 1] = [neighbor_key[i][0]]
                obj_count += 1
            else:
                group = [neighbor_key[i][0]] + [neighbor_key[i][j+1] for j in numpy.nonzero(neighbor_probs[i])[0]]
                try:
                    group_id = min([predicted_obj_id[g] for g in group if predicted_obj_id[g] > 0])
                    for g in group:
                        if predicted_obj_id[g]==0: #add to existing cluster
                            predicted_obj_id[g] = group_id
                            clusters[group_id].append(g)
                        elif predicted_obj_id[g]!=group_id: #merge two clusters
                            remove_id = predicted_obj_id[g]
                            clusters[group_id].extend(clusters[remove_id])
                            for r in clusters[remove_id]:
                                predicted_obj_id[r] = group_id
                            del clusters[remove_id]
                except ValueError: #all neighbors do not have ids assigned yet
                    clusters[obj_count + 1] = []
                    for g in group:
                        predicted_obj_id[g] = obj_count + 1
                        clusters[obj_count + 1].append(g)
                    obj_count += 1

        if AREA is None:
            output_cloud = numpy.array(point_orig_list[-len(update_list):])
            for i in range(len(update_list)):
                if viz == 'cluster':
                    obj_id = predicted_obj_id[-len(update_list)+i]
                elif viz == 'gt':
                    obj_id = gt_obj_id[-len(update_list)+i]
                if viz in ['cluster', 'gt']:
                    if not obj_id in obj_color:
                        obj_color[obj_id] = numpy.random.randint(0,255,3)
                    output_cloud[i,3:6] = obj_color[obj_id]
                elif viz=='cls':
                    output_cloud[i,3:6] = cls_color[gt_cls_id[-len(update_list)+i]]
                elif viz=='normals':
                    n = normals[-len(update_list)+i]
                    if n is None:
                        output_cloud[i, 3:6] = [100,100,100]
                    else:
                        output_cloud[i, 3:6] = n * 255
            publish_output(output_cloud)

    t = time.time() - t
    comp_time.append(t)
    print('Scan #%3d: %5d/%5d/%7d points obj_count %d/%d time %.3f'%(count_msg, len(update_list), numpy.sum(local_mask), len(point_id_map), len(clusters), obj_count, t))
    count_msg += 1
	
if AREA is None:
    rospy.init_node('baseline_seg')
    rospy.Subscriber('laser_cloud_surround',PointCloud2,cloud_surround_callback)
    rospy.Subscriber('slam_out_pose',PoseStamped,pose_callback)
    pubOutput = rospy.Publisher('output_cloud', PointCloud2, queue_size=1)
    rospy.spin()
else:
    bag = rosbag.Bag('data/guardian_centers_%s.bag' % AREA, 'r')
    poses = []
    for topic, msg, t in bag.read_messages(topics=['slam_out_pose']):
        poses.append([msg.pose.position.x, msg.pose.position.y])
    i = 0
    for topic, msg, t in bag.read_messages(topics=['laser_cloud_surround']):
        robot_position = poses[i]
        cloud_surround_callback(msg)
        i += 1

#ignore clutter in evaluation
valid_mask = numpy.array(gt_obj_id) > 0
predicted_obj_id = numpy.array(predicted_obj_id)
gt_obj_id = numpy.array(gt_obj_id)[valid_mask]
predicted_obj_id = predicted_obj_id[valid_mask]
point_orig_list = numpy.array(point_orig_list)[valid_mask]
normals = numpy.array([[0.5,0.5,0.5] if n is None else n for n in normals])[valid_mask]

obj_color = numpy.random.randint(0,255,(max(gt_obj_id)+1,3))
obj_color[0] = [100, 100, 100]
point_orig_list[:,3:6] = obj_color[gt_obj_id, :]
savePCD('viz/area%s_gt_obj_id.pcd' % AREA, point_orig_list)
obj_color = numpy.random.randint(0,255,(max(predicted_obj_id)+1,3))
obj_color[0] = [100, 100, 100]
point_orig_list[:,3:6] = obj_color[predicted_obj_id, :]
valid_mask = numpy.ones(len(predicted_obj_id), dtype=bool)
for i in set(predicted_obj_id):
    cluster_mask = predicted_obj_id==i
    if numpy.sum(cluster_mask) < 10:
        valid_mask[cluster_mask] = False
savePCD('viz/%s_area%s_predicted_obj_id.pcd' % (mode, AREA), point_orig_list[valid_mask])
#point_orig_list[:,3:6] = [class_to_color_rgb[c] for c in gt_cls_id]
#savePCD('viz/area%s_gt_cls_id.pcd' % AREA, point_orig_list)
#point_orig_list[:,3:6] = [class_to_color_rgb[c] for c in predicted_cls_id]
#savePCD('viz/%s_area%s_predicted_cls_id.pcd' % (mode, AREA), point_orig_list)
if mode=='normals':
    point_orig_list[:,3:6] = normals * 255
    savePCD('viz/area%s_normals.pcd' % (AREA), point_orig_list)

print("Avg Comp Time: %.3f" % numpy.mean(comp_time))
print("CPU Mem: %.2f" % (psutil.Process(os.getpid()).memory_info()[0] / 1.0e9))
print("GPU Mem: 0.0")
#acc, iou, avg_acc, avg_iou, stats = get_cls_id_metrics(gt_cls_id, predicted_cls_id)
acc = iou = avg_acc = avg_iou = 0
nmi, ami, ars, prc, rcl, iou, hom, com, vms = get_obj_id_metrics(gt_obj_id, predicted_obj_id)
print("NMI: %.3f AMI: %.3f ARS: %.3f PRC: %.3f RCL: %.3f IOU: %.3f HOM: %.3f COM: %.3f VMS: %.3f %d/%d clusters"% (nmi,ami,ars,prc, rcl, iou, hom,com,vms,len(numpy.unique(predicted_obj_id)),len(numpy.unique(gt_obj_id))))
print('all 0 0 0 %.3f 0 %.3f' % (acc, iou))
print('avg 0 0 0 %.3f 0 %.3f' % (avg_acc, avg_iou))
