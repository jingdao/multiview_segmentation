import sys
import numpy
import rosbag
from sensor_msgs import point_cloud2
from class_util import classes, class_to_id, class_to_color_rgb
from util import loadPCD, savePCD, get_cls_id_metrics, get_obj_id_metrics
from offline_architecture import PointNet, PointNet2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import networkx as nx
import itertools
import time
import psutil

VAL_AREA = 1
net_type = 'pointnet'
#mode = 'time'
#mode = 'space'
mode = 'single'
for i in range(len(sys.argv)-1):
	if sys.argv[i]=='--area':
		VAL_AREA = int(sys.argv[i+1])
	if sys.argv[i]=='--net':
		net_type = sys.argv[i+1]
	if sys.argv[i]=='--mode':
		mode = sys.argv[i+1]
save_viz = False
if '--viz' in sys.argv:
	save_viz = True

NUM_CLASSES = len(classes)
NUM_POINT = 2048
resolution = 0.1
local_range = 1.0
time_interval = 1
space_interval = 0.5
sample_state = numpy.random.RandomState(0)
count_msg = 0
obj_count = 0
comp_time = []
num_updates = []
clusters = {}
point_id_map = {}
point_orig_list = []
gt_obj_id = []
gt_cls_id = []
predicted_obj_id = []
predicted_cls_id = []

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
base_path = 'models'
MODEL_PATH = '%s/offline_%s_model_s3dis_%d.ckpt' % (base_path, net_type, VAL_AREA)
if net_type=='pointnet':
	net = PointNet(1,NUM_POINT,len(classes)) 
elif net_type=='pointnet2':
	net = PointNet2(1,NUM_POINT,len(classes)) 
saver = tf.train.Saver()
saver.restore(sess, MODEL_PATH)
print('Restored from %s'%MODEL_PATH)

def process_cloud(cloud, robot_position):
    global count_msg, obj_count
    t = time.time()
    pcd = []
    for p in point_cloud2.read_points(cloud, field_names=("x","y","z","r","g","b","o","c"), skip_nans=True):
        pcd.append(p)
    pcd = numpy.array(pcd)
    local_mask = numpy.sum((pcd[:,:2]-robot_position)**2, axis=1) < local_range * local_range
    pcd = pcd[local_mask, :]
    pcd[:,3:6] = pcd[:,3:6] / 255.0 - 0.5
    original_pcd = pcd.copy()
    minZ = pcd[:, 2].min()
    pcd[:,:2] -= robot_position
    pcd[:,2] -= minZ

    pcdi = [tuple(p) for p in (original_pcd[:,:3]/resolution).round().astype(int)]
    update_list = []
    for i in range(len(pcdi)):
        if not pcdi[i] in point_id_map:
            point_id_map[pcdi[i]] = len(point_orig_list)
            point_orig_list.append(original_pcd[i,:6].copy())
            update_list.append(pcdi[i])
            gt_obj_id.append(int(original_pcd[i,6]))
            gt_cls_id.append(int(original_pcd[i,7]))
            predicted_obj_id.append(0)

    if len(update_list)>0:
        stacked_points = numpy.array(point_orig_list[-len(update_list):])
        stacked_points[:,:2] -= robot_position
        stacked_points[:,2] -= minZ

        if mode=='single' or count_msg==0:
            num_batches = int(numpy.ceil(1.0 * len(stacked_points) / NUM_POINT))
            input_points = numpy.zeros((1, NUM_POINT, 6))
            for batch_id in range(num_batches):
                start_idx = batch_id * NUM_POINT
                end_idx = (batch_id + 1) * NUM_POINT
                valid_idx = min(len(stacked_points), end_idx)
                if end_idx <= len(stacked_points):
                    input_points[0, :valid_idx-start_idx] = stacked_points[start_idx:valid_idx,:]
                else:
                    input_points[0, :valid_idx-start_idx] = stacked_points[start_idx:valid_idx,:]
                    input_points[0, valid_idx-end_idx:] = stacked_points[sample_state.choice(range(len(stacked_points)), end_idx-valid_idx, replace=True),:]
                cls = sess.run(net.output, feed_dict={net.pointclouds_pl: input_points, net.is_training_pl: False})
                cls = cls[0].argmax(axis=1)
                predicted_cls_id.extend(cls[:valid_idx-start_idx])
        else:
            #allocate batch as 1024 points from current scan and 1024 context points from previous scans
            if mode=='time':
                num_context_points = sum(num_updates[-time_interval:])        
                context_points_offset = len(point_orig_list) - len(update_list) - num_context_points
                context_points = [point_orig_list[context_points_offset + i] for i in numpy.random.choice(num_context_points, NUM_POINT/2, replace=num_context_points<NUM_POINT/2)]
            elif mode=='space':
                all_points = numpy.array(point_orig_list[:-len(update_list)])
                context_points = []
                current_space_interval = space_interval
                while len(context_points)==0:
                    context_mask = numpy.sum((all_points[:,:2]-robot_position)**2, axis=1) < current_space_interval * current_space_interval
                    context_points = all_points[context_mask, :]
                    current_space_interval *= 2
                context_points = context_points[numpy.random.choice(len(context_points), NUM_POINT/2, replace=len(context_points)<NUM_POINT/2)]
            context_points = numpy.array(context_points)
            context_points[:, :2] -= robot_position
            context_points[:, 2] -= minZ
            num_batches = int(numpy.ceil(1.0 * len(stacked_points) / (NUM_POINT/2)))
            input_points = numpy.zeros((1, NUM_POINT, 6))
            for batch_id in range(num_batches):
                start_idx = batch_id * NUM_POINT/2
                end_idx = (batch_id + 1) * NUM_POINT/2
                valid_idx = min(len(stacked_points), end_idx)
                if end_idx <= len(stacked_points):
                    input_points[0, :valid_idx-start_idx] = stacked_points[start_idx:valid_idx,:]
                else:
                    input_points[0, :valid_idx-start_idx] = stacked_points[start_idx:valid_idx,:]
                    input_points[0, valid_idx-start_idx:NUM_POINT/2] = stacked_points[sample_state.choice(range(len(stacked_points)), end_idx-valid_idx, replace=True),:]
                input_points[0, NUM_POINT/2:, :] = context_points 
                cls = sess.run(net.output, feed_dict={net.pointclouds_pl: input_points, net.is_training_pl: False})
                cls = cls[0].argmax(axis=1)
                predicted_cls_id.extend(cls[:valid_idx-start_idx])

        neighbor_key = []
        neighbor_probs = []
        for k in update_list:
            nk = [point_id_map[k]]
            for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
                if offset!=(0,0,0):
                    kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
                    if kk in point_id_map and predicted_cls_id[point_id_map[k]] == predicted_cls_id[point_id_map[kk]]:
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

    t = time.time() - t
    comp_time.append(t)
    if len(update_list)>0:
        num_updates.append(len(update_list))
#    print('Scan #%3d: cur:%4d/%6d'%(count_msg, len(update_list), len(point_orig_list)))
    count_msg += 1

bag = rosbag.Bag('data/s3dis_%d.bag' % VAL_AREA, 'r')
poses = []
for topic, msg, t in bag.read_messages(topics=['slam_out_pose']):
    poses.append([msg.pose.position.x, msg.pose.position.y])
i = 0
for topic, msg, t in bag.read_messages(topics=['laser_cloud_surround']):
    process_cloud(msg, poses[i])
    i += 1

point_orig_list = numpy.array(point_orig_list)
predicted_obj_id = numpy.array(predicted_obj_id)
gt_obj_id = numpy.array(gt_obj_id)
predicted_cls_id = numpy.array(predicted_cls_id)
gt_cls_id = numpy.array(gt_cls_id)

if save_viz:
    obj_color = numpy.random.randint(0,255,(max(predicted_obj_id)+1,3))
    obj_color[0] = [100, 100, 100]
    point_orig_list[:,3:6] = obj_color[predicted_obj_id, :]
    savePCD('viz/%s_area%d_predicted_obj_id.pcd' % (net_type, VAL_AREA), point_orig_list)
    point_orig_list[:,3:6] = [class_to_color_rgb[c] for c in predicted_cls_id]
    savePCD('viz/%s_area%d_predicted_cls_id.pcd' % (net_type, VAL_AREA), point_orig_list)

print("Avg Comp Time: %.3f" % numpy.mean(comp_time))
print("CPU Mem: %.2f" % (psutil.Process(os.getpid()).memory_info()[0] / 1.0e9))
print("GPU Mem: %.1f" % (sess.run(tf.contrib.memory_stats.MaxBytesInUse()) / 1.0e6))
acc, iou, avg_acc, avg_iou, stats = get_cls_id_metrics(gt_cls_id, predicted_cls_id)
nmi, ami, ars, _, _, _, hom, com, vms = get_obj_id_metrics(gt_obj_id, predicted_obj_id)
print("NMI: %.3f AMI: %.3f ARS: %.3f HOM: %.3f COM: %.3f VMS: %.3f %d/%d clusters"% (nmi,ami,ars,hom,com,vms,len(numpy.unique(predicted_obj_id)),len(numpy.unique(gt_obj_id))))
print('all 0 0 0 %.3f 0 %.3f' % (acc, iou))
print('avg 0 0 0 %.3f 0 %.3f' % (avg_acc, avg_iou))
