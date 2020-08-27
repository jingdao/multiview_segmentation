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

VAL_AREA = 1
net_type = 'pointnet'
for i in range(len(sys.argv)-1):
	if sys.argv[i]=='--area':
		VAL_AREA = int(sys.argv[i+1])
	if sys.argv[i]=='--net':
		net_type = sys.argv[i+1]
save_viz = False
if '--viz' in sys.argv:
	save_viz = True

NUM_CLASSES = len(classes)
NUM_POINT = 2048
resolution = 0.1
local_range = 2
sample_state = numpy.random.RandomState(0)
count_msg = 0
point_id_map = {}
point_orig_list = []
gt_obj_id = []
gt_cls_id = []

def process_cloud(cloud, robot_position):
    global count_msg
    pcd = []
    for p in point_cloud2.read_points(cloud, field_names=("x","y","z","r","g","b","o","c"), skip_nans=True):
        pcd.append(p)
    pcd = numpy.array(pcd)
    local_mask = numpy.sum((pcd[:,:2]-robot_position)**2, axis=1) < local_range * local_range
    pcd = pcd[local_mask, :]
    pcd[:,3:6] = pcd[:,3:6] / 255.0 - 0.5
    original_pcd = pcd.copy()
    pcd[:,:2] -= robot_position
    pcd[:,2] -= pcd[:,2].min()

    pcdi = [tuple(p) for p in (original_pcd[:,:3]/resolution).round().astype(int)]
    update_list = []
    for i in range(len(pcdi)):
        if not pcdi[i] in point_id_map:
            point_id_map[pcdi[i]] = len(point_orig_list)
            point_orig_list.append(original_pcd[i,:6].copy())
            update_list.append(pcdi[i])
            gt_obj_id.append(int(original_pcd[i,6]))
            gt_cls_id.append(int(original_pcd[i,7]))
    print('Scan #%3d: cur:%4d'%(count_msg, len(update_list)))
    count_msg += 1

point_orig_list_path = 'viz/area%d_point_orig_list.npy' % VAL_AREA
if os.path.exists(point_orig_list_path):
    point_orig_list = numpy.load(point_orig_list_path)
    gt_obj_id = point_orig_list[:, 6].astype(int)
    gt_cls_id = point_orig_list[:, 7].astype(int)
    point_orig_list = point_orig_list[:, :6]
else:
    bag = rosbag.Bag('data/area%d.bag' % VAL_AREA, 'r')
    poses = []
    for topic, msg, t in bag.read_messages(topics=['slam_out_pose']):
        poses.append([msg.pose.position.x, msg.pose.position.y])
    i = 0
    for topic, msg, t in bag.read_messages(topics=['laser_cloud_surround']):
        process_cloud(msg, poses[i])
        i += 1

    point_orig_list = numpy.array(point_orig_list)
    point_orig_list[:,3:6] = (point_orig_list[:,3:6]+0.5)*255
    savePCD(point_orig_list_path.replace('.npy', '.pcd'), point_orig_list)
    point_orig_list = numpy.hstack((point_orig_list, numpy.reshape(gt_obj_id, [-1, 1]), numpy.reshape(gt_cls_id, [-1, 1])))
    numpy.save(point_orig_list_path, point_orig_list)
    point_orig_list = point_orig_list[:, :6]

predicted_obj_id_path = 'results/%s_area%d_predicted_obj_id.npy' % (net_type, VAL_AREA)
predicted_cls_id_path = 'results/%s_area%d_predicted_cls_id.npy' % (net_type, VAL_AREA)
if os.path.exists(predicted_obj_id_path) and os.path.exists(predicted_cls_id_path):
    predicted_obj_id = numpy.load(predicted_obj_id_path).astype(int)
    predicted_cls_id = numpy.load(predicted_cls_id_path).astype(int)
else:
    print('point_orig_list', point_orig_list.shape)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
#    base_path = '/home/jd/Desktop/learn_region_grow/models'
    base_path = 'models'
    MODEL_PATH = '%s/offline_%s_model%d.ckpt' % (base_path, net_type, VAL_AREA)
    if net_type=='pointnet':
        net = PointNet(1,NUM_POINT,len(classes)) 
    elif net_type=='pointnet2':
        net = PointNet2(1,NUM_POINT,len(classes)) 
    saver = tf.train.Saver()
    saver.restore(sess, MODEL_PATH)
    print('Restored from %s'%MODEL_PATH)

    predicted_cls_id = numpy.zeros(len(point_orig_list), dtype=int)
    grid_resolution = 1.0
    grid = numpy.round(point_orig_list[:,:2]/grid_resolution).astype(int)
    grid_set = set([tuple(g) for g in grid])
    print('Performing inference on %d grid sets...' % len(grid_set))
    for g in grid_set:
        grid_mask = numpy.all(grid==g, axis=1)
        grid_idx = numpy.nonzero(grid_mask)[0]
        num_batches = int(numpy.ceil((1.0 * len(grid_idx) / NUM_POINT)))
        for batch in range(num_batches):
            batch_grid_idx = grid_idx[batch*NUM_POINT : (batch+1)*NUM_POINT]
            grid_points = point_orig_list[batch_grid_idx, :]
            grid_points[:, 3:6] = grid_points[:, 3:6]/255.0 - 0.5
            centroid_xy = numpy.array(g)*grid_resolution
            centroid_z = grid_points[:,2].min()
            grid_points[:,:2] -= centroid_xy
            grid_points[:,2] -= centroid_z
            input_points = numpy.zeros((1, NUM_POINT, 6))
            input_points[0,:len(grid_points),:] = grid_points[:NUM_POINT,:6]
            input_points[0,len(grid_points):,:] = grid_points[0,:6]
            cls, = sess.run([net.output], feed_dict={net.pointclouds_pl: input_points, net.is_training_pl: False})
            cls = cls[0].argmax(axis=1)
            predicted_cls_id[batch_grid_idx] = cls[:len(grid_points)]

    edges = []
    point_voxels = numpy.round(point_orig_list[:,:3]/resolution).astype(int)
    point_id_map = {}
    for i in range(len(point_voxels)):
        k = tuple(point_voxels[i])
        point_id_map[k] = i
    for i in range(len(point_voxels)):
        k = tuple(point_voxels[i])
        for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
            if offset!=(0,0,0):
                kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
                if kk in point_id_map and predicted_cls_id[point_id_map[kk]]==predicted_cls_id[i]:
                    edges.append([i, point_id_map[kk]])

    #calculate connected components from edges
    print('Calculating connected components from %d edges %d vertices...' % (len(edges), len(point_voxels)))
    G = nx.Graph(edges)
    clusters = nx.connected_components(G)
    clusters = [list(c) for c in clusters]
    predicted_obj_id = numpy.zeros(len(point_voxels),dtype=int)
    min_cluster_size = 10
    cluster_id = 1
    for i in range(len(clusters)):
        if len(clusters[i]) > min_cluster_size:
            predicted_obj_id[clusters[i]] = cluster_id
            cluster_id += 1

    numpy.save(predicted_obj_id_path, predicted_obj_id)
    numpy.save(predicted_cls_id_path, predicted_cls_id)

if save_viz:
    obj_color = numpy.random.randint(0,255,(max(gt_obj_id)+1,3))
    obj_color[0] = [100, 100, 100]
    point_orig_list[:,3:6] = obj_color[gt_obj_id, :]
    savePCD('viz/area%d_gt_obj_id.pcd' % VAL_AREA, point_orig_list)
    obj_color = numpy.random.randint(0,255,(max(predicted_obj_id)+1,3))
    obj_color[0] = [100, 100, 100]
    point_orig_list[:,3:6] = obj_color[predicted_obj_id, :]
    savePCD('viz/%s_area%d_predicted_obj_id.pcd' % (net_type, VAL_AREA), point_orig_list)
    point_orig_list[:,3:6] = [class_to_color_rgb[c] for c in gt_cls_id]
    savePCD('viz/area%d_gt_cls_id.pcd' % VAL_AREA, point_orig_list)
    point_orig_list[:,3:6] = [class_to_color_rgb[c] for c in predicted_cls_id]
    savePCD('viz/%s_area%d_predicted_cls_id.pcd' % (net_type, VAL_AREA), point_orig_list)

print('Avg Comp Time: 0')
print('CPU Mem: 0')
print('GPU Mem: 0')
acc, iou, avg_acc, avg_iou, stats = get_cls_id_metrics(gt_cls_id, predicted_cls_id)
nmi, ami, ars, _, _, _, hom, com, vms = get_obj_id_metrics(gt_obj_id, predicted_obj_id)
print("NMI: %.3f AMI: %.3f ARS: %.3f HOM: %.3f COM: %.3f VMS: %.3f %d/%d clusters"% (nmi,ami,ars,hom,com,vms,len(numpy.unique(predicted_obj_id)),len(numpy.unique(gt_obj_id))))
print('all 0 0 0 %.3f 0 %.3f' % (acc, iou))
print('avg 0 0 0 %.3f 0 %.3f' % (avg_acc, avg_iou))
