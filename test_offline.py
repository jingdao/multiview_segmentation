import h5py 
import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
import sys
from class_util import classes
from offline_architecture import PointNet, PointNet2
from util import get_cls_id_metrics, get_obj_id_metrics
import networkx as nx
import itertools

NUM_POINT = 2048
NUM_CLASSES = len(classes)
resolution = 0.1

def loadFromH5(filename, load_labels=True):
	f = h5py.File(filename,'r')
	all_points = f['points'][:]
	count_room = f['count_room'][:]
	tmp_points = []
	idp = 0
	for i in range(len(count_room)):
		tmp_points.append(all_points[idp:idp+count_room[i], :])
		idp += count_room[i]
	f.close()
	room = []
	labels = []
	class_labels = []
	if load_labels:
		for i in range(len(tmp_points)):
			room.append(tmp_points[i][:,:-2])
			labels.append(tmp_points[i][:,-2].astype(int))
			class_labels.append(tmp_points[i][:,-1].astype(int))
		return room, labels, class_labels
	else:
		return tmp_points

VAL_AREA = 1
net_type = 'pointnet'
for i in range(len(sys.argv)):
    if sys.argv[i] == '--net':
        net_type = sys.argv[i+1]
    if sys.argv[i]=='--area':
        VAL_AREA = int(sys.argv[i+1])

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

gt_cls_id_arr = []
predicted_cls_id_arr = []
gt_obj_id_arr = []
predicted_obj_id_arr = []

all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%d.h5' % VAL_AREA)
for room_id in range(len(all_points)):
    points = all_points[room_id]
    gt_cls_id_arr.extend(all_cls_id[room_id])
    gt_obj_id_arr.extend(all_obj_id[room_id])

    predicted_cls_id = numpy.zeros(len(points), dtype=int)
    grid_resolution = 1.0
    grid = numpy.round(points[:,:2]/grid_resolution).astype(int)
    grid_set = set([tuple(g) for g in grid])
    print('Room %d: performing inference on %d grid sets...' % (room_id, len(grid_set)))
    for g in grid_set:
        grid_mask = numpy.all(grid==g, axis=1)
        grid_idx = numpy.nonzero(grid_mask)[0]
        num_batches = int(numpy.ceil((1.0 * len(grid_idx) / NUM_POINT)))
        for batch in range(num_batches):
            batch_grid_idx = grid_idx[batch*NUM_POINT : (batch+1)*NUM_POINT]
            grid_points = points[batch_grid_idx, :]
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
    point_voxels = numpy.round(points[:,:3]/resolution).astype(int)
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

    predicted_cls_id_arr.extend(predicted_cls_id)
    predicted_obj_id_arr.extend(predicted_obj_id)

print('Avg Comp Time: 0')
print('CPU Mem: 0')
print('GPU Mem: 0')
acc, iou, avg_acc, avg_iou, stats = get_cls_id_metrics(numpy.array(gt_cls_id_arr), numpy.array(predicted_cls_id_arr))
nmi, ami, ars, _, _, _, hom, com, vms = get_obj_id_metrics(numpy.array(gt_obj_id_arr), numpy.array(predicted_obj_id_arr))
print("NMI: %.3f AMI: %.3f ARS: %.3f HOM: %.3f COM: %.3f VMS: %.3f %d/%d clusters"% (nmi,ami,ars,hom,com,vms,len(numpy.unique(predicted_obj_id_arr)),len(numpy.unique(gt_obj_id_arr))))
print('all 0 0 0 %.3f 0 %.3f' % (acc, iou))
print('avg 0 0 0 %.3f 0 %.3f' % (avg_acc, avg_iou))

