import h5py
import sys
import numpy
import scipy
from class_util import classes, class_to_color_rgb, classes_gc, class_to_color_rgb_gc
from architecture import MCPNet, PointNet, PointNet2, VoxNet, SGPN
import itertools
import os
import psutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
import random
import math
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import rosbag
from sensor_msgs import point_cloud2
from util import savePCD, get_cls_id_metrics, get_obj_id_metrics

num_neighbors = 50
neighbor_radii = 0.3
feature_size = 6
NUM_CLASSES = len(classes)
resolution = 0.1
local_range = 2
save_viz = False

VAL_AREA = '5'
net_type = 'mcpnet'
dataset = 's3dis'
for i in range(len(sys.argv)):
	if sys.argv[i]=='--area':
		VAL_AREA = sys.argv[i+1]
	if sys.argv[i]=='--net':
		net_type = sys.argv[i+1]
	if sys.argv[i]=='--dataset':
		dataset = sys.argv[i+1]
		if dataset=='guardian_centers':
			local_range = 10
			NUM_CLASSES = 17
			feature_size = 3
			classes = classes_gc
			class_to_color_rgb = class_to_color_rgb_gc
	if sys.argv[i]=='--save':          
		save_viz = True

batch_size = 256 if net_type.startswith('mcpnet') else 1024
hidden_size = 200
embedding_size = 50
samples_per_instance = 16
sample_state = numpy.random.RandomState(0)
count_msg = 0
obj_count = 0
comp_time = []
point_id_map = {}
coarse_map = {}
point_orig_list = []
obj_id_list = []
cls_id_list = []
embedding_list = []
view_list = []
view_idx = []
agg_points = []
agg_obj_id = []
agg_cls_id = []
agg_count = []
normal_threshold = 0.9 #0.8
color_threshold = 0.005 #0.01
ed_threshold = 0.1
dp_threshold = 0.92 if net_type.startswith('mcpnet') else 0.98
accA = {}
accB = {}
accN = {}
normals = []
predicted_obj_id = []
predicted_cls_id = []
clusters = {}
USE_NORMAL = False

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
if net_type=='pointnet':
	net = PointNet(batch_size, feature_size, NUM_CLASSES)
if net_type=='pointnet2':
	net = PointNet2(batch_size, feature_size, NUM_CLASSES)
elif net_type=='voxnet':
	net = VoxNet(batch_size, feature_size, NUM_CLASSES)
elif net_type=='sgpn':
	net = SGPN(batch_size, feature_size, NUM_CLASSES)
elif net_type=='mcpnet_simple':
	num_neighbors = 0
	net = MCPNet(batch_size, num_neighbors, feature_size, hidden_size, embedding_size, NUM_CLASSES)
elif net_type=='mcpnet':
	net = MCPNet(batch_size, num_neighbors, feature_size, hidden_size, embedding_size, NUM_CLASSES)
saver = tf.train.Saver()
MODEL_PATH = 'models/%s_model_%s_%s.ckpt'%(net_type, dataset, VAL_AREA)
saver.restore(sess, MODEL_PATH)
print('Restored network from %s'%MODEL_PATH)

def process_cloud(cloud, robot_position):
	global count_msg, obj_count
	t = time.time()
	pcd = []
	for p in point_cloud2.read_points(cloud, field_names=("x","y","z","r","g","b","o","c"), skip_nans=True):
		pcd.append(p)
	pcd = numpy.array(pcd)
	local_mask = numpy.sum((pcd[:,:2]-robot_position)**2, axis=1) < local_range * local_range
	#only keep nonzero obj_id
	local_mask = numpy.logical_and(local_mask, pcd[:, 6] > 0)
	pcd = pcd[local_mask, :]
	#shuffle to remove ordering in Z-direction
	numpy.random.shuffle(pcd)
	pcd[:,3:6] = pcd[:,3:6] / 255.0 - 0.5
	original_pcd = pcd.copy()
	pcd[:,:2] -= robot_position
	minZ = pcd[:,2].min()
	pcd[:,2] -= minZ

	acc = 0.0
	pcdi = [tuple(p) for p in (original_pcd[:,:3]/resolution).round().astype(int)]
	update_list = []
	for i in range(len(pcdi)):
		if not pcdi[i] in point_id_map:
			point_id_map[pcdi[i]] = len(point_orig_list)
			point_orig_list.append(original_pcd[i,:6].copy())
			obj_id_list.append(int(original_pcd[i,6]))
			cls_id_list.append(int(original_pcd[i,7]))
			update_list.append(pcdi[i])
			normals.append(None)
			predicted_obj_id.append(0)
	
	if len(update_list)>0:
		if USE_NORMAL:
			for k in update_list:
				p = point_orig_list[point_id_map[k]][:3]
				for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
					kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
					if not kk in accN:
						accA[kk] = numpy.zeros((3,3))
						accB[kk] = numpy.zeros(3)
						accN[kk] = 0
					accA[kk] += numpy.outer(p,p)
					accB[kk] += p
					accN[kk] += 1
			
			for k in update_list:
				cov = accA[k] / accN[k] - numpy.outer(accB[k], accB[k]) / accN[k]**2
				U,S,V = numpy.linalg.svd(cov)
				normals[point_id_map[k]] = numpy.fabs(V[2])

		for k in update_list:
			idx = point_id_map[k]
			kk = tuple((point_orig_list[idx][:3]/neighbor_radii).round().astype(int))
			if not kk in coarse_map:
				coarse_map[kk] = []
			coarse_map[kk].append(idx)
		
		if net_type=='mcpnet' and num_neighbors>0:
			stacked_points = numpy.zeros((len(update_list), (num_neighbors+1)*feature_size))
			stacked_points[:,:feature_size] = numpy.array([p[:feature_size] for p in point_orig_list[-len(update_list):]])
			stacked_points[:, :2] = 0
			stacked_points[:,2] -= minZ
			for i in range(len(update_list)):
				idx = point_id_map[update_list[i]]
				p = point_orig_list[idx][:feature_size]
				k = tuple((p[:3]/neighbor_radii).round().astype(int))
				neighbors = []
				for offset in itertools.product(range(-1,2),range(-1,2),range(-1,2)):
					kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
					if kk in coarse_map:
						neighbors.extend(coarse_map[kk])
				neighbors = sample_state.choice(neighbors, num_neighbors, replace=len(neighbors)<num_neighbors)
				neighbors = numpy.array([point_orig_list[n][:feature_size] for n in neighbors])
				neighbors -= p
				stacked_points[i,feature_size:] = neighbors.reshape(1, num_neighbors*feature_size)
		else:
			stacked_points = numpy.array([p[:feature_size] for p in point_orig_list[-len(update_list):]])
			stacked_points[:,:2] -= robot_position
			stacked_points[:,2] -= minZ

		num_batches = int(math.ceil(1.0 * len(stacked_points) / batch_size))
		input_points = numpy.zeros((batch_size, stacked_points.shape[1]))
		for batch_id in range(num_batches):
			start_idx = batch_id * batch_size
			end_idx = (batch_id + 1) * batch_size
			valid_idx = min(len(stacked_points), end_idx)
			if end_idx <= len(stacked_points):
				input_points[:valid_idx-start_idx] = stacked_points[start_idx:valid_idx,:]
			else:
				input_points[:valid_idx-start_idx] = stacked_points[start_idx:valid_idx,:]
				padding_samples = sample_state.choice(range(len(stacked_points)), end_idx-valid_idx, replace=True)
				input_points[valid_idx-end_idx:] = stacked_points[padding_samples, :]
			if net_type in ['sgpn','mcpnet','mcpnet_simple']:
				emb_val, cls_val = sess.run([net.embeddings, net.class_output], {net.input_pl:input_points, net.is_training_pl:False})
				embedding_list.extend(emb_val[:valid_idx-start_idx])
			else:
				cls_val = sess.run(net.class_output, {net.input_pl:input_points, net.is_training_pl:False})
			predicted_cls_id.extend(numpy.argmax(cls_val[:valid_idx-start_idx],axis=1))
		acc = 1.0 * numpy.sum(numpy.equal(predicted_cls_id[-len(stacked_points):], cls_id_list[-len(stacked_points):])) / len(stacked_points)

		neighbor_key = []
		neighbor_probs = []
		for k in update_list:
			nk = [point_id_map[k]]
			l = 0
			if USE_NORMAL:
				for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
					if offset!=(0,0,0):
						kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
						if kk in point_id_map and \
							normals[point_id_map[kk]].dot(normals[point_id_map[k]]) > normal_threshold and \
							numpy.sum((point_orig_list[point_id_map[k]][3:6]-point_orig_list[point_id_map[kk]][3:6])**2) < color_threshold:
							nk.append(point_id_map[kk])
			elif net_type in ['sgpn','mcpnet','mcpnet_simple']:
				for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
					if offset!=(0,0,0):
						kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
						if kk in point_id_map:
							dot_prod = numpy.dot(embedding_list[point_id_map[k]], embedding_list[point_id_map[kk]])
							if dot_prod > dp_threshold:
								nk.append(point_id_map[kk])
			else:
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
	print('Scan #%3d: cur:%4d/%3d agg:%5d/%3d acc %.2f time %.3f'%(count_msg, len(update_list), len(set(obj_id_list[len(obj_id_list)-len(update_list):])), len(point_id_map), len(set(obj_id_list)), acc, t))
	count_msg += 1

bag = rosbag.Bag('data/%s_%s.bag' % (dataset, VAL_AREA), 'r')
poses = []
for topic, msg, t in bag.read_messages(topics=['slam_out_pose']):
	poses.append([msg.pose.position.x, msg.pose.position.y])
i = 0
for topic, msg, t in bag.read_messages(topics=['laser_cloud_surround']):
	process_cloud(msg, poses[i])
	i += 1
#	if i==1:
#		break

#ignore clutter in evaluation
valid_mask = numpy.array(obj_id_list) > 0
gt_obj_id = numpy.array(obj_id_list)[valid_mask]
predicted_obj_id = numpy.array(predicted_obj_id)[valid_mask]
gt_cls_id = numpy.array(cls_id_list)[valid_mask]
predicted_cls_id = numpy.array(predicted_cls_id)[valid_mask]
point_orig_list = numpy.array(point_orig_list)[valid_mask]
if len(embedding_list) > 0:
	embedding_list = numpy.array(embedding_list)[valid_mask]

if save_viz:
	obj_color = numpy.random.randint(0,255,(max(gt_obj_id)+1,3))
	obj_color[0] = [100, 100, 100]
	point_orig_list[:,3:6] = obj_color[gt_obj_id, :]
	savePCD('viz/area%s_gt_obj_id.pcd' % VAL_AREA, point_orig_list)
	obj_color = numpy.random.randint(0,255,(max(predicted_obj_id)+1,3))
	obj_color[0] = [100, 100, 100]
	point_orig_list[:,3:6] = obj_color[predicted_obj_id, :]
	valid_mask = numpy.ones(len(predicted_obj_id), dtype=bool)
	for i in set(predicted_obj_id):
		cluster_mask = predicted_obj_id==i
		if numpy.sum(cluster_mask) < 10:
			valid_mask[cluster_mask] = False
	savePCD('viz/%s_area%s_predicted_obj_id.pcd' % (net_type, VAL_AREA), point_orig_list[valid_mask])
	point_orig_list[:,3:6] = [class_to_color_rgb[c] for c in gt_cls_id]
	savePCD('viz/area%s_gt_cls_id.pcd' % VAL_AREA, point_orig_list)
	point_orig_list[:,3:6] = [class_to_color_rgb[c] for c in predicted_cls_id]
	savePCD('viz/%s_area%s_predicted_cls_id.pcd' % (net_type, VAL_AREA), point_orig_list)
	if len(embedding_list) > 0:
		X_embedded = PCA(n_components=3).fit_transform(embedding_list)
		obj_color = (X_embedded - X_embedded.min(axis=0)) / (X_embedded.max(axis=0) - X_embedded.min(axis=0)) * 255
		point_orig_list[:,3:6] = obj_color
		savePCD('viz/%s_area%s_embedding.pcd' % (net_type, VAL_AREA), point_orig_list)

print("Avg Comp Time: %.3f" % numpy.mean(comp_time))
print("CPU Mem: %.2f" % (psutil.Process(os.getpid()).memory_info()[0] / 1.0e9))
print("GPU Mem: %.1f" % (sess.run(tf.contrib.memory_stats.MaxBytesInUse()) / 1.0e6))
nmi, ami, ars, prc, rcl, iou, hom, com, vms = get_obj_id_metrics(gt_obj_id, predicted_obj_id)
print("NMI: %.3f AMI: %.3f ARS: %.3f PRC: %.3f RCL: %.3f IOU: %.3f HOM: %.3f COM: %.3f VMS: %.3f %d/%d clusters"% (nmi,ami,ars,prc, rcl, iou, hom,com,vms,len(numpy.unique(predicted_obj_id)),len(numpy.unique(gt_obj_id))))
acc, iou, avg_acc, avg_iou, stats = get_cls_id_metrics(gt_cls_id, predicted_cls_id, class_labels=classes, printout=False)
print('all 0 0 0 %.3f 0 %.3f' % (acc, iou))
print('avg 0 0 0 %.3f 0 %.3f' % (avg_acc, avg_iou))
