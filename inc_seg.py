import h5py 
import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
import math
import sys
from architecture import MCPNet, PointNet, PointNet2, VoxNet, SGPN
from class_util import classes, class_to_id, class_to_color_rgb
import itertools
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import scipy.stats
import psutil

import roslib
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PoseStamped
import std_msgs

robot_position = None
def pose_callback(msg):
	global robot_position
	robot_position = [msg.pose.position.x, msg.pose.position.y]

VAL_AREA = 1
net_type = 'mcpnet'
for i in range(len(sys.argv)-1):
	if sys.argv[i]=='--area':
		VAL_AREA = int(sys.argv[i+1])
	if sys.argv[i]=='--net':
		net_type = sys.argv[i+1]
mode = None
if '--color' in sys.argv:
	mode='color'
if '--cluster' in sys.argv:
	mode='cluster'
if '--classify' in sys.argv:
	mode='classify'

local_range = 2
resolution = 0.1
num_neighbors = 50
neighbor_radii = 0.3
batch_size = 256 if net_type.startswith('mcpnet') else 1024
hidden_size = 200
embedding_size = 50
dp_threshold = 0.9 if net_type.startswith('mcpnet') else 0.98
feature_size = 6
NUM_CLASSES = len(classes)

point_id_map = {}
coarse_map = {}
point_orig_list = []
gt_obj_id = []
gt_cls_id = []
predicted_obj_id = []
predicted_cls_id = []
embedding_list = []

count_msg = 0
clusters = {}
obj_count = 0
comp_time = []
numpy.random.seed(0)
sample_state = numpy.random.RandomState(0)
obj_color = {}

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

def cloud_surround_callback(cloud):
	global count_msg, obj_count
	t = time.time()
	pcd = []
	for p in point_cloud2.read_points(cloud, field_names=("x","y","z","r","g","b","o","c"), skip_nans=True):
		pcd.append(p)

	if robot_position is None:
		centroid = numpy.median(pcd[:,:2],axis=0)
	else:
		centroid = robot_position
	pcd = numpy.array(pcd)
	local_mask = numpy.sum((pcd[:,:2]-centroid)**2, axis=1) < local_range * local_range
	pcd = pcd[local_mask, :]
	pcd[:,3:6] = pcd[:,3:6] / 255.0 - 0.5
	original_pcd = pcd.copy()
	pcd[:,:2] -= centroid
	pcd[:,2] -= pcd[:,2].min()

	global_objects = [[] for i in range(len(classes))]
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
			predicted_obj_id.append(0)

	if len(update_list)>0:
		for k in update_list:
			idx = point_id_map[k]
			kk = tuple((point_orig_list[idx][:3]/neighbor_radii).round().astype(int))
			if not kk in coarse_map:
				coarse_map[kk] = []
			coarse_map[kk].append(idx)
		
		if net_type=='mcpnet' and num_neighbors>0:
			stacked_points = numpy.zeros((len(update_list), (num_neighbors+1)*6))
			stacked_points[:,:6] = numpy.array(point_orig_list[-len(update_list):])
			stacked_points[:,:2] -= robot_position
			for i in range(len(update_list)):
				idx = point_id_map[update_list[i]]
				p = point_orig_list[idx][:6]
				k = tuple((p[:3]/neighbor_radii).round().astype(int))
				neighbors = []
				for offset in itertools.product(range(-1,2),range(-1,2),range(-1,2)):
					kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
					if kk in coarse_map:
						neighbors.extend(coarse_map[kk])
				neighbors = sample_state.choice(neighbors, num_neighbors, replace=len(neighbors)<num_neighbors)
				neighbors = numpy.array([point_orig_list[n][:6] for n in neighbors])
				neighbors -= p
				stacked_points[i,6:] = neighbors.reshape(1, num_neighbors*6)
		else:
			stacked_points = numpy.array(point_orig_list[-len(update_list):])
			stacked_points[:,:2] -= robot_position

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
				input_points[valid_idx-end_idx:] = stacked_points[sample_state.choice(range(len(stacked_points)), end_idx-valid_idx, replace=True),:]
			if net_type in ['sgpn','mcpnet','mcpnet_simple']:
				emb_val, cls_val = sess.run([net.embeddings, net.class_output], {net.input_pl:input_points, net.is_training_pl:False})
				embedding_list.extend(emb_val[:valid_idx-start_idx])
			else:
				cls_val = sess.run(net.class_output, {net.input_pl:input_points, net.is_training_pl:False})
			predicted_cls_id.extend(numpy.argmax(cls_val[:valid_idx-start_idx],axis=1))
		acc = 1.0 * numpy.sum(numpy.equal(predicted_cls_id[-len(stacked_points):], gt_cls_id[-len(stacked_points):])) / len(stacked_points)
		for k in update_list:
			idx = point_id_map[k]
			global_objects[predicted_cls_id[idx]].append(point_orig_list[idx][:3])

		neighbor_key = []
		neighbor_probs = []
		for k in update_list:
			nk = [point_id_map[k]]
			if net_type in ['sgpn','mcpnet','mcpnet_simple']:
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

	if len(update_list) > 0:
		if mode=='color':
			output_cloud = numpy.array(point_orig_list[-len(update_list):])
			output_cloud[:,3:6] = (output_cloud[:,3:6]+0.5)*255
			publish_output(output_cloud)
		elif mode=='cluster':
#			output_cloud = numpy.array([global_features[point_id_map[k]] for k in point_id_map])
#			mask = numpy.ones(len(dt_id),dtype=bool)
#			for c in clusters:
#				if len(clusters[c]) < 10:
#					mask[clusters[c]] = False
#			output_cloud[:,3:6] = obj_color[[dt_id[point_id_map[k]] for k in point_id_map],:]
#			mask = [mask[point_id_map[k]] for k in point_id_map]
#			output_cloud = output_cloud[mask, :]
#			publish_output(output_cloud)
			output_cloud = numpy.array(point_orig_list[-len(update_list):])
			for i in range(len(update_list)):
				obj_id = predicted_obj_id[-len(update_list)+i]
				if not obj_id in obj_color:
					obj_color[obj_id] = numpy.random.randint(0,255,3)
				output_cloud[i,3:6] = obj_color[obj_id]
			publish_output(output_cloud)
		elif mode=='classify':
			for i in range(len(classes)):
				if len(global_objects[i])>0:
					header = std_msgs.msg.Header()
					header.stamp = rospy.Time.now()
					header.frame_id = '/map'
					msg = point_cloud2.create_cloud_xyz32(header,global_objects[i])
					pubObjects[i].publish(msg)

	t = time.time() - t
	comp_time.append(t)
	print('Scan #%3d: %5d/%5d/%7d points acc %.3f time %.3f'%(count_msg, len(update_list), numpy.sum(local_mask), len(point_id_map), acc, t))
	count_msg += 1
	

GPU_INDEX = 0
MODEL_PATH = 'models/%s_model%d.ckpt'%(net_type, VAL_AREA)
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
else:
	print('Invalid network type')
	sys.exit(1)
saver = tf.train.Saver()
saver.restore(sess, MODEL_PATH)
print('Restored network from %s'%MODEL_PATH)

rospy.init_node('inc_seg')
rospy.Subscriber('laser_cloud_surround',PointCloud2,cloud_surround_callback)
rospy.Subscriber('slam_out_pose',PoseStamped,pose_callback)
pubOutput = rospy.Publisher('output_cloud', PointCloud2, queue_size=1)
pubObjects = []
for c in classes:
	p = rospy.Publisher(c, PointCloud2, queue_size=1)
	pubObjects.append(p)
rospy.spin()

#calculate accuracy
def calculate_accuracy():
	stats = {}
	stats['all'] = {'tp':0, 'fp':0, 'fn':0} 
	for c in classes:
		stats[c] = {'tp':0, 'fp':0, 'fn':0} 
	for g in range(len(predicted_cls_id)):
		if gt_cls_id[g] == predicted_cls_id[g]:
			stats[classes[int(gt_cls_id[g])]]['tp'] += 1
			stats['all']['tp'] += 1
		else:
			stats[classes[int(gt_cls_id[g])]]['fn'] += 1
			stats['all']['fn'] += 1
			stats[classes[predicted_cls_id[g]]]['fp'] += 1
			stats['all']['fp'] += 1

	prec_agg = []
	recl_agg = []
	iou_agg = []
	print("%10s %6s %6s %6s %5s %5s %5s"%('CLASS','TP','FP','FN','PREC','RECL','IOU'))
	for c in sorted(stats.keys()):
		try:
			stats[c]['pr'] = 1.0 * stats[c]['tp'] / (stats[c]['tp'] + stats[c]['fp'])
		except ZeroDivisionError:
			stats[c]['pr'] = 0
		try:
			stats[c]['rc'] = 1.0 * stats[c]['tp'] / (stats[c]['tp'] + stats[c]['fn'])
		except ZeroDivisionError:
			stats[c]['rc'] = 0
		try:
			stats[c]['IOU'] = 1.0 * stats[c]['tp'] / (stats[c]['tp'] + stats[c]['fp'] + stats[c]['fn'])
		except ZeroDivisionError:
			stats[c]['IOU'] = 0
		if c not in ['all']:
			print("%10s %6d %6d %6d %5.3f %5.3f %5.3f"%(c,stats[c]['tp'],stats[c]['fp'],stats[c]['fn'],stats[c]['pr'],stats[c]['rc'],stats[c]['IOU']))
			prec_agg.append(stats[c]['pr'])
			recl_agg.append(stats[c]['rc'])
			iou_agg.append(stats[c]['IOU'])
	c = 'all'
	print("%10s %6d %6d %6d %5.3f %5.3f %5.3f"%('all',stats[c]['tp'],stats[c]['fp'],stats[c]['fn'],stats[c]['pr'],stats[c]['rc'],stats[c]['IOU']))
	print("%10s %6d %6d %6d %5.3f %5.3f %5.3f"%('avg',stats[c]['tp'],stats[c]['fp'],stats[c]['fn'],numpy.mean(prec_agg),numpy.mean(recl_agg),numpy.mean(iou_agg)))

					
print("Avg Comp Time: %.3f" % numpy.mean(comp_time))
print("CPU Mem: %.2f" % (psutil.Process(os.getpid()).get_memory_info()[0] / 1.0e9))
#sys.exit(1)
print("Computing stats, please wait ...")
nmi = normalized_mutual_info_score(gt_obj_id, predicted_obj_id)
ami = adjusted_mutual_info_score(gt_obj_id, predicted_obj_id)
ars = adjusted_rand_score(gt_obj_id, predicted_obj_id)
print("NMI: %.3f AMI: %.3f ARS: %.3f %d/%d clusters"% (nmi,ami,ars,len(numpy.unique(predicted_obj_id)),len(numpy.unique(gt_obj_id))))
calculate_accuracy()

