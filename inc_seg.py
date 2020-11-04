import h5py 
import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
import math
import sys
from architecture import MCPNet, PointNet, PointNet2, VoxNet, SGPN
from class_util import classes, class_to_color_rgb, classes_outdoor, class_to_color_rgb_outdoor
import itertools
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import scipy.stats
import psutil
import copy
from util import get_cls_id_metrics, get_obj_id_metrics

import roslib
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import std_msgs

robot_position = None
def pose_callback(msg):
	global robot_position
	robot_position = [msg.pose.position.x, msg.pose.position.y]

VAL_AREA = '1'
net_type = 'mcpnet'
dataset = 's3dis'
local_range = 2
feature_size = 6
USE_XY = True
for i in range(len(sys.argv)-1):
	if sys.argv[i]=='--area':
		VAL_AREA = sys.argv[i+1]
	if sys.argv[i]=='--net':
		net_type = sys.argv[i+1]
	if sys.argv[i]=='--dataset':
		dataset = sys.argv[i+1]
		if dataset=='outdoor':
			local_range = 10
			feature_size = 3
			classes = classes_outdoor
			class_to_color_rgb = class_to_color_rgb_outdoor
			USE_XY = False
mode = None
if '--color' in sys.argv:
	mode='color'
if '--cluster' in sys.argv:
	mode='cluster'
if '--classify' in sys.argv:
	mode='classify'
if '--boxes' in sys.argv:
	mode='boxes'

NUM_CLASSES = len(classes)
resolution = 0.1
num_neighbors = 50
neighbor_radii = 0.3
batch_size = 256 if net_type.startswith('mcpnet') else 1024
hidden_size = 200
embedding_size = 50
dp_threshold = 0.92 if net_type.startswith('mcpnet') else 0.98

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
obj_color = {0:[100,100,100]}

box_marker = None
text_marker = None
previousMarkers = 0

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

def updateMarkers(idx, pcd, prediction):
	global box_marker
	if box_marker is None:
		box_marker = Marker()
		box_marker.header.frame_id = "/map";
		box_marker.type = Marker.LINE_LIST;
		box_marker.lifetime = rospy.Duration();
		box_marker.color.a = 1.0;
		box_marker.action = Marker.ADD;
		box_marker.scale.x = 0.05;
		box_marker.pose.orientation.w = 1.0;
	marker = copy.copy(box_marker)
	marker.header.stamp = rospy.Time.now();
	marker.id = idx;
	marker.color.r = class_to_color_rgb[prediction][0] / 255.0;
	marker.color.g = class_to_color_rgb[prediction][1] / 255.0;
	marker.color.b = class_to_color_rgb[prediction][2] / 255.0;
	box = numpy.zeros((8,3))
	box[[0,1,4,5],0] = min([p[0] for p in pcd])
	box[[2,3,6,7],0] = max([p[0] for p in pcd])
	box[[0,2,4,6],1] = min([p[1] for p in pcd])
	box[[1,3,5,7],1] = max([p[1] for p in pcd])
	box[[0,1,2,3],2] = min([p[2] for p in pcd])
	box[[4,5,6,7],2] = max([p[2] for p in pcd])
	p1 = Point(*box[0,:])
	p2 = Point(*box[1,:])
	p3 = Point(*box[2,:])
	p4 = Point(*box[3,:])
	p5 = Point(*box[4,:])
	p6 = Point(*box[5,:])
	p7 = Point(*box[6,:])
	p8 = Point(*box[7,:])
	marker.points = [p1,p2,p1,p3,p2,p4,p3,p4,p1,p5,p2,p6,p3,p7,p4,p8,p5,p6,p5,p7,p6,p8,p7,p8]
	pubMarker.publish(marker);

def deleteMarkers(idx):
	marker = Marker()
	marker.header.frame_id = "/map";
	marker.header.stamp = rospy.Time.now();
	marker.id = idx;
	marker.action = Marker.DELETE;
	pubMarker.publish(marker);

def updateText(idx, pcd, prediction):
	global text_marker
	if text_marker is None:
		text_marker = Marker()
		text_marker.header.frame_id = "/map";
		text_marker.type = Marker.TEXT_VIEW_FACING;
		text_marker.action = Marker.ADD;
		text_marker.pose.orientation.w = 1.0;
		text_marker.scale.x = text_marker.scale.y = text_marker.scale.z = 0.5;
		text_marker.color.a = 1.0;
	marker = copy.copy(text_marker)
	marker.id = idx;
	marker.header.stamp = rospy.Time.now();
	marker.pose.position.x = numpy.mean([p[0] for p in pcd])
	marker.pose.position.y = numpy.mean([p[1] for p in pcd])
	marker.pose.position.z = max([p[2] for p in pcd])
	marker.text = classes[prediction];
	marker.color.r = class_to_color_rgb[prediction][0] / 255.0;
	marker.color.g = class_to_color_rgb[prediction][1] / 255.0;
	marker.color.b = class_to_color_rgb[prediction][2] / 255.0;
	pubMarker.publish(marker)

def cloud_surround_callback(cloud):
	global count_msg, obj_count, previousMarkers
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
	#only keep nonzero obj_id
	local_mask = numpy.logical_and(local_mask, pcd[:, 6] > 0)
	pcd = pcd[local_mask, :]
	if len(pcd)==0:
		return
	#shuffle to remove ordering in Z-direction
	numpy.random.shuffle(pcd)
	pcd[:,3:6] = pcd[:,3:6] / 255.0 - 0.5
	original_pcd = pcd.copy()
	pcd[:,:2] -= centroid
	minZ = pcd[:,2].min()
	pcd[:,2] -= minZ

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
			stacked_points = numpy.zeros((len(update_list), (num_neighbors+1)*feature_size))
			stacked_points[:,:feature_size] = numpy.array([p[:feature_size] for p in point_orig_list[-len(update_list):]])
			if USE_XY:
				stacked_points[:, :2] -= robot_position
			else:
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
		cluster_update = []
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
							cluster_update.append(g)
							clusters[group_id].append(g)
						elif predicted_obj_id[g]!=group_id: #merge two clusters
							remove_id = predicted_obj_id[g]
							clusters[group_id].extend(clusters[remove_id])
							for r in clusters[remove_id]:
								predicted_obj_id[r] = group_id
								cluster_update.append(r)
							del clusters[remove_id]
				except ValueError: #all neighbors do not have ids assigned yet
					clusters[obj_count + 1] = []
					for g in group:
						predicted_obj_id[g] = obj_count + 1
						clusters[obj_count + 1].append(g)
						cluster_update.append(g)
					obj_count += 1

	if len(update_list) > 0:
		if mode=='color':
			output_cloud = numpy.array(point_orig_list[-len(update_list):])
			output_cloud[:,3:6] = (output_cloud[:,3:6]+0.5)*255
#			for i in range(len(update_list)):
#				output_cloud[i,3:6] = class_to_color_rgb[predicted_cls_id[-len(update_list)+i]]
			publish_output(output_cloud)
		elif mode=='cluster':
			output_cloud = numpy.array([point_orig_list[i] for i in cluster_update])
			for i in range(len(cluster_update)):
				obj_id = predicted_obj_id[cluster_update[i]]
#				obj_id = gt_obj_id[cluster_update[i]]
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
		elif mode=='boxes':
			currentMarkers = 0
			for i in clusters:
				if len(clusters[i]) > 50:
					pcd = [point_orig_list[c] for c in clusters[i]]
					cls_id = [predicted_cls_id[c] for c in clusters[i]]
					prediction = scipy.stats.mode(cls_id)[0][0]
					if prediction==0:
						continue
					updateMarkers(currentMarkers, pcd, prediction)
					currentMarkers += 1
					updateText(currentMarkers, pcd, prediction)
					currentMarkers += 1
			for i in range(currentMarkers, previousMarkers):
				deleteMarkers(i)
			previousMarkers = currentMarkers


	t = time.time() - t
	comp_time.append(t)
	print('Scan #%3d: %5d/%5d/%7d points acc %.3f time %.3f'%(count_msg, len(update_list), numpy.sum(local_mask), len(point_id_map), acc, t))
	count_msg += 1
	

GPU_INDEX = 0
MODEL_PATH = 'models/%s_model_%s_%s.ckpt'%(net_type, dataset, VAL_AREA)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)

if net_type=='pointnet':
	net = PointNet(batch_size, feature_size, NUM_CLASSES)
elif net_type=='pointnet2':
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
	print('Invalid network type %s'%net_type)
	sys.exit(1)
saver = tf.train.Saver()
saver.restore(sess, MODEL_PATH)
print('Restored network from %s'%MODEL_PATH)

rospy.init_node('inc_seg')
rospy.Subscriber('laser_cloud_surround',PointCloud2,cloud_surround_callback)
rospy.Subscriber('slam_out_pose',PoseStamped,pose_callback)
pubOutput = rospy.Publisher('output_cloud', PointCloud2, queue_size=1)
pubMarker = rospy.Publisher('markers', Marker, queue_size=10)
pubObjects = []
for c in classes:
	p = rospy.Publisher(c, PointCloud2, queue_size=1)
	pubObjects.append(p)
rospy.spin()

#ignore clutter in evaluation
valid_mask = numpy.array(gt_obj_id) > 0
gt_obj_id = numpy.array(gt_obj_id)[valid_mask]
predicted_obj_id = numpy.array(predicted_obj_id)[valid_mask]
gt_cls_id = numpy.array(gt_cls_id)[valid_mask]
predicted_cls_id = numpy.array(predicted_cls_id)[valid_mask]

print("Avg Comp Time: %.3f" % numpy.mean(comp_time))
print("CPU Mem: %.2f" % (psutil.Process(os.getpid()).memory_info()[0] / 1.0e9))
print("GPU Mem: %.1f" % (sess.run(tf.contrib.memory_stats.MaxBytesInUse()) / 1.0e6))
nmi, ami, ars, prc, rcl, iou, hom, com, vms = get_obj_id_metrics(gt_obj_id, predicted_obj_id)
print("NMI: %.3f AMI: %.3f ARS: %.3f PRC: %.3f RCL: %.3f IOU: %.3f HOM: %.3f COM: %.3f VMS: %.3f %d/%d clusters"% (nmi,ami,ars,prc, rcl, iou, hom,com,vms,len(numpy.unique(predicted_obj_id)),len(numpy.unique(gt_obj_id))))
acc, iou, avg_acc, avg_iou, stats = get_cls_id_metrics(gt_cls_id, predicted_cls_id, class_labels=classes, printout=False)
print('all 0 0 0 %.3f 0 %.3f' % (acc, iou))
print('avg 0 0 0 %.3f 0 %.3f' % (avg_acc, avg_iou))
