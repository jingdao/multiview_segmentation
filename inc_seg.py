import h5py 
import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
import math
import sys
from pointnet_seg import PointNet
from raxnet import RaxNet
from class_util import classes, class_to_id, class_to_color_rgb
import itertools
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import scipy.stats

import roslib
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PoseStamped
import std_msgs
import rosbag

robot_position = None
def pose_callback(msg):
	global robot_position
	robot_position = [msg.pose.position.x, msg.pose.position.y]

VAL_AREA = 1
for i in range(len(sys.argv)-1):
	if sys.argv[i]=='--area':
		VAL_AREA = int(sys.argv[i+1])
mode = None
if '--color' in sys.argv:
	mode='color'
if '--instance' in sys.argv:
	mode='instance'
if '--semantic' in sys.argv:
	mode='semantic'
local_range = 2
resolution = 0.1
cls_threshold = 0.5 #0.5
normal_threshold = 0.8 #0.8
color_threshold = 0.01 #0.01
point_feature_map = {}
point_conf_map = {}
point_id_map = {}
view_id_map = {}
view_features = []
global_features = []
count_msg = 0
gt_labels = []
class_labels = []
class_probs = []
gt_id = []
dt_id = []
accA = {}
accB = {}
accN = {}
normals = []
clusters = {}
obj_count = 0
comp_time = []
numpy.random.seed(0)
obj_color = numpy.random.randint(0,255,(100000,3))
obj_color[0,:] = [200,200,200]

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
	pcd = numpy.array(pcd)
	original_pcd = pcd.copy()
	if robot_position is None:
		centroid = numpy.median(pcd[:,:2],axis=0)
	else:
		centroid = robot_position
	pcd[:,:2] -= centroid
	pcd[:,2] -= pcd[:,2].min()
	pcd[:,3:6] = pcd[:,3:6] / 255.0 - 0.5
	original_pcd[:,3:6] = original_pcd[:,3:6] / 255.0 - 0.5
	local_mask = numpy.sum(pcd[:,:2]**2, axis=1) < local_range * local_range
	pcd = pcd[local_mask, :]
	original_pcd = original_pcd[local_mask, :]
	if len(pcd) < NUM_POINT:
		return
	subset = numpy.random.choice(len(pcd), NUM_POINT, replace=False)
	pcd = pcd[subset, :]
	original_pcd = original_pcd[subset, :]
	order = numpy.argsort(pcd[:,2])
	input_channels = 6 if USE_RGB else 3
	pcd =  pcd[order, :input_channels].reshape((1,NUM_POINT,input_channels))
	original_pcd =  original_pcd[order, :]

	global_objects = [[] for i in range(len(classes))]
	update_list = {}
	acc = 0.0
	feed_dict = {net.pointclouds_pl: pcd, net.is_training_pl: False}
	ft, = sess.run([net.pool[-1]], feed_dict=feed_dict)
	view_features.append(ft.flatten())

	pcdi = [tuple(p) for p in (original_pcd[:,:3]/resolution).astype(int)]
	rft = numpy.zeros((NUM_POINT, POOL_SIZE, input_channels))
	vft = numpy.zeros((NUM_POINT, POOL_SIZE, ft.shape[-1]))
	for i in range(NUM_POINT):
		if not pcdi[i] in point_id_map:
			point_id_map[pcdi[i]] = len(point_id_map)
			gt_labels.append(original_pcd[i,-1])
			global_features.append(original_pcd[i,:6])
			class_labels.append(0)
			class_probs.append(numpy.zeros(len(classes)))
			gt_id.append(int(original_pcd[i,6]))
			dt_id.append(0)
			normals.append(numpy.zeros(3))
			update_list[pcdi[i]] = original_pcd[i,:3]
			point_feature_map[pcdi[i]] = [pcd[0,i,:input_channels]]
			view_id_map[pcdi[i]] = [len(view_features)-1]
		else:
			if len(point_feature_map[pcdi[i]]) < POOL_SIZE:
				point_feature_map[pcdi[i]].append(pcd[0,i,:input_channels])
				view_id_map[pcdi[i]].append(len(view_features)-1)
			else:
				k = numpy.random.randint(POOL_SIZE + 1)
				if k < POOL_SIZE:
					point_feature_map[pcdi[i]][k] = pcd[0,i,:input_channels]
					view_id_map[pcdi[i]][k] = len(view_features)-1
		N = len(point_feature_map[pcdi[i]])
		samples = range(N) + list(numpy.random.choice(N, POOL_SIZE-N, replace=True))
		rft[i,:,:] = numpy.array(point_feature_map[pcdi[i]])[samples, :]
		for j in range(POOL_SIZE):
			vft[i,j,:] = view_features[view_id_map[pcdi[i]][samples[j]]]

	cls = []
	conf = []
	for i in range(NUM_POINT / BATCH_SIZE2):
		output = sess.run(rnet.output, {
			rnet.point_input_pl:rft[i*BATCH_SIZE2:(i+1)*BATCH_SIZE2, :, :],
			rnet.view_input_pl:vft[i*BATCH_SIZE2:(i+1)*BATCH_SIZE2, :, :]
		})
		conf.extend(output)
		cls.extend(numpy.argmax(output, -1))
	conf = numpy.array(conf)
	conf = conf - conf.max(axis=1).reshape(-1,1)
	conf = numpy.exp(conf) / numpy.sum(numpy.exp(conf),axis=1).reshape(-1,1)
	acc = 1.0 * numpy.sum(cls == original_pcd[:,-1]) / NUM_POINT
	for i in range(NUM_POINT):
		global_objects[cls[i]].append(original_pcd[i,:3])
		class_labels[point_id_map[pcdi[i]]] = cls[i]
		class_probs[point_id_map[pcdi[i]]] = conf[i,:]
		point_conf_map[pcdi[i]] = numpy.max(conf[i])

	if len(update_list) > 0:
		for k in update_list:
			for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
				kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
				if not kk in accN:
					accA[kk] = numpy.zeros((3,3))
					accB[kk] = numpy.zeros(3)
					accN[kk] = 0
				p = update_list[k]
				accA[kk] += numpy.outer(p,p)
				accB[kk] += p
				accN[kk] += 1
		
		for k in update_list:
			cov = accA[k] / accN[k] - numpy.outer(accB[k], accB[k]) / accN[k]**2
			U,S,V = numpy.linalg.svd(cov)
			normals[point_id_map[k]] = numpy.fabs(V[2])

		neighbor_key = []
		neighbor_probs = []
		for k in update_list:
			nk = [point_id_map[k]]
			l = 0
			for offset in itertools.product([-2,-1,0,1,2],[-2,-1,0,1,2],[-2,-1,0,1,2]):
				if offset!=(0,0,0):
					kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
					if kk in point_id_map and \
						normals[point_id_map[kk]].dot(normals[point_id_map[k]]) > normal_threshold and \
						numpy.sum((global_features[point_id_map[k]][3:6]-global_features[point_id_map[kk]][3:6])**2) < color_threshold and \
						class_probs[point_id_map[kk]].dot(class_probs[point_id_map[k]]) > cls_threshold:
						nk.append(point_id_map[kk])
					l += 1
			if len(nk) > 1:
				neighbor_key.append(nk)
				neighbor_probs.append([True] * (len(nk)-1))
			else:
				dt_id[point_id_map[k]] = obj_count + 1
				clusters[obj_count + 1] = [point_id_map[k]]
				obj_count += 1

		for i in range(len(neighbor_probs)):
			if not numpy.any(neighbor_probs[i]): #isolated point
				dt_id[neighbor_key[i][0]] = obj_count + 1
				clusters[obj_count + 1] = [neighbor_key[i][0]]
				obj_count += 1
			else:
				group = [neighbor_key[i][0]] + [neighbor_key[i][j+1] for j in numpy.nonzero(neighbor_probs[i])[0]]
				try:
					group_id = min([dt_id[g] for g in group if dt_id[g] > 0])
					for g in group:
						if dt_id[g]==0: #add to existing cluster
							dt_id[g] = group_id
							clusters[group_id].append(g)
						elif dt_id[g]!=group_id: #merge two clusters
							remove_id = dt_id[g]
							clusters[group_id].extend(clusters[remove_id])
							for r in clusters[remove_id]:
								dt_id[r] = group_id
							del clusters[remove_id]
				except ValueError: #all neighbors do not have ids assigned yet
					clusters[obj_count + 1] = []
					for g in group:
						dt_id[g] = obj_count + 1
						clusters[obj_count + 1].append(g)
					obj_count += 1

	if len(update_list) > 0:
		if mode=='color':
			output_cloud = numpy.array([global_features[point_id_map[k]] for k in point_id_map])
			output_cloud[:,3:6] = (output_cloud[:,3:6]+0.5)*255
			publish_output(output_cloud)
		elif mode=='instance':
			output_cloud = numpy.array([global_features[point_id_map[k]] for k in point_id_map])
			mask = numpy.ones(len(dt_id),dtype=bool)
			for c in clusters:
				if len(clusters[c]) < 10:
					mask[clusters[c]] = False
			output_cloud[:,3:6] = obj_color[[dt_id[point_id_map[k]] for k in point_id_map],:]
			mask = [mask[point_id_map[k]] for k in point_id_map]
			output_cloud = output_cloud[mask, :]
			publish_output(output_cloud)
		elif mode=='semantic':
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
	

BATCH_SIZE = 1
GPU_INDEX = 0
MODEL_PATH = 'models/pointnet/model%d.ckpt' % VAL_AREA
MODEL_PATH2 = 'models/multiview/model%d.ckpt' % VAL_AREA
NUM_CLASSES = len(classes)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
np = tf.get_variable('np',[],dtype=tf.int32)
nr = tf.get_variable('nr',[],dtype=tf.bool)
saver = tf.train.Saver([np,nr])
saver.restore(sess, MODEL_PATH)
NUM_POINT, USE_RGB = sess.run([np,nr])
print("Creating PointNet with %d points %d classes"%(NUM_POINT, NUM_CLASSES))
net = PointNet(BATCH_SIZE, NUM_POINT, NUM_CLASSES, use_rgb = USE_RGB)
saver = tf.train.Saver()
saver.restore(sess, MODEL_PATH)
ps = tf.get_variable('ps',[],dtype=tf.int32)
saver = tf.train.Saver([ps])
saver.restore(sess, MODEL_PATH2)
POOL_SIZE = sess.run(ps)
BATCH_SIZE2 = 1024
rnet = RaxNet(BATCH_SIZE2, POOL_SIZE, 6 if USE_RGB else 3, 512, NUM_CLASSES)
print("Creating RaxNet with %d classes"%(NUM_CLASSES))
saver = tf.train.Saver(rnet.pkernel +  rnet.pbias +  rnet.vkernel +  rnet.vbias +  rnet.rkernel +  rnet.rbias)
saver.restore(sess, MODEL_PATH2)

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
	for g in range(len(class_labels)):
		if gt_labels[g] == class_labels[g]:
			stats[classes[int(gt_labels[g])]]['tp'] += 1
			stats['all']['tp'] += 1
		else:
			stats[classes[int(gt_labels[g])]]['fn'] += 1
			stats['all']['fn'] += 1
			stats[classes[class_labels[g]]]['fp'] += 1
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

					
gt_labels = numpy.array(gt_labels)
class_labels = numpy.array(class_labels)
calculate_accuracy()
print("Avg Comp Time: %.3f" % numpy.mean(comp_time))
nmi = normalized_mutual_info_score(gt_id, dt_id)
ami = adjusted_mutual_info_score(gt_id, dt_id)
ars = adjusted_rand_score(gt_id, dt_id)
print("NMI: %.3f AMI: %.3f ARS: %.3f %d/%d clusters"% (nmi,ami,ars,len(numpy.unique(dt_id)),len(numpy.unique(gt_id))))

min_cluster_size = 10
mask = numpy.zeros(len(dt_id),dtype=bool)
removed_clusters = 0
for c in clusters:
	if len(clusters[c]) < min_cluster_size:
		mask[clusters[c]] = True
		removed_clusters += 1
	M = scipy.stats.mode(class_labels[clusters[c]])[0][0]
	class_labels[clusters[c]] = M
dt_id = numpy.array(dt_id)
dt_id[mask] = 0
print("Filtered %d points (remaining %d clusters)"%(numpy.sum(mask), len(set(dt_id))))
calculate_accuracy()

