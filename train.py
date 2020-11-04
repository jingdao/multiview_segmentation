import h5py
import sys
import numpy
import scipy
from class_util import classes
from architecture import MCPNet, PointNet, PointNet2, VoxNet, SGPN
import itertools
import os
import psutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
import random
import math
import rosbag
from sensor_msgs import point_cloud2
from util import savePCD

def get_acc(emb,lb):
	correct = 0
	for i in range(len(lb)):
		dist = numpy.sum((emb[i] - emb)**2, axis=1)
		order = numpy.argsort(dist)
		correct += lb[i] == lb[order[1]]
	return 1.0 * correct / len(lb)

def get_anova(emb, lb):
	lid = list(set(lb))
	nf = emb.shape[1]
	class_mean = numpy.zeros((len(lid), nf))
	for i in range(len(lid)):
		class_mean[i] = emb[lb==lid[i]].mean(axis=0)
	overall_mean = emb.mean(axis=0)
	between_group = 0
	for i in range(len(lid)):
		num_in_group = numpy.sum(lb==lid[i])
		between_group += numpy.sum((class_mean[i] - overall_mean)**2) * num_in_group
	between_group /= (len(lid) - 1 + 1e-6)
	within_group = 0
	for i in range(len(lid)):
		within_group += numpy.sum((emb[lb==lid[i]] - class_mean[i])**2)
	within_group /= (len(lb) - len(lid) + 1e-6)
	F = 0 if within_group==0 else between_group / within_group
	return between_group, within_group, F

def get_even_sampling(labels, batch_size, samples_per_instance):
	pool = {}
	for i in set(labels):
		pool[i] = set(numpy.nonzero(labels==i)[0])
	idx = []
	while len(pool) > 0 and len(idx) < batch_size:
		k = pool.keys()
		c = k[numpy.random.randint(len(k))]	
		if len(pool[c]) > samples_per_instance:
			inliers = set(numpy.random.choice(list(pool[c]), samples_per_instance, replace=False))
			idx.extend(inliers)
			pool[c] -= inliers
		else:
			idx.extend(pool[c])
			del pool[c]
	return idx[:batch_size]

local_range = 2
resolution = 0.1
num_neighbors = 50
neighbor_radii = 0.3
feature_size = 6
max_epoch = 50
samples_per_instance = 16
train_subsample = 10
NUM_CLASSES = len(classes)

net_type = 'mcpnet'
dataset = 's3dis'
TRAIN_AREA = ['1','2','3','4','6']
VAL_AREA = ['5']
USE_XY = True
for i in range(len(sys.argv)-1):
	if sys.argv[i]=='--train-area':
		TRAIN_AREA = sys.argv[i+1].split(',')
	if sys.argv[i]=='--val-area':
		VAL_AREA = sys.argv[i+1].split(',')
	if sys.argv[i]=='--net':
		net_type = sys.argv[i+1]
	if sys.argv[i]=='--dataset':
		dataset = sys.argv[i+1]
		if dataset=='outdoor':
			local_range = 10
			NUM_CLASSES = 17
			feature_size = 3
			train_subsample = 1
			USE_XY = False
mode = None
if '--color' in sys.argv:
	mode='color'
if '--cluster' in sys.argv:
	mode='cluster'
if '--classify' in sys.argv:
	mode='classify'

batch_size = 256 if net_type.startswith('mcpnet') else 1024
hidden_size = 200
embedding_size = 50
dp_threshold = 0.9 if net_type.startswith('mcpnet') else 0.99

train_points,train_obj_id,train_cls_id = [],[],[]
val_points,val_obj_id,val_cls_id = [],[],[]
point_id_map = {}
coarse_map = {}
point_orig_list = []
agg_points = []
agg_obj_id = []
agg_cls_id = []
count_msg = 0
numpy.random.seed(0)
sample_state = numpy.random.RandomState(0)

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
	print('Invalid network type')
	sys.exit(1)
saver = tf.train.Saver()
MODEL_PATH = 'models/%s_model_%s_%s.ckpt'%(net_type, dataset, VAL_AREA[0])

def process_cloud(cloud, robot_position):
	global count_msg
	t = time.time()
	pcd = []
	for p in point_cloud2.read_points(cloud, field_names=("x","y","z","r","g","b","o","c"), skip_nans=True):
		pcd.append(p)
	pcd = numpy.array(pcd)
	local_mask = numpy.sum((pcd[:,:2]-robot_position)**2, axis=1) < local_range * local_range
	local_mask = numpy.logical_and(local_mask, pcd[:, 6] > 0)
	pcd = pcd[local_mask, :]
	pcd[:,3:6] = pcd[:,3:6] / 255.0 - 0.5
	original_pcd = pcd.copy()
	pcd[:,:2] -= robot_position
	minZ = pcd[:,2].min()
	pcd[:,2] -= minZ

	pcdi = [tuple(p) for p in (original_pcd[:,:3]/resolution).round().astype(int)]
	update_list = []
	for i in range(len(pcdi)):
		if not pcdi[i] in point_id_map:
			point_id_map[pcdi[i]] = len(point_orig_list)
			point_orig_list.append(original_pcd[i].copy())
			update_list.append(pcdi[i])

	for k in update_list:
		idx = point_id_map[k]
		kk = tuple((point_orig_list[idx][:3]/neighbor_radii).round().astype(int))
		if not kk in coarse_map:
			coarse_map[kk] = []
		coarse_map[kk].append(idx)
	
	if count_msg%train_subsample==0:
		pcd = numpy.array(point_orig_list[-len(update_list):])
		original_pcd = pcd.copy()
		pcd[:,:2] -= robot_position
		pcd[:,2] -= minZ
		point_samples = numpy.arange(len(pcd))
		point_samples = point_samples[sample_state.choice(len(point_samples), batch_size*2, replace=len(point_samples) < batch_size*2)]
		if net_type=='mcpnet' and num_neighbors>0:
			neighbor_array = []
			for i in range(len(point_samples)):
				p = original_pcd[point_samples[i],:feature_size]
				idx = point_id_map[tuple((p[:3]/resolution).round().astype(int))]
				k = tuple((point_orig_list[idx][:3]/neighbor_radii).round().astype(int))
				neighbors = []
				for offset in itertools.product(range(-1,2),range(-1,2),range(-1,2)):
					kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
					if kk in coarse_map:
						neighbors.extend(coarse_map[kk])
				neighbors = sample_state.choice(neighbors, num_neighbors, replace=len(neighbors)<num_neighbors)
				neighbors = numpy.array([point_orig_list[n][:feature_size] for n in neighbors])
				neighbors -= p
				neighbor_array.append(neighbors)
			stacked_points = numpy.hstack((pcd[point_samples,:feature_size], numpy.array(neighbor_array).reshape((len(point_samples), num_neighbors*feature_size))))
            if not USE_XY:
                stacked_points[:, :2] = 0
			agg_points.append(stacked_points)
		else:
			agg_points.append(pcd[point_samples,:feature_size])
		agg_obj_id.append(pcd[point_samples,6])
		agg_cls_id.append(pcd[point_samples,7])

	t = time.time() - t
	sys.stdout.write('Scan #%3d: cur:%4d agg:%5d time %.3f\r'%(count_msg, len(update_list), len(point_id_map),  t))
	sys.stdout.flush()
	count_msg += 1
	
AREAS = TRAIN_AREA + VAL_AREA
for area in AREAS:
	print('\nScanning area %s ...'%area)
	point_id_map = {}
	coarse_map = {}
	point_orig_list = []
	agg_points = []
	agg_obj_id = []
	agg_cls_id = []
	count_msg = 0
	bag = rosbag.Bag('data/%s_%s.bag' % (dataset, area), 'r')
	poses = []
	for topic, msg, t in bag.read_messages(topics=['slam_out_pose']):
		poses.append([msg.pose.position.x, msg.pose.position.y])
	n = 0
	for topic, msg, t in bag.read_messages(topics=['laser_cloud_surround']):
		process_cloud(msg, poses[n])
		n += 1
#		if n==5:
#			break
	obj_id_list = [i for j in agg_obj_id for i in j]
	cls_id_list = [i for j in agg_cls_id for i in j]
	u, c = numpy.unique(cls_id_list, return_counts=True)
	print('scans', n, 'obj', len(set(obj_id_list)), 'cls', u, c)
	if area in VAL_AREA:
		val_points.extend(agg_points) 
		val_obj_id.extend(agg_obj_id) 
		val_cls_id.extend(agg_cls_id) 
	if area in TRAIN_AREA:
		train_points.extend(agg_points) 
		train_obj_id.extend(agg_obj_id) 
		train_cls_id.extend(agg_cls_id) 

print('')
print('train',len(train_points),train_points[0].shape)
print('val',len(val_points), val_points[0].shape)
init = tf.global_variables_initializer()
sess.run(init, {})
for epoch in range(max_epoch):
	loss_arr = []
	cls_arr = []
	acc_arr = []
	bg_arr = []
	wg_arr = []
	f_arr = []
	for i in random.sample(xrange(len(train_points)), len(train_points)):
		idx = get_even_sampling(train_obj_id[i], batch_size,samples_per_instance)
		input_points = train_points[i][idx, :]
		input_labels = train_obj_id[i][idx]
		input_class = train_cls_id[i][idx]
		if net_type in ['sgpn','mcpnet','mcpnet_simple']:
			_, loss_val, cls_val, emb_val = sess.run([net.train_op, net.loss, net.class_acc, net.embeddings], {net.input_pl:input_points, net.label_pl:input_labels, net.class_pl:input_class, net.is_training_pl:True})
			acc = get_acc(emb_val, input_labels)
			bg,wg,f = get_anova(emb_val, input_labels)
		else:
			_, loss_val, cls_val = sess.run([net.train_op, net.loss, net.class_acc], {net.input_pl:input_points, net.labels_pl:input_class, net.is_training_pl:True})
			acc,bg,wg,f = 0,0,0,0
		loss_arr.append(loss_val)
		cls_arr.append(cls_val)
		acc_arr.append(acc)
		bg_arr.append(bg)
		wg_arr.append(wg)
		f_arr.append(f)
	print("Epoch %d loss %.2f cls %.3f acc %.2f bg %.2f wg %.2f F %.2f"%(epoch,numpy.mean(loss_arr),numpy.mean(cls_arr),numpy.mean(acc_arr),numpy.mean(bg_arr),numpy.mean(wg_arr),numpy.mean(f_arr)))

	if epoch%10==9:
		loss_arr = []
		cls_arr = []
		acc_arr = []
		bg_arr = []
		wg_arr = []
		f_arr = []
		for i in random.sample(xrange(len(val_points)), len(val_points)):
			idx = get_even_sampling(val_obj_id[i], batch_size,samples_per_instance)
			input_points = val_points[i][idx, :]
			input_labels = val_obj_id[i][idx]
			input_class = val_cls_id[i][idx]
			if net_type in ['sgpn','mcpnet','mcpnet_simple']:
				loss_val, cls_val, emb_val = sess.run([net.loss, net.class_acc, net.embeddings], {net.input_pl:input_points, net.label_pl:input_labels, net.class_pl: input_class, net.is_training_pl:False})
				acc = get_acc(emb_val, input_labels)
				bg,wg,f = get_anova(emb_val, input_labels)
			else:
				loss_val, cls_val = sess.run([net.loss, net.class_acc], {net.input_pl:input_points, net.labels_pl: input_class, net.is_training_pl:False})
				acc,bg,wg,f = 0,0,0,0
			loss_arr.append(loss_val)
			cls_arr.append(cls_val)
			acc_arr.append(acc)
			bg_arr.append(bg)
			wg_arr.append(wg)
			f_arr.append(f)
		print("Validation %d loss %.2f cls %.3f acc %.2f bg %.2f wg %.2f F %.2f"%(epoch,numpy.mean(loss_arr),numpy.mean(cls_arr),numpy.mean(acc_arr),numpy.mean(bg_arr),numpy.mean(wg_arr),numpy.mean(f_arr)))

saver.save(sess, MODEL_PATH)

