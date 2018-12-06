import tensorflow as tf
from metric_loss_ops import triplet_semihard_loss
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate

class MCPNet:
	def __init__(self,batch_size, neighbor_size, feature_size, hidden_size, embedding_size, num_class):
		self.class_pl = tf.placeholder(tf.int32, shape=(batch_size))
		self.label_pl = tf.placeholder(tf.int32, shape=(batch_size))
		self.input_pl = tf.placeholder(tf.float32, shape=(batch_size, feature_size*(neighbor_size+1)))
		self.is_training_pl = tf.placeholder(tf.bool, shape=())

		#NETWORK_WEIGHTS
		kernel1 = tf.get_variable('kernel1', [1,feature_size,hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias1 = tf.get_variable('bias1', [hidden_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		kernel2 = tf.get_variable('kernel2', [1,hidden_size,hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias2 = tf.get_variable('bias2', [hidden_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		kernel3 = tf.get_variable('kernel3', [feature_size+hidden_size if neighbor_size>0 else feature_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias3 = tf.get_variable('bias3', [hidden_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		kernel4 = tf.get_variable('kernel4', [hidden_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias4 = tf.get_variable('bias4', [embedding_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		kernel5 = tf.get_variable('kernel5', [hidden_size+embedding_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias5 = tf.get_variable('bias5', [hidden_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		kernel6 = tf.get_variable('kernel6', [hidden_size, num_class], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias6 = tf.get_variable('bias6', [num_class], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

		#MULTI-VIEW CONTEXT POOLING
		if neighbor_size > 0:
			neighbor_fc = tf.reshape(self.input_pl[:,feature_size:], [batch_size, neighbor_size, feature_size])
			neighbor_fc = tf.nn.conv1d(neighbor_fc, kernel1, 1, padding='VALID')
			neighbor_fc = tf.nn.bias_add(neighbor_fc, bias1)
			neighbor_fc = tf.nn.relu(neighbor_fc)
			neighbor_fc = tf.nn.conv1d(neighbor_fc, kernel2, 1, padding='VALID')
			neighbor_fc = tf.nn.bias_add(neighbor_fc, bias2)
			neighbor_fc = tf.nn.relu(neighbor_fc)
			neighbor_fc = tf.reduce_max(neighbor_fc, axis=1)
			concat = tf.concat(axis=1, values=[self.input_pl[:,:feature_size], neighbor_fc])
		else:
			concat = self.input_pl[:,:feature_size]

		#FEATURE EMBEDDING BRANCH (for instance label prediction)
		fc3 = tf.matmul(concat, kernel3)
		fc3 = tf.nn.bias_add(fc3, bias3)
		fc3 = tf.nn.relu(fc3)
		self.fc4 = tf.matmul(fc3, kernel4)
		self.fc4 = tf.nn.bias_add(self.fc4, bias4)
		self.embeddings = tf.nn.l2_normalize(self.fc4, dim=1)
		self.triplet_loss = triplet_semihard_loss(self.label_pl, self.embeddings)

		#CLASSIFICATION BRANCH (for class label prediction)
		self.pool = tf.reduce_max(fc3, axis=0)
		pool_diff = fc3 - tf.reshape(tf.tile(self.pool, [batch_size]), [batch_size, -1])
		upper_concat = tf.concat(axis=1, values=[pool_diff, self.fc4])
		fc5 = tf.matmul(upper_concat, kernel5)
		fc5 = tf.nn.bias_add(fc5, bias5)
		fc5 = tf.nn.relu(fc5)
		fc6 = tf.matmul(fc5, kernel6)
		self.class_output = tf.nn.bias_add(fc6, bias6)
		self.class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.class_output, labels=self.class_pl))
		correct = tf.equal(tf.argmax(self.class_output, -1), tf.to_int64(self.class_pl))
		self.class_acc = tf.reduce_mean(tf.cast(correct, tf.float32)) 

		#LOSS FUNCTIONS
		self.loss = self.triplet_loss + self.class_loss
		batch = tf.Variable(0)
		optimizer = tf.train.AdamOptimizer(0.001)
		self.train_op = optimizer.minimize(self.loss, global_step=batch)

class PointNet:
	def __init__(self,batch_size, feature_size, num_class):
		NUM_FEATURE_CHANNELS = [64,64,64,128,1024,256,128,512,256,num_class]
		self.kernel = [None]*9
		self.bias = [None]*9
		self.conv = [None]*5
		self.fc = [None]*3
		self.labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
		self.input_pl = tf.placeholder(tf.float32, shape=(batch_size, feature_size))
		self.is_training_pl = tf.placeholder(tf.bool, shape=())

		#CONVOLUTION LAYERS
		for i in range(5):
			self.kernel[i] = tf.get_variable('kernel'+str(i), [feature_size if i==0 else NUM_FEATURE_CHANNELS[i-1], NUM_FEATURE_CHANNELS[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.bias[i] = tf.get_variable('bias'+str(i), [NUM_FEATURE_CHANNELS[i]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.conv[i] = tf.matmul(self.input_pl if i==0 else self.conv[i-1], self.kernel[i])
			self.conv[i] = tf.nn.bias_add(self.conv[i], self.bias[i])
			self.conv[i] = tf.nn.relu(self.conv[i])

		#MAX POOLING
		self.pool = tf.reduce_max(self.conv[4], axis=0)
		self.pool = tf.reshape(self.pool, [1, NUM_FEATURE_CHANNELS[4]])
		for i in range(5,7):
			self.kernel[i] = tf.get_variable('kernel'+str(i), [NUM_FEATURE_CHANNELS[i-1], NUM_FEATURE_CHANNELS[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.bias[i] = tf.get_variable('bias'+str(i), [NUM_FEATURE_CHANNELS[i]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.pool = tf.matmul(self.pool, self.kernel[i])
			self.pool = tf.nn.bias_add(self.pool, self.bias[i])
			self.pool = tf.nn.relu(self.pool)
		self.pool = tf.tile(self.pool, [batch_size, 1])
		self.concat = tf.concat(axis=1, values=[self.conv[4], self.pool])

		#FULLY CONNECTED LAYERS
		for i in range(7,9):
			self.kernel[i] = tf.get_variable('kernel'+str(i), [NUM_FEATURE_CHANNELS[4]+NUM_FEATURE_CHANNELS[6] if i==7 else NUM_FEATURE_CHANNELS[i-1], NUM_FEATURE_CHANNELS[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.bias[i] = tf.get_variable('bias'+str(i), [NUM_FEATURE_CHANNELS[i]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.fc[i-7] = tf.matmul(self.concat if i==7 else self.fc[i-8], self.kernel[i])
			self.fc[i-7] = tf.nn.bias_add(self.fc[i-7], self.bias[i])

		self.fc[1] = tf.cond(self.is_training_pl, lambda: tf.nn.dropout(self.fc[1], keep_prob=0.7), lambda: self.fc[1])
		self.kernel[-1] = tf.get_variable('kernel9', [NUM_FEATURE_CHANNELS[-2], NUM_FEATURE_CHANNELS[-1]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias[-1] = tf.get_variable('bias9', [NUM_FEATURE_CHANNELS[-1]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.fc[2] = tf.matmul(self.fc[1], self.kernel[-1])
		self.fc[2] = tf.nn.bias_add(self.fc[2], self.bias[-1])

		#LOSS FUNCTIONS
		self.class_output = self.fc[2]
		self.class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.class_output, labels=self.labels_pl))
		correct = tf.equal(tf.argmax(self.class_output, -1), tf.to_int64(self.labels_pl))
		self.class_acc = tf.reduce_mean(tf.cast(correct, tf.float32)) 
		self.loss = self.class_loss
		batch = tf.Variable(0)
		optimizer = tf.train.AdamOptimizer(0.001)
		self.train_op = optimizer.minimize(self.loss, global_step=batch)

def sample_and_group(npoint, radius, nsample, xyz, points):
	new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
	idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
	grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
	grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
	if points is not None:
		grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
		new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
	else:
		new_points = grouped_xyz
	return new_xyz, new_points, idx, grouped_xyz

#PointNet Set Abstraction (SA) Module
def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope):
	data_format = 'NHWC'
	with tf.variable_scope(scope) as sc:
		new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points)
		kernel = [None]*len(mlp)
		bias = [None]*len(mlp)
		for i, num_out_channel in enumerate(mlp):
			kernel[i] = tf.get_variable('kernel'+str(i), [1,1,new_points.get_shape()[-1].value, num_out_channel], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			bias[i] = tf.get_variable('bias'+str(i), [num_out_channel], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			new_points = tf.nn.conv2d(new_points, kernel[i], [1, 1, 1, 1], padding='VALID')
			new_points = tf.nn.bias_add(new_points, bias[i])
			new_points = tf.nn.relu(new_points)
		new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
		new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
		return new_xyz, new_points, idx

#PointNet Feature Propogation (FP) Module
def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope):
	with tf.variable_scope(scope) as sc:
		dist, idx = three_nn(xyz1, xyz2)
		dist = tf.maximum(dist, 1e-10)
		norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
		norm = tf.tile(norm,[1,1,3])
		weight = (1.0/dist) / norm
		interpolated_points = three_interpolate(points2, idx, weight)

		if points1 is not None:
			new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
		else:
			new_points1 = interpolated_points
		new_points1 = tf.expand_dims(new_points1, 2)

		kernel = [None]*len(mlp)
		bias = [None]*len(mlp)
		for i, num_out_channel in enumerate(mlp):
			kernel[i] = tf.get_variable('kernel'+str(i), [1,1,new_points1.get_shape()[-1].value, num_out_channel], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			bias[i] = tf.get_variable('bias'+str(i), [num_out_channel], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			new_points1 = tf.nn.conv2d(new_points1, kernel[i], [1, 1, 1, 1], padding='VALID')
			new_points1 = tf.nn.bias_add(new_points1, bias[i])
			new_points1 = tf.nn.relu(new_points1)
		new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1

class PointNet2:
	def __init__(self,batch_size, feature_size, num_class):
		self.labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
		self.input_pl = tf.placeholder(tf.float32, shape=(batch_size, feature_size))
		self.is_training_pl = tf.placeholder(tf.bool, shape=())
		l0_xyz = tf.reshape(self.input_pl[:,:3], [1,batch_size,3])
		l0_points = None

		# Layer 1
		l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=self.is_training_pl, bn_decay=0, scope='layer1')
		l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=self.is_training_pl, bn_decay=0, scope='layer2')
		l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=64, radius=0.4, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=self.is_training_pl, bn_decay=0, scope='layer3')
		l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=16, radius=0.8, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=self.is_training_pl, bn_decay=0, scope='layer4')

		# Feature Propagation layers
		l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], self.is_training_pl, 0, scope='fa_layer1')
		l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], self.is_training_pl, 0, scope='fa_layer2')
		l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], self.is_training_pl, 0, scope='fa_layer3')
		l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], self.is_training_pl, 0, scope='fa_layer4')

		# FC layers
		l0_points = tf.reshape(l0_points, [batch_size, 128])
		kernel1 = tf.get_variable('kernel1', [128, 128], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias1 = tf.get_variable('bias1', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		fc1 = tf.matmul(l0_points, kernel1)
		fc1 = tf.nn.bias_add(fc1, bias1)
		fc1 = tf.nn.relu(fc1)
		kernel2 = tf.get_variable('kernel2', [128, num_class], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias2 = tf.get_variable('bias2', [num_class], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		fc2 = tf.matmul(fc1, kernel2)
		self.class_output = tf.nn.bias_add(fc2, bias2)

		#LOSS FUNCTIONS
		self.class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.class_output, labels=self.labels_pl))
		correct = tf.equal(tf.argmax(self.class_output, -1), tf.to_int64(self.labels_pl))
		self.class_acc = tf.reduce_mean(tf.cast(correct, tf.float32)) 
		self.loss = self.class_loss
		batch = tf.Variable(0)
		optimizer = tf.train.AdamOptimizer(0.001)
		self.train_op = optimizer.minimize(self.loss, global_step=batch)

class VoxNet:
	def __init__(self,batch_size, feature_size, num_class):
		self.labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
		self.input_pl = tf.placeholder(tf.float32, shape=(batch_size, feature_size))
		self.is_training_pl = tf.placeholder(tf.bool, shape=())

		#FORM VOXEL GRID
		voxel_idx = tf.identity(self.input_pl)
		voxel_idx = tf.stack([(voxel_idx[:,0]+2)*8,(voxel_idx[:,1]+2)*8,(voxel_idx[:,2]/6)*32], axis=1)
		voxel_idx = tf.to_int32(voxel_idx)
		voxel_idx = tf.clip_by_value(voxel_idx, 0, 31)
		self.voxel_grid = tf.sparse_to_dense(voxel_idx, [32,32,32], 1.0, validate_indices=False)
		self.voxel_grid = tf.reshape(self.voxel_grid, [1,32,32,32,1])

		#3D CONVOLUTIONS
		kernel1 = tf.get_variable('kernel1', [5,5,5,1,32], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias1 = tf.get_variable('bias1', [32], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		conv1 = tf.nn.conv3d(self.voxel_grid, kernel1, [1,2,2,2,1], "VALID", data_format='NDHWC')
		conv1 = tf.nn.bias_add(conv1, bias1)
		conv1 = tf.nn.relu(conv1)
		kernel2 = tf.get_variable('kernel2', [3,3,3,32,32], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias2 = tf.get_variable('bias2', [32], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		conv2 = tf.nn.conv3d(conv1, kernel2, [1,1,1,1,1], "VALID", data_format='NDHWC')
		conv2 = tf.nn.bias_add(conv2, bias2)
		conv2 = tf.nn.relu(conv2)
		pool2 = tf.nn.max_pool3d(conv2,ksize=[1,12,12,12,1],strides=[1,12,12,12,1], padding="VALID", name='pool')
		pool2 = tf.tile(tf.reshape(pool2, [1,32]), [batch_size, 1])

		#CONCATENATE VOXEL FEATURES WITH POINT CLOUD
		self.concat = tf.concat(axis=1, values=[self.input_pl, pool2])
		kernel3 = tf.get_variable('kernel3', [feature_size+32, 128], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias3 = tf.get_variable('bias3', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		fc3 = tf.matmul(self.concat, kernel3)
		fc3 = tf.nn.bias_add(fc3, bias3)
		fc3 = tf.nn.relu(fc3)
		kernel4 = tf.get_variable('kernel4', [128, num_class], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias4 = tf.get_variable('bias4', [num_class], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		fc4 = tf.matmul(fc3, kernel4)
		self.class_output = tf.nn.bias_add(fc4, bias4)

		#LOSS FUNCTIONS
		self.class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.class_output, labels=self.labels_pl))
		correct = tf.equal(tf.argmax(self.class_output, -1), tf.to_int64(self.labels_pl))
		self.class_acc = tf.reduce_mean(tf.cast(correct, tf.float32)) 
		self.loss = self.class_loss
		batch = tf.Variable(0)
		optimizer = tf.train.AdamOptimizer(0.001)
		self.train_op = optimizer.minimize(self.loss, global_step=batch)

class SGPN:
	def __init__(self,batch_size, feature_size, num_class):
		NUM_FEATURE_CHANNELS = [64,64,64,128,1024,256,128,512,256]
		self.kernel = [None]*14
		self.bias = [None]*14
		self.conv = [None]*5
		self.fc = [None]*5
		self.label_pl = tf.placeholder(tf.int32, shape=(batch_size))
		self.class_pl = tf.placeholder(tf.int32, shape=(batch_size))
		self.input_pl = tf.placeholder(tf.float32, shape=(batch_size, feature_size))
		self.is_training_pl = tf.placeholder(tf.bool, shape=())

		#CONVOLUTION LAYERS
		for i in range(5):
			self.kernel[i] = tf.get_variable('kernel'+str(i), [feature_size if i==0 else NUM_FEATURE_CHANNELS[i-1], NUM_FEATURE_CHANNELS[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.bias[i] = tf.get_variable('bias'+str(i), [NUM_FEATURE_CHANNELS[i]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.conv[i] = tf.matmul(self.input_pl if i==0 else self.conv[i-1], self.kernel[i])
			self.conv[i] = tf.nn.bias_add(self.conv[i], self.bias[i])
			self.conv[i] = tf.nn.relu(self.conv[i])

		#MAX POOLING
		self.pool = tf.reduce_max(self.conv[4], axis=0)
		self.pool = tf.reshape(self.pool, [1, NUM_FEATURE_CHANNELS[4]])
		for i in range(5,7):
			self.kernel[i] = tf.get_variable('kernel'+str(i), [NUM_FEATURE_CHANNELS[i-1], NUM_FEATURE_CHANNELS[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.bias[i] = tf.get_variable('bias'+str(i), [NUM_FEATURE_CHANNELS[i]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.pool = tf.matmul(self.pool, self.kernel[i])
			self.pool = tf.nn.bias_add(self.pool, self.bias[i])
			self.pool = tf.nn.relu(self.pool)
		self.pool = tf.tile(self.pool, [batch_size, 1])
		self.concat = tf.concat(axis=1, values=[self.conv[4], self.pool])

		#FULLY CONNECTED LAYERS
		for i in range(7,9):
			self.kernel[i] = tf.get_variable('kernel'+str(i), [NUM_FEATURE_CHANNELS[4]+NUM_FEATURE_CHANNELS[6] if i==7 else NUM_FEATURE_CHANNELS[i-1], NUM_FEATURE_CHANNELS[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.bias[i] = tf.get_variable('bias'+str(i), [NUM_FEATURE_CHANNELS[i]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.fc[i-7] = tf.matmul(self.concat if i==7 else self.fc[i-8], self.kernel[i])
			self.fc[i-7] = tf.nn.bias_add(self.fc[i-7], self.bias[i])

		#CLASSIFICATION OUTPUT BRANCH
		self.kernel[9] = tf.get_variable('kernel9', [256, 128], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias[9] = tf.get_variable('bias9', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.fc[2] = tf.matmul(self.fc[1], self.kernel[9])
		self.fc[2] = tf.nn.bias_add(self.fc[2], self.bias[9])
		self.fc[2] = tf.nn.relu(self.fc[2])
		self.kernel[10] = tf.get_variable('kernel10', [128, num_class], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias[10] = tf.get_variable('bias10', [num_class], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.fc[2] = tf.matmul(self.fc[2], self.kernel[10])
		self.class_output = tf.nn.bias_add(self.fc[2], self.bias[10])

		#CONFIDENCE OUTPUT BRANCH
		self.kernel[11] = tf.get_variable('kernel11', [256, 128], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias[11] = tf.get_variable('bias11', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.fc[3] = tf.matmul(self.fc[1], self.kernel[11])
		self.fc[3] = tf.nn.bias_add(self.fc[3], self.bias[11])
		self.fc[3] = tf.nn.relu(self.fc[3])
		self.kernel[12] = tf.get_variable('kernel12', [128, 1], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias[12] = tf.get_variable('bias12', [1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.fc[3] = tf.matmul(self.fc[3], self.kernel[12])
		self.conf_output = tf.nn.bias_add(self.fc[3], self.bias[12])

		#SIMILARITY MATRIX OUTPUT BRANCH
		self.kernel[13] = tf.get_variable('kernel13', [256, 128], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias[13] = tf.get_variable('bias13', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.fc[4] = tf.matmul(self.fc[1], self.kernel[13])
		self.fc[4] = tf.nn.bias_add(self.fc[4], self.bias[13])
		self.embeddings = tf.nn.relu(self.fc[4])
		r = tf.reshape(tf.reduce_sum(self.embeddings*self.embeddings, axis=1), [batch_size, 1])
		D = r - 2 * tf.matmul(self.embeddings, tf.transpose(self.embeddings)) + tf.transpose(r)
		self.sim_output = tf.maximum(10 * D, 0)
		self.embeddings = tf.nn.l2_normalize(self.embeddings, dim=1)

		#DOUBLE-HINGE LOSS FOR SIMILARITY MATRIX
		same_group = tf.tile(tf.reshape(self.label_pl,[batch_size,1]),[1,batch_size])==tf.tile(tf.reshape(self.label_pl,[1,batch_size]),[batch_size,1])
		same_class = tf.tile(tf.reshape(self.class_pl,[batch_size,1]),[1,batch_size])==tf.tile(tf.reshape(self.class_pl,[1,batch_size]),[batch_size,1])
		pos =  tf.multiply(tf.cast(same_group, tf.float32), self.sim_output) # minimize distances if in the same group
		mask1 = tf.cast(tf.logical_and(tf.logical_not(same_group), same_class), tf.float32)
		neg_samesem = 10.0 * tf.multiply(mask1, tf.maximum(tf.subtract(10.0, self.sim_output), 0))
		mask2 = tf.cast(tf.logical_not(tf.logical_or(same_group, same_class)), tf.float32)
		neg_diffsem = tf.multiply(mask2, tf.maximum(tf.subtract(80.0, self.sim_output), 0))
		self.sim_loss  = neg_samesem + neg_diffsem + pos
		self.sim_loss = tf.reduce_mean(self.sim_loss)

		#LOSS FUNCTIONS
		self.class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.class_output, labels=self.class_pl))
		correct = tf.equal(tf.argmax(self.class_output, -1), tf.to_int64(self.class_pl))
		self.class_acc = tf.reduce_mean(tf.cast(correct, tf.float32)) 
		self.loss = self.class_loss + self.sim_loss
		batch = tf.Variable(0)
		optimizer = tf.train.AdamOptimizer(0.001)
		self.train_op = optimizer.minimize(self.loss, global_step=batch)
