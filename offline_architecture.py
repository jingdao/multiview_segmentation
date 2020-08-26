import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate

BASE_LEARNING_RATE = 2e-4
NUM_FEATURE_CHANNELS = [64,64,128,512]
NUM_CONV_LAYERS = len(NUM_FEATURE_CHANNELS)
NUM_FC_CHANNELS = [512]
NUM_FC_LAYERS = len(NUM_FC_CHANNELS)
class PointNet():
	def __init__(self,batch_size,num_point,num_class):
		#inputs
		input_channels = 6
		self.pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, input_channels))
		self.labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))	
		self.is_training_pl = tf.placeholder(tf.bool, shape=())
		self.input = tf.expand_dims(self.pointclouds_pl,-1)
		self.conv = [None] * NUM_CONV_LAYERS
		self.kernel = [None] * NUM_CONV_LAYERS
		self.bias = [None] * NUM_CONV_LAYERS
		self.fc = [None] * (NUM_FC_LAYERS + 1)
		self.fc_weights = [None] * (NUM_FC_LAYERS + 1)
		self.fc_bias = [None] * (NUM_FC_LAYERS + 1)
		self.pool = [None]
		self.tile = [None] * 2

		#hidden layers
		for i in range(NUM_CONV_LAYERS):
			self.kernel[i] = tf.get_variable('kernel'+str(i), [1,input_channels if i==0 else 1, 1 if i==0 else NUM_FEATURE_CHANNELS[i-1], NUM_FEATURE_CHANNELS[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.bias[i] = tf.get_variable('bias'+str(i), [NUM_FEATURE_CHANNELS[i]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.conv[i] = tf.nn.conv2d(self.input if i==0 else self.conv[i-1], self.kernel[i], [1, 1, 1, 1], padding='VALID')
			self.conv[i] = tf.nn.bias_add(self.conv[i], self.bias[i])
			self.conv[i] = tf.nn.relu(self.conv[i])

		self.pool[0] = tf.nn.max_pool(self.conv[-1],ksize=[1, num_point, 1, 1],strides=[1, num_point, 1, 1], padding='VALID', name='pool'+str(i))
		self.tile[0] = tf.tile(tf.reshape(self.pool[0],[batch_size,-1,NUM_FEATURE_CHANNELS[-1]]) , [1,1,num_point])
		self.tile[0] = tf.reshape(self.tile[0],[batch_size,num_point,-1])
		self.tile[0] = tf.reshape(self.conv[-1],[batch_size,num_point,-1]) - self.tile[0]
		self.tile[1] = tf.reshape(self.conv[1], [batch_size, num_point, -1])
		self.concat = tf.concat(axis=2, values=self.tile)

		def batch_norm_template(inputs, is_training, moments_dims):
			with tf.variable_scope('bn') as sc:
				num_channels = inputs.get_shape()[-1].value
				beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
					name='beta', trainable=True)
				gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
					name='gamma', trainable=True)
				batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
				ema = tf.train.ExponentialMovingAverage(decay=0.9)
				ema_apply_op = tf.cond(is_training,
					lambda: ema.apply([batch_mean, batch_var]),
					lambda: tf.no_op())

				def mean_var_with_update():
					with tf.control_dependencies([ema_apply_op]):
						return tf.identity(batch_mean), tf.identity(batch_var)

				mean, var = tf.cond(is_training,
					mean_var_with_update,
					lambda: (ema.average(batch_mean), ema.average(batch_var)))
				normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
			return normed

		self.fc[0] = self.concat
		for i in range(NUM_FC_LAYERS):
			self.fc_weights[i] = tf.get_variable('fc_weights'+str(i), [1,self.fc[i].get_shape().as_list()[2], NUM_FC_CHANNELS[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.fc_bias[i] = tf.get_variable('fc_bias'+str(i), [NUM_FC_CHANNELS[i]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.fc[i+1] = tf.nn.conv1d(self.fc[i], self.fc_weights[i], 1, padding='VALID')
			self.fc[i+1] = tf.nn.bias_add(self.fc[i+1], self.fc_bias[i])
			self.fc[i+1] = batch_norm_template(self.fc[i+1],self.is_training_pl,[0,])
			self.fc[i+1] = tf.nn.relu(self.fc[i+1])

		#output
		self.fc_weights[-1] = tf.get_variable('fc_weights'+str(NUM_FC_LAYERS), [1,self.fc[-1].get_shape().as_list()[2], num_class], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.fc_bias[-1] = tf.get_variable('fc_bias'+str(NUM_FC_LAYERS), [num_class], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.output = tf.nn.conv1d(self.fc[-1], self.fc_weights[-1], 1, padding='VALID')
		self.output = tf.nn.bias_add(self.output, self.fc_bias[-1])

		#loss functions
		self.class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.labels_pl))
		self.loss = self.class_loss
		self.correct = tf.equal(tf.argmax(self.output, -1), tf.to_int64(self.labels_pl))
		self.class_acc = tf.reduce_mean(tf.cast(self.correct, tf.float32)) 

		#optimizer
		self.batch = tf.Variable(0)
		self.learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE,self.batch,500,0.5,staircase=True)
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss, global_step=self.batch)

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
		new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]s
	return new_points1


class PointNet2():
	def __init__(self,batch_size, num_point, num_class):
		input_channels = 6
		self.pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, input_channels))
		self.labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
		self.is_training_pl = tf.placeholder(tf.bool, shape=())
		l0_xyz = self.pointclouds_pl[:,:,:3]
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
		l0_points = tf.reshape(l0_points, [batch_size, num_point, 128])
		kernel1 = tf.get_variable('kernel1', [1, 128, 128], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias1 = tf.get_variable('bias1', [128], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		fc1 = tf.nn.conv1d(l0_points, kernel1, 1, padding='VALID')	
		fc1 = tf.nn.bias_add(fc1, bias1)
		fc1 = tf.nn.relu(fc1)
		kernel2 = tf.get_variable('kernel2', [1, 128, num_class], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		bias2 = tf.get_variable('bias2', [num_class], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		fc2 = tf.nn.conv1d(fc1, kernel2, 1, padding='VALID')	
		self.output = tf.nn.bias_add(fc2, bias2)

		#LOSS FUNCTIONS
		self.class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.labels_pl))
		correct = tf.equal(tf.argmax(self.output, -1), tf.to_int64(self.labels_pl))
		self.class_acc = tf.reduce_mean(tf.cast(correct, tf.float32)) 
		self.loss = self.class_loss
		batch = tf.Variable(0)
		optimizer = tf.train.AdamOptimizer(0.001)
		self.train_op = optimizer.minimize(self.loss, global_step=batch)

