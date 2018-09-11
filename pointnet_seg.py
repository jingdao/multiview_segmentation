import numpy
import tensorflow as tf
import sys

BASE_LEARNING_RATE = 2e-4
NUM_CONV_LAYERS = 4
NUM_FEATURE_CHANNELS = [64,64,128,512]
NUM_FC_CHANNELS = [512]
NUM_FC_LAYERS = len(NUM_FC_CHANNELS)

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

class PointNet():
	def __init__(self,batch_size,num_point,num_class,use_rgb=False):
		#inputs
		input_channels = 6 if use_rgb else 3
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
			self.conv[i] = batch_norm_template(self.conv[i],self.is_training_pl,[0,1])
			self.conv[i] = tf.nn.relu(self.conv[i])

		self.pool[0] = tf.nn.max_pool(self.conv[-1],ksize=[1, num_point, 1, 1],strides=[1, num_point, 1, 1], padding='VALID', name='pool'+str(i))
		self.tile[0] = tf.tile(tf.reshape(self.pool[0],[batch_size,-1,NUM_FEATURE_CHANNELS[-1]]) , [1,1,num_point])
		self.tile[0] = tf.reshape(self.tile[0],[batch_size,num_point,-1])
		self.tile[1] = tf.reshape(self.conv[1], [batch_size, num_point, -1])
		self.concat = tf.concat(axis=2, values=self.tile)

		self.fc[0] = self.concat
		for i in range(NUM_FC_LAYERS):
			self.fc_weights[i] = tf.get_variable('fc_weights'+str(i), [1,self.fc[i].get_shape().as_list()[2], NUM_FC_CHANNELS[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.fc_bias[i] = tf.get_variable('fc_bias'+str(i), [NUM_FC_CHANNELS[i]], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.fc[i+1] = tf.nn.conv1d(self.fc[i], self.fc_weights[i], 1, padding='VALID')
			self.fc[i+1] = tf.nn.bias_add(self.fc[i+1], self.fc_bias[i])
			self.fc[i+1] = batch_norm_template(self.fc[i+1],self.is_training_pl,[0,])
			self.fc[i+1] = tf.nn.relu(self.fc[i+1])

		#output
#		self.fc[-1] = tf.nn.dropout(self.fc[-1], 0.7)
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
