import tensorflow as tf
import sys

NUM_P_LAYERS = 3
NUM_V_LAYERS = 3
NUM_R_LAYERS = 2
P_CHANNELS = 128
V_CHANNELS = 256
R_CHANNELS = 512

print("RaxNet",NUM_P_LAYERS,NUM_V_LAYERS,NUM_R_LAYERS,P_CHANNELS,V_CHANNELS)

class RaxNet():
	def __init__(self,batch_size, pool_size, point_feature_size, view_feature_size, output_size):
		self.label_pl = tf.placeholder(tf.int32, shape=(batch_size))
		self.point_input_pl = tf.placeholder(tf.float32, shape=(batch_size,pool_size,point_feature_size))
		self.view_input_pl = tf.placeholder(tf.float32, shape=(batch_size,pool_size,view_feature_size))
		self.pkernel = [None] * NUM_P_LAYERS
		self.pbias = [None] * NUM_P_LAYERS
		self.vkernel = [None] * NUM_V_LAYERS
		self.vbias = [None] * NUM_V_LAYERS
		self.rkernel = [None] * NUM_R_LAYERS
		self.rbias = [None] * NUM_R_LAYERS

		for i in range(NUM_P_LAYERS):
			self.pkernel[i] = tf.get_variable('pkernel'+str(i), [1,point_feature_size if i==0 else P_CHANNELS, P_CHANNELS], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.pbias[i] = tf.get_variable('pbias'+str(i), [P_CHANNELS], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			fc1 = tf.nn.conv1d(self.point_input_pl if i==0 else fc1, self.pkernel[i], 1, padding='VALID')
			fc1 = tf.nn.bias_add(fc1, self.pbias[i])
			fc1 = tf.nn.relu(fc1)
		fc1x = tf.reduce_max(fc1, axis=1)
		fc1m = tf.reduce_mean(fc1, axis=1)

		for i in range(NUM_V_LAYERS):
			self.vkernel[i] = tf.get_variable('vkernel'+str(i), [1,view_feature_size if i==0 else V_CHANNELS, V_CHANNELS], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.vbias[i] = tf.get_variable('vbias'+str(i), [V_CHANNELS], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			fc2 = tf.nn.conv1d(self.view_input_pl if i==0 else fc2, self.vkernel[i], 1, padding='VALID')
			fc2 = tf.nn.bias_add(fc2, self.vbias[i])
			fc2 = tf.nn.relu(fc2)
		fc2x = tf.reduce_max(fc2, axis=1)
		fc2m = tf.reduce_mean(fc2, axis=1)

#		fc3 = tf.concat(axis=1, values=[fc1x,fc2x])
#		fc3 = tf.concat(axis=1, values=[fc1m,fc2m])
		fc3 = tf.concat(axis=1, values=[fc1x,fc2x,fc1m,fc2m])
		for i in range(NUM_R_LAYERS):
			input_channels = fc3.get_shape().as_list()[1] if i==0 else R_CHANNELS
			output_channels = R_CHANNELS if i < NUM_R_LAYERS - 1 else output_size
			self.rkernel[i] = tf.get_variable('rkernel'+str(i), [input_channels, output_channels], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.rbias[i] = tf.get_variable('rbias'+str(i), [output_channels], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			fc3 = tf.matmul(fc3, self.rkernel[i])
			fc3 = tf.nn.bias_add(fc3, self.rbias[i])
			if i < NUM_R_LAYERS - 1:
				fc3 = tf.nn.relu(fc3)

		self.output = fc3
		self.class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits( logits=self.output, labels=self.label_pl)
		self.class_loss = tf.reduce_mean(self.class_loss)
		class_correct = tf.equal(tf.argmax(self.output, -1), tf.to_int64(self.label_pl))
		self.class_acc = tf.reduce_mean(tf.cast(class_correct, tf.float32))
		batch = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(2e-4,batch,20000,0.5,staircase=True)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		self.train_op = optimizer.minimize(self.class_loss, global_step=batch)

