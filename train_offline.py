import h5py 
import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
import sys
from class_util import classes
from offline_architecture import PointNet, PointNet2

BATCH_SIZE = 100
NUM_POINT = 2048
NUM_CLASSES = len(classes)
MAX_EPOCH = 50
VAL_STEP = 10
GPU_INDEX = 0

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

def jitter_data(points, labels):
	output_points = points.copy()
	output_labels = labels.copy()
	for i in range(len(points)):
		if numpy.random.randint(2):
			output_points[i,:,0] = -output_points[i,:,0]
		if numpy.random.randint(2):
			output_points[i,:,1] = -output_points[i,:,1]
		C = numpy.random.rand() * 0.5 + 0.75
		T = numpy.random.rand(3) * 0.4 - 0.2
		output_points[i,:,:3] = output_points[i,:,:3] * C + T
	return output_points, output_labels

if __name__=='__main__':

	VAL_AREA = 1
	net = 'pointnet'
	for i in range(len(sys.argv)):
		if sys.argv[i] == '--net':
			net = sys.argv[i+1]
		if sys.argv[i]=='--area':
			VAL_AREA = int(sys.argv[i+1])
	MODEL_PATH = 'models/offline_%s_model%d.ckpt' % (net, VAL_AREA)

	#arrange points into batches of 2048x6
	train_points = []
	train_labels = []
	val_points = []
	val_labels = []
	for AREA in [1,2,3,4,5,6]:
		all_points,all_obj_id,all_cls_id = loadFromH5('data/s3dis_area%d.h5' % AREA)
		for room_id in range(len(all_points)):
			points = all_points[room_id]
			cls_id = all_cls_id[room_id]

			grid_resolution = 1.0
			grid = numpy.round(points[:,:2]/grid_resolution).astype(int)
			grid_set = set([tuple(g) for g in grid])
			for g in grid_set:
				grid_mask = numpy.all(grid==g, axis=1)
				grid_points = points[grid_mask, :6]
				centroid_xy = numpy.array(g)*grid_resolution
				centroid_z = grid_points[:,2].min()
				grid_points[:,:2] -= centroid_xy
				grid_points[:,2] -= centroid_z
				grid_labels = cls_id[grid_mask]

				subset = numpy.random.choice(len(grid_points), NUM_POINT*2, replace=len(grid_points)<NUM_POINT*2)
				if AREA==VAL_AREA:
					val_points.append(grid_points[subset])
					val_labels.append(grid_labels[subset])
				else:
					train_points.append(grid_points[subset])
					train_labels.append(grid_labels[subset])

	train_points = numpy.array(train_points)
	train_labels = numpy.array(train_labels)
	val_points = numpy.array(val_points)
	val_labels = numpy.array(val_labels)
	print('Train Points',train_points.shape)
	print('Train Labels',train_labels.shape)
	print('Validation Points',val_points.shape)
	print('Validation Labels',val_labels.shape)

	with tf.Graph().as_default():
		with tf.device('/gpu:'+str(GPU_INDEX)):
			if net == 'pointnet2':
				net = PointNet2(BATCH_SIZE,NUM_POINT,NUM_CLASSES)
			else:
				net = PointNet(BATCH_SIZE,NUM_POINT,NUM_CLASSES) 
			saver = tf.train.Saver()

			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			config.allow_soft_placement = True
			config.log_device_placement = False
			sess = tf.Session(config=config)
			init = tf.global_variables_initializer()
			sess.run(init, {net.is_training_pl: True})


			for epoch in range(MAX_EPOCH):
				#shuffle data
				idx = numpy.arange(len(train_labels))
				numpy.random.shuffle(idx)
				shuffled_points = train_points[idx, :, :]
				shuffled_labels = train_labels[idx, :]
				input_points = numpy.zeros((BATCH_SIZE, NUM_POINT, 6))
				input_labels = numpy.zeros((BATCH_SIZE, NUM_POINT))

				#split into batches
				num_batches = int(len(train_labels) / BATCH_SIZE)
				class_loss = []
				class_acc = []
				inner_loss = []
				for batch_id in range(num_batches):
					start_idx = batch_id * BATCH_SIZE
					end_idx = (batch_id + 1) * BATCH_SIZE
					if NUM_POINT == shuffled_points.shape[1]:
						input_points[:] = shuffled_points[start_idx:end_idx,:,:]
						input_labels = shuffled_labels[start_idx:end_idx,:]
					else:
						for i in range(BATCH_SIZE):
							subset = numpy.random.choice(shuffled_points.shape[1], NUM_POINT, replace=False)
							input_points[i,:,:] = shuffled_points[start_idx+i, subset, :]
							input_labels[i,:] = shuffled_labels[start_idx+i,subset]
					input_points, input_labels = jitter_data(input_points, input_labels)
					feed_dict = {net.pointclouds_pl: input_points,
						net.labels_pl: input_labels,
						net.is_training_pl: True}
					
					a1,l1,_ = sess.run([net.class_acc,net.class_loss,net.train_op], feed_dict=feed_dict)
					class_acc.append(a1)
					class_loss.append(l1)
				print('Epoch: %d Loss: %.3f (cls %.3f)'%(epoch,numpy.mean(class_loss), numpy.mean(class_acc)))

				if epoch % VAL_STEP == VAL_STEP - 1:
					#get validation loss
					num_batches = int(len(val_labels) / BATCH_SIZE)
					class_loss = []
					class_acc = []
					inner_loss = []
					for batch_id in range(num_batches):
						start_idx = batch_id * BATCH_SIZE
						end_idx = (batch_id + 1) * BATCH_SIZE
						if NUM_POINT == val_points.shape[1]:
							input_points[:] = val_points[start_idx:end_idx,:,:]
							input_labels = val_labels[start_idx:end_idx,:]
						else:
							for i in range(BATCH_SIZE):
								subset = numpy.random.choice(val_points.shape[1], NUM_POINT, replace=False)
								input_points[i,:,:] = val_points[start_idx+i, subset, :]
								input_labels[i,:] = val_labels[start_idx+i,subset]
						feed_dict = {net.pointclouds_pl: input_points,
							net.labels_pl: input_labels,
							net.is_training_pl: False}
						a1,l1 = sess.run([net.class_acc,net.class_loss], feed_dict=feed_dict)
						class_acc.append(a1)
						class_loss.append(l1)
					print('Validation: %d Loss: %.3f (cls %.3f)'%(epoch,numpy.mean(class_loss), numpy.mean(class_acc)))
			#save trained model
			saver.save(sess, MODEL_PATH)
	
