import numpy
import rospy
import itertools
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs
from sensor_msgs import point_cloud2
from tf.transformations import quaternion_matrix, quaternion_from_matrix
from tf2_geometry_msgs import do_transform_pose
import tf
import time
from util import savePLY, downsample
import rosbag
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
import sys
import os
import pickle
import itertools

init_time = None
translation = None
rotation = None
global_cloud = []
pose_arr = []
count_msg = 0
previous_time = None

downsample_resolution = 0.1
label_resolution = 0.2
start_time = 0
end_time = numpy.inf
subsample = 1

AREA = None
for i in range(len(sys.argv)-1):
	if sys.argv[i]=='--area':
		AREA = sys.argv[i+1]

if AREA=='5_0':
    velodyne_to_faro = numpy.array([
        [-0.075, 0.997, -0.021, 198.928],
        [-0.997, -0.075, -0.011, 18.381],
        [-0.012, 0.020, 1.000, -0.096],
        [0.000, 0.000, 0.000, 1.000],
    ])
    start_time = 100
    subsample = 5
elif AREA=='7_0':
    velodyne_to_faro = numpy.array([
        [0.887, 0.459, -0.054, 206.789],
        [-0.461, 0.887, -0.035, -5.584],
        [0.032, 0.056, 0.998, -0.154],
        [0.000, 0.000, 0.000, 1.000],
    ])
    start_time = 45
    subsample = 5
elif AREA=='9_1':
    velodyne_to_faro = numpy.array([
        [-0.010, 0.999, -0.046, 195.002],
        [-1.000, -0.012, -0.028, -15.897],
        [-0.028, 0.046, 0.999, -0.533],
        [0.000, 0.000, 0.000, 1.000],
    ])
    end_time = 100
    subsample = 5
elif AREA=='10_2':
    velodyne_to_faro = numpy.array([
        [0.676, 0.737, -0.008, 204.603],
        [-0.736, 0.675, -0.051, -34.668],
        [-0.032, 0.040, 0.999, -0.642],
        [0.000, 0.000, 0.000, 1.000],
    ])
    subsample = 5
elif AREA=='10_6':
    velodyne_to_faro = numpy.array([
        [0.998, 0.069, 0.011, 249.252],
        [-0.068, 0.997, -0.025, -39.141],
        [-0.013, 0.024, 1.000, -0.147],
        [0.000, 0.000, 0.000, 1.000],
    ])
    end_time = 60
    subsample = 5
elif AREA=='16_4':
    velodyne_to_faro = numpy.array([
        [-0.470, 0.881, 0.058, 222.270],
        [-0.882, -0.471, 0.009, 15.934],
        [0.036, -0.047, 0.998, -0.082],
        [0.000, 0.000, 0.000, 1.000],
    ])
    subsample = 5
else:
    velodyne_to_faro = None
faro_offset = numpy.array([234.40, -8.49, 0])
if velodyne_to_faro is not None:
    bag = rosbag.Bag('data/guardian_centers_%s.bag' % AREA, 'w')

#get ground truth instance labels
pkl_path = 'viz/gt_obj_map.pkl'
pkl_path2 = 'viz/gt_cls_map.pkl'
if os.path.exists(pkl_path):
    gt_obj_map = pickle.load(open(pkl_path, 'rb'))
    gt_cls_map = pickle.load(open(pkl_path2, 'rb'))
else:
    gt_labels = numpy.load('data/guardian_centers_concrete/processed/guardian_centers_concrete.npy')
    gt_obj_map = {}
    gt_cls_map = {}
    gt_voxels = [tuple(p) for p in numpy.round(gt_labels[:, :3] / label_resolution).astype(int)]
    for i in range(len(gt_voxels)):
        if not gt_voxels[i] in gt_obj_map:
            gt_obj_map[gt_voxels[i]] = gt_labels[i, 6]
            gt_cls_map[gt_voxels[i]] = gt_labels[i, 7]
    pickle.dump(gt_obj_map, open(pkl_path, 'wb'))
    pickle.dump(gt_cls_map, open(pkl_path2, 'wb'))
print('Loaded gt_map', len(gt_obj_map), len(gt_cls_map))
instance_set = set()

def process_cloud(msg):
    global init_time, count_msg, previous_time
    if count_msg % subsample != 0:
        count_msg += 1
        return
    count_msg += 1
    if init_time is None:
        init_time = msg.header.stamp.to_sec()
        print('init_time', init_time)
    elif msg.header.stamp.to_sec() - init_time < start_time:
        return
    elif msg.header.stamp.to_sec() - init_time > end_time:
        return
    try:
        listener.waitForTransform("map", "base_link", msg.header.stamp, rospy.Duration(1.0))
        translation, rotation = listener.lookupTransform("map", "base_link", msg.header.stamp)
    except Exception as e:
        print(e)
        return
    
    if translation is None:
        pass
    else:
        data = numpy.frombuffer(msg.data, dtype=numpy.float32)
        pcd = data.reshape(-1,8)[:,:3]
        nan_mask = numpy.any(numpy.isnan(pcd), axis=1)
        pcd = pcd[numpy.logical_not(nan_mask), :]
        x_mask = numpy.abs(pcd[:,0]) < 20
        y_mask = numpy.abs(pcd[:,1]) < 20
        z_mask = numpy.abs(pcd[:,2]) < 20
        pcd = pcd[x_mask & y_mask & z_mask, :]

        #apply transformation
        pcd = downsample(pcd, downsample_resolution)
        T = numpy.array(translation)
        R = quaternion_matrix(rotation)[:3,:3]
        pcd = R.dot(pcd.transpose()).transpose() + T
        
        if velodyne_to_faro is None:
            global_cloud.extend(pcd)
            dt = (msg.header.stamp.to_sec() - init_time)
            print("t:%.2f"%dt, pcd.shape, count_msg)
        else:
            if previous_time is None:
                secs = msg.header.stamp.to_sec()
            else:
                secs = previous_time + 0.1
            previous_time = secs
            t = rospy.Time.from_sec(secs)
            msg.header.frame_id = '/map'
            msg.header.stamp = t

            #apply transformation from velodyne frame to FARO frame
            pcd = velodyne_to_faro[:3, :3].dot(pcd.T).T + velodyne_to_faro[:3, 3]

            #get instance labels
            instance_labels = numpy.zeros(len(pcd))
            class_labels = numpy.zeros(len(pcd))
            pcd_voxels = [tuple(p) for p in numpy.round(pcd[:, :3] / label_resolution).astype(int)]
            for i in range(len(pcd_voxels)):
                k = pcd_voxels[i]
                if k in gt_obj_map:
                    instance_labels[i] = gt_obj_map[k]
                    class_labels[i] = gt_cls_map[k]
                    instance_set.add(gt_obj_map[k])
                    continue
                for offset in itertools.product(range(-1,2),range(-1,2),range(-1,2)):
                    kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
                    if kk in gt_obj_map:
                        instance_labels[i] = gt_obj_map[kk]
                        class_labels[i] = gt_cls_map[kk]
                        instance_set.add(gt_obj_map[kk])
                        break

            #apply offset for visualization
            pcd -= faro_offset

            T = TransformStamped()
            T.header = msg.header
            T.transform.translation.x = velodyne_to_faro[0, 3] - faro_offset[0]
            T.transform.translation.y = velodyne_to_faro[1, 3] - faro_offset[1]
            T.transform.translation.z = velodyne_to_faro[2, 3] - faro_offset[2]
            q = quaternion_from_matrix(velodyne_to_faro)
            T.transform.rotation.x = q[0]
            T.transform.rotation.y = q[1]
            T.transform.rotation.z = q[2]
            T.transform.rotation.w = q[3]

            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = translation[0]
            pose.pose.position.y = translation[1]
            pose.pose.position.z = translation[2]
            pose.pose.orientation.x = rotation[0]
            pose.pose.orientation.y = rotation[1]
            pose.pose.orientation.z = rotation[2]
            pose.pose.orientation.w = rotation[3]
            pose = do_transform_pose(pose, T)
            pose_arr.append(pose)
            bag.write('slam_out_pose',pose,t=t)

            path = Path()
            path.header = msg.header
            path.poses = pose_arr
            bag.write('trajectory',path,t=t)

            fields = [
                PointField('x',0,PointField.FLOAT32,1),
                PointField('y',4,PointField.FLOAT32,1),
                PointField('z',8,PointField.FLOAT32,1),
                PointField('r',12,PointField.UINT8,1),
                PointField('g',13,PointField.UINT8,1),
                PointField('b',14,PointField.UINT8,1),
                PointField('o',15,PointField.INT32,1),
                PointField('c',19,PointField.UINT8,1),
            ]
            pcd_with_labels = numpy.zeros((len(pcd), 8))
            pcd_with_labels[:, :3] = pcd
            pcd_with_labels[:, 3:6] = 255
            pcd_with_labels[:, 6] = instance_labels
            pcd_with_labels[:, 7] = class_labels
            pcd_with_labels = point_cloud2.create_cloud(msg.header,fields, pcd_with_labels)
            bag.write('laser_cloud_surround',pcd_with_labels,t=t)

            dt = (msg.header.stamp.to_sec() - init_time)
            print("t:%.2f"%dt, pcd.shape, count_msg, len(instance_set), 'instances', numpy.sum(instance_labels>0), 'labels')

rospy.init_node('bag_converter')
listener = tf.TransformListener()
subscribed_topic = '/full_cloud_projected'
global_cloud_sub = rospy.Subscriber(subscribed_topic, PointCloud2, process_cloud)
print('Listening to %s ...' % subscribed_topic)
rospy.spin()

if velodyne_to_faro is None:
    global_cloud = numpy.array(global_cloud)
    print(global_cloud.shape)
    global_cloud = numpy.hstack((global_cloud, numpy.zeros((len(global_cloud),3))))
    print(global_cloud.shape)
    global_cloud = downsample(global_cloud, 0.05)
    print(global_cloud.shape)
    savePLY('viz/global_cloud.ply', global_cloud)
else:
    bag.close()
