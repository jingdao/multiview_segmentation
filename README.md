Dynamic Laser Scanning Dataset for Multi-view Incremental Segmentation
========

Supplementary material (dynamic laser scanning dataset) for the RAL paper
*Multi-view Incremental Segmentation of 3D Point Clouds for Mobile Robots*.

Prerequisites
-----
1. numpy
2. scipy
3. scikit-learn
4. tensorflow
5. ROS
6. h5py

Ray Tracing
-----

Obtain the S3DIS dataset from [here](http://buildingparser.stanford.edu/dataset.html),
which is from the paper *3D Semantic Parsing of Large-Scale Indoor Spaces* by Armeni et al..
Run the following code to generate a ROS bag file as a result of the scan simulation.

	#combine all rooms from Area 3 to a single HDF5 file
	python building_parser_combined.py --area 3

	#perform ray-tracing to generate a bag file
	python raytrace_dynamic.py --area 3


Scan Data
-----

- The point cloud data contains 3D coordinates (XYZ), color (RGB), object instance ID (O), and class ID (C) for each scan point.
- The bag file structure is as follows:

Topic | Data Type | Description
--- | --- | ---
laser_cloud_surround | sensors_msgs/PointCloud2 | Array of (X,Y,Z,R,G,B,O,C) tuples
slam_out_pose | geometry_msgs/PoseStamped | Robot pose at each scan point
trajectory | nav_msgs/Path | Array of sequences of robot poses

Usage
------

	#optional: train a model from scratch
	#select MCPNet as the network architecture (other options are pointnet,pointnet2,voxnet,sgpn)
	#select Area 3 as validation set (Areas 1,2,4,5,6 as training set)
	#model will be saved in models/mcpnet_model3.ckpt
	python train.py --net mcpnet --dataset s3dis --train-area 1,2,4,5,6 --val-area 3

    #optional: train a model for the outdoor dataset
    python train.py --net mcpnet --dataset outdoor --train-area 5_0,7_0,9_1,10_2,10_6 --val-area 16_4

	#start the ROS node for incremental segmentation
	#select Area 3 as the validation dataset
	#select MCPNet as the network architecture
	#use the flag --color to publish original color point cloud scans
	#use the flag --cluster to publish clustering results
	#use the flag --classify to publish classification results
	python inc_seg.py --net mcpnet --area 3 --dataset s3dis
	
	#use RViz as a visualization tool
	rviz -d inc_seg.rviz
	
	#publish the laser scan data from a ROS bag file
	rosbag play data/s3dis_3.bag

Evaluation
---------

    #calculate evaluation metrics after replaying from ROS bag files
    for i in 1 2 3 4 5 6
    do
        python -u inc_seg_replay.py --net mcpnet --area $i --dataset s3dis --save >> results/result_mcpnet.txt
    done

Evaluation for offline methods
---------
	#train an offline PointNet model from scratch
    for i in 1 2 3 4 5 6
    do
        python -u train_offline.py --net pointnet --area $i > results/log_train_offline_pointnet_area$i.txt
    done

	#calculate evaluation metrics using offline PointNet (with room segmentation)
    for i in 1 2 3 4 5 6
    do
        python -u test_offline.py --net pointnet --area $i >> results/result_offline_room_pointnet.txt
    done

    #calculate evaluation metrics using offline PointNet
    for i in 1 2 3 4 5 6
    do
        python -u offline_seg.py --net pointnet --area $i >> results/result_offline_pointnet.txt
    done

    #calculate evaluation metrics using semi-offline PointNet
    for i in 1 2 3 4 5 6
    do
        python -u semi_offline_seg.py --mode space --area $i >> results/result_semi_offline_space.txt
    done

Non-learning baseline methods
--------
    #online segmentation using normal vectors
    python baseline_seg.py --mode normals --dataset outdoor --area 16_4

Reference
--------

	@article{Chen2019,
		author = {Chen,Jingdao and Cho, Yong K. and Kira, Zsolt},
		title = {Multi-view Incremental Segmentation of 3D Point Clouds for Mobile Robots},
		journal = {IEEE Robotics and Automation Letters},
		year = {2019},
	}
	
Links
-----

[IEEE Robotics and Automation Letters published version](https://ieeexplore.ieee.org/document/8624392)

[arxiv preprint version](https://arxiv.org/abs/1902.06768)
	
Screenshots
-----

RGB-mapped laser scanned point cloud

![screenshot1](results/screenshot1.png?raw=true)

Clustering results

![screenshot2](results/screenshot2.png?raw=true)

Classification results

![screenshot3](results/screenshot3.png?raw=true)
