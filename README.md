Dynamic Laser Scanning Dataset for Multi-view Incremental Segmentation
========

Supplementary material (dynamic laser scanning dataset) for the ICRA submission
*Multi-view Incremental Segmentation of 3D Point Clouds*.

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
- The bag file structures is as follows:

Topic | Data Type | Description
--- | --- | ---
laser_cloud_surround | sensors_msgs/PointCloud2 | Array of (X,Y,Z,R,G,B,O,C) tuples
slam_out_pose | geometry_msgs/PoseStamped | Robot pose at each scan point
trajectory | nav_msgs/Path | Array of sequences of robot poses

Usage
------

	#start the ROS node for incremental segmentation
	#select Area 3 as the validation dataset
	#use the flag --color to publish original color point cloud scans
	#use the flag --instance to publish instance segmentation results
	#use the flag --semantic to publish semantic segmentation results
	python inc_seg.py --area 3
	
	#use RViz as a visualization tool
	rviz -d inc_seg.rviz
	
	#publish the laser scan data from a ROS bag file
	rosbag play data/area3.bag
	
Screenshots
-----

RGB-mapped laser scanned point cloud

![screenshot1](results/screenshot1.png?raw=true)

Instance segmentation results

![screenshot2](results/screenshot2.png?raw=true)

Semantic segmentation results

![screenshot3](results/screenshot3.png?raw=true)
