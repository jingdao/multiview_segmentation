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

Scan Data
-----

- Download bag files from the following URLs to the data directory
	- https://www.dropbox.com/s/fka0si2xcr78fms/area1.bag?dl=0
	- https://www.dropbox.com/s/xro6c5sf7wniei5/area3.bag?dl=0
	- https://www.dropbox.com/s/6vd1jw0096ka771/area4.bag?dl=0
	- https://www.dropbox.com/s/uxqps4o6rd5lk4m/area5.bag?dl=0
	- https://www.dropbox.com/s/k0a3aos9h3dz6u6/area6.bag?dl=0
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
