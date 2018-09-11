#!/usr/bin/python

import sys
VAL_AREA = 1
for i in range(len(sys.argv)-1):
	if sys.argv[i]=='--area':
		VAL_AREA = int(sys.argv[i+1])
import numpy
import h5py
import glob
from class_util import classes, class_to_id, class_to_color_rgb
import os

gt_points = []
idr = 0
ido = 0
downsample_resolution = 0.05
pointset = set()

for room in glob.glob('data/Stanford3dDataset_v1.2/Area_%d/*'%VAL_AREA):
	if not os.path.isdir(room):
		continue
	print(room)
	room_points = []

	#Process each object in the room
	for obj in glob.glob('%s/Annotations/*.txt'%room):
		obj_type = obj.split('/')[-1].split('_')[0]
		if not obj_type in class_to_id:
			continue

		#read all points from this object
		points = numpy.loadtxt(obj)
		xyz = (points[:,:3] / downsample_resolution).astype(int)
		for i in range(len(xyz)):
			key = tuple(xyz[i])
			if not key in pointset:
				pointset.add(key)
				q = list(points[i]) + [ido, class_to_id[obj_type]]
				gt_points.append(q)
		print(idr, ido, len(points), len(gt_points))
		ido += 1

	idr += 1
#	break

gt_points = numpy.array(gt_points)
h5_fout = h5py.File('data/area%d.h5'%VAL_AREA,'w')
h5_fout.create_dataset(
	'points', data=gt_points,
	compression='gzip', compression_opts=4,
	dtype=numpy.float32)
h5_fout.close()

