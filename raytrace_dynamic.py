#!/usr/bin/python

import numpy
import h5py
import sys
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import math
import heapq

import roslib
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import rosbag
import std_msgs

VAL_AREA = 1
for i in range(len(sys.argv)-1):
	if sys.argv[i]=='--area':
		VAL_AREA = int(sys.argv[i+1])
data_dir = 'data/Stanford3dDataset_v1.2/Area_%d' % VAL_AREA
resolution = 0.1
resolution_2d = 0.2
scan_height = 1
scan_range = 5
local_range = 2
horiz_res = 1
vert_res = 1
vert_min = -45
vert_max = 80
min_scan_height = 0.1
max_scan_height = 1.4
num_samples = 2048

def shortest_path(x1,y1, x2,y2, drivable):
	closed = set()
	opened = set()
	cameFrom = {}
        F = {(x1,y1):math.sqrt((x1-x2)**2 + (y1-y2)**2)}
	G = {(x1,y1):0}
        H = {(x1,y1):math.sqrt((x1-x2)**2 + (y1-y2)**2)}
	PQ = []
	start = (x1,y1)
	goal = (x2,y2)
	heapq.heappush(PQ, [F[start], start])
	opened.add(start)
	while len(PQ) > 0:
		_,current = heapq.heappop(PQ)
		if current == goal:
			path = [current]
			while current in cameFrom:
				current = cameFrom[current]
				path.append(current)
			path.reverse()
			return path[1:]
		closed.add(current)
		opened.remove(current)
		neighbors = [
			(current[0]-1,current[1]),
			(current[0]+1,current[1]),
			(current[0],current[1]-1),
			(current[0],current[1]+1),
			(current[0]-1,current[1]-1),
			(current[0]-1,current[1]+1),
			(current[0]+1,current[1]-1),
			(current[0]+1,current[1]+1),
		]
		for i in range(len(neighbors)):
			n = neighbors[i]
			cost = 1 if i < 4 else 1.4142135623730951
			if not n in drivable or n in closed:
				continue
			if not n in opened:
				G[n] = G[current] + cost
				H[n] = math.sqrt((n[0]-x2)**2 + (n[1]-y2)**2)
				F[n] = G[n] + H[n]
				cameFrom[n] = current
				opened.add(n)
				heapq.heappush(PQ, [F[n], n])
			elif G[current] + cost < G[n]:
				G[n] = G[current] + cost
				F[n] = G[n] + H[n]
				cameFrom[n] = current
				#decrease key
				for i in range(len(PQ)):
					if PQ[i][1]==n:
						PQ[i][0] = F[n]
						heapq._siftdown(PQ, 0, i)

def batch_shortest_path(x0,y0, targets, drivable):
	closed = set()
	opened = set()
	target_set = set([tuple(t) for t in targets])
	G = {(x0,y0):0}
	cameFrom = {}
	PQ = []
	start = (x0,y0)
	heapq.heappush(PQ, [G[start], start])
	opened.add(start)
	while len(PQ) > 0:
		_,current = heapq.heappop(PQ)
		closed.add(current)
		opened.remove(current)
		if len(closed.intersection(target_set)) == len(target_set):
			distances = []
			paths = []
			for t in targets:
				gval = G[tuple(t)]
				distances.append(gval)
				current = tuple(t)
				path = [current]
				while current in cameFrom:
					current = cameFrom[current]
					path.append(current)
				path.reverse()
				paths.append(path[1:])
			return distances, paths
		neighbors = [
			(current[0]-1,current[1]),
			(current[0]+1,current[1]),
			(current[0],current[1]-1),
			(current[0],current[1]+1),
			(current[0]-1,current[1]-1),
			(current[0]-1,current[1]+1),
			(current[0]+1,current[1]-1),
			(current[0]+1,current[1]+1),
		]
		for i in range(len(neighbors)):
			n = neighbors[i]
			cost = 1 if i < 4 else 1.4142135623730951
			if not n in drivable or n in closed:
				continue
			if not n in opened:
				G[n] = G[current] + cost
				cameFrom[n] = current
				opened.add(n)
				heapq.heappush(PQ, [G[n], n])
			elif G[current] + cost < G[n]:
				G[n] = G[current] + cost
				cameFrom[n] = current
				#decrease key
				for i in range(len(PQ)):
					if PQ[i][1]==n:
						PQ[i][0] = G[n]
						heapq._siftdown(PQ, 0, i)

def two_opt(pairwise_dist, path,i,j):
	if j==len(path)-1:
		if pairwise_dist[path[i-1]][path[-1]] + 1e-6 < pairwise_dist[path[i-1]][path[i]]:
			return path[:i] + path[i:len(path)][::-1]
		else:
			return None
	else:
		if pairwise_dist[path[i-1]][path[j]] + pairwise_dist[path[i]][path[j+1]] + 1e-6 < \
			pairwise_dist[path[i-1]][path[i]] + pairwise_dist[path[j]][path[j+1]]:
			return path[:i] + path[i:j+1][::-1] + path[j+1:]
		else:
			return None

def two_opt_loop(pairwise_dist, path):
	updated = True
	while updated:
		updated = False
		for i in range(1,len(path)-1):
			for j in range(i+1,len(path)):
				new_path = two_opt(pairwise_dist, path, i, j)
				if new_path is not None:
					path = new_path
					updated = True
	return path

def fitness(pairwise_dist, G):
	f = 0
	for i in range(len(G)-1):
		f += pairwise_dist[G[i]][G[i+1]]
	return f

f = h5py.File('data/area%d.h5'%VAL_AREA, 'r')
points = f['points'][:]
f.close()
raster_height = {}
for p in points:
	key = tuple((p[:2]/ resolution_2d).astype(int))
	if not key in raster_height or p[2] < raster_height[key]:
		raster_height[key] = p[2]
floor_height = {}
for k in raster_height:
	floor_height[k] = raster_height[k]
	for i in range(-2,3):
		for j in range(-2,3):
			kk = (k[0]+i, k[1]+j)
			if kk in raster_height and raster_height[kk] < floor_height[k]:
				floor_height[k] = raster_height[kk]
occupied = set()
for p in points:
	key = tuple((p[:2]/ resolution_2d).astype(int))
	if p[2] < floor_height[key] + max_scan_height and p[2] > floor_height[key] + min_scan_height:
		occupied.add(key)
minX = min(p[0] for p in occupied)
minY = min(p[1] for p in occupied)
maxX = max(p[0] for p in occupied)
maxY = max(p[1] for p in occupied)
width = maxX - minX + 1
height = maxY - minY + 1
raster = numpy.zeros((height,width),dtype=bool)
for p in occupied:
	raster[p[1]-minY, p[0]-minX] = True

pointkeys = [tuple(p) for p in numpy.round(points[:,:3]/resolution).astype(int)]
pointmap = {}
for i in range(len(points)):
	if not pointkeys[i] in pointmap:
		pointmap[pointkeys[i]] = points[i]
print('Loaded points',len(pointmap))
scan_spots = numpy.loadtxt(data_dir+'/scan_spots.txt')
path = numpy.round(scan_spots/resolution_2d).astype(int) - [minX, minY]
print('Loaded scan_spots',len(scan_spots))

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

navigable = set()
unoccupied = numpy.nonzero(raster==0)
unoccupied = set(zip(unoccupied[1], unoccupied[0]))
occupied = numpy.nonzero(raster==1)
occupied = set(zip(occupied[1], occupied[0]))
Q = [tuple(path[0])]
while len(Q) > 0:
	q = Q[-1]
	del Q[-1]
	if q in navigable:
		continue
	navigable.add(q)
	n1 = (q[0]-1, q[1])
	n2 = (q[0]+1, q[1])
	n3 = (q[0], q[1]-1)
	n4 = (q[0], q[1]+1)
	if n1 in unoccupied and not n1 in navigable:
		Q.append(n1)
	if n2 in unoccupied and not n2 in navigable:
		Q.append(n2)
	if n3 in unoccupied and not n3 in navigable:
		Q.append(n3)
	if n4 in unoccupied and not n4 in navigable:
		Q.append(n4)

path = numpy.array(filter(lambda x:tuple(x) in navigable,path))
print('Valid scan_spots',len(path))
N = len(path)
pairwise_dist = {}
for i in range(N):
	pairwise_dist[i] = {}
for i in range(N-1):
	candidates = path[i+1:]
	dist,_ = batch_shortest_path(path[i][0],path[i][1], candidates, navigable)
	for j in range(i+1, N):
		pairwise_dist[i][j] = dist[j-1-i]
		pairwise_dist[j][i] = dist[j-1-i]
print('Calculated pairwise_dist')
best_dist = None
best_path = None
for i in range(10):
	gene = [0] + list(numpy.random.choice(range(1,N),N-1,replace=False))
	gene = two_opt_loop(pairwise_dist, gene)
	d = fitness(pairwise_dist, gene)
	if best_dist is None or d < best_dist:
		best_dist = d
		best_path = [path[j] for j in gene]
print('Optimized path %.0f -> %.0f'%(fitness(pairwise_dist, range(N)),best_dist))
path = best_path

def plotLine3D(x0,y0,z0,x1,y1,z1,pointmap):
	sx, sy, sz = numpy.sign([x1-x0, y1-y0, z1-z0])
	dx, dy, dz = numpy.abs([x1-x0, y1-y0, z1-z0])
	dm = max(dx,dy,dz)
	x1 = y1 = z1 = dm/2

	for i in range(dm):
		if (x0,y0,z0) in pointmap:
			return (x0,y0,z0)
		x1 -= dx
		if x1 < 0:
			x1 += dm
			x0 += sx
		y1 -= dy
		if y1 < 0:
			y1 += dm
			y0 += sy
		z1 -= dz
		if z1 < 0:
			z1 += dm
			z0 += sz
	return None

def collect_scan_discrete(x, y, z, pointmap, scan_range):
	new_scan = set()
	for i in numpy.arange(0,360,horiz_res):
		for j in numpy.arange(vert_min,vert_max,vert_res):
			x1 = int(numpy.round(x / resolution))
			y1 = int(numpy.round(y / resolution))
			z1 = int(numpy.round(z / resolution))
			rx = numpy.cos(j * numpy.pi / 180) * numpy.cos(i * numpy.pi / 180)
			ry = numpy.cos(j * numpy.pi / 180) * numpy.sin(i * numpy.pi / 180)
			rz = numpy.sin(j * numpy.pi / 180)
			x2 = int(numpy.round((x + rx * scan_range)/resolution))
			y2 = int(numpy.round((y + ry * scan_range)/resolution))
			z2 = int(numpy.round((z + rz * scan_range)/resolution))
			collision = plotLine3D(x1,y1,z1,x2,y2,z2,pointmap)
			if collision is not None:
				new_scan.add(collision)
	return new_scan

rospy.init_node('raytrace_dynamic')
t = rospy.Time.now().to_sec()
bag = rosbag.Bag('data/area%d.bag'%VAL_AREA, 'w')

def publish_cloud(cloud, t):
	header = std_msgs.msg.Header()
	header.stamp = t
	header.frame_id = '/map'
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
	msg = point_cloud2.create_cloud(header,fields, cloud)
	bag.write('laser_cloud_surround',msg,t=t)

def publish_pose(x1, y1, x2, y2, z2, t):
	x1 = (x1+minX)*resolution_2d
	y1 = (y1+minY)*resolution_2d
	x2 = (x2+minX)*resolution_2d
	y2 = (y2+minY)*resolution_2d
	msg = PoseStamped()
	msg.header = std_msgs.msg.Header()
	msg.header.stamp = t
	msg.header.frame_id = '/map'
	msg.pose.position.x = x2
	msg.pose.position.y = y2
	msg.pose.position.z = z2
	v = numpy.array([x2-x1, -(y2-y1)]).astype(float)
	v /= numpy.linalg.norm(v)
	theta = numpy.arctan2(v[1],v[0])
	msg.pose.orientation.x = 0
	msg.pose.orientation.y = 0
	msg.pose.orientation.z = 0
	msg.pose.orientation.w = 1
	if not hasattr(publish_pose, 'path'):
		publish_pose.path = []
	publish_pose.path.append(msg)
	bag.write('slam_out_pose',msg,t=t)
	msg = Path()
	msg.header = std_msgs.msg.Header()
	msg.header.stamp = t
	msg.header.frame_id = '/map'
	msg.poses = publish_pose.path
	bag.write('trajectory',msg,t=t)

for j in range(len(path)):
	source = path[j]
	if tuple(source) in navigable:
		break

trajectory_x = []
trajectory_y = []
gt_points = []
scan_offset = []
for i in range(j+1, len(path)):
	target = path[i]
	if not tuple(target) in navigable:
		print('Skip',source, target)
		continue
	current_path = shortest_path(source[0], source[1], target[0], target[1], navigable)
	for j in range(len(current_path)):
		xi = current_path[j][0]
		yi = current_path[j][1]
		trajectory_x.append(xi)
		trajectory_y.append(yi)
		x = (xi+minX) * resolution_2d
		y = (yi+minY) * resolution_2d
		try:
			z = floor_height[(xi+minX,yi+minY)] + scan_height
		except KeyError:
			continue
		if len(trajectory_x) >= 2:
			new_scan = collect_scan_discrete(x,y,z,pointmap,scan_range)
			xyz = numpy.array(list(new_scan)) * resolution
			rgboc = numpy.array([pointmap[p][[3,4,5,6,7]] for p in new_scan])
			pcd = numpy.hstack((xyz, rgboc))
			if len(pcd) < 0.25 * num_samples:
				continue
			print('Processed %d/%d: %d points @ %.1f'%(i, len(path), len(pcd), z))
			publish_pose(trajectory_x[-2], trajectory_y[-2], trajectory_x[-1], trajectory_y[-1], z, rospy.Time.from_sec(t))
			publish_cloud(pcd, rospy.Time.from_sec(t))
			t += 0.1
			local_mask = numpy.sum((pcd[:,:2] - [x,y])**2, axis=1) < local_range * local_range
			pcd = pcd[local_mask]
			if len(pcd) > num_samples:
				samples = numpy.random.choice(len(pcd), num_samples, False)
			else:
				samples = range(len(pcd)) + list(numpy.random.choice(len(pcd), num_samples-len(pcd), True))
			pcd = pcd[samples, :]
			pcd[:,:3] -= [x,y,z]
			pcd[:,3:6] = pcd[:,3:6] / 255.0 - 0.5
			gt_points.append(pcd)
			scan_offset.append([x,y,z])
		ax.clear()
		plt.imshow(raster, interpolation='none', origin='lower')
		plt.scatter(xi, yi, color='r', s=20)
		plt.scatter(target[0], target[1], color='g', s=20)
		plt.plot(trajectory_x, trajectory_y, 'b-')
		plt.pause(0.025)
	source = target

bag.close()

