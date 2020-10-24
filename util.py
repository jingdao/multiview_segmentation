import numpy
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from class_util import classes
import time

def loadPLY(filename):
	vertices = []
	faces = []
	numV = 0
	numF = 0
	f = open(filename,'r')
	while True:
		l = f.readline()
		if l.startswith('element vertex'):
			numV = int(l.split()[2])
		elif l.startswith('element face'):
			numF = int(l.split()[2])
		elif l.startswith('end_header'):
			break
	for i in range(numV):
		l = f.readline()
		vertices.append([float(j) for j in l.split()])
	for i in range(numF):
		l = f.readline()
		faces.append([int(j) for j in l.split()[1:4]])
	f.close()
	return numpy.array(vertices),numpy.array(faces)

def savePLY(filename, points, faces=None):
	f = open(filename,'w')
	f.write("""ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar r
property uchar g
property uchar b
""" % len(points))
	if faces is None:
		f.write("end_header\n")
	else:
		f.write("""
element face %d
property list uchar int vertex_index
end_header
""" % (len(faces)))
	for p in points:
		f.write("%f %f %f %d %d %d\n"%(p[0],p[1],p[2],p[3],p[4],p[5]))
	if not faces is None:
		for p in faces:
			f.write("3 %d %d %d\n"%(p[0],p[1],p[2]))
	f.close()
	print('Saved to %s: (%d points)'%(filename, len(points)))

def loadPCD(filename):
	pcd = open(filename,'r')
	for l in pcd:
		if l.startswith('DATA'):
			break
	points = []
	for l in pcd:
		ll = l.split()
		x = float(ll[0])
		y = float(ll[1])
		z = float(ll[2])
		if len(ll)>3:
			rgb = int(ll[3])
			b = rgb & 0xFF
			g = (rgb >> 8) & 0xFF
			r = (rgb >> 16) & 0xFF
			points.append([x,y,z,r,g,b])
		else:
			points.append([x,y,z])
	pcd.close()
	points = numpy.array(points)
	return points

def savePCD(filename,points):
	if len(points)==0:
		return
	f = open(filename,"w")
	l = len(points)
	header = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F I
COUNT 1 1 1 1
WIDTH %d
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS %d
DATA ascii
""" % (l,l)
	f.write(header)
	for p in points:
		rgb = (int(p[3]) << 16) | (int(p[4]) << 8) | int(p[5])
		f.write("%f %f %f %d\n"%(p[0],p[1],p[2],rgb))
	f.close()
	print('Saved %d points to %s' % (l,filename))

def get_cls_id_metrics(gt_cls_id, predicted_cls_id, class_labels=classes, printout=False):
    stats = {}
    stats['all'] = {'tp':0, 'fp':0, 'fn':0} 
    for c in class_labels:
        stats[c] = {'tp':0, 'fp':0, 'fn':0} 
    for g in range(len(predicted_cls_id)):
        if gt_cls_id[g] == predicted_cls_id[g]:
            stats[class_labels[int(gt_cls_id[g])]]['tp'] += 1
            stats['all']['tp'] += 1
        else:
            stats[class_labels[int(gt_cls_id[g])]]['fn'] += 1
            stats['all']['fn'] += 1
            stats[class_labels[predicted_cls_id[g]]]['fp'] += 1
            stats['all']['fp'] += 1

    prec_agg = []
    recl_agg = []
    iou_agg = []
    if printout:
        print("%15s %6s %6s %6s %5s %5s %5s"%('CLASS','TP','FP','FN','PREC','RECL','IOU'))
    for c in sorted(stats.keys()):
        try:
            stats[c]['pr'] = 1.0 * stats[c]['tp'] / (stats[c]['tp'] + stats[c]['fp'])
        except ZeroDivisionError:
            stats[c]['pr'] = 0
        try:
            stats[c]['rc'] = 1.0 * stats[c]['tp'] / (stats[c]['tp'] + stats[c]['fn'])
        except ZeroDivisionError:
            stats[c]['rc'] = 0
        try:
            stats[c]['IOU'] = 1.0 * stats[c]['tp'] / (stats[c]['tp'] + stats[c]['fp'] + stats[c]['fn'])
        except ZeroDivisionError:
            stats[c]['IOU'] = 0
        if c not in ['all']:
            if printout:			
                print("%15s %6d %6d %6d %5.3f %5.3f %5.3f"%(c,stats[c]['tp'],stats[c]['fp'],stats[c]['fn'],stats[c]['pr'],stats[c]['rc'],stats[c]['IOU']))
            prec_agg.append(stats[c]['pr'])
            recl_agg.append(stats[c]['rc'])
            iou_agg.append(stats[c]['IOU'])

    if printout:
        c = 'all'
        print("%15s %6d %6d %6d %5.3f %5.3f %5.3f"%('all',stats[c]['tp'],stats[c]['fp'],stats[c]['fn'],stats[c]['pr'],stats[c]['rc'],stats[c]['IOU']))
        print("%15s %6d %6d %6d %5.3f %5.3f %5.3f"%('avg',stats[c]['tp'],stats[c]['fp'],stats[c]['fn'],numpy.mean(prec_agg),numpy.mean(recl_agg),numpy.mean(iou_agg)))

    acc = stats['all']['pr']
    iou = stats['all']['IOU']
    avg_acc = numpy.mean(prec_agg)
    avg_iou = numpy.mean(iou_agg)
    return acc, iou, avg_acc, avg_iou, stats

def get_obj_id_metrics(gt_obj_id, predicted_obj_id):
    nmi = normalized_mutual_info_score(gt_obj_id, predicted_obj_id)
    ami = adjusted_mutual_info_score(gt_obj_id, predicted_obj_id)
    ars = adjusted_rand_score(gt_obj_id, predicted_obj_id)
    hom = homogeneity_score(gt_obj_id, predicted_obj_id)
    com = completeness_score(gt_obj_id, predicted_obj_id)
    vms = 2 * hom * com / (hom + com)

    unique_id, count = numpy.unique(gt_obj_id, return_counts=True)
    # only calculate instance metrics if small number of instances
    if len(unique_id) < 100:
        gt_match = 0
        dt_match = numpy.zeros(predicted_obj_id.max(), dtype=bool)
        mean_iou = []
        for k in range(len(unique_id)):
            i = unique_id[numpy.argsort(count)][::-1][k]
            best_iou = 0
            for j in range(1, predicted_obj_id.max()+1):
                if not dt_match[j-1]:
                    iou = 1.0 * numpy.sum(numpy.logical_and(gt_obj_id==i, predicted_obj_id==j)) / numpy.sum(numpy.logical_or(gt_obj_id==i, predicted_obj_id==j))
                    best_iou = max(best_iou, iou)
                    if iou > 0.5:
                        dt_match[j-1] = True
                        gt_match += 1
                        break
            mean_iou.append(best_iou)
        prc = numpy.mean(dt_match)
        rcl = 1.0 * gt_match / len(set(gt_obj_id))
        mean_iou = numpy.mean(mean_iou)
    else:
        prc = rcl = mean_iou = numpy.nan

    return nmi, ami, ars, prc, rcl, mean_iou, hom, com, vms 

def downsample(cloud, resolution=0.1):
    voxel_set = set()
    output_cloud = []
    voxels = [tuple(k) for k in numpy.round(cloud[:, :3]/resolution).astype(int)]
    for i in range(len(voxels)):
        if not voxels[i] in voxel_set:
            output_cloud.append(cloud[i])
            voxel_set.add(voxels[i])
    return numpy.array(output_cloud) 
