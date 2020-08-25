#!/usr/bin/python
import numpy
import os
nets = ['normal','pointnet','pointnet2','sgpn','voxnet','mcpnet_simple','mcpnet']
table1 = []
table2 = []
table3 = []
use_network_size = True
display_name = {
	'normal': "Normal + color",
	'pointnet': "PointNet",
	'pointnet2': "PointNet++",
	'voxnet' : 'VoxNet',
	'sgpn': 'SGPN',
	'mcpnet_simple': 'Proposed',
	'mcpnet': 'Proposed + MCP'
}

for n in nets:
	try:
		network_size  =	os.path.getsize('%s_model1.ckpt.data-00000-of-00001'%n) / 1.0e6
	except OSError:
		network_size = 0
	f = open('result_%s.txt'%n,'r')
	acc = []
	aacc = []
	iou = []
	aiou = []
	nmi = []
	ami = []
	ars = []
	t = []
	cpu = []
	gpu = []
	for l in f:
		if l.strip().startswith('all'):
			acc.append(float(l.split()[4]))	
			iou.append(float(l.split()[6]))	
		if l.strip().startswith('avg'):
			aacc.append(float(l.split()[4]))	
			aiou.append(float(l.split()[6]))	
		if l.startswith('NMI'):
			nmi.append(float(l.split()[1]))	
			ami.append(float(l.split()[3]))	
			ars.append(float(l.split()[5]))	
		if l.startswith('Avg Comp'):
			t.append(float(l.split()[3]))	
		if l.startswith('CPU'):
			cpu.append(float(l.split()[2]))	
		if l.startswith('GPU'):
			gpu.append(float(l.split()[2]))	
	nmi = numpy.mean(nmi)*100
	ami = numpy.mean(ami)*100
	ars = numpy.mean(ars)*100
	acc = numpy.mean(acc)*100
	aacc = numpy.mean(aacc)*100
	iou = numpy.mean(iou)*100
	aiou = numpy.mean(aiou)*100
	t = numpy.mean(t)
	cpu = numpy.mean(cpu)
	gpu = numpy.mean(gpu)
	table1.append([n,aiou,iou,aacc,acc])
	table2.append([n,nmi,ami,ars])
	f.close()
	if use_network_size:
		table3.append([n,t,network_size,cpu])
	else:
		table3.append([n,t,gpu,cpu])

for t in table1:
	print('%s & %.1f & %.1f & %.1f & %.1f \\\\'%(display_name[t[0]],t[1],t[2],t[3],t[4]))
print('')
for t in table2:
	print('%s & %.1f & %.1f & %.1f \\\\'%(display_name[t[0]],t[1],t[2],t[3]))
print('')
for t in table3:
	print('%s & %.3f & %.1f & %.2f \\\\'%(display_name[t[0]],t[1], t[2], t[3]))

for n in nets:
	try:
		f = open('/home/jd/Documents/multiview_segmentation/results/%s.txt'%n,'r')
	except IOError:
		continue
	acc = []
	for l in f:
		if l.strip().startswith('all'):
			acc.append(float(l.split()[10]))	
	print(n,numpy.mean(acc))
	f.close()

