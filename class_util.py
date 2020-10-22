
#list of all classes
classes = ['clutter', 'board', 'bookcase', 'beam', 'chair', 'column', 'door', 'sofa', 'table', 'window', 'ceiling', 'floor', 'wall']

#integer ID for each class
class_to_id = {classes[i] : i for i in range(len(classes))}

#minimum percentage of object points that has to be in a cell 
point_ratio_threshold = {
	'clutter': 0,
	'board': 0.1,
	'bookcase': 0.5,
	'beam': 0.1,
	'chair': 0.5,
	'column': 0.5,
	'door': 0.5,
	'sofa': 0.1,
	'table': 0.1,
	'window': 0.5,
	'ceiling': 0.01,
	'floor': 0.01,
	'wall': 0.01,
}

#color mapping for semantic segmentation
class_to_color_rgb = {
	0: (200,200,200), #clutter
	1: (0,100,100), #board
	2: (255,0,0), #bookcase
	3: (255,200,200), #beam
	4: (0,0,100), #chair
	5: (0,255,255), #column
	6: (0,100,0), #door
	7: (255,0,255), #sofa
	8: (50,50,50), #table
	9: (0,255,0), #window
	10: (255,255,0), #ceiling
	11: (0,0,255), #floor
	12: (255,165,0), #wall
}

classes_gc = ['clutter', 'floor', 'beam', 'wall', 'girder', 'damaged_slab', 'column', 'debris', 'car', 'damaged_beam', 'damaged_column', 'ground', 'slab', 'damaged_wall', 'container', 'tree', 'door']
class_to_color_rgb_gc = {
    0: (100, 100, 100),
    1: (3, 230, 80),
    2: (243, 251, 221),
    3: (132, 191, 73),
    4: (188, 164, 129),
    5: (32, 215, 23),
    6: (230, 182, 149),
    7: (168, 23, 26),
    8: (186, 187, 31),
    9: (89, 217, 98),
    10: (204, 160, 198), 
    11: (47, 224, 131),
    12: (156, 141, 227),
    13: (188, 221, 112),
    14: (191, 14, 147),
    15: (59, 237, 58),
    16: (184, 252, 88),
}
#{'girder': 4, 'damaged_slab': 5, 'damaged_wall': 13, 'floor': 1, 'wall': 3, 'car': 8, 'tree': 15, 'column': 6, 'damaged_column': 10, 'beam': 2, 'debris': 7, 'clutter': 0, 'door': 16, 'damaged_beam': 9, 'container': 14, 'slab': 12, 'ground': 11}

if __name__=='__main__':
#    import numpy
#    from util import loadPLY
#    gt_labels = numpy.load('data/guardian_centers_concrete/processed/guardian_centers_concrete.npy')
#    pcd,_ = loadPLY('data/guardian_centers_concrete/processed/guardian_centers_concrete_cls_id.ply')
#    class_to_color_rgb_gc = {}
#    for i in range(len(gt_labels)):
#        cid = gt_labels[i,7]
#        if cid not in class_to_color_rgb_gc:
#            class_to_color_rgb_gc[cid] = tuple(pcd[i, 3:6].astype(int))
#    print(class_to_color_rgb_gc)

    import matplotlib.pyplot as plt
    plt.figure()
    for i in [10,11,12,6,9,2,8,7,4,5,3,1,0]:
        c = class_to_color_rgb[i]
        c = (1.0*c[0]/255, 1.0*c[1]/255, 1.0*c[2]/255)
        plt.scatter(0,0,color=c,label=classes[i],s=200)
    plt.legend(ncol=7,prop={'size': 16})
    plt.figure()
    for i in [11,1,2,3,4,6,7,8,12,5,9,10,13,0,14,15,16]:
        c = class_to_color_rgb_gc[i]
        c = (1.0*c[0]/255, 1.0*c[1]/255, 1.0*c[2]/255)
        plt.scatter(0,0,color=c,label=classes_gc[i],s=200)
    plt.legend(ncol=7,prop={'size': 16})
    plt.show()
