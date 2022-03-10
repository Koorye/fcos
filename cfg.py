import random


classes = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

colors = [(random.randrange(0,256), 
           random.randrange(0,256), 
           random.randrange(0,256)) for _ in range(len(classes))]

scales=(8, 16, 32, 64, 128)