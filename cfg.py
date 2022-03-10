import numpy as np
import random


# 类别列表
classes = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

# 每个类别的颜色
colors = [(random.randrange(0, 256),
           random.randrange(0, 256),
           random.randrange(0, 256)) for _ in range(len(classes))]

# 图片的尺寸
size = (512, 800)

# 图片的缩放比例
scales = (8, 16, 32, 64, 128)

# 每一级特征的最大回归距离范围
m = (-np.inf, 64, 128, 256, 512, np.inf)
