import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import VOCDetection

from dataset import VOCDataset


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Use device:', device)

test_set = VOCDataset(VOCDetection(
    root='data', image_set='val', download=False))


for i in range(100):
    print(i)
    img, _,_,_ = test_set[i]
    img = test_set.untrans(img)
    img = img.permute(1,2,0).detach().numpy()
    plt.imshow(img)
    plt.show()