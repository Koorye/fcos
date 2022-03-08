import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection

from dataset import VOCDataset

batch_size = 4
epochs = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_set = VOCDataset(VOCDetection(root='data', image_set='train', download=False))
test_set = VOCDataset(VOCDetection(root='data', image_set='val', download=False))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

for ep in range(epochs):
    pass
