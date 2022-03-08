import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from tqdm import tqdm
from visdom import Visdom

from dataset import VOCDataset
from fcos import FCOS
from loss import LocLoss, CenterLoss, ClsLoss
from loss_v2 import Loss

batch_size = 1
lr=1e-4
epochs = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_set = VOCDataset(VOCDetection(root='data', image_set='train', download=False))
test_set = VOCDataset(VOCDetection(root='data', image_set='val', download=False))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = FCOS(n_classes=20).to(device)
# loc_criterion = LocLoss()
# center_criterion = CenterLoss()
# cls_criterion = ClsLoss()
criterion = Loss(20)
optim = Adam(model.parameters(), lr=lr)

viz = Visdom()
viz.line([0],[0],win='train_loss', opts=dict(title='Train Loss'))

for ep in range(epochs):

    model.train()    
    total_loss, count = 0., 0

    for index, (imgs, loc_maps, center_maps, cls_maps) in enumerate(tqdm(train_loader)):
        imgs = imgs.to(device)
        loc_maps_pred, center_maps_pred, cls_maps_pred = model(imgs)

        for i in range(5):
            loc_map_pred = loc_maps_pred[i]
            center_map_pred = center_maps_pred[i]
            cls_map_pred = cls_maps_pred[i]

            loc_map_gt = loc_maps[i].to(device)
            center_map_gt = center_maps[i].to(device)
            cls_map_gt = cls_maps[i].to(device)

            # loc_loss = loc_criterion(loc_map_pred, loc_map_gt)
            # center_loss = center_criterion(center_map_pred, center_map_gt)
            # cls_loss = cls_criterion(cls_map_pred, cls_map_gt)
            # loss = loc_loss + center_loss + cls_loss
            loss = criterion(cls_map_pred,loc_map_pred,center_map_pred,cls_map_gt,loc_map_gt,center_map_gt)

            optim.zero_grad()
            # loc_loss.backward()
            # center_loss.backward()
            # cls_loss.backward()
            loss.backward()
            optim.step()

            # total_loss += loss.item()
            # count += 1
            # if count == 100:
                # viz.line([total_loss/count],[ep*len(train_loader)+index],'train_loss',update='append')
                # total_loss,count = 0., 0
            