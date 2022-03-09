import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from tqdm import tqdm
from visdom import Visdom

from dataset import VOCDataset
from fcos import FCOS
from loss import LocLoss, CenterLoss, ClsLoss
from utils import heatmap2rgb, heatmaps2rgb


batch_size = 1
lr = 1e-3
weight_decay = 1e-4
epochs = 14
history_weights = None

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Use device:', device)

train_set = VOCDataset(VOCDetection(root='data', image_set='train', download=False))

# 挑选了一张比较有特征的图片用来测试
test_set = VOCDataset(VOCDetection(root='data', image_set='val', download=False))
img_test, loc_maps_test, center_maps_test, cls_maps_test = test_set[48]
img_test = img_test.unsqueeze(0).to(device)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

model = FCOS(n_classes=20).to(device)
loc_criterion = LocLoss()
center_criterion = CenterLoss()
cls_criterion = ClsLoss()

# optim = SGD(model.parameters(), lr=lr,
            # momentum=momentem, weight_decay=weight_decay)
optim = Adam(model.parameters(), lr=lr)
lr_scheduler = MultiStepLR(optim, [10, 12], gamma=.1)


history_ep = 0
if history_weights:
    weight_dict = torch.load(f'weights/{history_weights}')
    history_ep = weight_dict['epoch']
    model.load_state_dict(weight_dict['model'])
    optim.load_state_dict(weight_dict['optim'])
    lr_scheduler.load_state_dict(weight_dict['lr_scheduler'])

viz = Visdom()
viz.line([[0, 0, 0, 0]], [0], win='train_loss', opts=dict(
    title='Train Loss', legend=['total', 'loc', 'center', 'cls']))

for ep in range(history_ep, epochs):

    model.train()
    total_loss, loc_loss, center_loss, cls_loss, count = 0., 0., 0., 0., 0

    # 数据格式为列表，包含每一层的特征 -> (b,c,h,w)
    for index, (imgs, loc_maps, center_maps, cls_maps) in enumerate(tqdm(train_loader)):
        imgs = imgs.to(device)

        # 返回列表，包含每一层的回归结果 -> (b,c,h,w)
        loc_maps_pred, center_maps_pred, cls_maps_pred = model(imgs)

        # 对每一层进行训练后叠加一起回传
        loss = 0.
        for i in range(5):
            # 取每一层的回归结果
            loc_map_pred = loc_maps_pred[i]
            center_map_pred = center_maps_pred[i]
            cls_map_pred = cls_maps_pred[i]

            # 取每一层的GT
            loc_map_gt = loc_maps[i].to(device)
            center_map_gt = center_maps[i].to(device)
            cls_map_gt = cls_maps[i].to(device)

            loc_loss_ = loc_criterion(loc_map_pred, loc_map_gt)
            center_loss_ = center_criterion(center_map_pred, center_map_gt)
            cls_loss_ = cls_criterion(cls_map_pred, cls_map_gt)

            print(loc_loss_.item(), center_loss_.item(), cls_loss_.item())
            loss_ = loc_loss_ + center_loss_ + cls_loss_
            loss = loss + loss_

            loc_loss += loc_loss_.item()
            center_loss += center_loss_.item()
            cls_loss += cls_loss_.item()
            total_loss += loss_.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

        count += 1
        if count == 50 or index == len(train_loader) - 1:
            if index == len(train_loader) - 1:
                final_loss = total_loss / count
            viz.line([[total_loss/count,
                       loc_loss/count,
                       center_loss/count,
                       cls_loss/count]],
                     [ep*len(train_loader)+index], 'train_loss', update='append')
            total_loss, loc_loss, center_loss, cls_loss, count = 0., 0., 0., 0., 0

            viz.image(train_set.untrans(img_test[0]),
                      win='img', opts=dict(title='Image', width=300, height=250))
            layer = 2                
            viz.images(heatmaps2rgb(loc_maps_test[layer]), 
                       win='loc_gt', nrow=1, 
                       opts=dict(title='Loc GT', width=100, height=400))
            viz.image(heatmap2rgb(center_maps_test[layer]), 
                      win='center_gt', 
                      opts=dict(title='Center GT', width=100, height=100))
            viz.images(heatmaps2rgb(cls_maps_test[layer]), 
                       win='cls_gt', nrow=4, 
                       opts=dict(title='Cls GT', width=400, height=500))
            # 取第layer层第0个batch的特征图(较小尺寸的物体)绘制heatmap
            with torch.no_grad():
                loc_maps_test_pred, center_maps_test_pred, cls_maps_test_pred = model(img_test) 

                viz.images(heatmaps2rgb(loc_maps_test_pred[layer][0]), 
                           win='loc_pred', nrow=1, 
                           opts=dict(title='Loc Pred',width=100,height=400))
                viz.image(heatmap2rgb(center_maps_test_pred[layer][0]), 
                          win='center_pred', 
                          opts=dict(title='Center Pred', width=100, height=100))
                viz.images(heatmaps2rgb(cls_maps_test_pred[layer][0]),
                           win='cls_pred', nrow=4, 
                           opts=dict(title='Cls Pred', width=400, height=500))


    lr_scheduler.step()

    torch.save({
        'epoch': ep,
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }, f'weights/FCOS_epoch{ep}_loss{final_loss}.pth')

    torch.cuda.empty_cache()
