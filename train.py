from logging import root
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from tqdm import tqdm
from visdom import Visdom

from dataset import VOCDataset
from fcos import FCOS
from loss import Loss
from utils import cls2onehot, heatmap2rgb, heatmaps2rgb, draw_boxes, decode_heatmaps
from cfg import scales, m, size


batch_size = 2
show_every = 20
lr = 1e-4
epochs = 14
history_weights = 'FCOS_epoch3_loss1.6426.pth'

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Use device:', device)

train_set = VOCDataset(root='data', train=True, size=size, scales=scales, m=m)

# 挑选了一张比较有特征的图片用来测试
test_set = VOCDataset(root='data', train=False, size=size, scales=scales, m=m)
img_test, loc_maps_test, center_maps_test, cls_maps_test, _ = test_set[48]
img_test = img_test.unsqueeze(0).to(device)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

model = FCOS(n_classes=20).to(device)
criterion = Loss()

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

for ep in range(history_ep+1, epochs+1):

    model.train()
    loc_loss_total, center_loss_total, cls_loss_total, count = 0., 0., 0., 0

    # 数据格式为列表，包含每一层的特征 -> (b,c,h,w)
    for index, (imgs, loc_maps, center_maps, cls_maps, masks) in enumerate(tqdm(train_loader, desc=f'Epoch{ep}')):
        imgs = imgs.to(device)

        # 返回列表，包含每一层的回归结果 -> (b,c,h,w)
        loc_maps_pred, center_maps_pred, cls_maps_pred = model(imgs)

        # 对每一层进行训练后叠加一起回传
        loc_loss, center_loss, cls_loss = criterion(loc_maps_pred, loc_maps,
                  center_maps_pred, center_maps,
                  cls_maps_pred, cls_maps, 
                  masks)
        
        loss = loc_loss + center_loss + cls_loss

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        loc_loss_total += loc_loss.item()
        center_loss_total += center_loss.item()
        cls_loss_total += cls_loss.item()

        count += 1
        if count == show_every or index == len(train_loader) - 1:
            total_loss = loc_loss_total + center_loss_total + cls_loss_total

            if index == len(train_loader) - 1:
                final_loss = total_loss / count

            viz.line([[total_loss/count,
                       loc_loss_total/count,
                       center_loss_total/count,
                       cls_loss_total/count]],
                     [(ep-1)*len(train_loader)+index], 'train_loss', update='append')
            loc_loss_total, center_loss_total, cls_loss_total, count = 0., 0., 0., 0

            layer = 2
            viz.images(heatmaps2rgb(loc_maps_test[layer], normalize=True),
                       win='loc_gt', nrow=1,
                       opts=dict(title='Loc GT', width=100, height=400))
            viz.image(heatmap2rgb(center_maps_test[layer]),
                      win='center_gt',
                      opts=dict(title='Center GT', width=100, height=100))
            viz.images(heatmaps2rgb(cls2onehot(cls_maps_test[layer], 20)),
                       win='cls_gt', nrow=4,
                       opts=dict(title='Cls GT', width=400, height=500))

            # 取第layer层第0个batch的特征图(较小尺寸的物体)绘制heatmap
            with torch.no_grad():
                loc_maps_test_pred, center_maps_test_pred, cls_maps_test_pred = model(img_test)

                viz.images(heatmaps2rgb(loc_maps_test_pred[layer][0], normalize=True),
                           win='loc_pred', nrow=1,
                           opts=dict(title='Loc Pred', width=100, height=400))
                viz.image(heatmap2rgb(center_maps_test_pred[layer][0]),
                          win='center_pred',
                          opts=dict(title='Center Pred', width=100, height=100))
                viz.images(heatmaps2rgb(cls_maps_test_pred[layer][0]),
                           win='cls_pred', nrow=4,
                           opts=dict(title='Cls Pred', width=400, height=500))

                boxes_gt = decode_heatmaps(loc_maps_test,
                                           center_maps_test,
                                           cls_maps_test,
                                           scales=scales,
                                           use_nms=True)
                boxes_pred = decode_heatmaps(loc_maps_test_pred,
                                             center_maps_test_pred,
                                             cls_maps_test_pred,
                                             scales=scales,
                                             use_nms=True)
                viz.image(draw_boxes(img_test[0], boxes_gt, trans=train_set.untrans),
                          win='box_gt', opts=dict(title='Box GT', width=500, height=400))
                viz.image(draw_boxes(img_test[0], boxes_pred, trans=train_set.untrans),
                          win='box_pred', opts=dict(title='Box Pred', width=500, height=400))

    lr_scheduler.step()

    torch.save({
        'epoch': ep,
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }, f'weights/FCOS_epoch{ep}_loss{final_loss:.4f}.pth')

    torch.cuda.empty_cache()
