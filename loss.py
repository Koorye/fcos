import torch
from torch import nn


class LocLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, loc_map_pred, loc_map_gt):
        l_pred, t_pred, r_pred, b_pred = torch.split(loc_map_pred,1,1)
        l_gt, t_gt, r_gt, b_gt = torch.split(loc_map_gt,1,1)

        area_pred = (l_pred+r_pred) * (t_pred+b_pred)
        area_gt = (l_gt+r_gt) * (t_gt+b_gt)

        w_union = torch.min(r_pred, r_gt) + torch.min(l_pred, l_gt)
        h_union = torch.min(b_pred, b_gt) + torch.min(t_pred, t_gt)

        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect

        iou_loss = -torch.log((area_intersect+1)/(area_union+1))

        return iou_loss.mean()

class CenterLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, center_map_pred, center_map_gt):
        return nn.BCEWithLogitsLoss(reduce='sum')(center_map_pred.squeeze(1), center_map_gt)
    
class ClsLoss(nn.Module):
    def __init__(self, alpha=2., gamma=4.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, cls_map_pred, cls_map_gt):
        n_classes = cls_map_pred.size(1)
        loss = 0.
        for i in range(n_classes):
            loss_ = -self.alpha * cls_map_gt[:,i] * (1-cls_map_pred[:,i])**self.gamma * torch.log(cls_map_pred[:,i]) \
                   -self.alpha * (1-cls_map_gt[:,i]) * cls_map_pred[:,i]**self.gamma * torch.log(1-cls_map_pred[:,i])
            loss_ = loss_.mean()
            loss = loss + loss_
        return loss
