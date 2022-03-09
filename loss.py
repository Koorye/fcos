import torch
from torch import nn


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LocLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, loc_map_pred, loc_map_gt):
        l_pred, t_pred, r_pred, b_pred = torch.split(loc_map_pred,1,1)
        l_gt, t_gt, r_gt, b_gt = torch.split(loc_map_gt,1,1)
        positive_idx = l_gt > 0
        if len(l_pred[positive_idx]) == 0:
            return torch.tensor([0]).float().to(device)

        area_pred = (l_pred+r_pred) * (t_pred+b_pred)
        area_gt = (l_gt+r_gt) * (t_gt+b_gt)

        w_union = torch.min(r_pred, r_gt) + torch.min(l_pred, l_gt)
        h_union = torch.min(b_pred, b_gt) + torch.min(t_pred, t_gt)

        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect

        iou = torch.clamp((area_intersect+1) / (area_union+1), 1e-4, 1)
        iou_loss = -torch.log(iou)

        return iou_loss[positive_idx].mean()

class CenterLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, center_map_pred, center_map_gt):
        center_map_pred = torch.clamp(center_map_pred, 1e-4, 1-1e-4)

        positive_idx = center_map_gt > 0
        if len(center_map_pred[positive_idx]) == 0:
            return torch.tensor([0]).float().to(device)
            
        loss = -center_map_gt * torch.log(center_map_pred) - (1-center_map_gt) * torch.log(1-center_map_pred)

        return loss[positive_idx].mean()
    
class ClsLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, cls_map_pred, cls_map_gt):
        cls_map_pred = torch.clamp(cls_map_pred, 1e-4, 1-1e-4)

        loss = -self.alpha * cls_map_gt * (1-cls_map_pred)**self.gamma * torch.log(cls_map_pred) \
            -(1-self.alpha) * (1-cls_map_gt) * cls_map_pred**self.gamma * torch.log(1-cls_map_pred)

        # (b,n_classes,h,w)
        n_classes = loss.size(1)
        
        return loss.mean() * n_classes
