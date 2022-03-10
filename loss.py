import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss

from utils import cls2onehot


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LocLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, loc_map_pred, loc_map_gt):
        """
        回归损失，采用GIoU Loss
        : param loc_map_pred <tensor>: (n,4)
        : param loc_map_gt <tensor>: (n,4)
        """
        
        # (n_positive_samples,)
        l_pred, t_pred, r_pred, b_pred = loc_map_pred[:,0], loc_map_pred[:,1], loc_map_pred[:,2], loc_map_pred[:,3]
        l_gt, t_gt, r_gt, b_gt = loc_map_gt[:,0], loc_map_gt[:,1], loc_map_gt[:,2], loc_map_gt[:,3]
        l_max, t_max, r_max, b_max = torch.max(l_pred, l_gt), torch.max(t_pred, t_gt), torch.max(r_pred, r_gt), torch.max(b_pred, b_gt)

        # (n_positive_samples,)
        area_pred = (l_pred+r_pred) * (t_pred+b_pred)
        area_gt = (l_gt+r_gt) * (t_gt+b_gt)
        area_g = (l_max+r_max) * (t_max+b_max)

        # (n_positive_samples,)
        w_union = torch.min(r_pred, r_gt) + torch.min(l_pred, l_gt)
        h_union = torch.min(b_pred, b_gt) + torch.min(t_pred, t_gt)

        # (n_positive_samples,)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect

        # (n_positive_samples,)
        iou = area_intersect / area_union
        giou = iou - (area_g-area_union) / area_g.clamp_(1e-6)
        giou_loss = 1 - giou
        # iou_loss = -torch.log(iou.clamp_(1e-6))

        return giou_loss.sum()

class CenterLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, center_map_pred, center_map_gt):
        """
        中心度损失，采用BCE Loss
        : param center_map_pred <tensor>: (n,)
        : param center_map_gt <tensor>: (n,)
        """
        
        return F.binary_cross_entropy(center_map_pred, center_map_gt, reduction='sum')
    
class ClsLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, cls_map_pred, cls_map_gt):
        """
        类别损失，采用Focal Loss
        : param cls_map_pred <tensor>: (n,n_classes)
        : param cls_map_gt <tensor>: (n,)
        """

        # (n, n_classes)
        n_clses = cls_map_pred.size(1)
        cls_map_gt = cls2onehot(cls_map_gt, n_clses)

        gamma = 2.
        alpha = .25

        pt = cls_map_pred * cls_map_gt + (1.-cls_map_pred) * (1.-cls_map_gt)
        w = alpha * cls_map_gt + (1.-alpha) * (1-cls_map_gt)
        loss = -w * torch.pow((1.0-pt), gamma) * pt.log()
        
        return loss.sum()

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_criterion = LocLoss()
        self.center_criterion = CenterLoss()
        self.cls_criterion = ClsLoss()
    
    def forward(self, loc_maps_pred, loc_maps_gt, 
                center_maps_pred, center_maps_gt, 
                cls_maps_pred, cls_maps_gt, masks):
        """
        对所有预测值进行多尺度损失计算和汇总
        : param loc_maps_pred <list<tensor>>: [(b,4,h,w), ...]
        : param loc_maps_gt <list<tensor>>: [(b,4,h,w), ...]
        : param center_maps_pred <list<tensor>>: [(b,h,w), ...]
        : param center_maps_gt <list<tensor>>: [(b,h,w), ...]
        : param cls_maps_pred <list<tensor>>: [(b,n_cls,h,w), ...]
        : param cls_maps_gt <list<tensor>>: [(b,h,w), ...]
        : param masks <list<tensor>>: [(b,h,w), ...]
        : return <tuple<tensor>, <tensor>, <tensor>>: (1,), (1,), (1,) 回归、中心度、类别损失
        """
        
        n_layers = len(loc_maps_pred)
        
        loc_map_pred_f, loc_map_gt_f = [], []
        center_map_pred_f, center_map_gt_f = [], []
        cls_map_pred_f, cls_map_gt_f = [], []
        mask_f = []

        for l in range(n_layers):
            # (b,4,h,w)
            loc_map_pred, loc_map_gt = loc_maps_pred[l], loc_maps_gt[l] 
            loc_map_pred = loc_map_pred.permute(0,2,3,1).contiguous().view(-1,4)
            loc_map_gt = loc_map_gt.permute(0,2,3,1).contiguous().view(-1,4)
            loc_map_pred_f.append(loc_map_pred)
            loc_map_gt_f.append(loc_map_gt)

            # (b,h,w)
            center_map_pred, center_map_gt = center_maps_pred[l], center_maps_gt[l]
            center_map_pred = center_map_pred.contiguous().view(-1)
            center_map_gt = center_map_gt.contiguous().view(-1)
            center_map_pred_f.append(center_map_pred)
            center_map_gt_f.append(center_map_gt)

            # (b,n_classes,h,w), (b,h,w)
            cls_map_pred, cls_map_gt = cls_maps_pred[l], cls_maps_gt[l]
            n_classes = cls_map_pred.size(1)
            cls_map_pred = cls_map_pred.permute(0,2,3,1).contiguous().view(-1,n_classes)
            cls_map_gt = cls_map_gt.view(-1)
            cls_map_pred_f.append(cls_map_pred)
            cls_map_gt_f.append(cls_map_gt)
            
            # (n,)
            mask = masks[l].contiguous().view(-1)
            mask_f.append(mask)

        # (n,4)
        loc_map_pred_f, loc_map_gt_f = torch.cat(loc_map_pred_f), torch.cat(loc_map_gt_f)
        # (n,)
        center_map_pred_f, center_map_gt_f = torch.cat(center_map_pred_f), torch.cat(center_map_gt_f)
        # (n,n_classes), (n,)
        cls_map_pred_f, cls_map_gt_f = torch.cat(cls_map_pred_f), torch.cat(cls_map_gt_f)
        # (n,)
        mask_f = torch.cat(mask_f)
        n_pos = torch.sum(mask_f).clamp(1)

        loc_map_pred_f, loc_map_gt_f = loc_map_pred_f[mask_f], loc_map_gt_f[mask_f]        
        center_map_pred_f, center_map_gt_f = center_map_pred_f[mask_f], center_map_gt_f[mask_f]        
        
        loc_map_pred_f, loc_map_gt_f = loc_map_pred_f.to(device), loc_map_gt_f.to(device)
        center_map_pred_f, center_map_gt_f = center_map_pred_f.to(device), center_map_gt_f.to(device)
        cls_map_pred_f, cls_map_gt_f = cls_map_pred_f.to(device), cls_map_gt_f.to(device)
        
        loc_loss = self.loc_criterion(loc_map_pred_f, loc_map_gt_f) / n_pos
        center_loss = self.center_criterion(center_map_pred_f, center_map_gt_f) / n_pos
        cls_loss = self.cls_criterion(cls_map_pred_f, cls_map_gt_f) / n_pos

        return loc_loss, center_loss, cls_loss
        