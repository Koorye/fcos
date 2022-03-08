from torch import nn
import torch
class Loss(nn.Module):
  def __init__(self,num_class,alpha=2,gamma=4, lambda_reg = 1, lambda_center=1):
    super(Loss,self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.lambda_reg = lambda_reg
    self.lambda_center = lambda_center
    self.num_class = num_class
  def _focal_loss(self,predict,target):
    # pos_mask = target.gt(0).float()
    # neg_mask = 1 - pos_mask
    # pos_prob = pos_mask.mean()
    # neg_prob = neg_mask.mean()

    loss = -self.alpha*target*torch.pow(1-predict,self.gamma)*torch.log(predict) \
            -self.alpha*(1-target)*torch.pow(predict,self.gamma)*torch.log(1-predict) 
    
    loss = loss.mean()
    #print(target.max())
    #print(target.min())
    return loss
    
  def _cls_loss(self,predict,target):
    # Instead of training multi classifier, we train C binary classifier, therefore
    # the loss would be calculated at each classifier and sum together
    
    loss = 0
    for i in range(self.num_class):
      loss += self._focal_loss(predict[:,i,:,:],target[:,i,:,:])
    return loss
    
  def _reg_loss(self,predict,target):
    d1_gt, d2_gt, d3_gt, d4_gt = torch.split(target,1,1)
    d1_pred, d2_pred, d3_pred, d4_pred = torch.split(predict,1,1)
    area_gt = (d1_gt+d3_gt)*(d2_gt+d4_gt)
    area_pred = (d1_pred+d3_pred)*(d2_pred+d4_pred)
    w_union = torch.min(d3_gt,d3_pred)+ torch.min(d1_gt,d1_pred)
    h_union = torch.min(d2_gt,d2_pred)+ torch.min(d4_gt, d4_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    iou_loss = -torch.log((area_intersect+1)/(area_union+1))
    return iou_loss

  def _center_loss(self,predict,target):
    return nn.BCEWithLogitsLoss(reduce='sum')(predict.squeeze(1),target)
  
  def forward(self,cls_pred,reg_pred,center_pred,cls_gt,reg_gt,center_gt):
    pos_ids = cls_gt.gt(0).float()#1 - cls_gt.lt(1).float()
    #num_pos = pos_ids.sum()
    cls_loss = self._cls_loss(cls_pred,cls_gt)
    reg_loss = (self._reg_loss(reg_pred,reg_gt)*pos_ids).mean()
    center_loss = self._center_loss(center_pred,center_gt)
    total_loss = (cls_loss + self.lambda_reg*reg_loss) + self.lambda_center*center_loss
    return total_loss
    