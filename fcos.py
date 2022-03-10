import torch
from torch import nn
from torchvision.models import resnet50


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = resnet50(pretrained=True)
        self.layer0 = nn.Sequential(*list(self.backbone.children())[:4])
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
    
    def forward(self, x):
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c3,c4,c5

class FPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 = nn.Upsample(scale_factor=2)

        self.down1 = nn.Sequential(
            nn.Conv2d(256,256,1),
            nn.Conv2d(256,256,3,stride=2,padding=1),
            nn.Conv2d(256,256,1),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(256,256,1),
            nn.Conv2d(256,256,3,stride=2,padding=1),
            nn.Conv2d(256,256,1),
        )

        self.trans1 = nn.Conv2d(2048, 256, 1)
        self.trans2 = nn.Conv2d(1024, 256, 1)
        self.trans3 = nn.Conv2d(512, 256, 1)

        self.smooth1 = nn.Conv2d(256,256,1)
        self.smooth2 = nn.Conv2d(256,256,1)
        
        self.apply(self.weight_init_)
    
    def forward(self, x):
        c3, c4, c5 = x
        p5 = self.trans1(c5)
        p4 = self.trans2(c4) + self.up1(p5)
        p3 = self.trans3(c3) + self.up2(p4)
        
        p4 = self.smooth1(p4)
        p3 = self.smooth1(p3)

        p6 = self.down1(p5)
        p7 = self.down2(p6)

        return p3, p4, p5, p6, p7
    
    def weight_init_(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight, a=1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

class Head(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes
        self.conv_cls = nn.Sequential(
            nn.Conv2d(256,256,1),
            nn.GroupNorm(32, 256), 
            nn.ReLU(inplace=True),

            nn.Conv2d(256,256,1),
            nn.GroupNorm(32, 256), 
            nn.ReLU(inplace=True),

            nn.Conv2d(256,256,1),
            nn.GroupNorm(32, 256), 
            nn.ReLU(inplace=True),

            nn.Conv2d(256,256,1),
            nn.GroupNorm(32, 256), 
            nn.ReLU(inplace=True),
        )
        self.conv_reg = nn.Sequential(
            nn.Conv2d(256,256,1),
            nn.GroupNorm(32, 256), 
            nn.ReLU(inplace=True),

            nn.Conv2d(256,256,1),
            nn.GroupNorm(32, 256), 
            nn.ReLU(inplace=True),

            nn.Conv2d(256,256,1),
            nn.GroupNorm(32, 256), 
            nn.ReLU(inplace=True),

            nn.Conv2d(256,256,1),
            nn.GroupNorm(32, 256), 
            nn.ReLU(inplace=True),
        )

        self.cls_branch = nn.Conv2d(256, self.n_classes, 1)
        self.center_branch = nn.Conv2d(256,1,1)
        self.reg_branch = nn.Conv2d(256,4,1)

        self.apply(self.weight_init_)
    
    def forward(self, x, s):
        cls_f = self.conv_cls(x)
        reg_f = self.conv_reg(x)

        cls_map = self.cls_branch(cls_f)
        center_map = self.center_branch(reg_f)
        loc_map = self.reg_branch(reg_f)

        loc_map = torch.exp(s*loc_map)
        center_map = center_map.sigmoid()
        cls_map = cls_map.sigmoid()

        return loc_map, center_map, cls_map

    def weight_init_(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, std=.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

class FCOS(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.backbone = Backbone()
        self.fpn = FPN()
        self.head = Head(n_classes=n_classes)
        
        self.si = nn.Parameter(torch.ones(5).float())
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.fpn(features)

        loc_maps, center_maps, cls_maps = [], [], []

        for idx, feature in enumerate(features):
            loc_map, center_map, cls_map = self.head(feature, self.si[idx])
            loc_maps.append(loc_map)
            center_maps.append(center_map.squeeze(1))
            cls_maps.append(cls_map)
        
        return loc_maps, center_maps, cls_maps
