import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision import transforms as T

from cfg import classes


class VOCDataset(Dataset):
    def __init__(self, base, scales=(8, 16, 32, 64, 128)):
        super().__init__()

        self.base = base
        self.scales = scales

        self.size = (800, 1024)

        self.trans = T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            T.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        
        self.untrans = T.Compose([
            T.Normalize([-.485/.229, -.456/.224, -.406/.225], [1/.229, 1/.224, 1/.225]),
            T.ToPILImage(),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]
        # (cls, xmin, ymin, xmax, ymax)
        boxes = self._parse_target_dict(target)
        img, boxes = self._resize_and_sort(img, boxes)

        loc_maps, center_maps, cls_maps = [], [], []
        for scale in self.scales:
            # 将x,y根据sclae重置尺寸并取整
            # (n_boxes,5)
            boxes_ = np.array(boxes)
            boxes_[:,1:] = boxes_[:,1:] / scale
            boxes_ = boxes_.astype(int)

            loc_map, center_map, cls_map = self._gen_heatmap(boxes_, scale)
            loc_maps.append(loc_map)
            center_maps.append(center_map)
            cls_maps.append(cls_map)

        return img, loc_maps, center_maps, cls_maps

    def _parse_target_dict(self, target):
        """
        解析dict，转换成list[(cls,xmin,ymin,xmax,ymax)]的形式
        """

        boxes = []
        for obj in target['annotation']['object']:
            cls = obj['name']
            cls = classes.index(cls)
            box = obj['bndbox']
            xmin, ymin, xmax, ymax = int(box['xmin']), \
                int(box['ymin']), int(box['xmax']), int(box['ymax'])
            box = (cls, xmin, ymin, xmax, ymax)
            boxes.append(box)
        return boxes

    def _resize_and_sort(self, img, boxes):
        """
        将图片和box重置到目标尺寸，并根据box面积从小到大排序
        """

        w_origin, h_origin = img.size
        img = self.trans(img)
        h, w = self.size
        for ind, box in enumerate(boxes):
            boxes[ind] = (box[0],
                          int(box[1]*w/w_origin),
                          int(box[2]*h/h_origin),
                          int(box[3]*w/w_origin),
                          int(box[4]*h/h_origin))
        boxes.sort(key=lambda x: x[3]*x[4])
        return img, boxes

    def _gen_heatmap(self, boxes, scale):
        h, w = (np.array(self.size) / scale).astype(int)

        loc_map = np.zeros((4, h, w)).astype(float)
        center_map = np.zeros((h, w)).astype(float)
        cls_map = np.zeros((len(classes), h, w))

        # 表示heatmap中已经设置过的位置
        all_mask = np.zeros((h, w))

        x, y = np.meshgrid(range(w), range(h))

        for box in boxes:
            cls, xmin, ymin, xmax, ymax = box

            # 在heatmap大小的zero矩阵中填充一个box大小的矩形作为mask
            tmp_mask = np.zeros((h, w)).astype(int)
            tmp_mask[ymin:ymax+1, xmin:xmax+1] = 1

            # 只保留总mask中还没被其余box覆盖过的位置
            # 由于box面积从小到大排序
            # 面积更小的box将优先被分配特征信息
            tmp_mask = np.where(tmp_mask > all_mask, 1, 0)

            # 将当前box加入总mask中
            all_mask[ymin:ymax+1, xmin:xmax+1] = 1

            cls_map[cls, :, :] += tmp_mask.copy()

            # 计算mask内的锚点相对于left/top/right/bottom的距离
            l = x - xmin
            t = y - ymin
            r = xmax - x
            b = ymax - y
            l[l < 0] = 0
            t[t < 0] = 0
            r[r < 0] = 0
            b[b < 0] = 0
            loc_map[0, :, :] += l * tmp_mask
            loc_map[1, :, :] += t * tmp_mask
            loc_map[2, :, :] += r * tmp_mask
            loc_map[3, :, :] += b * tmp_mask

            # 根据公式(min(l,r)*min(t,b))/(max(l,r)*max(t,b))
            # 计算centerness
            min_lr = np.where(l > r, r, l)
            inv_max_lr = np.where(l > r, 1/(l+1e-8), 1/(r+1e-8))
            min_tb = np.where(t > b, b, t)
            inv_max_tb = np.where(t > b, 1/(t+1e-8), 1/(b+1e-8))
            center_map += np.sqrt(min_lr*inv_max_lr*min_tb*inv_max_tb)

        loc_map = torch.Tensor(loc_map)
        center_map = torch.Tensor(center_map)
        cls_map = torch.Tensor(cls_map)

        return loc_map, center_map, cls_map


if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt

    train_set = VOCDataset(VOCDetection(root='data',
                                        image_set='train',
                                        download=False))
    img, loc_maps, center_maps, cls_maps = train_set[0]
    img = train_set.untrans(img)

    for i in [0,2,4]:
        plt.subplot(3,4,1)
        plt.title('Image')
        plt.imshow(img)

        plt.subplot(3,4,5)
        plt.title('Loc Heatmap Left')
        sns.heatmap(loc_maps[i][0])
        plt.subplot(3,4,6)
        plt.title('Loc Heatmap Top')
        sns.heatmap(loc_maps[i][1])
        plt.subplot(3,4,7)
        plt.title('Loc Heatmap Right')
        sns.heatmap(loc_maps[i][2])
        plt.subplot(3,4,8)
        plt.title('Loc Heatmap Bottom')
        sns.heatmap(loc_maps[i][3])

        plt.subplot(3,4,9)
        plt.title('Centerness Heatmap')
        sns.heatmap(center_maps[i])

        plt.subplot(3,4,10)
        plt.title('Class Heatmap People')
        sns.heatmap(cls_maps[i][14])

        plt.tight_layout()
        plt.show()

        plt.clf()
