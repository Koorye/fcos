import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from torchvision.ops import nms

from cfg import classes, colors


def cls2onehot(cls_map, n_clses):
    """
    将cls_map转换为one hot形式
    : param cls_map <tensor>: (n,) or (h,w) or (b,h,w)
    : param n_clses <int>: 类别数
    : return <tensor>: (n,n_clses) or (n_clses,h,w) or (b,n_clses,h,w)
    """
    
    hw = False
    if len(cls_map.size()) == 2:
        hw = True
        h,w = cls_map.size()
        cls_map = cls_map.flatten()

    bhw = False
    if len(cls_map.size()) == 3:
        bhw = True
        b,h,w = cls_map.size()
        cls_map = cls_map.flatten()
        
    cls_idx = torch.arange(1, n_clses+1).to(cls_map.device).unsqueeze(0)
    cls_map = cls_map.unsqueeze(1)

    cls_map = (cls_map == cls_idx).float()
    
    if hw:
        cls_map = cls_map.view(h,w,n_clses).permute(2,0,1)
    if bhw:
        cls_map = cls_map.view(b,h,w,n_clses).permute(0,3,1,2)
    
    return cls_map    

@torch.no_grad()
def heatmap2rgb(heatmap, theme='jet', normalize=False, scale=1.):
    """
    将二维heatmap转换为rgb图片
    : param heatmap <tensor>: (h,w)
    : return <tensor>: (c,h,w)
    """

    heatmap = heatmap.clone().detach().cpu().numpy()

    cm = plt.get_cmap(theme)
    if normalize:
        heatmap = (heatmap - np.min(heatmap)) / \
            (np.max(heatmap) - np.min(heatmap) + 1e-6)
    else:
        heatmap *= scale
    mapped_data = cm(heatmap)

    # (h,w,c)
    img = np.array(mapped_data)
    img = img[:, :, :3]
    img = torch.tensor(img).permute(2, 0, 1)

    return img


@torch.no_grad()
def heatmaps2rgb(heatmaps, theme='jet', normalize=False, scale=1.):
    """
    将批量二维heatmap转换为rgb图片
    : param heatmaps <tensor>: (b,h,w)
    : return <tensor>: (b,c,h,w)
    """

    out_imgs = []
    for heatmap in heatmaps:
        out_imgs.append(heatmap2rgb(
            heatmap.clone().detach().cpu(), theme, normalize, scale))

    return torch.stack(out_imgs)


@torch.no_grad()
def decode_heatmap(loc_map, center_map, cls_map, scale, thresh=.1, use_nms=False, nms_thresh=.5):
    """
    将heatmap解码为预测框
    : param loc_map <tensor>: (4,h,w)
    : param center_map <tensor>: (h,w)
    : param cls_map <tensor>: (n_classes,h,w) or (h,w)
    : param scale <float>: 缩放比
    : param thresh <float>: 大于阈值的置信度将被认为是预测框
    : param use_nms <bool>: 是否使用nms
    : param nms_thresh <float>: nms的IoU阈值
    : return <tensor>: (n,6) -> cls,score,xmin,ymin,xmax,ymax
    """

    if len(cls_map.size()) == 2:
        cls_map = cls2onehot(cls_map, 20)
    n_classes, h, w = cls_map.size()

    yc, xc = torch.meshgrid(torch.arange(h), torch.arange(w))
    l, t, r, b = loc_map.clone().detach().cpu()

    # (h,w)
    score, cls = cls_map.max(0)
    score *= center_map
    
    cls = cls[score > thresh]
    if len(cls) == 0:
        return torch.tensor([])
    
    l = l[score > thresh]
    t = t[score > thresh]
    r = r[score > thresh]
    b = b[score > thresh]
    yc, xc = yc[score > thresh], xc[score > thresh]

    score = score[score > thresh]

    xmin, ymin, xmax, ymax = xc-l, yc-t, xc+r, yc+b
    xmin *= scale
    ymin *= scale
    xmax *= scale
    ymax *= scale

    boxes = torch.stack([cls,score,xmin,ymin,xmax,ymax], 1)

    if use_nms:
        box_locs, box_scores = boxes[:, 2:], boxes[:, 1]
        keep = nms(box_locs, box_scores, iou_threshold=nms_thresh)
        boxes = boxes[keep]

    return boxes


@torch.no_grad()
def decode_heatmaps(loc_maps, center_maps, cls_maps, scales, thresh=.1, use_nms=False, nms_thresh=.5):
    """
    将批量heatmap解码为预测框
    : param loc_maps <list<tensor>>: [(4,h,w), ...]
    : param center_map <list<tensor>>: [(h,w), ...]
    : param cls_map <list<tensor>>: [(n_classes,h,w), ...] or [(h,w), ...]
    : param scales <list<float>>: 每个heatmap的缩放比
    : param thresh <float>: 大于阈值的置信度将被认为是预测框
    : param use_nms <bool>: 是否使用nms
    : param nms_thresh <float>: nms的IoU阈值
    : return <tensor>: (n,6) -> cls,score,xmin,ymin,xmax,ymax
    """

    boxes = []
    for i in range(len(loc_maps)):
        loc_map, center_map, cls_map = loc_maps[i], center_maps[i], cls_maps[i]

        loc_map = loc_map.clone().detach().cpu().squeeze()
        center_map = center_map.clone().detach().cpu().squeeze()
        cls_map = cls_map.clone().detach().cpu().squeeze()

        boxes_ = decode_heatmap(
            loc_map, center_map, cls_map, scales[i], thresh, use_nms, nms_thresh)

        if len(boxes_) > 0:
            boxes.append(boxes_)

    if len(boxes) == 0:
        return torch.tensor(boxes)

    boxes = torch.cat(boxes)

    if use_nms:
        box_locs, box_scores = boxes[:, 2:], boxes[:, 1]
        keep = nms(box_locs, box_scores, iou_threshold=nms_thresh)
        boxes = boxes[keep]

    return boxes


@torch.no_grad()
def draw_boxes(img, boxes, show=False, trans=None):
    """
    : param img <tensor>: (c,h,w)
    : param boxes <tensor>: (n,6)
    : param show <bool>: 是否展示
    : param untrans <torchvision.transforms>: 是否对图片进行转换
    : return img <tensor>: (c,h,w)
    """
    if trans is not None:
        img = trans(img)

    img = img.clone().detach().cpu().permute(1, 2, 0).numpy()
    if img.max() <= 1.:
        img *= 255
        img = np.uint8(img)

    for box in boxes:
        cls_idx, score, xmin, ymin, xmax, ymax = box
        cls_idx = int(cls_idx)
        score = float(score)
        xmin, ymin, xmax, ymax = math.ceil(xmin), math.ceil(
            ymin), math.ceil(xmax), math.ceil(ymax)
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                            colors[cls_idx], int(math.ceil(score*5)))
        img = cv2.rectangle(img, (xmin, ymin), (xmin+120,
                            ymin+12), colors[cls_idx], -1)
        img = cv2.putText(img, f'{classes[cls_idx]}:{score:.3f}', (
            xmin, ymin+10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)

    if show:
        plt.imshow(img)
        plt.show()

    if isinstance(img, cv2.UMat):
        img = img.get()
        
    img = torch.from_numpy(img).permute(2, 0, 1) / 255.
    return img

