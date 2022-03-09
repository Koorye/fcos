import matplotlib.pyplot as plt
import numpy as np
import torch

def heatmap2rgb(heatmap, theme='jet'):
    """
    : heatmap <tensor>: (h,w)
    : return <tensor>: (c,h,w)
    """

    heatmap = heatmap.detach().cpu().numpy()

    cm = plt.get_cmap(theme)
    normed_data = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-6)
    mapped_data = cm(normed_data)

    # (h,w,c)
    img = np.array(mapped_data)
    img = img[:,:,:3]
    img = torch.tensor(img).permute(2, 0, 1)
    
    return img


def heatmaps2rgb(heatmaps):
    """
    : heatmaps <tensor>: (b,h,w)
    : return <tensor>: (b,c,h,w)
    """

    out_imgs = []
    for heatmap in heatmaps:
        out_imgs.append(heatmap2rgb(heatmap))

    return torch.stack(out_imgs)

