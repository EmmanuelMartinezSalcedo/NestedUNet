import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target, threshold=0.5, smooth=1e-5):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    output = output > threshold
    target = target > threshold
    
    intersection = (output & target).sum(axis=(1,2,3))  # Asumiendo (B,C,H,W)
    union = (output | target).sum(axis=(1,2,3))
    
    # Evitar división por cero
    iou = np.where(union == 0, 1.0, (intersection + smooth) / (union + smooth))
    
    return np.mean(iou)



def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
