import torch
import torch.nn as nn
from torch.nn import functional as F

import torch
import numpy as np


class IOULoss(nn.Module):
    def __init__(self, smooth: float=1e-6):
        super().__init__()
        self._smooth = smooth

    def forward(self, y_pred, y_true):
        # You can comment out this line if you are passing tensors of equal shape
        # But if you are passing output from UNet or something it will most probably
        # be with the BATCH x 1 x H x W shape
        outputs = y_pred.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

        intersection = (outputs & y_true).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | y_true).float().sum((1, 2))  # Will be zzero if both are 0

        iou = (intersection + self._smooth) / (union + self._smooth)  # We smooth our devision to avoid 0/0

        thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

        return thresholded  # Or thresholded.mean() if you are interested in average across the batch


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha: float = 10.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()

