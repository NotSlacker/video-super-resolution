import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def charbonnier_loss(input, target, eps, reduction="mean"):
    output = torch.sqrt((input - target) ** 2 + eps ** 2)

    if reduction is not None:
        if reduction == "mean":
            output = output.mean()
        elif reduction == "sum":
            output = output.sum()
        else:
            raise ValueError(f"{reduction} is not a valid value for reduction")

    return output

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()

        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        return charbonnier_loss(input, target, self.eps, self.reduction)
