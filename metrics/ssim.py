import torch
import torch.nn as nn
import torch.nn.functional as F

from piq import ssim as ssim_metric


class SsimMetric(nn.Module):
    def __init__(self, *ssim_args, **ssim_kwargs):
        super().__init__()

        self.ssim_args = ssim_args
        self.ssim_kwargs = ssim_kwargs

    def forward(self, input, target):
        return ssim_metric(input, target, *self.ssim_args, **self.ssim_kwargs)
