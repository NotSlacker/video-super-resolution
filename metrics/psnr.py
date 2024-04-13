import torch
import torch.nn as nn
import torch.nn.functional as F

from piq import psnr as psnr_metric


class PsnrMetric(nn.Module):
    def __init__(self, *psnr_args, **psnr_kwargs):
        super().__init__()

        self.psnr_args = psnr_args
        self.psnr_kwargs = psnr_kwargs

    def forward(self, input, target):
        return psnr_metric(input, target, *self.psnr_args, **self.psnr_kwargs)
