from dataclasses import dataclass

from torch.nn import Module

from .psnr import PsnrMetric, psnr_metric
from .ssim import SsimMetric, ssim_metric

implemented = {
    "psnr": PsnrMetric,
    "ssim": SsimMetric,
}

@dataclass
class MetricWrapper:
    name: str
    func: Module
