from dataclasses import dataclass

from torch.nn import Module

from .charbonnier import CharbonnierLoss, charbonnier_loss
from .spectral import SpectralLoss, spectral_loss

implemented = {
    "charbonnier": CharbonnierLoss,
    "spectral": SpectralLoss,
}

@dataclass
class LossWrapper:
    name: str
    func: Module
    weight: float
