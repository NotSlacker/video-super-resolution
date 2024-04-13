import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.contrib import extract_tensor_patches

Y_QUANT = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)

Y_PERCEPTUAL = torch.tensor(
    [
        [
            0.01840373,
            0.58999538,
            0.99887572,
            1.35731124,
            1.51378647,
            1.59615814,
            1.60794642,
            1.59768487,
        ],
        [
            0.54538216,
            0.85649201,
            1.15733541,
            1.48270578,
            1.60280835,
            1.63332936,
            1.6204338,
            1.60154382,
        ],
        [
            0.97357237,
            1.15783744,
            1.37431718,
            1.6003753,
            1.66259869,
            1.65741183,
            1.62199538,
            1.60600917,
        ],
        [
            1.32983939,
            1.48646416,
            1.60701991,
            1.70982965,
            1.70294495,
            1.63899384,
            1.62378931,
            1.61771147,
        ],
        [
            1.51241334,
            1.61619203,
            1.6781811,
            1.69350727,
            1.68488638,
            1.62312361,
            1.60460837,
            1.59458518,
        ],
        [
            1.59135561,
            1.67050663,
            1.67614207,
            1.69062219,
            1.6559363,
            1.61859251,
            1.58166148,
            1.57357605,
        ],
        [
            1.62406025,
            1.64713876,
            1.63513799,
            1.6238076,
            1.6050742,
            1.57471174,
            1.5501197,
            1.54399399,
        ],
        [
            1.6153739,
            1.60502484,
            1.59157153,
            1.58655249,
            1.57093044,
            1.56278208,
            1.54476926,
            1.52678157,
        ],
    ],
    dtype=torch.float32,
)

Y_WEIGHT = Y_PERCEPTUAL / Y_QUANT

T_DCT_ORTHO = torch.tensor(
    [
        [
            0.353553384542,
            0.353553384542,
            0.353553384542,
            0.353553384542,
            0.353553384542,
            0.353553384542,
            0.353553384542,
            0.353553384542,
        ],
        [
            0.490392625332,
            0.415734797716,
            0.277785092592,
            0.097545161843,
            -0.097545161843,
            -0.277785092592,
            -0.415734797716,
            -0.490392625332,
        ],
        [
            0.461939752102,
            0.191341727972,
            -0.191341727972,
            -0.461939752102,
            -0.461939752102,
            -0.191341727972,
            0.191341727972,
            0.461939752102,
        ],
        [
            0.415734797716,
            -0.097545146942,
            -0.490392625332,
            -0.277785122395,
            0.277785122395,
            0.490392625332,
            0.097545146942,
            -0.415734797716,
        ],
        [
            0.353553384542,
            -0.353553384542,
            -0.353553384542,
            0.353553384542,
            0.353553384542,
            -0.353553384542,
            -0.353553384542,
            0.353553384542,
        ],
        [
            0.277785092592,
            -0.490392625332,
            0.097545191646,
            0.415734827518,
            -0.415734827518,
            -0.097545191646,
            0.490392625332,
            -0.277785092592,
        ],
        [
            0.191341713071,
            -0.461939752102,
            0.461939752102,
            -0.191341713071,
            -0.191341713071,
            0.461939752102,
            -0.461939752102,
            0.191341713071,
        ],
        [
            0.097545117140,
            -0.277785152197,
            0.415734797716,
            -0.490392655134,
            0.490392655134,
            -0.415734797716,
            0.277785152197,
            -0.097545117140,
        ],
    ],
    dtype=torch.float32,
)


def spectral_loss(input, target, transform=T_DCT_ORTHO, weight=None, reduction="mean"):
    # input = extract_tensor_patches(input, 8, 8)
    # target = extract_tensor_patches(target, 8, 8)

    B, C, H, W = input.shape

    input = input.reshape(B, C, -1, W // 8, 8).transpose(-2, -3).reshape(B, C, -1, 8, 8)
    target = target.reshape(B, C, -1, W // 8, 8).transpose(-2, -3).reshape(B, C, -1, 8, 8)

    input_transformed = transform @ input @ transform.t()
    target_transformed = transform @ target @ transform.t()

    output = (input_transformed - target_transformed) ** 2

    if weight is not None:
        output = output * weight.to(output)

    if reduction is not None:
        if reduction == "mean":
            output = output.mean()
        elif reduction == "sum":
            output = output.sum()
        else:
            raise ValueError(f"{reduction} is not a valid value for reduction")

    return output


class SpectralLoss(nn.Module):
    def __init__(self, transform=T_DCT_ORTHO, weight=Y_WEIGHT, reduction="mean"):
        super().__init__()

        self.register_buffer("transform", transform)

        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return spectral_loss(input, target, self.transform, self.weight, self.reduction)
