import numpy as np
import torch
import torchvision.transforms.v2 as T

import utils


class VideoRandomCrop(T.Transform):
    def __init__(self, size):
        self.size = size

    def __call__(self, lr, hr):
        lr_h, lr_w = lr.shape[-2:]
        hr_h, hr_w = hr.shape[-2:]

        scale_h, scale_w = hr_h // lr_h, hr_w // lr_w

        i, j, h, w = T.RandomCrop.get_params(lr, output_size=self.size)

        lr = T.functional.crop(lr, i, j, h, w)
        hr = T.functional.crop(hr, i * scale_h, j * scale_w, h * scale_h, w * scale_w)

        return lr, hr


class VideoRandomHorizontalFlip(T.Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, lr, hr):
        if torch.rand(1) < self.p:
            lr = T.functional.hflip(lr)
            hr = T.functional.hflip(hr)

        return lr, hr


class VideoRandomVerticalFlip(T.Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, lr, hr):
        if torch.rand(1) < self.p:
            lr = T.functional.vflip(lr)
            hr = T.functional.vflip(hr)

        return lr, hr


class VideoToDtype(T.Transform):
    def __init__(self, dtype=torch.float32, scale=True):
        self.dtype = dtype
        self.scale = scale

    def __call__(self, lr, hr):
        lr = T.functional.to_dtype(lr, self.dtype, self.scale)
        hr = T.functional.to_dtype(hr, self.dtype, self.scale)

        return lr, hr


class VideoRgbToYuv400(T.Transform):
    def __init__(self, color_matrix="709", color_range="full"):
        self.color_matrix = color_matrix
        self.color_range = color_range

    def __call__(self, lr, hr):
        lr = utils.rgb_to_y(lr, self.color_matrix, self.color_range)
        hr = utils.rgb_to_y(hr, self.color_matrix, self.color_range)

        return lr, hr


class VideoRgbToYuv444(T.Transform):
    def __init__(self, color_matrix="709", color_range="full"):
        self.color_matrix = color_matrix
        self.color_range = color_range

    def __call__(self, lr, hr):
        lr = utils.rgb_to_yuv(lr, self.color_matrix, self.color_range)
        hr = utils.rgb_to_yuv(hr, self.color_matrix, self.color_range)

        return lr, hr
