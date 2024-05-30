import torch
import torch.nn as nn
import torchinfo

from functools import partial

from .common import Residual, ConvShuffle, MultiScaleAttention


class Net(nn.Module):
    def __init__(self, n_frames, n_channels, scale_factor, **kwargs):
        super().__init__()

        self.n_frames = n_frames
        self.n_channels = n_channels
        self.scale_factor = scale_factor

        self.fe = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, padding=1),
            nn.LeakyReLU(0.1),
        )

        self.proc = nn.Sequential(
            Residual(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.LeakyReLU(0.1),
                MultiScaleAttention(16),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.LeakyReLU(0.1),
            ),
            Residual(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.LeakyReLU(0.1),
                MultiScaleAttention(16),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.LeakyReLU(0.1),
            ),
        )

        self.out = ConvShuffle(16, n_channels, 3, padding=1, scale_factor=scale_factor)

        self.interpolate = partial(
            nn.functional.interpolate, scale_factor=scale_factor, mode="bilinear"
        )

    def forward(self, inputs):
        B, T, C, H, W = inputs.shape

        inputs = inputs.view(B, -1, H, W)
        outputs = self.inference(inputs)
        outputs = outputs.view(B, -1, C, H * self.scale_factor, W * self.scale_factor)

        return outputs

    def inference(self, x):
        x = torch.split(x, self.n_channels, dim=-3)
        r = x[1]

        f = self.fe(r)

        f = self.proc(f)
        f = self.out(f)

        x = self.interpolate(r) + f

        return x


if __name__ == "__main__":
    n_frames = 3
    n_channels = 1
    scale_factor = 3

    model = Net(n_frames, n_channels, scale_factor)

    inputs = torch.rand(1, n_frames, n_channels, 360, 640)
    outputs = model(inputs)

    torchinfo.summary(model, input_size=inputs.shape)

    # torch.onnx.export(model, inputs, "onnx/baseline_attention.onnx")
