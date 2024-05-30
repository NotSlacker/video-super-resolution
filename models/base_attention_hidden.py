import torch
import torch.nn as nn
import torchinfo

from functools import partial

from common import Residual, ConvShuffle, MultiScaleAttention


class Net(nn.Module):
    def __init__(self, n_frames, n_channels, scale_factor, n_hiddens, **kwargs):
        super().__init__()

        self.n_frames = n_frames
        self.n_channels = n_channels
        self.scale_factor = scale_factor
        self.n_hiddens = n_hiddens

        self.fe = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(n_channels, 16, 3, padding=1),
                nn.LeakyReLU(0.1),
            )
            for i in range(self.n_frames)
        )

        self.agg_forward = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.agg_backward = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.agg = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, padding=1),
            nn.LeakyReLU(0.1),
        )

        self.agg_hidden = nn.Sequential(
            nn.Conv2d(16 + n_hiddens, 16, 3, padding=1),
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

        self.out_hidden = nn.Sequential(
            nn.Conv2d(16, n_hiddens, 3, padding=1),
            nn.LeakyReLU(0.1),
        )

        self.interpolate = partial(
            nn.functional.interpolate, scale_factor=scale_factor, mode="bilinear"
        )

    def forward(self, inputs):
        B, T, C, H, W = inputs.shape

        h = torch.zeros(B, self.n_hiddens, H, W, device=inputs.device)
        outputs = []

        for i in range(T - self.n_frames + 1):
            x = inputs[:, i : i + self.n_frames, :, :, :]

            x = x.view(B, -1, H, W)
            x, h = self.inference(x, h)
            x = x.view(B, -1, C, H * self.scale_factor, W * self.scale_factor)

            outputs.append(x)

        outputs = torch.cat(outputs, 1)

        return outputs

    def inference(self, x, h):
        x = torch.split(x, self.n_channels, dim=-3)
        r = x[1]

        f = [self.fe[i](x[i]) for i in range(self.n_frames)]

        f = self.agg(
            torch.cat(
                [
                    self.agg_forward(torch.cat([f[0], f[1]], dim=-3)),
                    self.agg_backward(torch.cat([f[2], f[1]], dim=-3)),
                ],
                dim=-3,
            )
        )
        f = self.agg_hidden(torch.cat([h, f], dim=-3))

        f = self.proc(f)

        h = self.out_hidden(f)
        f = self.out(f)

        x = self.interpolate(r) + f

        return x, h


if __name__ == "__main__":
    T = 1

    n_frames = 3
    n_channels = 1
    scale_factor = 2
    n_hiddens = 16

    model = Net(n_frames, n_channels, scale_factor, n_hiddens)

    inputs = torch.rand(1, T + n_frames - 1, n_channels, 360, 640)
    outputs = model(inputs)

    torchinfo.summary(model, input_size=inputs.shape)

    # torch.onnx.export(model, inputs, "onnx/baseline_attention_hidden.onnx")
