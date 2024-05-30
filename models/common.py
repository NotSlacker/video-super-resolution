import torch
import torch.nn as nn
import torch.nn.functional as F


def ICNR(tensor, initializer, scale_factor, *initializer_args, **initializer_kwargs):
    scale_factor_squared = scale_factor * scale_factor

    assert tensor.shape[0] % scale_factor_squared == 0, (
        "The size of the first dimension: "
        f"tensor.shape[0] = {tensor.shape[0]}"
        " is not divisible by square of scale_factor: "
        f"scale_factor = {scale_factor}"
    )

    sub_kernel = torch.empty(tensor.shape[0] // scale_factor_squared, *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *initializer_args, **initializer_kwargs)

    return sub_kernel.repeat_interleave(scale_factor_squared, dim=0)


class Print(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)

        return x


class ConvShuffle(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        scale_factor=2,
    ):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * scale_factor * scale_factor,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.PixelShuffle(scale_factor),
        )

        weight = ICNR(
            self.body[0].weight,
            initializer=nn.init.kaiming_normal_,
            scale_factor=scale_factor,
        )
        self.body[0].weight.data.copy_(weight)

    def forward(self, x):
        x = self.body(x)

        return x


class Residual(nn.Module):
    def __init__(self, *layers, res_scale=1.0):
        super().__init__()

        self.body = nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, x):
        x = x + self.body(x) * self.res_scale

        return x


class OmniGroupAttention(nn.Module):
    def __init__(self, n_channels=16):
        super().__init__()

        self.n_channels = n_channels

        self.attention_filters = nn.ModuleList(
            [
                nn.Conv2d(n_channels, n_channels, (1, 1), padding=(0, 0)),
                nn.Conv2d(n_channels, n_channels, (1, 5), padding=(0, 2)),
                nn.Conv2d(n_channels, n_channels, (5, 1), padding=(2, 0)),
                nn.Conv2d(n_channels, n_channels, (3, 3), padding=(1, 1)),
            ]
        )
        self.attention_ch = nn.Sequential(
            nn.Conv2d(n_channels * 2, n_channels, 1, bias=False), nn.Sigmoid()
        )
        self.attention_sp = nn.Sequential(nn.Conv2d(2, 1, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        mask = []

        for group, filter in zip(
            torch.split(x, self.n_channels, dim=-3), self.attention_filters
        ):
            g = filter(group)

            ch_avg = torch.mean(g, dim=(-1, -2), keepdim=True)
            ch_max = torch.amax(g, dim=(-1, -2), keepdim=True)

            sp_avg = torch.mean(g, dim=-3, keepdim=True)
            sp_max = torch.amax(g, dim=-3, keepdim=True)

            ch = torch.cat([ch_avg, ch_max], dim=-3)
            sp = torch.cat([sp_avg, sp_max], dim=-3)

            ch = self.attention_ch(ch)
            sp = self.attention_sp(sp)

            mask.append(ch * sp)

        mask = torch.cat(mask, dim=-3)

        x = x * mask

        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels, n_hiddens=16):
        super().__init__()

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.att = nn.Sequential(
            nn.Conv2d(n_channels, n_hiddens, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(n_hiddens, n_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        self.out = nn.Sigmoid()

    def forward(self, x):
        f_avg = self.avg(x)
        f_max = self.max(x)

        f_att = self.att(f_avg) + self.att(f_max)

        x = x * self.out(f_att)

        return x


class SpatialAttention(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.avg = nn.AvgPool2d(3, padding=1, stride=1, count_include_pad=False)
        self.max = nn.MaxPool2d(3, padding=1, stride=1)

        self.att = nn.Sequential(
            nn.Conv2d(
                n_channels * 2, n_channels * 2, 3, padding=2, groups=n_channels * 2
            ),
            nn.Conv2d(n_channels * 2, n_channels, 1),
        )

        self.out = nn.Sigmoid()

    def forward(self, x):
        f_avg = self.avg(x)
        f_max = self.max(x)

        f_att = self.att(torch.cat([f_avg, f_max], dim=-3))

        x = x * self.out(f_att)

        return x


class DualAttention(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.att_ch = ChannelAttention(n_channels)
        self.att_sp = SpatialAttention(n_channels)

        self.out = nn.Sequential(
            nn.Conv2d(n_channels * 2, n_channels, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        f_ch = self.att_ch(x)
        f_sp = self.att_sp(x)

        x = x + self.out(torch.cat([f_ch, f_sp], dim=-3))

        return x


class MultiScaleAttention(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.down_0 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            ChannelAttention(n_channels),
        )
        self.down_1 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1, stride=2),
            nn.LeakyReLU(0.1),
            ChannelAttention(n_channels),
        )
        self.down_2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1, stride=2),
            nn.LeakyReLU(0.1),
            ChannelAttention(n_channels),
        )

        self.out_0 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1, groups=n_channels),
            nn.Conv2d(n_channels, n_channels, 1),
            nn.LeakyReLU(0.1),
        )
        self.out_1 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1, groups=n_channels),
            nn.Conv2d(n_channels, n_channels, 1),
            nn.LeakyReLU(0.1),
        )
        self.out_2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1, groups=n_channels),
            nn.Conv2d(n_channels, n_channels, 1),
            nn.LeakyReLU(0.1),
        )

        self.out = nn.Sequential(
            nn.Conv2d(n_channels * 2, n_channels, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        d_0 = self.down_0(x)
        h_0, w_0 = d_0.size(-2), d_0.size(-1)

        d_1 = self.down_1(x)
        h_1, w_1 = d_1.size(-2), d_1.size(-1)

        d_2 = self.down_2(d_1)

        u_1 = F.interpolate(d_2, size=(h_1, w_1), mode="nearest")
        d_1 = d_1 + u_1

        u_0 = F.interpolate(d_1, size=(h_0, w_0), mode="nearest")
        d_0 = d_0 + u_0

        o_2 = F.interpolate(self.out_2(d_2), size=(h_0, w_0), mode="nearest")
        o_1 = F.interpolate(self.out_1(d_1), size=(h_0, w_0), mode="nearest")
        o_0 = self.out_0(d_0)

        x = o_0 + o_1 + o_2

        return x


class GroupAggregation(nn.Module):
    def __init__(self, n_channels, n_ratio=2):
        super().__init__()

        self.agg_in = nn.Conv2d(n_channels * 2, n_channels, 1)

        self.agg_filters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(n_channels, n_channels, 1),
                    nn.Conv2d(
                        n_channels,
                        n_channels,
                        3,
                        padding=1,
                        dilation=1,
                        groups=n_channels,
                    ),
                    nn.Conv2d(n_channels, n_channels // n_ratio, 1),
                ),
                nn.Sequential(
                    nn.Conv2d(n_channels, n_channels, 1),
                    nn.Conv2d(
                        n_channels,
                        n_channels,
                        3,
                        padding=2,
                        dilation=2,
                        groups=n_channels,
                    ),
                    nn.Conv2d(n_channels, n_channels // n_ratio, 1),
                ),
                nn.Sequential(
                    nn.Conv2d(n_channels, n_channels, 1),
                    nn.Conv2d(
                        n_channels,
                        n_channels,
                        3,
                        padding=4,
                        dilation=4,
                        groups=n_channels,
                    ),
                    nn.Conv2d(n_channels, n_channels // n_ratio, 1),
                ),
            ]
        )

        self.agg_out = nn.Conv2d(n_channels // n_ratio * 3, n_channels, 1)

    def forward(self, x):
        x = self.agg_in(x)
        x = torch.cat([f(x) for f in self.agg_filters], dim=-3)
        x = self.agg_out(x)

        return x


class SpatialGroupAttention(nn.Module):
    def __init__(self, n_channels, n_groups=4):
        super().__init__()

        self.n_channels = n_channels
        self.n_groups = n_groups

        self.process = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1, groups=n_channels),
            nn.Conv2d(n_channels, n_channels, 1),
            nn.ReLU(),
        )

        self.filters = nn.ModuleList(
            [
                nn.Conv2d(n_channels, n_channels // n_groups, (1, 1), padding=(0, 0)),
                nn.Conv2d(n_channels, n_channels // n_groups, (1, 5), padding=(0, 2)),
                nn.Conv2d(n_channels, n_channels // n_groups, (5, 1), padding=(2, 0)),
                nn.Conv2d(n_channels, n_channels // n_groups, (3, 3), padding=(1, 1)),
            ]
        )

        self.mask = nn.Sequential(
            nn.Conv2d((n_channels // n_groups) * len(self.filters), n_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        m = torch.cat(
            [filter(x) for filter in self.filters],
            dim=-3,
        )

        x = self.process(x)
        m = self.mask(m)

        x = x * m

        return x


class DynamicAttention(nn.Module):
    def __init__(self, in_channels, out_channels, n_channels=32, n_hiddens=128):
        super().__init__()

        self.n_channels = n_channels

        self.preprocess = nn.Conv2d(in_channels, n_channels, 1)

        self.body = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 1),
        )

        self.attention = nn.Sequential(
            Residual(
                SpatialAttention(n_channels),
                nn.Conv2d(n_channels, n_channels, 1),
            ),
        )

        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(-3, -1),
            nn.Linear(in_channels, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_channels),
            nn.Softmax(-1),
        )

        self.head = nn.Conv2d(n_channels, out_channels, 1)

    def forward(self, x):
        s = self.scale(x)

        x = self.preprocess(x)

        b = self.body(x)
        a = self.attention(x)

        x = b * s[..., None, None] + a * (1 - s)[..., None, None]

        x = self.head(x)

        return x
