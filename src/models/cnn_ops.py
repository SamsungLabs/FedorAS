# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import torch.nn as nn

class GenericConv(nn.Module):
    def __init__(self, in_channels, out_channels, *args, norm=nn.BatchNorm2d, act=nn.ReLU, **kwargs):
        super().__init__()

        kwargs.setdefault('bias', not bool(norm))
        self.conv = nn.Conv2d(in_channels, out_channels, *args, **kwargs)

        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = None

        if act is not None:
            self.act = act()
        else:
            self.act = None

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SE(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.c1 = nn.Conv2d(channels, channels, 1)
        self.c2 = nn.Conv2d(channels, channels, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        self.c1.reset_parameters()
        self.c2.reset_parameters()

    def forward(self, x):
        y = self.pool(x)
        y = self.c1(y)
        y = self.relu(y)
        y = self.c2(y)
        return self.sigmoid(y) * x


class StdConv(GenericConv):
    def __init__(self, kernel, channels, out_channels=None):
        if out_channels is None:
            out_channels = channels
        super().__init__(channels, out_channels, kernel, padding=kernel//2)


class MBConv(nn.Module):
    def __init__(self, kernel, channels, expansion):
        super().__init__()
        self.channels = channels

        expanded = int(channels * expansion)
        self.conv1 = GenericConv(channels, expanded, kernel, padding=kernel//2)
        self.se = SE(expanded)
        self.conv2 = GenericConv(expanded, channels, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.se(y)
        y = self.conv2(y)
        return x + y

class DSConv(nn.Module):
    def __init__(self, kernel, channels, expansion=1, out_channels=None):
        super().__init__()
        self.channels = channels

        expanded = int(channels * expansion)

        out_channels = channels if out_channels is None else out_channels
        self.conv1 = GenericConv(channels, expanded, kernel, padding=kernel//2, groups=min(channels,expanded))
        self.conv2 = GenericConv(expanded, out_channels, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return y + x
