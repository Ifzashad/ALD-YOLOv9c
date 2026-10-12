"""ALD-YOLOv9c reconstruction: manuscript Figures 6, 7 and 9.

Kernel count/reduction ratios are explicit reconstruction choices, not recovered
original hyperparameters. SPDX-License-Identifier: GPL-3.0-only
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class DynConvLayer(nn.Module):
    """Per-image softmax mixture of kernels, followed by BatchNorm and SiLU."""
    def __init__(self, c1, c2, k=3, s=1, experts=4, reduction=4):
        super().__init__()
        if k % 2 != 1 or experts < 1 or reduction < 1:
            raise ValueError('Use odd kernels, positive experts and reduction')
        self.c1, self.c2, self.k, self.s = c1, c2, k, s
        self.experts = experts
        # Name "weight" ensures the upstream optimizer includes the expert bank.
        self.weight = nn.Parameter(torch.empty(experts, c2, c1, k, k))
        hidden = max(4, c1 // reduction)
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(c1, hidden, 1), nn.ReLU(),
                                    nn.Conv2d(hidden, experts, 1))
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        for w in self.weight:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

    def routing_weights(self, x):
        return self.router(x).flatten(1).softmax(1)

    def forward(self, x):
        b, c, h, w = x.shape
        alpha = self.routing_weights(x)
        mixed = (alpha @ self.weight.flatten(1)).reshape(b * self.c2, c, self.k, self.k)
        out = F.conv2d(x.reshape(1, b * c, h, w), mixed,
                       stride=self.s, padding=self.k // 2, groups=b)
        out = out.reshape(b, self.c2, out.shape[-2], out.shape[-1])
        return self.act(self.bn(out))


class FANet(nn.Module):
    """Shared channel MLP, then summed 3/5/7 spatial convolutions (Eqs. 3-7)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False),
                                 nn.ReLU(), nn.Conv2d(hidden, channels, 1, bias=False))
        self.spatial = nn.ModuleList(nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
                                     for k in (3, 5, 7))

    def forward(self, x):
        channel = (self.mlp(F.adaptive_avg_pool2d(x, 1)) +
                   self.mlp(F.adaptive_max_pool2d(x, 1))).sigmoid()
        x = x * channel
        pooled = torch.cat((x.amax(1, keepdim=True), x.mean(1, keepdim=True)), 1)
        spatial = sum(conv(pooled) for conv in self.spatial).sigmoid()
        return x * spatial
