"""Wise-IoU v1 distance attention and optional v3 non-monotonic focusing.

v1 follows manuscript Eqs. 8-10. v3 is the default interpretation of its
non-monotonic focusing prose. Its alpha/delta/EMA settings are reconstruction
choices. Reference: https://arxiv.org/abs/2301.10051
SPDX-License-Identifier: GPL-3.0-only
"""
import torch
from torch import nn


class WiseIoU(nn.Module):
    def __init__(self, version='wiou_v3', alpha=1.9, delta=3.0, half_life=7000):
        super().__init__()
        if version not in ('wiou_v1', 'wiou_v3'):
            raise ValueError(version)
        if alpha <= 1 or delta <= 0 or half_life <= 0:
            raise ValueError('Invalid Wise-IoU hyperparameters')
        self.version, self.alpha, self.delta = version, alpha, delta
        self.momentum = 1 - 0.5 ** (1 / half_life)
        self.register_buffer('iou_mean', torch.tensor(1.0))

    def forward(self, pred, target):
        # FP32 geometry prevents squared-distance overflow under mixed precision.
        p, t = pred.float(), target.float()
        whp = (p[:, 2:] - p[:, :2]).clamp(min=0)
        wht = (t[:, 2:] - t[:, :2]).clamp(min=0)
        inter_wh = (torch.minimum(p[:, 2:], t[:, 2:]) -
                    torch.maximum(p[:, :2], t[:, :2])).clamp(min=0)
        inter = inter_wh.prod(-1, keepdim=True)
        union = whp.prod(-1, keepdim=True) + wht.prod(-1, keepdim=True) - inter
        iou = inter / union.clamp(min=1e-7)
        raw = 1 - iou
        centers = ((p[:, :2] + p[:, 2:]) - (t[:, :2] + t[:, 2:])) / 2
        enclosing = torch.maximum(p[:, 2:], t[:, 2:]) - torch.minimum(p[:, :2], t[:, :2])
        denom = enclosing.square().sum(-1, keepdim=True).detach().clamp(min=1e-7)
        loss = (centers.square().sum(-1, keepdim=True) / denom).exp() * raw
        if self.version == 'wiou_v3' and raw.numel():
            # Validation is no_grad: it must not contaminate the training mean.
            if self.training and torch.is_grad_enabled():
                with torch.no_grad():
                    self.iou_mean.lerp_(raw.detach().mean(), self.momentum)
            beta = raw.detach() / self.iou_mean.clamp(min=1e-7)
            # Log-space avoids overflow as beta grows late in training.
            log_alpha = torch.log(raw.new_tensor(self.alpha))
            focus = (torch.log(beta.clamp(min=1e-12)) -
                     torch.log(raw.new_tensor(self.delta)) -
                     (beta - self.delta) * log_alpha).exp()
            loss = loss * focus
        return loss, iou
