"""Paper operator: DCA -> PA -> DSA; explicit, independent ablation switches."""
import torch
from torch import nn
from torch.nn import functional as F


class DynamicChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        def mlp():
            return nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False),
                                 nn.ReLU(), nn.Conv2d(hidden, channels, 1, bias=False))
        self.shared = mlp()
        self.modulator = nn.Sequential(mlp(), nn.Sigmoid())

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        base = self.shared(avg) + self.shared(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(base * self.modulator(avg))


class PositionAttention(nn.Module):
    def __init__(self, channels, grid_size):
        super().__init__()
        dim, hidden = max(channels // 4, 16), max(channels // 2, 8)
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, grid_size, grid_size) * .02)
        self.pos_proj = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.BatchNorm2d(hidden),
                                      nn.ReLU(), nn.Conv2d(hidden, 1, 1))
        self.feat_proj = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.BatchNorm2d(hidden),
                                       nn.ReLU(), nn.Conv2d(hidden, 1, 1))

    def forward(self, x):
        prior = F.interpolate(self.pos_embedding, size=x.shape[-2:], mode='bilinear',
                              align_corners=False)
        return torch.sigmoid(self.pos_proj(prior) + self.feat_proj(x))


class DynamicSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2, 2, k, padding=k//2, groups=2, bias=False),
                          nn.Conv2d(2, 1, 1, bias=False)) for k in (3, 5, 7)])
        self.weights = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(2, 8, 1),
                                     nn.ReLU(), nn.Conv2d(8, 3, 1), nn.Softmax(dim=1))

    def forward(self, x):
        descriptor = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), 1)
        weights = self.weights(descriptor)
        return torch.sigmoid(sum(branch(descriptor) * weights[:, i:i+1]
                                 for i, branch in enumerate(self.branches)))


class DPCAM(nn.Module):
    def __init__(self, channels, grid_size, dca=True, pa=True, dsa=True):
        super().__init__()
        self.dca = DynamicChannelAttention(channels) if dca else None
        self.pa = PositionAttention(channels, grid_size) if pa else None
        self.dsa = DynamicSpatialAttention() if dsa else None

    def forward(self, x):
        for module in (self.dca, self.pa, self.dsa):
            if module is not None:
                x = x * module(x)
        return x


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 8, 8)
        self.shared = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(),
                                    nn.Conv2d(hidden, channels, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        x = x * torch.sigmoid(self.shared(F.adaptive_avg_pool2d(x, 1)) +
                              self.shared(F.adaptive_max_pool2d(x, 1)))
        return x * torch.sigmoid(self.spatial(torch.cat((x.mean(1, keepdim=True),
                                                        x.amax(1, keepdim=True)), 1)))


def make_attention(kind, channels, grid_size):
    variants = {'dpcam': (True, True, True), 'dca_dsa': (True, False, True),
                'pa_only': (False, True, False)}
    if kind == 'none':
        return nn.Identity()
    if kind == 'cbam':
        return CBAM(channels)
    if kind not in variants:
        raise ValueError(f'Unknown attention: {kind}')
    return DPCAM(channels, grid_size, *variants[kind])
