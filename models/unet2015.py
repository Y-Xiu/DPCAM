"""Ronneberger et al. 2015, Fig. 1: valid convolutions and cropped skips.

The PCB adapter mirrors the input boundary and crops logits, so every model can
predict the same 512x512 target. The core itself retains the 572 -> 388 geometry.
This is a paper-based PyTorch implementation, not the authors' Caffe package.
"""
import torch
from torch import nn
from torch.nn import functional as F


def center_crop(x, size):
    h, w = size
    top, left = (x.shape[-2]-h)//2, (x.shape[-1]-w)//2
    if top < 0 or left < 0:
        raise ValueError('Cannot crop to a larger shape')
    return x[..., top:top+h, left:left+w]


def double_conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3), nn.ReLU(inplace=True))


class UNet2015Core(nn.Module):
    def __init__(self, num_classes=5, in_channels=3):
        super().__init__()
        widths = [64, 128, 256, 512, 1024]
        self.encoder = nn.ModuleList(double_conv(a, b) for a, b in
                                     zip([in_channels]+widths[:-1], widths))
        self.up = nn.ModuleList(nn.ConvTranspose2d(a, b, 2, stride=2)
                                for a, b in zip(widths[:0:-1], widths[-2::-1]))
        self.decoder = nn.ModuleList(double_conv(2*c, c) for c in widths[-2::-1])
        self.head = nn.Conv2d(64, num_classes, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        skips = []
        for index, block in enumerate(self.encoder):
            x = block(x)
            if index < 4:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, block, skip in zip(self.up, self.decoder, reversed(skips)):
            x = up(x)
            x = block(torch.cat([center_crop(skip, x.shape[-2:]), x], dim=1))
        return self.head(x)


class UNet2015(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.core = UNet2015Core(num_classes)

    @staticmethod
    def padded_size(n):
        # Valid encoder sizes are 16*k+12; core output is input minus 184.
        return ((n+184-12+15)//16)*16+12

    def forward(self, x):
        size = x.shape[-2:]
        for dim, n in zip((-2, -1), size):
            if n < 2:
                raise ValueError('Mirror padding requires dimensions >= 2')
            target = self.padded_size(n)
            left = (target-n)//2
            indices = torch.arange(-left, target-left, device=x.device) % (2*(n-1))
            indices = torch.minimum(indices, 2*(n-1)-indices)
            x = x.index_select(dim, indices)
        return center_crop(self.core(x), size)
