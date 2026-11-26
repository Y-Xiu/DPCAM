"""Shared U-shaped decoder with explicitly attached attention at each fusion block."""
import torch
from torch import nn
from torch.nn import functional as F
from .attention import make_attention


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU(),
                         nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU())


class PlainEncoder(nn.Module):
    """Explicit replacement for ConvNeXt in the no-backbone ablation."""
    channels = [64, 128, 256, 512]
    reductions = [4, 8, 16, 32]

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(DoubleConv(3, 32), nn.MaxPool2d(2))
        self.blocks = nn.ModuleList([DoubleConv(a, b)
                                    for a, b in zip([32, 64, 128, 256], self.channels)])

    def forward(self, x):
        x = self.stem(x)
        features = []
        for block in self.blocks:
            x = block(F.max_pool2d(x, 2))
            features.append(x)
        return features


class Fusion(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, attention, grid_size):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)
        self.attention = make_attention(attention, out_ch, grid_size)

    def forward(self, x, skip=None):
        size = skip.shape[-2:] if skip is not None else (x.shape[-2]*2, x.shape[-1]*2)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat((x, skip), 1)
        return self.attention(self.conv(x))


class PaperUNet(nn.Module):
    def __init__(self, backbone='convnext_base', attention='dpcam', num_classes=5,
                 image_size=512, pretrained=True):
        super().__init__()
        if backbone == 'plain':
            self.encoder = PlainEncoder()
            channels, reductions = self.encoder.channels, self.encoder.reductions
        else:
            import timm
            self.encoder = timm.create_model(backbone, features_only=True,
                                              out_indices=(0, 1, 2, 3), pretrained=pretrained)
            channels = self.encoder.feature_info.channels()
            reductions = self.encoder.feature_info.reduction()
        # Query actual feature channels; no hand-written HRNet/EfficientNet assumptions.
        self.feature_channels, self.feature_reductions = channels, reductions
        self.decoder = nn.ModuleList()
        current = channels[-1]
        for skip_ch, reduction in zip(channels[-2::-1], reductions[-2::-1]):
            self.decoder.append(Fusion(current, skip_ch, skip_ch, attention,
                                       max(2, image_size // reduction)))
            current = skip_ch
        self.final_fusion = Fusion(current, 0, 64, attention,
                                    max(2, image_size * 2 // reductions[0]))
        self.head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(32, num_classes, 1))

    def forward(self, x):
        features = self.encoder(x)
        y = features[-1]
        for block, skip in zip(self.decoder, features[-2::-1]):
            y = block(y, skip)
        y = self.head(self.final_fusion(y))
        return F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
