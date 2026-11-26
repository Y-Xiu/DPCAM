"""PSPNet with flexible backbone support"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam import CBAM
from .backbones import BackboneFactory


class PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module"""
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.out_channels = out_channels

        self.pool_blocks = nn.ModuleList()
        for pool_size in pool_sizes:
            self.pool_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        size = x.shape[-2:]
        pool_outputs = [x]

        for pool_block in self.pool_blocks:
            pool_out = pool_block(x)
            pool_out = F.interpolate(pool_out, size=size, mode='bilinear', align_corners=False)
            pool_outputs.append(pool_out)

        return torch.cat(pool_outputs, dim=1)


class PSPNet(nn.Module):
    """PSPNet with flexible backbone support"""
    def __init__(self, num_classes=5, backbone='resnet50', use_cbam=False, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_cbam = use_cbam

        # 创建backbone
        self.backbone, encoder_channels = BackboneFactory.create_backbone(
            backbone, pretrained=pretrained, out_indices=(3,)
        )

        backbone_channels = encoder_channels[0]
        ppm_out_channels = backbone_channels // 4
        decoder_in_channels = backbone_channels + ppm_out_channels * 4
        decoder_hidden_channels = 512

        # Pyramid Pooling Module
        self.ppm = PyramidPoolingModule(backbone_channels, ppm_out_channels, pool_sizes=[1, 2, 3, 6])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_in_channels, decoder_hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(decoder_hidden_channels, num_classes, 1)
        )

        self.cbam = CBAM(decoder_hidden_channels) if use_cbam else None

    def forward(self, x):
        size = x.shape[-2:]

        # Backbone
        features = self.backbone(x)
        x = features[0]

        # Pyramid Pooling
        x = self.ppm(x)

        # Decoder
        x = self.decoder(x)

        if self.cbam is not None:
            x = self.cbam(x)

        # Upsample to original size
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        return x
