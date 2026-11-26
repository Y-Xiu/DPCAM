"""DeepLabV3 and DeepLabV3+ with flexible backbone support"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam import CBAM
from .backbones import BackboneFactory


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.aspp_blocks = nn.ModuleList()
        for rate in rates:
            self.aspp_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]

        res = [self.conv1x1(x)]
        for aspp_block in self.aspp_blocks:
            res.append(aspp_block(x))

        pool = self.image_pool(x)
        pool = F.interpolate(pool, size=size, mode='bilinear', align_corners=False)
        res.append(pool)

        res = torch.cat(res, dim=1)
        res = self.project(res)
        return res


class DeepLabV3(nn.Module):
    """DeepLabV3 with flexible backbone support"""
    def __init__(self, num_classes=5, backbone='resnet50', use_cbam=False, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_cbam = use_cbam

        # 创建backbone
        self.backbone, encoder_channels = BackboneFactory.create_backbone(
            backbone, pretrained=pretrained, out_indices=(3,)
        )

        in_channels = encoder_channels[0]
        aspp_out_channels = 256

        # ASPP
        self.aspp = ASPP(in_channels, aspp_out_channels, rates=[6, 12, 18])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_out_channels, aspp_out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_out_channels, num_classes, 1)
        )

        self.cbam = CBAM(aspp_out_channels) if use_cbam else None

    def forward(self, x):
        size = x.shape[-2:]

        # Backbone
        features = self.backbone(x)
        x = features[0]

        # ASPP
        x = self.aspp(x)

        if self.cbam is not None:
            x = self.cbam(x)

        # Decoder
        x = self.decoder(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        return x


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with flexible backbone support"""
    def __init__(self, num_classes=5, backbone='resnet50', use_cbam=False, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_cbam = use_cbam

        # 创建backbone with multiple outputs
        self.backbone, encoder_channels = BackboneFactory.create_backbone(
            backbone, pretrained=pretrained, out_indices=(1, 3)
        )

        low_level_channels = encoder_channels[0]
        high_level_channels = encoder_channels[1]
        low_level_proj_channels = 48
        aspp_out_channels = 256

        # ASPP
        self.aspp = ASPP(high_level_channels, aspp_out_channels, rates=[6, 12, 18])

        # Low-level feature projection
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_proj_channels, 1, bias=False),
            nn.BatchNorm2d(low_level_proj_channels),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_out_channels + low_level_proj_channels, aspp_out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_out_channels, aspp_out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_out_channels, num_classes, 1)
        )

        self.cbam = CBAM(aspp_out_channels) if use_cbam else None

    def forward(self, x):
        size = x.shape[-2:]

        # Backbone
        features = self.backbone(x)
        low_level = features[0]
        high_level = features[1]

        # ASPP
        x = self.aspp(high_level)

        if self.cbam is not None:
            x = self.cbam(x)

        # Upsample and concatenate with low-level features
        x = F.interpolate(x, size=low_level.shape[-2:], mode='bilinear', align_corners=False)
        low_level = self.low_level_proj(low_level)
        x = torch.cat([x, low_level], dim=1)

        # Decoder
        x = self.decoder(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        return x
