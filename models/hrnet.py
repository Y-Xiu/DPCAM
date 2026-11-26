"""HRNet with flexible backbone support"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam import CBAM
from .backbones import BackboneFactory


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_cbam=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_channels) if use_cbam else None

    def forward(self, x):
        x = self.conv(x)
        if self.cbam is not None:
            x = self.cbam(x)
        return x


class HRNet(nn.Module):
    """HRNet with flexible backbone support"""
    def __init__(self, num_classes=5, backbone='hrnet_w18', use_cbam=False, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_cbam = use_cbam

        # 创建backbone
        self.backbone, encoder_channels = BackboneFactory.create_backbone(
            backbone, pretrained=pretrained, out_indices=(0, 1, 2, 3)
        )

        total_channels = sum(encoder_channels)
        fusion_channels = 256

        # Decoder - fuse multi-scale features
        self.decoder = nn.Sequential(
            nn.Conv2d(total_channels, fusion_channels, 1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
        )

        # Final layers
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(fusion_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )

        self.cbam = CBAM(fusion_channels) if use_cbam else None

    def forward(self, x):
        size = x.shape[-2:]

        # Encoder
        features = self.backbone(x)

        # Upsample all features to the same resolution (highest resolution)
        target_size = features[0].shape[-2:]
        aligned_features = [features[0]]

        for i in range(1, len(features)):
            feat = features[i]
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)

        # Concatenate all features
        x = torch.cat(aligned_features, dim=1)

        # Decoder
        x = self.decoder(x)

        if self.cbam is not None:
            x = self.cbam(x)

        # Final upsampling
        x = self.final_up(x)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        return x
