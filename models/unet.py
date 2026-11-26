"""UNet and variants with flexible backbone support"""
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


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_cbam=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels, use_cbam=use_cbam)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """UNet with flexible backbone support"""
    def __init__(self, num_classes=5, backbone='resnet50', use_cbam=False, pretrained=True, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_cbam = use_cbam

        # 创建backbone
        self.backbone, encoder_channels = BackboneFactory.create_backbone(
            backbone, pretrained=pretrained, out_indices=(0, 1, 2, 3)
        )

        # Decoder
        self.up4 = UpBlock(encoder_channels[3] + encoder_channels[2], encoder_channels[2], use_cbam=use_cbam)
        self.up3 = UpBlock(encoder_channels[2] + encoder_channels[1], encoder_channels[1], use_cbam=use_cbam)
        self.up2 = UpBlock(encoder_channels[1] + encoder_channels[0], encoder_channels[0], use_cbam=use_cbam)
        self.up1 = UpBlock(encoder_channels[0], 64, use_cbam=use_cbam)

        # Final upsampling to original resolution
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]
        # Encoder
        features = self.backbone(x)
        e1, e2, e3, e4 = features

        # Decoder
        d4 = self.up4(e4, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, None)

        # Final upsampling to original resolution
        d0 = self.final_up(d1)

        # Output
        out = self.final_conv(d0)
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out


class UNetPlusPlus(nn.Module):
    """UNet++ with flexible backbone support"""
    def __init__(self, num_classes=5, backbone='resnet50', use_cbam=False, pretrained=True, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_cbam = use_cbam

        # 创建backbone
        self.backbone, encoder_channels = BackboneFactory.create_backbone(
            backbone, pretrained=pretrained, out_indices=(0, 1, 2, 3)
        )

        # Nested decoder blocks
        self.up4_1 = UpBlock(encoder_channels[3] + encoder_channels[2], encoder_channels[2], use_cbam=use_cbam)
        self.up3_1 = UpBlock(encoder_channels[2] + encoder_channels[1], encoder_channels[1], use_cbam=use_cbam)
        self.up2_1 = UpBlock(encoder_channels[1] + encoder_channels[0], encoder_channels[0], use_cbam=use_cbam)
        self.up1_1 = UpBlock(encoder_channels[0], 64, use_cbam=use_cbam)

        self.up3_2 = UpBlock(encoder_channels[2] + encoder_channels[1], encoder_channels[1], use_cbam=use_cbam)
        self.up2_2 = UpBlock(encoder_channels[1] + encoder_channels[0], encoder_channels[0], use_cbam=use_cbam)
        self.up1_2 = UpBlock(encoder_channels[0], 64, use_cbam=use_cbam)

        self.up2_3 = UpBlock(encoder_channels[1] + encoder_channels[0], encoder_channels[0], use_cbam=use_cbam)
        self.up1_3 = UpBlock(encoder_channels[0], 64, use_cbam=use_cbam)

        self.up1_4 = UpBlock(64 + 64, 64, use_cbam=use_cbam)

        # Final upsampling to original resolution
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]
        # Encoder
        features = self.backbone(x)
        e1, e2, e3, e4 = features

        # Nested decoder
        d4_1 = self.up4_1(e4, e3)
        d3_1 = self.up3_1(d4_1, e2)
        d2_1 = self.up2_1(d3_1, e1)
        d1_1 = self.up1_1(d2_1, None)

        d3_2 = self.up3_2(d3_1, e2)
        d2_2 = self.up2_2(d2_1, e1)
        d1_2 = self.up1_2(d2_2, None)

        d2_3 = self.up2_3(d2_2, e1)
        d1_3 = self.up1_3(d2_3, None)

        d1_4 = self.up1_4(d1_3, d1_1)

        # Final upsampling to original resolution
        d0 = self.final_up(d1_4)
        out = self.final_conv(d0)
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out


class SegNet(nn.Module):
    """SegNet with flexible backbone support"""
    def __init__(self, num_classes=5, backbone='resnet50', use_cbam=False, pretrained=True, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_cbam = use_cbam

        # 创建backbone
        self.backbone, encoder_channels = BackboneFactory.create_backbone(
            backbone, pretrained=pretrained, out_indices=(0, 1, 2, 3)
        )

        # Decoder
        self.up4 = UpBlock(encoder_channels[3] + encoder_channels[2], encoder_channels[2], use_cbam=use_cbam)
        self.up3 = UpBlock(encoder_channels[2] + encoder_channels[1], encoder_channels[1], use_cbam=use_cbam)
        self.up2 = UpBlock(encoder_channels[1] + encoder_channels[0], encoder_channels[0], use_cbam=use_cbam)
        self.up1 = UpBlock(encoder_channels[0], 64, use_cbam=use_cbam)

        # Final upsampling to original resolution
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]
        # Encoder
        features = self.backbone(x)
        e1, e2, e3, e4 = features

        # Decoder
        d4 = self.up4(e4, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, None)

        # Final upsampling to original resolution
        d0 = self.final_up(d1)

        # Output
        out = self.final_conv(d0)
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out
