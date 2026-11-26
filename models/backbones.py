"""Backbone factory for encoder selection"""
import torch.nn as nn
import timm


class BackboneFactory:
    """Backbone工厂，支持多种encoder选择"""

    # ResNet系列 (timm features_only=True output)
    RESNET_BACKBONES = {
        'resnet18': {'channels': [64, 64, 128, 256], 'model': 'resnet18'},
        'resnet34': {'channels': [64, 64, 128, 256], 'model': 'resnet34'},
        'resnet50': {'channels': [64, 256, 512, 1024], 'model': 'resnet50'},
        'resnet101': {'channels': [64, 256, 512, 1024], 'model': 'resnet101'},
    }

    # EfficientNet系列 (timm features_only=True output)
    EFFICIENTNET_BACKBONES = {
        'efficientnet_b0': {'channels': [16, 24, 40, 112], 'model': 'efficientnet_b0'},
        'efficientnet_b1': {'channels': [16, 24, 40, 112], 'model': 'efficientnet_b1'},
        'efficientnet_b2': {'channels': [16, 32, 48, 120], 'model': 'efficientnet_b2'},
        'efficientnet_b3': {'channels': [24, 32, 48, 136], 'model': 'efficientnet_b3'},
    }

    # ConvNeXt系列
    CONVNEXT_BACKBONES = {
        'convnext_tiny': {'channels': [96, 192, 384, 768], 'model': 'convnext_tiny'},
        'convnext_small': {'channels': [96, 192, 384, 768], 'model': 'convnext_small'},
        'convnext_base': {'channels': [128, 256, 512, 1024], 'model': 'convnext_base'},
        'convnext_large': {'channels': [192, 384, 768, 1536], 'model': 'convnext_large'},
    }

    # HRNet系列 (timm features_only=True output)
    HRNET_BACKBONES = {
        'hrnet_w18': {'channels': [64, 128, 256, 512], 'model': 'hrnet_w18'},
        'hrnet_w32': {'channels': [64, 128, 256, 512], 'model': 'hrnet_w32'},
        'hrnet_w48': {'channels': [64, 128, 256, 512], 'model': 'hrnet_w48'},
    }

    # 所有可用backbone
    ALL_BACKBONES = {
        **RESNET_BACKBONES,
        **EFFICIENTNET_BACKBONES,
        **CONVNEXT_BACKBONES,
        **HRNET_BACKBONES,
    }

    @staticmethod
    def get_available_backbones():
        """获取所有可用backbone"""
        return list(BackboneFactory.ALL_BACKBONES.keys())

    @staticmethod
    def get_backbone_info(backbone_name):
        """获取backbone信息"""
        if backbone_name not in BackboneFactory.ALL_BACKBONES:
            raise ValueError(f"Backbone '{backbone_name}' not found")
        return BackboneFactory.ALL_BACKBONES[backbone_name]

    @staticmethod
    def create_backbone(backbone_name, pretrained=True, out_indices=(0, 1, 2, 3)):
        """
        创建backbone

        Args:
            backbone_name: backbone名称
            pretrained: 是否使用预训练权重
            out_indices: 输出层索引

        Returns:
            backbone: timm模型
            channels: 对应out_indices的输出通道数
        """
        if backbone_name not in BackboneFactory.ALL_BACKBONES:
            raise ValueError(
                f"Backbone '{backbone_name}' not found. "
                f"Available: {BackboneFactory.get_available_backbones()}"
            )

        info = BackboneFactory.ALL_BACKBONES[backbone_name]
        model_name = info['model']
        all_channels = info['channels']

        # 创建backbone
        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices
        )

        # 根据out_indices提取对应的通道数
        channels = [all_channels[i] for i in out_indices]

        return backbone, channels

    @staticmethod
    def get_backbone_category(backbone_name):
        """获取backbone类别"""
        if backbone_name in BackboneFactory.RESNET_BACKBONES:
            return 'resnet'
        elif backbone_name in BackboneFactory.EFFICIENTNET_BACKBONES:
            return 'efficientnet'
        elif backbone_name in BackboneFactory.CONVNEXT_BACKBONES:
            return 'convnext'
        elif backbone_name in BackboneFactory.HRNET_BACKBONES:
            return 'hrnet'
        else:
            return 'unknown'

    @staticmethod
    def get_backbones_by_category(category):
        """按类别获取backbone"""
        if category == 'resnet':
            return list(BackboneFactory.RESNET_BACKBONES.keys())
        elif category == 'efficientnet':
            return list(BackboneFactory.EFFICIENTNET_BACKBONES.keys())
        elif category == 'convnext':
            return list(BackboneFactory.CONVNEXT_BACKBONES.keys())
        elif category == 'hrnet':
            return list(BackboneFactory.HRNET_BACKBONES.keys())
        else:
            return []
