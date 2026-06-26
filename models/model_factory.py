"""Model factory for creating segmentation models with flexible backbone support"""
import torch.nn as nn
from .unet import UNet, UNetPlusPlus, SegNet
from .deeplabv3 import DeepLabV3, DeepLabV3Plus
from .pspnet import PSPNet
from .efficient_unet import EfficientNetUNet, ResNetUNet
from .hrnet import HRNet
from .convnext_unet import ConvNeXtUNet
from .backbones import BackboneFactory
import torch


# 模型注册表
MODEL_REGISTRY = {
    # 基础模型
    'unet': {'class': UNet, 'category': 'basic', 'default_backbone': 'resnet50'},
    'unet++': {'class': UNetPlusPlus, 'category': 'basic', 'default_backbone': 'resnet50'},
    'segnet': {'class': SegNet, 'category': 'basic', 'default_backbone': 'resnet50'},

    # 多尺度模型
    'deeplabv3': {'class': DeepLabV3, 'category': 'multi-scale', 'default_backbone': 'resnet50'},
    'deeplabv3+': {'class': DeepLabV3Plus, 'category': 'multi-scale', 'default_backbone': 'resnet50'},
    'pspnet': {'class': PSPNet, 'category': 'multi-scale', 'default_backbone': 'resnet50'},

    # 高效模型
    'efficientnet_unet': {'class': EfficientNetUNet, 'category': 'efficient', 'default_backbone': 'efficientnet_b0'},
    'resnet_unet': {'class': ResNetUNet, 'category': 'efficient', 'default_backbone': 'resnet50'},

    # 高分辨率模型
    'hrnet': {'class': HRNet, 'category': 'high-resolution', 'default_backbone': 'hrnet_w18'},
    'convnext_unet': {'class': ConvNeXtUNet, 'category': 'high-resolution', 'default_backbone': 'convnext_base'},
}


def get_available_models():
    """获取所有可用模型列表"""
    return list(MODEL_REGISTRY.keys())


def get_models_by_category(category=None):
    """按类别获取模型"""
    if category is None:
        return MODEL_REGISTRY

    return {name: info for name, info in MODEL_REGISTRY.items() if info['category'] == category}


def get_model_info(model_name):
    """获取模型信息"""
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found")
    return MODEL_REGISTRY[model_name]


def get_default_backbone(model_name):
    """获取模型的默认backbone"""
    info = get_model_info(model_name)
    return info['default_backbone']


def create_model(model_name='convnext_unet', num_classes=5, backbone=None, use_cbam=True, 
                 pretrained=True, attention_type='cbam'):
    """
    创建分割模型

    Args:
        model_name: 模型名称，见MODEL_REGISTRY
        num_classes: 类别数
        backbone: backbone名称，如果为None则使用默认backbone
        use_cbam: 是否使用注意力机制（兼容旧接口）
        pretrained: 是否使用预训练权重
        attention_type: 注意力类型 - 'cbam', 'dpcam', 'dpcam_lite', 'pa_only'

    Returns:
        model: 分割模型
    """
    model_name = model_name.lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {get_available_models()}"
        )

    # 如果没有指定backbone，使用默认backbone
    if backbone is None:
        backbone = get_default_backbone(model_name)

    backbone = backbone.lower()

    # 验证backbone
    if backbone not in BackboneFactory.get_available_backbones():
        raise ValueError(
            f"Backbone '{backbone}' not found. Available backbones: {BackboneFactory.get_available_backbones()}"
        )

    model_class = MODEL_REGISTRY[model_name]['class']
    
    # 检查模型是否支持attention_type参数
    import inspect
    sig = inspect.signature(model_class.__init__)
    if 'attention_type' in sig.parameters:
        model = model_class(
            num_classes=num_classes,
            backbone=backbone,
            use_cbam=use_cbam,
            pretrained=pretrained,
            attention_type=attention_type
        )
    else:
        # 旧模型不支持attention_type
        model = model_class(
            num_classes=num_classes,
            backbone=backbone,
            use_cbam=use_cbam,
            pretrained=pretrained
        )

    return model


def get_model_config_string(model_name, backbone, use_cbam, attention_type='cbam'):
    """获取模型配置字符串"""
    if use_cbam and attention_type != 'cbam':
        return f"{model_name}_backbone_{backbone}_{attention_type}"
    return f"{model_name}_backbone_{backbone}_cbam_{use_cbam}"
