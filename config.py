import os


DATA_CONFIG = {
    'train_images_dir': './data/train/images',
    'train_masks_dir': './data/train/masks',
    'val_images_dir': './data/val/images',
    'val_masks_dir': './data/val/masks',
    'test_images_dir': './data/test/images',
    'test_masks_dir': './data/test/masks',
    'image_size': 512,
}


MODEL_CONFIG = {
    'num_classes': 5,
    'model_name': 'unet',
    'backbone': 'resnet50',
    'use_cbam': False,
    'attention_type': 'cbam',
    'pretrained': True,
}


AVAILABLE_MODELS = {

    'unet': 'Basic UNet',
    'unet++': 'UNet++ with nested skip connections',
    'segnet': 'SegNet with unpooling',


    'deeplabv3': 'DeepLabV3 with ASPP',
    'deeplabv3+': 'DeepLabV3+ with encoder-decoder',
    'pspnet': 'PSPNet with pyramid pooling',


    'efficientnet_unet': 'EfficientNet-UNet (lightweight)',
    'resnet_unet': 'ResNet-UNet (standard)',


    'hrnet': 'HRNet (high-resolution)',
    'convnext_unet': 'ConvNeXt-UNet (modern)',
}


TRAIN_CONFIG = {
    'batch_size': 2,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,

    'num_workers': 4,
    'device': 'cuda',
    'mixed_precision': False,
    'gradient_accumulation_steps': 1,
}


OPTIMIZER_CONFIG = {
    'type': 'adamw',  # 'adam', 'adamw', 'sgd'
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'betas': (0.9, 0.999),
}


SCHEDULER_CONFIG = {
    'type': 'cosine',  # 'cosine', 'step', 'exponential'
    'T_max': 100,
    'eta_min': 1e-6,
}


LOSS_CONFIG = {
    'type': 'combined',  # 'cross_entropy', 'dice', 'focal', 'combined'
    'losses': {
        'dice': {
            'type': 'dice',
            'smooth': 1.0,
            'ignore_index': -100,
        },
        'focal': {
            'type': 'focal',
            'alpha': None,  # 可选: [0.25, 0.75, 1.0, 1.0, 1.5] 为每个类别设置权重
            'gamma': 2.0,
            'ignore_index': -100,
        }
    },
    'weights': {
        'dice': 0.5,   # Dice Loss权重
        'focal': 0.5,  # Focal Loss权重
    }
}


AUGMENTATION_CONFIG = {
    'train': {
        'random_flip': True,
        'random_rotate': True,
        'random_brightness': True,
        'random_contrast': True,
    },
    'val': {
        'random_flip': False,
        'random_rotate': False,
        'random_brightness': False,
        'random_contrast': False,
    }
}


CHECKPOINT_CONFIG = {
    'save_dir': './checkpoints',
    'save_interval': 10,
    'keep_best': True,
    'best_metric': 'val_iou',
}

LOG_CONFIG = {
    'log_dir': './logs',
    'tensorboard_dir': './runs',
    'log_interval': 10,
}


INFERENCE_CONFIG = {
    'model_path': './checkpoints/best_model.pth',
    'batch_size': 1,
    'num_workers': 0,
    'device': 'cuda',
}


def get_model_save_dir(model_name, backbone, use_cbam):

    dir_name = f"{model_name}_backbone_{backbone}_cbam_{use_cbam}"
    return os.path.join(CHECKPOINT_CONFIG['save_dir'], dir_name)


def get_log_save_dir(model_name, backbone, use_cbam):

    dir_name = f"{model_name}_backbone_{backbone}_cbam_{use_cbam}"
    return os.path.join(LOG_CONFIG['log_dir'], dir_name)


def get_tensorboard_dir(model_name, backbone, use_cbam):

    dir_name = f"{model_name}_backbone_{backbone}_cbam_{use_cbam}"
    return os.path.join(LOG_CONFIG['tensorboard_dir'], dir_name)
