import os
import torch
import logging
import argparse
from torch.utils.data import DataLoader
from models import create_model
from dataset import PCBDefectDataset, get_test_transform
from metrics import dice_coef, iou_score, recall_score
from config import (
    DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, CHECKPOINT_CONFIG,
    get_model_save_dir, AVAILABLE_MODELS
)


def str_to_bool(v):
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_model(model_path=None, model_name=None, backbone=None, use_cbam=None, attention_type=None):

    logger = setup_logging()
    device = torch.device(TRAIN_CONFIG['device'] if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {device}")


    if model_path is None:
        if model_name is None:
            model_name = MODEL_CONFIG['model_name']
        if backbone is None:
            from models import get_default_backbone
            backbone = backbone or get_default_backbone(model_name)
        if use_cbam is None:
            use_cbam = MODEL_CONFIG['use_cbam']
        if attention_type is None:
            attention_type = MODEL_CONFIG.get('attention_type', 'cbam')


        if use_cbam and attention_type != 'cbam':
            checkpoint_dir = os.path.join(CHECKPOINT_CONFIG['save_dir'], 
                                         f"{model_name}_backbone_{backbone}_{attention_type}")
        else:
            checkpoint_dir = get_model_save_dir(model_name, backbone, use_cbam)
        model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    else:

        if os.path.exists(model_path):
            checkpoint_temp = torch.load(model_path, map_location=device)
            model_name = checkpoint_temp.get('model_name', MODEL_CONFIG['model_name'])
            backbone = checkpoint_temp.get('backbone', MODEL_CONFIG['backbone'])
            use_cbam = checkpoint_temp.get('use_cbam', MODEL_CONFIG['use_cbam'])
            attention_type = checkpoint_temp.get('attention_type', 'cbam')
        else:
            model_name = model_name or MODEL_CONFIG['model_name']
            if backbone is None:
                from models import get_default_backbone
                backbone = get_default_backbone(model_name)
            use_cbam = use_cbam if use_cbam is not None else MODEL_CONFIG['use_cbam']
            attention_type = attention_type or MODEL_CONFIG.get('attention_type', 'cbam')

    logger.info(f"Model: {model_name}, Backbone: {backbone}")
    logger.info(f"Attention: {attention_type if use_cbam else 'None'}")


    model = create_model(
        model_name=model_name,
        num_classes=MODEL_CONFIG['num_classes'],
        backbone=backbone,
        use_cbam=use_cbam,
        pretrained=False,
        attention_type=attention_type
    ).to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.error(f"Model path {model_path} not found!")
        return

    model.eval()


    logger.info("Loading test dataset...")
    test_dataset = PCBDefectDataset(
        DATA_CONFIG['test_images_dir'],
        DATA_CONFIG['test_masks_dir'],
        transform=get_test_transform(image_size=DATA_CONFIG['image_size']),
        is_train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=True
    )

    logger.info(f"Test samples: {len(test_dataset)}")


    logger.info("Testing...")
    total_dice = 0
    total_iou = 0
    total_mpa = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            dice = dice_coef(outputs, masks)
            iou = iou_score(outputs, masks)
            recall = recall_score(outputs, masks)
            mpa = recall

            total_dice += dice
            total_iou += iou
            total_mpa += mpa

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Batch [{batch_idx+1}/{len(test_loader)}]")


    avg_dice = total_dice / len(test_loader)
    avg_iou = total_iou / len(test_loader)
    avg_mpa = total_mpa / len(test_loader)

    logger.info("\n" + "=" * 50)
    logger.info("Test Results:")
    logger.info(f"  mDice: {avg_dice:.4f}")
    logger.info(f"  mIoU: {avg_iou:.4f}")
    logger.info(f"  mPA: {avg_mpa:.4f}")
    logger.info("=" * 50)

    return {
        'mdice': avg_dice,
        'miou': avg_iou,
        'mpa': avg_mpa
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PCB defect segmentation model')
    parser.add_argument('--model', type=str, default=None,
                        choices=list(AVAILABLE_MODELS.keys()),
                        help='Model name to test')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Backbone for encoder')
    parser.add_argument('--use-cbam', type=str_to_bool, default=None,
                        help='Whether model uses attention (True/False)')
    parser.add_argument('--attention-type', type=str, default=None,
                        choices=['cbam', 'dpcam', 'dpcam_lite', 'pa_only'],
                        help='Attention mechanism type: cbam, dpcam, dpcam_lite, pa_only')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')

    args = parser.parse_args()

    test_model(model_path=args.checkpoint, model_name=args.model, backbone=args.backbone, 
              use_cbam=args.use_cbam, attention_type=args.attention_type)
