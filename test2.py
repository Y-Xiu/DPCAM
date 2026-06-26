import os
import torch
import logging
import argparse
import numpy as np
import time
import pandas as pd
from torch.utils.data import DataLoader
from models import create_model
from dataset import PCBDefectDataset, get_test_transform
from metrics import dice_coef, iou_score, recall_score
from config import (
    DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, CHECKPOINT_CONFIG,
    get_model_save_dir, AVAILABLE_MODELS
)

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


DEFECT_TYPES = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4'
}

DEFECT_NAMES = {
    0: 'Background',
    1: 'Defect_Type1',
    2: 'Defect_Type2',
    3: 'Defect_Type3',
    4: 'Defect_Type4'
}


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


def calculate_per_class_iou(outputs, masks, num_classes):
    """Calculate IoU for each class"""
    class_iou = {}

    outputs = torch.argmax(outputs, dim=1)

    for class_idx in range(num_classes):
        outputs_class = (outputs == class_idx).long()
        masks_class = (masks == class_idx).long()

        intersection = (outputs_class * masks_class).sum()
        union = (outputs_class + masks_class).sum() - intersection

        if union == 0:
            iou = 0.0 if intersection == 0 else 1.0
        else:
            iou = intersection.item() / union.item()

        class_iou[class_idx] = iou

    return class_iou


def calculate_per_class_dice(outputs, masks, num_classes):
    """Calculate Dice coefficient for each class"""
    class_dice = {}

    outputs = torch.argmax(outputs, dim=1)

    for class_idx in range(num_classes):
        outputs_class = (outputs == class_idx).long()
        masks_class = (masks == class_idx).long()

        intersection = (outputs_class * masks_class).sum()
        total = outputs_class.sum() + masks_class.sum()

        if total == 0:
            dice = 0.0 if intersection == 0 else 1.0
        else:
            dice = 2 * intersection.item() / total.item()

        class_dice[class_idx] = dice

    return class_dice


def calculate_per_class_precision_recall(outputs, masks, num_classes):
    """Calculate Precision and Recall for each class"""
    class_metrics = {}

    outputs = torch.argmax(outputs, dim=1)

    for class_idx in range(num_classes):
        outputs_class = (outputs == class_idx).long()
        masks_class = (masks == class_idx).long()

        tp = (outputs_class * masks_class).sum().item()
        fp = (outputs_class * (1 - masks_class)).sum().item()
        fn = ((1 - outputs_class) * masks_class).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        class_metrics[class_idx] = {
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    return class_metrics


def test_model_per_class(model_path=None, model_name=None, backbone=None, use_cbam=None, attention_type=None):
    """
    Test model and report performance for each defect type
    """
    logger = setup_logging()
    device = torch.device(TRAIN_CONFIG['device'] if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {device}")

    # Load or create model configuration
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

    # Create and load model
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

    # Load test dataset
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

    num_classes = MODEL_CONFIG['num_classes']

    # Initialize storage for per-class metrics
    per_class_iou_list = {i: [] for i in range(num_classes)}
    per_class_dice_list = {i: [] for i in range(num_classes)}
    per_class_precision_list = {i: [] for i in range(num_classes)}
    per_class_recall_list = {i: [] for i in range(num_classes)}
    per_class_tp_list = {i: [] for i in range(num_classes)}
    per_class_fp_list = {i: [] for i in range(num_classes)}
    per_class_fn_list = {i: [] for i in range(num_classes)}

    # Overall metrics storage
    all_dice = []
    all_iou = []

    logger.info("\nTesting per-class performance...")

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            # Calculate overall metrics
            dice = dice_coef(outputs, masks)
            iou = iou_score(outputs, masks)
            all_dice.append(dice.item() if torch.is_tensor(dice) else dice)
            all_iou.append(iou.item() if torch.is_tensor(iou) else iou)

            # Calculate per-class metrics
            class_iou = calculate_per_class_iou(outputs, masks, num_classes)
            class_dice = calculate_per_class_dice(outputs, masks, num_classes)
            class_metrics = calculate_per_class_precision_recall(outputs, masks, num_classes)

            for class_idx in range(num_classes):
                per_class_iou_list[class_idx].append(class_iou[class_idx])
                per_class_dice_list[class_idx].append(class_dice[class_idx])
                per_class_precision_list[class_idx].append(class_metrics[class_idx]['precision'])
                per_class_recall_list[class_idx].append(class_metrics[class_idx]['recall'])
                per_class_tp_list[class_idx].append(class_metrics[class_idx]['tp'])
                per_class_fp_list[class_idx].append(class_metrics[class_idx]['fp'])
                per_class_fn_list[class_idx].append(class_metrics[class_idx]['fn'])

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Batch [{batch_idx+1}/{len(test_loader)}]")

    # Calculate statistics
    overall_dice = np.mean(all_dice)
    overall_iou = np.mean(all_iou)

    # Per-class statistics
    per_class_stats = []

    logger.info("\n" + "=" * 100)
    logger.info("Per-Class Performance Report")
    logger.info("=" * 100)
    logger.info(f"{'Class':<20} {'Name':<20} {'IoU':<15} {'Dice':<15} {'Precision':<15} {'Recall':<15}")
    logger.info("-" * 100)

    for class_idx in range(num_classes):
        class_name = DEFECT_NAMES.get(class_idx, f'Class_{class_idx}')
        mean_iou = np.mean(per_class_iou_list[class_idx])
        std_iou = np.std(per_class_iou_list[class_idx])
        mean_dice = np.mean(per_class_dice_list[class_idx])
        std_dice = np.std(per_class_dice_list[class_idx])
        mean_precision = np.mean(per_class_precision_list[class_idx])
        mean_recall = np.mean(per_class_recall_list[class_idx])

        logger.info(f"{class_idx:<20} {class_name:<20} {mean_iou:.4f}±{std_iou:.4f}  "
                   f"{mean_dice:.4f}±{std_dice:.4f}  {mean_precision:.4f}      {mean_recall:.4f}")

        per_class_stats.append({
            'Class_ID': class_idx,
            'Class_Name': class_name,
            'IoU_Mean': mean_iou,
            'IoU_Std': std_iou,
            'Dice_Mean': mean_dice,
            'Dice_Std': std_dice,
            'Precision': mean_precision,
            'Recall': mean_recall,
            'TP_Total': int(np.sum(per_class_tp_list[class_idx])),
            'FP_Total': int(np.sum(per_class_fp_list[class_idx])),
            'FN_Total': int(np.sum(per_class_fn_list[class_idx])),
        })

    logger.info("-" * 100)
    logger.info(f"{'Overall':<20} {'mIoU':<20} {overall_iou:.4f}")
    logger.info(f"{'Overall':<20} {'mDice':<20} {overall_dice:.4f}")
    logger.info("=" * 100)

    # Save results to CSV
    df_per_class = pd.DataFrame(per_class_stats)
    csv_filename = f'per_class_results_{model_name}_{backbone}_{attention_type if use_cbam else "no_attention"}.csv'
    df_per_class.to_csv(csv_filename, index=False)
    logger.info(f"\nPer-class results saved to {csv_filename}")

    # Create detailed report
    report_filename = f'per_class_report_{model_name}_{backbone}_{attention_type if use_cbam else "no_attention"}.txt'
    with open(report_filename, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("PCB Defect Segmentation - Per-Class Performance Report\n")
        f.write("=" * 100 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Backbone: {backbone}\n")
        f.write(f"Attention: {attention_type if use_cbam else 'None'}\n")
        f.write(f"Test Samples: {len(test_dataset)}\n")
        f.write("=" * 100 + "\n\n")

        f.write("Summary Statistics:\n")
        f.write(f"  Overall mIoU: {overall_iou:.4f}\n")
        f.write(f"  Overall mDice: {overall_dice:.4f}\n\n")

        f.write("Per-Class Metrics:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Class':<15} {'Name':<20} {'IoU':<18} {'Dice':<18} {'Precision':<15} {'Recall':<15}\n")
        f.write("-" * 100 + "\n")

        for stat in per_class_stats:
            f.write(f"{stat['Class_ID']:<15} {stat['Class_Name']:<20} "
                   f"{stat['IoU_Mean']:.4f}±{stat['IoU_Std']:.4f}  "
                   f"{stat['Dice_Mean']:.4f}±{stat['Dice_Std']:.4f}  "
                   f"{stat['Precision']:.4f}      {stat['Recall']:.4f}\n")

        f.write("\n" + "-" * 100 + "\n")
        f.write("Confusion Matrix Statistics (TP, FP, FN):\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Class':<15} {'Name':<20} {'TP':<12} {'FP':<12} {'FN':<12}\n")
        f.write("-" * 100 + "\n")

        for stat in per_class_stats:
            f.write(f"{stat['Class_ID']:<15} {stat['Class_Name']:<20} "
                   f"{stat['TP_Total']:<12} {stat['FP_Total']:<12} {stat['FN_Total']:<12}\n")

    logger.info(f"Detailed report saved to {report_filename}")

    return {
        'model': model_name,
        'backbone': backbone,
        'attention': attention_type if use_cbam else 'None',
        'overall_iou': overall_iou,
        'overall_dice': overall_dice,
        'per_class_stats': per_class_stats,
        'per_class_df': df_per_class
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PCB defect segmentation model per-class performance')
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

    test_model_per_class(model_path=args.checkpoint, model_name=args.model, backbone=args.backbone,
                        use_cbam=args.use_cbam, attention_type=args.attention_type)
