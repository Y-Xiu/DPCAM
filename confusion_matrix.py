import os
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models import create_model
from dataset import PCBDefectDataset, get_test_transform
from config import (
    DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, CHECKPOINT_CONFIG,
    get_model_save_dir
)


DEFAULT_CLASS_NAMES = ['background', 'bent', 'melt', 'missing', 'scratch']
#DEFAULT_CLASS_NAMES = ['background', 'good', 'exc_solder', 'poor_solder', 'spike']


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def update_confusion_matrix(cm, y_true, y_pred, num_classes):
    mask = (y_true >= 0) & (y_true < num_classes)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    hist = np.bincount(
        num_classes * y_true.astype(int) + y_pred.astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    cm += hist
    return cm


def plot_confusion_matrix(cm, class_names, title, save_path, normalize=False):
    cm_to_plot = cm.astype(np.float64)
    if normalize:
        row_sum = cm_to_plot.sum(axis=1, keepdims=True)
        cm_to_plot = np.divide(cm_to_plot, row_sum, where=row_sum > 0)

    # Paper-friendly font sizes (do not change computation logic)
    label_fontsize = 20
    tick_fontsize = 20
    cell_fontsize = 20
    title_fontsize = 20
    label_pad = -8 # smaller -> closer to the matrix

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_to_plot, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_xlabel('Predicted Label', fontsize=label_fontsize, labelpad=label_pad)
    ax.set_ylabel('True Label', fontsize=label_fontsize, labelpad=label_pad)
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')

    thresh = cm_to_plot.max() * 0.5 if cm_to_plot.size > 0 else 0.0
    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            if normalize:
                text = f"{cm_to_plot[i, j]:.3f}"
            else:
                text = f"{int(cm_to_plot[i, j])}"
            ax.text(
                j, i, text,
                ha='center', va='center',
                color='white' if cm_to_plot[i, j] > thresh else 'black',
                fontsize=cell_fontsize
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate pixel-level confusion matrix for segmentation')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default=MODEL_CONFIG['model_name'], help='Model name')
    parser.add_argument('--backbone', type=str, default=MODEL_CONFIG['backbone'], help='Backbone name')
    parser.add_argument('--use-cbam', type=str_to_bool, default=MODEL_CONFIG['use_cbam'], help='Use attention module')
    parser.add_argument('--attention-type', type=str, default=MODEL_CONFIG.get('attention_type', 'cbam'),
                        choices=['cbam', 'dpcam', 'dpcam_lite', 'pa_only'], help='Attention type')
    parser.add_argument('--batch-size', type=int, default=TRAIN_CONFIG['batch_size'], help='Batch size for inference')
    parser.add_argument('--output-dir', type=str, default='./inference_results/confusion_matrix', help='Output directory')
    parser.add_argument('--image-format', type=str, default='pdf', choices=['png', 'svg', 'pdf'],
                        help='Image format for confusion matrix (png, svg, or pdf)')
    parser.add_argument('--class-names', type=str, default=None,
                        help='Comma-separated class names, e.g. "background,good,exc_solder,spike,no_good"')
    args = parser.parse_args()

    logger = setup_logging()
    device = torch.device(TRAIN_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.class_names:
        class_names = [x.strip() for x in args.class_names.split(',')]
    else:
        class_names = DEFAULT_CLASS_NAMES

    num_classes = MODEL_CONFIG['num_classes']
    if len(class_names) != num_classes:
        logger.warning(
            f"class names length ({len(class_names)}) != num_classes ({num_classes}), "
            f"fallback to index names."
        )
        class_names = [f'class_{i}' for i in range(num_classes)]

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if args.use_cbam and args.attention_type != 'cbam':
            checkpoint_dir = os.path.join(
                CHECKPOINT_CONFIG['save_dir'],
                f"{args.model}_backbone_{args.backbone}_{args.attention_type}"
            )
        else:
            checkpoint_dir = get_model_save_dir(args.model, args.backbone, args.use_cbam)
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')

    model = create_model(
        model_name=args.model,
        num_classes=num_classes,
        backbone=args.backbone,
        use_cbam=args.use_cbam,
        pretrained=False,
        attention_type=args.attention_type
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    test_dataset = PCBDefectDataset(
        DATA_CONFIG['test_images_dir'],
        DATA_CONFIG['test_masks_dir'],
        transform=get_test_transform(image_size=DATA_CONFIG['image_size']),
        is_train=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=True
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            gts = masks.cpu().numpy()
            for gt, pred in zip(gts, preds):
                cm = update_confusion_matrix(cm, gt.flatten(), pred.flatten(), num_classes)

    img_ext = args.image_format.lower()
    count_png = os.path.join(args.output_dir, f'confusion_matrix_count.{img_ext}')
    norm_png = os.path.join(args.output_dir, f'confusion_matrix_normalized.{img_ext}')
    npy_path = os.path.join(args.output_dir, 'confusion_matrix_count.npy')
    csv_path = os.path.join(args.output_dir, 'confusion_matrix_count.csv')

    plot_confusion_matrix(cm, class_names, 'Confusion Matrix (Pixel Count)', count_png, normalize=False)
    plot_confusion_matrix(cm, class_names, '', norm_png, normalize=True)
    np.save(npy_path, cm)
    np.savetxt(csv_path, cm, delimiter=',', fmt='%d')

    logger.info(f"Saved: {count_png}")
    logger.info(f"Saved: {norm_png}")
    logger.info(f"Saved: {npy_path}")
    logger.info(f"Saved: {csv_path}")


if __name__ == '__main__':
    main()
