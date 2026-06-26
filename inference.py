import os
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from models import create_model
from config import MODEL_CONFIG, INFERENCE_CONFIG, get_model_save_dir, AVAILABLE_MODELS


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


class BatchInferencer:
    def __init__(self, model_path=None, model_name=None, backbone=None, use_cbam=None, attention_type=None, device=None):
        self.device = device or torch.device(INFERENCE_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path

        if self.model_path is None:
            if model_name is None:
                model_name = 'unet'
            if backbone is None:
                backbone = 'convnext_base'
            if use_cbam is None:
                use_cbam = True
            if attention_type is None:
                attention_type = 'dpcam'

            from config import CHECKPOINT_CONFIG
            if use_cbam and attention_type != 'cbam':
                checkpoint_dir = os.path.join(CHECKPOINT_CONFIG['save_dir'], 
                                             f"{model_name}_backbone_{backbone}_{attention_type}")
            else:
                checkpoint_dir = get_model_save_dir(model_name, backbone, use_cbam)
            self.model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        else:

            if os.path.exists(self.model_path):
                checkpoint_temp = torch.load(self.model_path, map_location=self.device)
                model_name = checkpoint_temp.get('model_name', MODEL_CONFIG['model_name'])
                backbone = checkpoint_temp.get('backbone', MODEL_CONFIG['backbone'])
                use_cbam = checkpoint_temp.get('use_cbam', MODEL_CONFIG['use_cbam'])
                attention_type = checkpoint_temp.get('attention_type', 'cbam')
            else:
                model_name = model_name or 'unet'
                if backbone is None:
                    backbone = 'convnext_base'
                use_cbam = use_cbam if use_cbam is not None else True
                attention_type = attention_type or 'dpcam'

        self.model_name = model_name
        self.backbone = backbone
        self.use_cbam = use_cbam
        self.attention_type = attention_type


        self.model = create_model(
            model_name=self.model_name,
            num_classes=MODEL_CONFIG['num_classes'],
            backbone=self.backbone,
            use_cbam=self.use_cbam,
            pretrained=False,
            attention_type=self.attention_type
        ).to(self.device)

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model from {self.model_path}")
            print(f"  Model: {self.model_name}, Backbone: {self.backbone}")
            print(f"  Attention: {self.attention_type if self.use_cbam else 'None'}")
        else:
            print(f"✗ Warning: Model path {self.model_path} not found!")

        self.model.eval()


        self.class_names = {
            0: 'background',
            1: 'good',
            2: 'exc_solder',
            3: 'poor_solder',
            4: 'spike'
        }

        self.color_map = {
            0: (0, 0, 0),           # background - black
            1: (0, 255, 0),         # good - green
            2: (255, 255, 0),       # exc_solder - yellow
            3: (0, 165, 255),       # poor_solder - orange
            4: (0, 0, 255)          # spike - red
        }

    def preprocess(self, image_path):
        """预处理图像"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        image = self._normalize_image(image)

        # HWC -> CHW
        image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)

        return image.to(self.device), original_shape

    def _normalize_image(self, image_rgb):
        image = image_rgb.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return image

    def _predict_full_rgb(self, image_norm):
        image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(image_tensor)
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        return pred_mask

    def predict(self, image_path):

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        image_norm = self._normalize_image(image)
        pred_mask = self._predict_full_rgb(image_norm)

        return pred_mask, original_shape

    def _colorize_mask(self, mask):

        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in self.color_map.items():
            colored_mask[mask == class_id] = color
        return colored_mask

    def _find_gt_mask_path(self, image_file, test_masks_dir):

        if test_masks_dir is None:
            return None
        base = os.path.splitext(image_file)[0]
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            p = os.path.join(test_masks_dir, base + ext)
            if os.path.exists(p):
                return p
        return None

    def visualize_prediction(self, image_path, pred_mask, gt_mask=None):


        original_image = cv2.imread(str(image_path))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)


        if gt_mask is None:
            gt_mask = np.zeros(pred_mask.shape, dtype=np.uint8)
        gt_colored_mask = self._colorize_mask(gt_mask)


        pred_colored_mask = self._colorize_mask(pred_mask)
        alpha = 0.5
        overlay = cv2.addWeighted(original_image, 1 - alpha, pred_colored_mask, alpha, 0)


        result = np.hstack([original_image, gt_colored_mask, overlay])

        return result, gt_colored_mask, pred_colored_mask

    def get_statistics(self, pred_mask):

        unique, counts = np.unique(pred_mask, return_counts=True)
        total_pixels = pred_mask.size

        stats = {}
        for class_id, count in zip(unique, counts):
            percentage = (count / total_pixels) * 100
            stats[self.class_names[class_id]] = {
                'pixels': int(count),
                'percentage': float(percentage)
            }

        return stats

    def process_test_images(self, test_images_dir, output_dir='./inference_results', test_masks_dir=None):

        if test_masks_dir is None or not os.path.exists(test_masks_dir):
            raise ValueError(
                f"GT mask directory is required for the second panel. "
                f"Provided: {test_masks_dir}"
            )

        os.makedirs(output_dir, exist_ok=True)


        masks_dir = os.path.join(output_dir, 'masks')
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)


        image_files = sorted([
            f for f in os.listdir(test_images_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])

        if not image_files:
            print(f"✗ No images found in {test_images_dir}")
            return

        print(f"\n{'='*60}")
        print(f"Processing {len(image_files)} test images...")
        print(f"Using GT masks from: {test_masks_dir}")
        print(f"{'='*60}\n")

        all_stats = {}
        results_summary = []


        for image_file in tqdm(image_files, desc="Inferencing"):
            image_path = os.path.join(test_images_dir, image_file)

            try:

                pred_mask, original_shape = self.predict(image_path)


                stats = self.get_statistics(pred_mask)
                all_stats[image_file] = stats


                mask_filename = image_file.replace('.jpg', '.png').replace('.jpeg', '.png')
                mask_save_path = os.path.join(masks_dir, mask_filename)
                cv2.imwrite(mask_save_path, pred_mask.astype(np.uint8))


                gt_mask_path = self._find_gt_mask_path(image_file, test_masks_dir)
                if gt_mask_path is None:
                    raise ValueError(f"GT mask not found for image: {image_file}")

                gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is None:
                    raise ValueError(f"Failed to read GT mask: {gt_mask_path}")
                if gt_mask.shape != pred_mask.shape:
                    gt_mask = cv2.resize(
                        gt_mask,
                        (pred_mask.shape[1], pred_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )


                viz_result, gt_colored_mask, pred_colored_mask = self.visualize_prediction(image_path, pred_mask, gt_mask=gt_mask)
                viz_filename = f"viz_{mask_filename}"
                viz_save_path = os.path.join(viz_dir, viz_filename)
                viz_result_bgr = cv2.cvtColor(viz_result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(viz_save_path, viz_result_bgr)


                results_summary.append({
                    'image': image_file,
                    'gt_mask_path': gt_mask_path if gt_mask_path is not None else "",
                    'mask_saved': mask_save_path,
                    'visualization_saved': viz_save_path,
                    'statistics': stats
                })

            except Exception as e:
                print(f"\n✗ Error processing {image_file}: {str(e)}")
                results_summary.append({
                    'image': image_file,
                    'error': str(e)
                })


        stats_file = os.path.join(output_dir, 'statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\n✓ Statistics saved to {stats_file}")


        summary_file = os.path.join(output_dir, 'results_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"✓ Results summary saved to {summary_file}")


        self._print_summary(output_dir, len(image_files), all_stats)

    def _print_summary(self, output_dir, total_images, all_stats):

        print(f"\n{'='*60}")
        print("Processing Summary")
        print(f"{'='*60}")
        print(f"Total images processed: {total_images}")
        print(f"Output directory: {output_dir}")
        print(f"\nOutput files:")
        print(f"  - Masks: {os.path.join(output_dir, 'masks')}")
        print(f"  - Visualizations: {os.path.join(output_dir, 'visualizations')}")
        print(f"  - Statistics: {os.path.join(output_dir, 'statistics.json')}")
        print(f"  - Summary: {os.path.join(output_dir, 'results_summary.json')}")


        if all_stats:
            print(f"\n{'='*60}")
            print("Average Statistics Across All Images")
            print(f"{'='*60}")

            class_stats = {}
            for image_stats in all_stats.values():
                for class_name, stats in image_stats.items():
                    if class_name not in class_stats:
                        class_stats[class_name] = {'pixels': 0, 'percentage': 0, 'count': 0}
                    class_stats[class_name]['pixels'] += stats['pixels']
                    class_stats[class_name]['percentage'] += stats['percentage']
                    class_stats[class_name]['count'] += 1

            for class_name, stats in class_stats.items():
                avg_percentage = stats['percentage'] / stats['count']
                print(f"{class_name:15s}: {avg_percentage:6.2f}%")

        print(f"{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch inference for PCB defect segmentation')
    parser.add_argument('--test-dir', type=str, default='./data/test/images',
                        help='Path to test images directory')
    parser.add_argument('--mask-dir', type=str, default=None,
                        help='Path to GT masks directory (optional). If omitted, auto infer from test-dir parent.')
    parser.add_argument('--model', type=str, default='unet',
                        choices=list(AVAILABLE_MODELS.keys()),
                        help='Model name to use')
    parser.add_argument('--backbone', type=str, default='convnext_base',
                        help='Backbone for encoder')
    parser.add_argument('--use-cbam', type=str_to_bool, default=True,
                        help='Whether model uses attention (True/False)')
    parser.add_argument('--attention-type', type=str, default='dpcam',
                        choices=['cbam', 'dpcam', 'dpcam_lite', 'pa_only'],
                        help='Attention mechanism type: cbam, dpcam, dpcam_lite, pa_only')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./inference_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default=INFERENCE_CONFIG['device'],
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()


    if not os.path.exists(args.test_dir):
        print(f"✗ Test directory not found: {args.test_dir}")
        return


    mask_dir = args.mask_dir
    if mask_dir is None:
        parent = os.path.dirname(args.test_dir.rstrip('/'))
        candidate = os.path.join(parent, 'masks')
        if os.path.exists(candidate):
            mask_dir = candidate


    inferencer = BatchInferencer(
        model_path=args.checkpoint,
        model_name=args.model,
        backbone=args.backbone,
        use_cbam=args.use_cbam,
        attention_type=args.attention_type,
        device=torch.device(args.device)
    )


    inferencer.process_test_images(args.test_dir, args.output_dir, test_masks_dir=mask_dir)


if __name__ == '__main__':
    main()
