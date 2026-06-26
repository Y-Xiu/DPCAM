import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import create_model
from dataset import PCBDefectDataset, get_train_transform, get_val_transform
from metrics import dice_coef, iou_score, recall_score
from losses import get_loss_function  # 新增：导入损失函数
from config import (
    DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, OPTIMIZER_CONFIG,
    SCHEDULER_CONFIG, LOSS_CONFIG, CHECKPOINT_CONFIG, LOG_CONFIG,
    get_model_save_dir, get_log_save_dir, get_tensorboard_dir, AVAILABLE_MODELS
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



def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class Trainer:
    def __init__(self, model_name='unet', backbone=None, use_cbam=True, attention_type='cbam', config_dict=None):
        self.model_name = model_name.lower()
        self.backbone = backbone
        self.use_cbam = use_cbam
        self.attention_type = attention_type
        self.device = torch.device(TRAIN_CONFIG['device'] if torch.cuda.is_available() else 'cpu')

        if self.backbone is None:
            from models import get_default_backbone
            self.backbone = get_default_backbone(self.model_name)


        self.checkpoint_dir = get_model_save_dir(self.model_name, self.backbone, self.use_cbam)

        if self.use_cbam and self.attention_type != 'cbam':
            self.checkpoint_dir = os.path.join(CHECKPOINT_CONFIG['save_dir'], 
                                              f"{self.model_name}_backbone_{self.backbone}_{self.attention_type}")
        self.log_dir = get_log_save_dir(self.model_name, self.backbone, self.use_cbam)
        if self.use_cbam and self.attention_type != 'cbam':
            self.log_dir = os.path.join(LOG_CONFIG['log_dir'], 
                                       f"{self.model_name}_backbone_{self.backbone}_{self.attention_type}")
        self.tensorboard_dir = get_tensorboard_dir(self.model_name, self.backbone, self.use_cbam)
        if self.use_cbam and self.attention_type != 'cbam':
            self.tensorboard_dir = os.path.join(LOG_CONFIG['tensorboard_dir'], 
                                               f"{self.model_name}_backbone_{self.backbone}_{self.attention_type}")


        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        self.logger = setup_logging(self.log_dir)

        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Model: {self.model_name}, Backbone: {self.backbone}")
        self.logger.info(f"Attention: {self.attention_type if self.use_cbam else 'None'}")
        self.logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
        self.logger.info(f"Log dir: {self.log_dir}")
        self.logger.info(f"TensorBoard dir: {self.tensorboard_dir}")

        # TensorBoard writer
        self.writer = SummaryWriter(self.tensorboard_dir)


        self.model = create_model(
            model_name=self.model_name,
            num_classes=MODEL_CONFIG['num_classes'],
            backbone=self.backbone,
            use_cbam=self.use_cbam,
            pretrained=MODEL_CONFIG['pretrained'],
            attention_type=self.attention_type
        ).to(self.device)

        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")


        self.criterion = get_loss_function(LOSS_CONFIG)
        if LOSS_CONFIG['type'] == 'combined':
            self.logger.info(f"Using combined loss: {list(LOSS_CONFIG['losses'].keys())} with weights {LOSS_CONFIG['weights']}")
        else:
            self.logger.info(f"Using loss: {LOSS_CONFIG['type']}")
        

        self.use_combined_loss = LOSS_CONFIG['type'] == 'combined'


        if OPTIMIZER_CONFIG['type'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=OPTIMIZER_CONFIG['lr'],
                weight_decay=OPTIMIZER_CONFIG['weight_decay'],
                betas=OPTIMIZER_CONFIG['betas']
            )
        elif OPTIMIZER_CONFIG['type'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=OPTIMIZER_CONFIG['lr'],
                weight_decay=OPTIMIZER_CONFIG['weight_decay']
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=OPTIMIZER_CONFIG['lr'],
                weight_decay=OPTIMIZER_CONFIG['weight_decay'],
                momentum=0.9
            )


        if SCHEDULER_CONFIG['type'].lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=SCHEDULER_CONFIG['T_max'],
                eta_min=SCHEDULER_CONFIG['eta_min']
            )
        elif SCHEDULER_CONFIG['type'].lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )

        self.best_val_iou = 0
        self.best_epoch = 0
        self.global_step = 0

    def train_epoch(self, train_loader, epoch):

        self.model.train()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        total_mpa = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            

            if self.use_combined_loss:
                loss, loss_dict = self.criterion(outputs, masks)
            else:
                loss = self.criterion(outputs, masks)
                loss_dict = None

            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                dice = dice_coef(outputs, masks)
                iou = iou_score(outputs, masks)
                mpa = recall_score(outputs, masks)

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            total_mpa += mpa


            if (batch_idx + 1) % LOG_CONFIG['log_interval'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_dice = total_dice / (batch_idx + 1)
                avg_iou = total_iou / (batch_idx + 1)
                avg_mpa = total_mpa / (batch_idx + 1)

                log_msg = (
                    f"Epoch [{epoch+1}/{TRAIN_CONFIG['num_epochs']}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.4f} mDice: {avg_dice:.4f} mIoU: {avg_iou:.4f} mPA: {avg_mpa:.4f}"
                )

                if loss_dict is not None:
                    loss_parts = ' '.join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                    log_msg += f" ({loss_parts})"
                
                self.logger.info(log_msg)

                self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                self.writer.add_scalar('train/dice', avg_dice, self.global_step)
                self.writer.add_scalar('train/iou', avg_iou, self.global_step)
                self.writer.add_scalar('train/mpa', avg_mpa, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        epoch_loss = total_loss / len(train_loader)
        epoch_dice = total_dice / len(train_loader)
        epoch_iou = total_iou / len(train_loader)
        epoch_mpa = total_mpa / len(train_loader)

        return epoch_loss, epoch_dice, epoch_iou, epoch_mpa

    def validate(self, val_loader, epoch):

        self.model.eval()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        total_mpa = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)

                if self.use_combined_loss:
                    loss, _ = self.criterion(outputs, masks)
                else:
                    loss = self.criterion(outputs, masks)

                dice = dice_coef(outputs, masks)
                iou = iou_score(outputs, masks)
                mpa = recall_score(outputs, masks)

                total_loss += loss.item()
                total_dice += dice
                total_iou += iou
                total_mpa += mpa

        val_loss = total_loss / len(val_loader)
        val_dice = total_dice / len(val_loader)
        val_iou = total_iou / len(val_loader)
        val_mpa = total_mpa / len(val_loader)

        self.writer.add_scalar('val/loss', val_loss, epoch)
        self.writer.add_scalar('val/dice', val_dice, epoch)
        self.writer.add_scalar('val/iou', val_iou, epoch)
        self.writer.add_scalar('val/mpa', val_mpa, epoch)

        self.logger.info(
            f"Validation - Loss: {val_loss:.4f} mDice: {val_dice:.4f} mIoU: {val_iou:.4f} "
            f"mPA: {val_mpa:.4f}"
        )

        return val_loss, val_dice, val_iou, val_mpa

    def save_checkpoint(self, epoch, val_iou, is_best=False):

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_iou': val_iou,
            'model_name': self.model_name,
            'backbone': self.backbone,
            'use_cbam': self.use_cbam,
            'attention_type': self.attention_type,  # 新增：保存注意力类型
        }


        latest_path = os.path.join(self.checkpoint_dir, 'latest_model.pth')
        torch.save(checkpoint, latest_path)


        if (epoch + 1) % CHECKPOINT_CONFIG['save_interval'] == 0:
            periodic_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(checkpoint, periodic_path)
            self.logger.info(f"Saved checkpoint: {periodic_path}")


        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")

    def train(self):

        self.logger.info("=" * 50)
        self.logger.info("Starting training...")
        self.logger.info("=" * 50)


        self.logger.info("Loading datasets...")
        train_dataset = PCBDefectDataset(
            DATA_CONFIG['train_images_dir'],
            DATA_CONFIG['train_masks_dir'],
            transform=get_train_transform(image_size=DATA_CONFIG['image_size']),
            is_train=True
        )
        val_dataset = PCBDefectDataset(
            DATA_CONFIG['val_images_dir'],
            DATA_CONFIG['val_masks_dir'],
            transform=get_val_transform(image_size=DATA_CONFIG['image_size']),
            is_train=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=True,
            num_workers=TRAIN_CONFIG['num_workers'],
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=False,
            num_workers=TRAIN_CONFIG['num_workers'],
            pin_memory=True
        )

        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Val samples: {len(val_dataset)}")


        start_time = time.time()

        for epoch in range(TRAIN_CONFIG['num_epochs']):
            self.logger.info(f"\nEpoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}")


            train_loss, train_dice, train_iou, train_mpa = self.train_epoch(train_loader, epoch)


            val_loss, val_dice, val_iou, val_mpa = self.validate(val_loader, epoch)


            self.scheduler.step()


            is_best = val_iou > self.best_val_iou
            if is_best:
                self.best_val_iou = val_iou
                self.best_epoch = epoch

            self.save_checkpoint(epoch, val_iou, is_best=is_best)

            self.writer.add_scalars('epoch/loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('epoch/dice', {
                'train': train_dice,
                'val': val_dice
            }, epoch)
            self.writer.add_scalars('epoch/iou', {
                'train': train_iou,
                'val': val_iou
            }, epoch)
            self.writer.add_scalars('epoch/mpa', {
                'train': train_mpa,
                'val': val_mpa
            }, epoch)

        total_time = time.time() - start_time
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Training completed!")
        self.logger.info(f"Best epoch: {self.best_epoch + 1}, Best Val IoU: {self.best_val_iou:.4f}")
        self.logger.info(f"Total training time: {total_time / 3600:.2f} hours")
        self.logger.info("=" * 50)

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train PCB defect segmentation model')
    parser.add_argument('--model', type=str, default=MODEL_CONFIG['model_name'],
                        choices=list(AVAILABLE_MODELS.keys()),
                        help=f'Model to use. Available: {", ".join(AVAILABLE_MODELS.keys())}')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Backbone for encoder. If None, use default backbone for the model')
    parser.add_argument('--use-cbam', type=str_to_bool, default=MODEL_CONFIG['use_cbam'],
                        help='Whether to use attention module (True/False)')
    parser.add_argument('--attention-type', type=str, default=MODEL_CONFIG.get('attention_type', 'cbam'),
                        choices=['cbam', 'dpcam', 'dpcam_lite', 'pa_only'],
                        help='Attention mechanism type: cbam (original), dpcam, dpcam_lite, pa_only (position attention only)')
    parser.add_argument('--batch-size', type=int, default=TRAIN_CONFIG['batch_size'],
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['num_epochs'],
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=OPTIMIZER_CONFIG['lr'],
                        help='Learning rate')
    parser.add_argument('--device', type=str, default=TRAIN_CONFIG['device'],
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    TRAIN_CONFIG['batch_size'] = args.batch_size
    TRAIN_CONFIG['num_epochs'] = args.epochs
    TRAIN_CONFIG['device'] = args.device
    OPTIMIZER_CONFIG['lr'] = args.lr

    trainer = Trainer(model_name=args.model, backbone=args.backbone, 
                     use_cbam=args.use_cbam, attention_type=args.attention_type)
    trainer.train()


if __name__ == '__main__':
    main()
