"""Shared runtime helpers for the two-task trainers (no legacy training CLI)."""
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from metrics import ConfusionMatrix
from models import logits_of
from models.attention import DPCAM, CBAM


def device_for(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu explicitly')
    return torch.device(name)

def architecture_info(model):
    modules = [(n, m) for n, m in model.named_modules() if isinstance(m, (DPCAM, CBAM))]
    return {'model_class': type(model).__module__+'.'+type(model).__name__,
            'initialization': getattr(model, 'initialization_provenance', {'kind': 'see run config'}),
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'attention': [{'path': n, 'type': type(m).__name__,
                           'dca': getattr(m, 'dca', None) is not None,
                           'pa': getattr(m, 'pa', None) is not None,
                           'dsa': getattr(m, 'dsa', None) is not None} for n, m in modules]}

def environment():
    import timm, torchvision
    return {'python': sys.version, 'platform': platform.platform(), 'torch': str(torch.__version__),
            'torchvision': torchvision.__version__, 'timm': timm.__version__,
            'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}

def epoch_pass(model, loader, criterion, device, optimizer=None, max_batches=None,
               classes=None, output_dir=None):
    model.train(optimizer is not None)
    total_loss, samples = 0., 0
    confusion = ConfusionMatrix(len(classes))
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for index, (images, masks, ids) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            images, masks = images.to(device), masks.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = logits_of(model, images)
            loss = criterion(logits, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss in batch {index}')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            confusion.update(logits.detach(), masks)
            total_loss += loss.item()*len(images)
            samples += len(images)
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                for name, pred in zip(ids, logits.argmax(1).cpu().numpy()):
                    Image.fromarray(pred.astype(np.uint8)).save(output_dir/(name+'.png'))
    if not samples:
        raise ValueError('No samples processed')
    return {'loss': total_loss/samples, 'samples': samples, **confusion.compute(classes)}, confusion.matrix
