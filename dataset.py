import json
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from common import ROOT, seed_worker


class SegmentationDataset(Dataset):
    def __init__(self, dataset, split, size=512):
        manifest = json.loads((ROOT/'data/processed'/dataset/'manifest.json').read_text())
        self.records = [r for r in manifest['records'] if r['split'] == split]
        self.size, self.training = size, split == 'train'
        self.mean = torch.tensor([.485, .456, .406])[:, None, None]
        self.std = torch.tensor([.229, .224, .225])[:, None, None]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        with Image.open(ROOT/r['image']) as f:
            im = f.convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR)
        with Image.open(ROOT/r['mask']) as f:
            mask = f.convert('L').resize((self.size, self.size), Image.Resampling.NEAREST)
        if self.training:
            for operation in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM):
                if random.random() < .5:
                    im, mask = im.transpose(operation), mask.transpose(operation)
            if random.random() < .5:
                angle = random.uniform(-15, 15)
                im = im.rotate(angle, Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
                mask = mask.rotate(angle, Image.Resampling.NEAREST, fillcolor=0)
            if random.random() < .5:
                im = ImageEnhance.Brightness(im).enhance(random.uniform(.8, 1.2))
                im = ImageEnhance.Contrast(im).enhance(random.uniform(.8, 1.2))
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float()/255
        y = torch.from_numpy(np.array(mask, dtype=np.int64))
        return (x-self.mean)/self.std, y, r['id']


def make_loader(config, split, seed, batch_size=None, workers=None):
    manifest = json.loads((ROOT/'data/processed'/config['dataset']/'manifest.json').read_text())
    if manifest['classes'] != config['classes']:
        raise ValueError('Configuration class order differs from the dataset manifest')
    data = SegmentationDataset(config['dataset'], split, config['image_size'])
    generator = torch.Generator().manual_seed(seed)
    batch = batch_size or config['training']['batch_size']
    # Keep batch size >=2 for global-pooling BatchNorm in comparison networks.
    drop_last = split == 'train' and len(data) % batch == 1
    return DataLoader(data, batch_size=batch, shuffle=split == 'train',
                      num_workers=config['training']['workers'] if workers is None else workers,
                      worker_init_fn=seed_worker, generator=generator, drop_last=drop_last,
                      pin_memory=torch.cuda.is_available())
