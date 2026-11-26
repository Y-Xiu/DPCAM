import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = ('train', 'val', 'test')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg['image_size'] % 32:
        raise ValueError('image_size must be divisible by 32')
    training = cfg['training']
    supported = {'optimizer': 'AdamW', 'scheduler': 'constant',
                 'selection': 'min_val_loss', 'mixed_precision': False}
    for key, value in supported.items():
        if training[key] != value:
            raise ValueError(f'Unsupported training option {key}={training[key]!r}')
    if training['epochs'] < 1 or training['batch_size'] < 2:
        raise ValueError('epochs >= 1 and training batch_size >= 2 required')
    if len(cfg['classes']) != len(set(cfg['classes'])) or cfg['classes'][0] != 'background':
        raise ValueError('Class names must be unique and start with background')
    return cfg


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
