"""Freeze the matched two-task SolDef matrix: task1 placement + task2 solder state."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, digest, write_json
from paper_protocol.data import audit

VARIANTS = ('none', 'cbam', 'dpcam', 'no_dca', 'no_pa', 'no_dsa',
            'equal_spatial', 'residual_cbam', 'residual_dpcam')
SEEDS = (42, 43, 44, 45, 46)
TASKS = {
    'task1': {
        'dataset': 'soldef_task1',
        'manifest': 'data/processed/soldef_task1/manifest.json',
        'classes': ['background', 'good', 'no_good'],
        'description': 'top-view SMT component placement (good / no_good)',
    },
    'task2': {
        'dataset': 'soldef',
        'manifest': 'data/processed/soldef/manifest.json',
        'classes': ['background', 'good', 'exc_solder', 'poor_solder', 'spike'],
        'description': '45-degree solder joint state (good / excessive / insufficient / spike)',
    },
}


def attention_for(variant):
    if variant == 'cbam' or variant == 'residual_cbam':
        return 'cbam'
    if variant == 'none':
        return 'none'
    return 'dpcam'


def config_for(task, spec, variant, audits):
    return {
        'experiment': f'twotask_{task}_{variant}',
        'protocol_id': 'two_task_v1',
        'task': task,
        'task_description': spec['description'],
        'dataset': spec['dataset'],
        'classes': spec['classes'],
        'image_size': 512,
        'split_seed': 42,
        'split_protocol': 'source_group',
        'model': {
            'name': 'paper_unet',
            'backbone': 'convnext_base',
            'attention': attention_for(variant),
            'pretrained': True,
            'feature_indices': [0, 1, 2, 3],
        },
        'model_provenance': (
            'Original ConvNeXt-Base/decoder and DPCAM operators; explicit study_variant '
            'defines removals or residual candidate; common parameters shared before attention'),
        'training': {
            'epochs': 150,
            'batch_size': 2,
            'lr': 0.0001,
            'optimizer': 'AdamW',
            'weight_decay': 1e-05,
            'scheduler': 'cosine',
            'workers': 4,
            'cpu_threads': 4,
            'selection': 'min_val_loss',
            'mixed_precision': False,
            'min_lr': 1e-06,
        },
        'loss': {
            'alpha': 0.25,
            'gamma': 2.0,
            'dice_weight': 0.5,
            'focal_weight': 0.5,
            'smooth': 1e-06,
        },
        'loss_protocol': 'softmax_one_vs_rest_binary_focal_mean_over_classes',
        'metric_protocol': 'global_confusion_macro_all_classes_empty_zero_v1',
        'position_encoding': 'stage resolution at image_size; bilinear interpolation',
        'profiling': {'batch_size': 1, 'warmup': 10, 'iterations': 100},
        'manifest': spec['manifest'],
        'manifest_sha256': audits[task]['manifest_sha256'],
        'loss_numerics': 'logsumexp/log_softmax; same softmax one-vs-rest Dice/Focal formula; no hard probability clamp',
        'evaluation_scope': 'train/validation only; test frozen',
        'study_variant': variant,
        'initialization_protocol': (
            'Construct common no-attention model first; append attention with a separate seeded RNG'),
    }


def main():
    audits = {task: audit(ROOT / spec['manifest']) for task, spec in TASKS.items()}
    entries = []
    for task, spec in TASKS.items():
        for variant in VARIANTS:
            cfg = config_for(task, spec, variant, audits)
            path = ROOT / 'configs/two_task' / f'{cfg["experiment"]}.json'
            if path.exists() and json.loads(path.read_text()) != cfg:
                raise ValueError('Existing config differs: ' + str(path))
            write_json(path, cfg)
            for seed in SEEDS:
                entries.append({
                    'task': task,
                    'dataset': spec['dataset'],
                    'variant': variant,
                    'seed': seed,
                    'config': str(path.relative_to(ROOT)),
                    'config_sha256': digest(path),
                })
    plan = {
        'protocol_id': 'two_task_v1',
        'source_paper': 'https://doi.org/10.3390/jmmp8030117',
        'tasks': {task: {k: v for k, v in spec.items()} for task, spec in TASKS.items()},
        'variants': list(VARIANTS),
        'seeds': list(SEEDS),
        'epochs': 150,
        'jobs': entries,
        'data_limitations': [
            'Task 1: 228 top-view originals with one good/no_good component region each; '
            'two deterministic 1440 crops per source, resized to 512.',
            'Task 2: 200 45-degree originals with four-class joint annotations; '
            'two 1024 crops per source, resized to 512.',
            'Both manifests use source-image-level 70/15/15 splits with no shared source.',
            'Test splits are frozen and are not used for checkpoint or hyperparameter selection.',
        ],
        'selection': 'minimum validation loss; paired training seeds only',
    }
    plan_path = ROOT / 'configs/two_task/plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Frozen plan differs: ' + str(plan_path))
    write_json(plan_path, plan)
    write_json(ROOT / 'reports/two_task/data_audit.json',
               {'source_paper': 'https://doi.org/10.3390/jmmp8030117', 'tasks': audits})
    print(json.dumps({t: {k: audits[t][k] for k in ('samples', 'class_presence',
                                                     'sources_in_multiple_splits',
                                                     'manifest_sha256')}
                      for t in TASKS}, ensure_ascii=False, indent=2))
    print(f'Frozen: {len(entries)} jobs = 2 tasks x {len(VARIANTS)} variants x {len(SEEDS)} seeds')


if __name__ == '__main__':
    main()
