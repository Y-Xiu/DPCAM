"""Rebuild from original annotations, deterministic crops and source-group splits."""
import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from common import ROOT, SPLITS, digest, write_json

CLASSES = {'soldef': ['background', 'good', 'exc_solder', 'poor_solder', 'spike'],
           'soldef_task1': ['background', 'good', 'no_good']}


def grouped_stratify(groups, seed=42):
    """Iterative multilabel allocation, exact source counts and deterministic ties."""
    rng = random.Random(seed)
    remaining = list(groups)
    rng.shuffle(remaining)
    n = len(groups)
    capacity = dict(zip(SPLITS, [int(.7*n), int(.15*n), n-int(.7*n)-int(.15*n)]))
    totals = Counter(label for g in groups for label in g['labels'])
    desired = {s: {label: count*ratio for label, count in totals.items()}
               for s, ratio in zip(SPLITS, (.7, .15, .15))}
    assignment = {}
    while remaining:
        freq = Counter(label for g in remaining for label in g['labels'])
        rare = min(freq, key=lambda label: (freq[label], label))
        candidates = [g for g in remaining if rare in g['labels']]
        for group in candidates:
            eligible = [s for s in SPLITS if capacity[s] > 0]
            scores = {s: (desired[s][rare], sum(desired[s][k] for k in group['labels']),
                          capacity[s]) for s in eligible}
            best = max(scores.values())
            split = rng.choice([s for s in eligible if scores[s] == best])
            assignment[group['id']] = split
            capacity[split] -= 1
            for label in group['labels']:
                desired[split][label] -= 1
            remaining.remove(group)
    return assignment


def write_sample(dataset, name, source_id, image, mask, crop, raw_image, raw_mask, notes):
    staging = ROOT / 'data' / 'processed' / dataset / 'staging'
    staging.mkdir(parents=True, exist_ok=True)
    ip, mp = staging / f'{name}.jpg', staging / f'{name}.png'
    if not cv2.imwrite(str(ip), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise IOError(ip)
    if not cv2.imwrite(str(mp), mask):
        raise IOError(mp)
    labels = sorted(int(x) for x in np.unique(mask) if x != 0)
    if not set(labels) <= set(range(1, len(CLASSES[dataset]))):
        raise ValueError(f'Invalid mask labels: {labels}')
    return {'id': name, 'source_id': source_id, 'labels': labels, 'crop_xyxy': crop,
            'raw_image': str(raw_image.relative_to(ROOT)),
            'raw_annotation': str(raw_mask.relative_to(ROOT)) if raw_mask else None,
            'source_sha256': digest(raw_image), 'notes': notes}


def prepare_soldef(source):
    raw = ROOT / 'data/raw/soldef'
    raw.mkdir(parents=True, exist_ok=True)
    records, groups = [], []
    for annotation in sorted(source.glob('*.json')):
        d = json.loads(annotation.read_text())
        image_path = annotation.with_suffix('.jpg')
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if any(s['label'] not in CLASSES['soldef'][1:] for s in d['shapes']):
            raise ValueError(f'Unsupported label in {annotation}; do not silently remap no_good')
        shutil.copy2(annotation, raw/annotation.name)
        shutil.copy2(image_path, raw/image_path.name)
        im = cv2.imread(str(raw/image_path.name))
        h, w = im.shape[:2]
        if h < 1024 or w < 2048 or len(d['shapes']) < 2:
            raise ValueError(f'Two non-overlapping 1024 crops unavailable: {annotation}')
        mask = np.zeros((h, w), np.uint8)
        centers = []
        # Stable class priority: thin spike annotations override solder regions.
        for shape in sorted(d['shapes'], key=lambda s: CLASSES['soldef'].index(s['label'])):
            points = np.asarray(shape['points'])
            cv2.fillPoly(mask, [points.astype(np.int32)], CLASSES['soldef'].index(shape['label']))
            centers.append(points.mean(0))
        centers = np.array(sorted(centers, key=lambda x: x[0]))
        cut = int(np.argmax(np.diff(centers[:, 0]))) + 1
        left, right = centers[:cut].mean(0), centers[cut:].mean(0)
        boundary = int(np.clip((left[0]+right[0])/2, 1024, w-1024))
        boxes = []
        for side, center in [('left', left), ('right', right)]:
            lo, hi = (0, boundary-1024) if side == 'left' else (boundary, w-1024)
            x = int(np.clip(center[0]-512, lo, hi))
            y = int(np.clip(center[1]-512, 0, h-1024))
            boxes.append([x, y, x+1024, y+1024])
            records.append(write_sample('soldef', annotation.stem+'_'+side, annotation.stem,
                                         im[y:y+1024, x:x+1024], mask[y:y+1024, x:x+1024],
                                         boxes[-1], raw/image_path.name, raw/annotation.name,
                                         {'side': side, 'jitter': False}))
        cropped_pixels = Counter()
        for x, y, x2, y2 in boxes:
            cropped_pixels.update(dict(zip(*np.unique(mask[y:y2, x:x2], return_counts=True))))
        labels = sorted({str(c) for r in records[-2:] for c in r['labels']}) or ['normal']
        lost = {CLASSES['soldef'][c]: int((mask == c).sum())-int(cropped_pixels.get(c, 0))
                for c in range(1, 5)}
        for r in records[-2:]:
            r['notes']['source_foreground_pixels_outside_crops'] = lost
        groups.append({'id': annotation.stem, 'labels': labels})
    return records, groups


def prepare_soldef_task1(source, crop_size=1440, jitter=100, replicates=2):
    """Task 1: top-view component placement, labels good / no_good.

    One annotated component per source image. Two deterministic square crops are
    taken around the component centroid and later resized to the shared 512 input.
    The crop is full-height because the annotated pad+component region can be
    wider than 1440 px; this preserves the complete height and most of the width.
    """
    raw = ROOT / 'data/raw/soldef_task1'
    raw.mkdir(parents=True, exist_ok=True)
    records, groups = [], []
    for annotation in sorted(source.glob('*.json')):
        d = json.loads(annotation.read_text())
        image_path = annotation.with_suffix('.jpg')
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        labels = {s['label'] for s in d['shapes']}
        if labels - set(CLASSES['soldef_task1'][1:]) or len(labels) != 1:
            raise ValueError(f'Unexpected task-1 labels in {annotation}: {labels}')
        if len(d['shapes']) != 1:
            raise ValueError(f'Expected one component annotation: {annotation}')
        if annotation.resolve() != (raw / annotation.name).resolve():
            shutil.copy2(annotation, raw / annotation.name)
        if image_path.resolve() != (raw / image_path.name).resolve():
            shutil.copy2(image_path, raw / image_path.name)
        im = cv2.imread(str(raw / image_path.name))
        h, w = im.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        points = np.asarray(d['shapes'][0]['points'])
        cv2.fillPoly(mask, [points.astype(np.int32)],
                     CLASSES['soldef_task1'].index(d['shapes'][0]['label']))
        cx, cy = points.mean(0)
        for m in range(replicates):
            dx = int(round((m - (replicates - 1) / 2) * 2 * jitter))
            x = int(np.clip(cx + dx - crop_size / 2, 0, w - crop_size))
            y = int(np.clip(cy - crop_size / 2, 0, h - crop_size))
            if crop_size >= h:
                y = 0
            box = [x, y, x + crop_size, y + crop_size]
            records.append(write_sample('soldef_task1', f'{annotation.stem}_{m}',
                                        annotation.stem,
                                        im[y:y + crop_size, x:x + crop_size],
                                        mask[y:y + crop_size, x:x + crop_size],
                                        box, raw / image_path.name, raw / annotation.name,
                                        {'task': 'component_placement',
                                         'jitter_x': dx, 'jitter': True}))
        crop_labels = sorted({str(c) for r in records[-replicates:] for c in r['labels']})
        groups.append({'id': annotation.stem, 'labels': crop_labels or ['normal']})
    return records, groups


def finish(dataset, records, groups, seed, crop_size=1024):
    if not records:
        raise ValueError('Empty source data')
    assignment = grouped_stratify(groups, seed)
    root = ROOT/'data/processed'/dataset
    for r in records:
        split = assignment[r['source_id']]
        r['split'] = split
        for kind, suffix in [('images', '.jpg'), ('masks', '.png')]:
            dst = root/split/kind/(r['id']+suffix); dst.parent.mkdir(parents=True, exist_ok=True)
            (root/'staging'/(r['id']+suffix)).replace(dst)
            r['image' if kind == 'images' else 'mask'] = str(dst.relative_to(ROOT))
            r['image_sha256' if kind == 'images' else 'mask_sha256'] = digest(dst)
    (root/'staging').rmdir()
    write_json(root/'manifest.json', {'dataset': dataset, 'classes': CLASSES[dataset],
                                      'split_seed': seed, 'split_unit': 'source_image',
                                      'ratios': [.7, .15, .15], 'crop_size': crop_size,
                                      'records': records})
    report = audit(dataset)
    write_json(ROOT/'reports'/f'data_{dataset}.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def audit(dataset):
    path = ROOT/'data/processed'/dataset/'manifest.json'
    manifest = json.loads(path.read_text())
    seen = {key: {} for key in ['source_id', 'source_sha256', 'image_sha256']}
    counts, class_counts, source_counts = Counter(), {s: Counter() for s in SPLITS}, {}
    checked_raw = {}
    for r in manifest['records']:
        split = r['split']; counts[split] += 1
        if r['raw_image'] not in checked_raw:
            checked_raw[r['raw_image']] = digest(ROOT/r['raw_image'])
        if checked_raw[r['raw_image']] != r['source_sha256']:
            raise ValueError(f'Raw source changed: {r["raw_image"]}')
        for key, mapping in seen.items():
            if r[key] in mapping and mapping[r[key]] != split:
                raise ValueError(f'Cross-split leakage ({key}): {r["id"]}')
            mapping[r[key]] = split
        ip, mp = ROOT/r['image'], ROOT/r['mask']
        if digest(ip) != r['image_sha256'] or digest(mp) != r['mask_sha256']:
            raise ValueError(f'Data changed: {r["id"]}')
        im, mask = cv2.imread(str(ip)), cv2.imread(str(mp), 0)
        expected_size = (manifest.get('crop_size', 1024),) * 2
        if im.shape[:2] != expected_size or mask.shape != expected_size:
            raise ValueError(f'Invalid dimensions: {ip}')
        labels = sorted(int(c) for c in np.unique(mask) if c)
        if labels != r['labels']:
            raise ValueError(f'Incorrect manifest classes: {mp}')
        class_counts[split].update(manifest['classes'][c] for c in labels)
        if not labels:
            class_counts[split]['normal'] += 1
        source_counts[r['source_id']] = split
    for split in SPLITS:
        for kind, key in [('images', 'image'), ('masks', 'mask')]:
            actual = {str(p.relative_to(ROOT)) for p in (path.parent/split/kind).iterdir() if p.is_file()}
            expected = {r[key] for r in manifest['records'] if r['split'] == split}
            if actual != expected:
                raise ValueError(f'Untracked or missing files: {dataset}/{split}/{kind}')
        if any(class_counts[split][c] == 0 for c in manifest['classes'][1:]):
            raise ValueError(f'Missing foreground class in {dataset}/{split}')
    return {'dataset': dataset, 'manifest_sha256': digest(path), 'samples': dict(counts),
            'sources': dict(Counter(source_counts.values())), 'classes': class_counts,
            'cross_split_source_overlap': 0, 'cross_split_image_hash_overlap': 0,
            'image_mask_pairs_verified': len(manifest['records'])}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['soldef', 'soldef_task1', 'all'], default='all')
    p.add_argument('--source-root', type=Path, default=ROOT.parent)
    p.add_argument('--soldef-source', type=Path)
    p.add_argument('--soldef-task1-source', type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--audit-only', action='store_true')
    args = p.parse_args()
    datasets = ('soldef', 'soldef_task1') if args.dataset == 'all' else (args.dataset,)
    for dataset in datasets:
        if args.audit_only:
            print(json.dumps(audit(dataset), ensure_ascii=False, indent=2)); continue
        dest = ROOT/'data/processed'/dataset
        if dest.exists():
            raise FileExistsError(f'{dest}: refusing to overwrite an existing dataset')
        crop_size = 1024
        if dataset == 'soldef':
            recs, groups = prepare_soldef(args.soldef_source or args.source_root/'pcbdata1/dataset2')
        elif dataset == 'soldef_task1':
            recs, groups = prepare_soldef_task1(args.soldef_task1_source or args.source_root/'pcbdata1/dataset1')
            crop_size = 1440
        finish(dataset, recs, groups, args.seed, crop_size)
