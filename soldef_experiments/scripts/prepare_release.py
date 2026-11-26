"""Create local frozen configs after data preparation; never train or select models."""
import argparse
import datetime
import json
from pathlib import Path
import runpy
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_membership(manifest, reference):
    """Compare immutable source membership independently of JPEG encoder versions."""
    for key in ('dataset', 'classes', 'split_seed', 'split_unit', 'ratios', 'crop_size'):
        if manifest.get(key) != reference.get(key):
            raise ValueError(f'Dataset metadata differs: {key}')
    keys = ('id', 'source_id', 'source_sha256', 'split', 'crop_xyxy', 'labels')
    def records(value):
        rows = value['records']
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate crop ID')
        return sorted(({k: r[k] for k in keys} for r in rows), key=lambda r: r['id'])
    if records(manifest) != records(reference):
        raise ValueError('Source images, crop geometry, class labels or split membership differ from the released protocol')


def check_fresh(root):
    for name in ('two_task_runs', 'comparison_runs', 'reports/two_task', 'reports/comparison_study'):
        folder = root / name
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f'{name} already contains run/freeze/setup output. Use a fresh checkout for a new cohort.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only', action='store_true', help='Check prepared data and published memberships without rewriting configs')
    args = p.parse_args()
    from paper_protocol.data import audit
    for dataset in ('soldef_task1', 'soldef'):
        path = ROOT / f'data/processed/{dataset}/manifest.json'
        if not path.is_file():
            raise FileNotFoundError(f'{path}: run prepare_data.py first (see README)')
        reference = json.loads((ROOT / f'data/splits/{dataset}.json').read_text())
        verify_membership(json.loads(path.read_text()), reference)
        print(dataset, audit(path)['samples'])
    if args.check_only:
        return
    check_fresh(ROOT)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = ROOT / 'reports/config_backups' / stamp
    for name in ('two_task', 'comparison_study'):
        source = ROOT / 'configs' / name
        if source.exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(backup / name))
    try:
        runpy.run_module('two_task_study.prepare', run_name='__main__')
        runpy.run_module('comparison_study.prepare', run_name='__main__')
    except BaseException:
        for name in ('two_task', 'comparison_study'):
            source = ROOT / 'configs' / name
            if source.exists():
                shutil.rmtree(source)
            if (backup / name).exists():
                shutil.move(str(backup / name), str(source))
        raise
    print('Prepared 90 ablations and 60 comparisons. No training or model evaluation was started.')
    print('Previous config snapshot:', backup.relative_to(ROOT))


if __name__ == '__main__':
    main()
