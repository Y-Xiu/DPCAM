"""Fetch SHA256-pinned assets without installing anything or replacing unknown files."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(path, record):
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}')
    if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
        raise ValueError(f'Size/SHA256 mismatch; existing file left unchanged: {path}')


def destination(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes selected root: {relative}')
    return path


def fetch(name, record, records, root, cache=None, offline=False):
    path = destination(root, record['path'])
    if path.exists() or path.is_symlink():
        verify(path, record)
        return
    cached = cache / path.name if cache else None
    if cached and cached.is_file():
        verify(cached, record)
    elif record.get('derived_from'):
        source_name = record['derived_from']
        source_record = records[source_name]
        source = destination(root, source_record['path'])
        fetch(source_name, source_record, records, root, cache, offline)
        sys.path.insert(0, str(ROOT))
        from modern_study.convert_mit import convert
        convert(source, path, source_record['sha256'], record['sha256'])
        verify(path, record)
        return
    elif record.get('manual'):
        raise FileNotFoundError(f"Download the author classification weight from {record['url']}; "
                                f"place it at {path} or use --cache-dir containing {path.name}")
    elif offline:
        raise FileNotFoundError(f'Not found in local cache: {path.name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    # A private temporary file is checked before being published. Hard-link creation
    # fails if another process creates the target; an existing target is never replaced.
    with tempfile.TemporaryDirectory(prefix='.asset-', dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        if cached and cached.is_file():
            shutil.copyfile(cached, staged)
        else:
            print(f"Downloading {name}: {record['url']}", flush=True)
            request = urllib.request.Request(record['url'], headers={'User-Agent': 'DPCAM-reproduction/1'})
            with urllib.request.urlopen(request, timeout=120) as response, staged.open('wb') as output:
                shutil.copyfileobj(response, output)
        verify(staged, record)
        os.link(staged, path)


def main():
    lock = json.loads((ROOT / 'configs/assets.lock.json').read_text())
    records = lock['assets']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=list(records), default=lock['default_assets'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--cache-dir', type=Path, help='Reuse files with the recorded basename after SHA256 validation')
    parser.add_argument('--offline', action='store_true', help='Only import cached files or perform MiT conversion; no downloads')
    parser.add_argument('--check', action='store_true', help='Read-only SHA256 validation; no downloads, conversion or writes')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    errors = []
    for name in args.only:
        record = records[name]
        if args.list:
            print(f"{name}: {record['path']} ({record['kind']}, {record['bytes']} bytes)")
            continue
        try:
            path = destination(args.root.resolve(), record['path'])
            if args.check:
                verify(path, record)
            else:
                fetch(name, record, records, args.root.resolve(), args.cache_dir, args.offline)
            print(f'OK {name}: {record["sha256"]}')
        except Exception as error:
            errors.append(name)
            print(f'ERROR {name}: {error}')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
