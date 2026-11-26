"""Fetch pinned upstream repositories, or validate existing copies without network."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(directory, *args):
    return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def destination(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes selected root: {relative}')
    return path


def verify(path, record):
    if path.is_symlink() or not (path / '.git').is_dir():
        raise ValueError(f'Expected a standalone Git checkout: {path}')
    if git(path, 'rev-parse', 'HEAD') != record['commit']:
        raise ValueError(f'Unexpected commit: {path}')
    for name, expected in record['file_sha256'].items():
        file = destination(path, name)
        if not file.is_file() or file.is_symlink() or digest(file) != expected:
            raise ValueError(f'Missing or changed author source: {file}')
    # Ignore compiled caches/build products, but reject extra importable source.
    for file in path.rglob('*.py'):
        relative = file.relative_to(path)
        if any(part in {'.git', '__pycache__', 'build', 'dist'} for part in relative.parts):
            continue
        if relative.as_posix() not in record['file_sha256']:
            raise ValueError(f'Unpinned Python source: {file}')


def fetch(path, record):
    if path.exists() or path.is_symlink():
        verify(path, record)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.source-', dir=path.parent) as temporary:
        checkout = Path(temporary) / 'repo'
        checkout.mkdir()
        git(checkout, 'init', '--quiet')
        git(checkout, 'remote', 'add', 'origin', record['remote'])
        git(checkout, 'fetch', '--quiet', '--depth=1', 'origin', record['commit'])
        git(checkout, 'checkout', '--quiet', '--detach', 'FETCH_HEAD')
        verify(checkout, record)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Refusing to overwrite existing destination: {path}')
        checkout.rename(path)


def main():
    lock = json.loads((ROOT / 'configs/sources.lock.json').read_text())['sources']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=list(lock), default=list(lock))
    parser.add_argument('--root', type=Path, default=ROOT, help='Destination/validation root; default repository root')
    parser.add_argument('--check', action='store_true', help='Read-only verification; no network or writes')
    parser.add_argument('--list', action='store_true', help='Print pinned sources and licenses, then exit')
    args = parser.parse_args()
    errors = []
    for name in args.only:
        record = lock[name]
        if args.list:
            print(f"{name}: {record['commit']} | {record['license']} | {record['license_url']}")
            continue
        try:
            path = destination(args.root.resolve(), record['path'])
            if args.check:
                verify(path, record)
            else:
                print(f"{name}: {record['license']}\nLicense: {record['license_url']}", flush=True)
                fetch(path, record)
            print(f'OK {name}: commit and {len(record["file_sha256"])} source/license hashes')
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            errors.append(name)
            print(f'ERROR {name}: {error}')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
