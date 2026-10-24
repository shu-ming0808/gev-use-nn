from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUTPUT = ROOT / 'outputs'
SEED = 20260929
EAS_CRS = '+proj=lcc +lat_1=25 +lat_2=40 +lat_0=32.5 +lon_0=120 +datum=WGS84 +units=m +no_defs'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def identity(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def verify_bundle():
    manifest = read(ROOT / 'manifest.json')
    for name, digest in manifest['files'].items():
        p = (ROOT / name).resolve()
        if ROOT not in p.parents or not p.is_file() or sha(p) != digest:
            raise ValueError(f'Bundle missing/changed: {name}')
    return manifest

def completed(folder, protocol=None):
    report_path = Path(folder) / 'run_report.json'
    if not report_path.is_file():
        return None
    report = read(report_path)
    if not report.get('workflow_completed'):
        return None
    if protocol is not None and report.get('protocol_id') != protocol:
        raise ValueError('Existing run belongs to a different experiment; use a new experiment name.')
    for name, digest in report['output_sha256'].items():
        if sha(Path(folder) / name) != digest:
            raise ValueError(f'Completed output changed: {folder}/{name}')
    return report

def attempt_folder(parent):
    parent = Path(parent); parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, 10000):
        path = parent / f'attempt_{i:03d}'
        if not path.exists():
            return path
    raise ValueError('Too many attempts')

def latest_completed(parent, protocol=None):
    for folder in sorted(Path(parent).glob('attempt_*'), reverse=True):
        result = completed(folder, protocol)
        if result:
            return folder, result
    return None

@contextmanager
def lock(folder):
    """OS lock releases after crashes; a leftover file is not a stale lock."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'run.lock').open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            try: msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as e: raise RuntimeError(f'Another runner is active: {folder}') from e
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt': msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(f, fcntl.LOCK_UN)

def raw_root():
    return ROOT.parent / '論文資料' / 'ERA5' / 'global_predictors_1976_2025'
