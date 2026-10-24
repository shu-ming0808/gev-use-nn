"""Update portable code only, with verified inputs, backup and no workflow launch."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f'Unsafe bundle path: {relative}')
    return path


@contextmanager
def run_lock(root):
    # Same OS lock as the runner; a leftover lock file does not mean active work.
    with (root / 'outputs/run.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def source_files(repo):
    template = repo / 'tools/thesis_run_bundle'
    excluded = {'__pycache__', '.venv', '.pytest_cache'}
    files = {p.relative_to(template).as_posix(): p for p in template.rglob('*')
             if p.is_file() and not excluded.intersection(p.relative_to(template).parts)}
    files.update({p.relative_to(repo).as_posix(): p for p in (repo / 'src').rglob('*.py')
                  if not excluded.intersection(p.relative_to(repo).parts)})
    files['notebooks/nn_mle_full_workflow_comparison.ipynb'] = (
        repo / 'simulation/taiwan_validation/nn_mle_full_workflow_comparison.ipynb')
    forbidden = {'inputs', 'models', 'data', 'outputs', 'archives', 'global_preprocessing'}
    if any(Path(name).parts[0] in forbidden or name == 'manifest.json' for name in files):
        raise ValueError('Source plan contains protected data/results')
    return files


def update(destination, *, apply=False, repo=REPO):
    root = destination.resolve()
    repo = repo.resolve()
    if root == repo or repo in root.parents:
        raise ValueError('Portable destination must be outside checkout')
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    with run_lock(root):
        for name, digest in manifest['files'].items():
            path = inside(root, name)
            if not path.is_file() or sha(path) != digest:
                raise ValueError(f'Existing bundle missing/changed: {name}; no update applied')
        plan = []
        for name, source in sorted(source_files(repo).items()):
            target = inside(root, name)
            digest = sha(source)
            if target.exists() and name not in manifest['files'] and sha(target) != digest:
                raise ValueError(f'Unmanaged local file would be overwritten: {name}')
            if not target.exists() or sha(target) != digest:
                plan.append((name, source, digest))
        report = dict(destination=str(root),apply=apply,changed=[n for n,_,_ in plan],
                      protected_data_results_models_unchanged=True,workflow_started=False)
        if not apply or not plan:
            return report
        # Stage all source bytes before touching the destination. Keep old files
        # and manifest in a unique archive for recovery after interruptions.
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        backup = root / 'archives' / ('code_update_' + stamp)
        backup.mkdir(parents=True, exist_ok=False)
        shutil.copy2(manifest_path, backup / 'manifest.json')
        for name, _, _ in plan:
            target = inside(root, name)
            if target.exists():
                saved = inside(backup, name)
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
        with tempfile.TemporaryDirectory(prefix='thesis_bundle_update_') as temporary:
            stage = Path(temporary)
            for name, source, digest in plan:
                staged = stage / name
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, staged)
                if sha(staged) != digest:
                    raise RuntimeError(f'Source changed while staging: {name}')
            try:
                for name, _, digest in plan:
                    target = inside(root, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(stage / name, target)
                    if sha(target) != digest:
                        raise RuntimeError(f'Copy verification failed: {name}')
                new_manifest = dict(manifest, files=dict(manifest['files']))
                for name, _, digest in plan:
                    new_manifest['files'][name] = digest
                # Verify unchanged model/input entries as well as new code.
                for name, digest in new_manifest['files'].items():
                    if sha(inside(root, name)) != digest:
                        raise RuntimeError(f'Post-update verification failed: {name}')
                temp_manifest = root / 'manifest.json.update_tmp'
                temp_manifest.write_text(json.dumps(new_manifest, indent=2), encoding='utf-8')
                temp_manifest.replace(manifest_path)
            except Exception:
                for name, _, _ in plan:
                    saved = inside(backup, name)
                    target = inside(root, name)
                    if saved.exists():
                        shutil.copy2(saved, target)
                    elif target.is_file():
                        target.unlink()  # Only this update's new, scoped files.
                shutil.copy2(backup / 'manifest.json', manifest_path)
                raise
        report.update(backup=str(backup), manifest_files=len(new_manifest['files']))
        (backup / 'update_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Default only reports the proposed code changes')
    args = parser.parse_args()
    print(json.dumps(update(args.destination, apply=args.apply), ensure_ascii=False, indent=2))
