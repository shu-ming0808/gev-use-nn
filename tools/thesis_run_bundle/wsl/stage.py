"""Verify/copy managed code and prepared inputs to Linux; never copy raw weather."""
import hashlib
import json
from pathlib import Path
import shutil
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def stage(source, target):
    source = source.resolve(); target = target.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Linux workspace must be separate from source bundle')
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8-sig'))
    for name, checksum in manifest['files'].items():
        p = (source/name).resolve()
        if source not in p.parents or not p.is_file() or digest(p) != checksum:
            raise ValueError(f'Source manifest failed: {name}')
    target.mkdir(parents=True, exist_ok=True)
    previous = target/'source.json'
    if previous.exists() and json.loads(previous.read_text())['source'] != str(source):
        raise ValueError('Workspace belongs to another mounted bundle; use a separate runtime directory')
    out = source/'outputs_wsl'; out.mkdir(exist_ok=True)
    # Do not update code while a Linux workflow is using it.
    import fcntl
    with (out/'run.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        link = target/'outputs'
        if link.is_symlink():
            if link.resolve() != out.resolve(): raise ValueError('Wrong output link')
        elif link.exists(): raise ValueError('Refuse to replace existing Linux outputs')
        else: link.symlink_to(out, target_is_directory=True)
        for name, checksum in manifest['files'].items():
            dest = target/name
            if target not in dest.resolve().parents: raise ValueError('Unsafe destination')
            if not dest.exists() or digest(dest) != checksum:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source/name, dest)
        shutil.copy2(source/'manifest.json', target/'manifest.json')
        previous.write_text(json.dumps({'source': str(source)}), encoding='utf-8')
        # Verified historical real EAS can calibrate simulations, not stand in
        # for a new Linux timing result. Keep its original report unchanged.
        if not (out/'eas').exists():
            for folder in sorted((source/'outputs/eas').glob('attempt_*'), reverse=True):
                path = folder/'run_report.json'
                if not path.exists(): continue
                report = json.loads(path.read_text(encoding='utf-8-sig'))
                if not report.get('workflow_completed'): continue
                for name, checksum in report['output_sha256'].items():
                    if Path(name).name != name or digest(folder/name) != checksum:
                        raise ValueError('Historical EAS output verification failed')
                dest = out/'eas'/folder.name
                dest.mkdir(parents=True)
                for name in [*report['output_sha256'], 'run_report.json']:
                    shutil.copy2(folder/name, dest/name)
                (dest/'import_note.json').write_text(json.dumps({
                    'source':str(folder),'historical_windows_calibration_only':True,
                    'not_a_linux_timing_run':True}), encoding='utf-8')
                break
    print('Linux workspace:', target)
    print('Separate outputs:', out)


if __name__ == '__main__':
    stage(Path(sys.argv[1]), Path(sys.argv[2]))
