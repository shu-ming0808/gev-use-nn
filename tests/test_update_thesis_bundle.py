import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'update_thesis_bundle', Path(__file__).resolve().parents[1] / 'tools/update_thesis_bundle.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    repo = tmp_path / 'repo'; dest = tmp_path / 'bundle'
    sources = {
        'tools/thesis_run_bundle/run.py': 'new runner',
        'src/taiwan_simulation/mle_estimator.py': 'new mle',
        'simulation/taiwan_validation/nn_mle_full_workflow_comparison.ipynb': '{}',
    }
    old = {'run.py': 'old runner', 'models/bound.pt': 'unchanged weights',
           'inputs/eas/data.npy': 'unchanged data', 'outputs/eas/result.csv': 'old result'}
    for root, files in ((repo, sources), (dest, old)):
        for name, contents in files.items():
            p = root / name; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(contents, encoding='utf-8')
    manifest = dict(version=1, files={n: module.sha(dest/n) for n in old if not n.startswith('outputs/')})
    (dest/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return repo, dest


def test_dry_run_does_not_replace_files(tmp_path):
    repo, dest = fixture(tmp_path)
    report = module.update(dest, repo=repo)
    assert report['apply'] is False
    assert (dest/'run.py').read_text() == 'old runner'
    assert not (dest/'archives').exists()


def test_update_preserves_data_results_and_archives_code(tmp_path):
    repo, dest = fixture(tmp_path)
    before = {n: module.sha(dest/n) for n in ('models/bound.pt','inputs/eas/data.npy','outputs/eas/result.csv')}
    report = module.update(dest, repo=repo, apply=True)
    assert (dest/'run.py').read_text() == 'new runner'
    assert (Path(report['backup'])/'run.py').read_text() == 'old runner'
    assert all(module.sha(dest/n) == h for n,h in before.items())
    manifest = json.loads((dest/'manifest.json').read_text())
    assert all(module.sha(dest/n) == h for n,h in manifest['files'].items())
    assert module.update(dest, repo=repo)['changed'] == []


def test_rejects_modified_bundle_before_writing(tmp_path):
    repo, dest = fixture(tmp_path)
    (dest/'run.py').write_text('user edit')
    with pytest.raises(ValueError, match='Existing bundle missing/changed'):
        module.update(dest, repo=repo, apply=True)
    assert (dest/'run.py').read_text() == 'user edit'
    assert not (dest/'archives').exists()


def test_rejects_outside_manifest_path(tmp_path):
    repo, dest = fixture(tmp_path)
    with pytest.raises(ValueError, match='Unsafe bundle path'):
        module.inside(dest, '../outside.txt')
