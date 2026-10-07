from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import compare_cheng_nn_mixed_sampling as study
from cheng_nn_simulation import ParameterRanges, SimulationConfig, build_input_features
from train_cheng_nn import atomic_json, file_sha256


def test_exact_balance_and_zero_structure():
    rng = np.random.default_rng(19)
    labels = study.balanced_scenarios(400, rng)
    np.testing.assert_array_equal(np.bincount(labels), [100]*4)
    with pytest.raises(ValueError):
        study.balanced_scenarios(401, rng)
    x, targets, c, scale, y = study.mixed_chunk(rng, labels, SimulationConfig(), ParameterRanges())
    assert x.shape == (400, 17) and y.shape == (400, 50)
    assert np.all(c[np.isin(labels, [0, 2]), 1] == 0)
    assert np.all(c[np.isin(labels, [0, 1]), 3] == 0)
    assert np.all(c[np.isin(labels, [1, 3]), 1] != 0)
    assert np.all(c[np.isin(labels, [2, 3]), 3] != 0)
    np.testing.assert_allclose(targets[:, 1] * scale[:, 1], c[:, 1], rtol=1e-6)
    np.testing.assert_allclose(targets[:, 3], c[:, 3], rtol=1e-6)
    np.testing.assert_allclose(targets[:, 0]*scale[:,1]+scale[:,0], c[:,0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(targets[:, 2]+np.log(scale[:,1]), c[:,2], rtol=1e-5, atol=1e-6)
    expected, original_scale = build_input_features(y)
    np.testing.assert_allclose(x, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(scale, original_scale)
    t = SimulationConfig().centered_time[None, :]
    mu = c[:,0,None] + c[:,1,None]*t
    sigma = np.exp(c[:,2,None] + c[:,3,None]*t)
    assert (1+c[:,4,None]*(y-mu)/sigma > 0).all()


def test_generation_reproducible_and_completed_split_is_verified(tmp_path):
    args = ('train', 40, 42, SimulationConfig(), ParameterRanges())
    a = study.generate_split(tmp_path/'a', *args)
    b = study.generate_split(tmp_path/'b', *args)
    assert a['files_sha256'] == b['files_sha256']
    assert study.generate_split(tmp_path/'a', *args) == a
    with pytest.raises(ValueError):
        study.generate_split(tmp_path/'a', 'train', 40, 43, SimulationConfig(), ParameterRanges())
    damaged = tmp_path/'a/train/scenario_id.npy'
    with damaged.open('ab') as f:
        f.write(b'corrupted')
    with pytest.raises(ValueError):
        study.generate_split(tmp_path/'a', *args)


def test_partial_generation_is_preserved(tmp_path):
    partial = tmp_path/'.train.partial'
    partial.mkdir()
    (partial/'keep.txt').write_text('unfinished')
    study.generate_split(tmp_path, 'train', 8, 42, SimulationConfig(), ParameterRanges())
    assert (partial/'keep.txt').read_text() == 'unfinished'
    assert (tmp_path/'train/metadata.json').is_file()


def test_grouped_metrics_preserve_all_cases():
    truth = np.zeros((8,5))
    prediction = np.tile(np.arange(1,6), (8,1))
    labels = np.repeat(np.arange(4), 2)
    rows = study.scenario_metrics(truth, prediction, labels, 17, 1)
    assert len(rows) == 20
    for row in rows:
        j = study.COEFFICIENT_NAMES.index(row['coefficient'])
        assert row['n_cases'] == 2 and row['RMSE'] == j+1
    prediction = prediction.astype(float)
    prediction[0,0] = np.nan
    with pytest.raises(ValueError):
        study.scenario_metrics(truth, prediction, labels, 17, 1)


def test_paired_inputs_training_and_test_is_never_loaded(tmp_path, monkeypatch):
    output, data = tmp_path/'results', tmp_path/'data'
    manifest = study.prepare(output, data, sizes=(40, 8, 8))
    assert len(manifest['runs']) == 6
    assert set(manifest['features']) == {'17','21'}
    study.validate_manifest(manifest)
    x17 = np.load(manifest['features']['17']['train']['path'])
    x21 = np.load(manifest['features']['21']['train']['path'])
    np.testing.assert_array_equal(x17[:,:11], x21[:,:11])
    assert x17.shape == (40,17) and x21.shape == (40,21)
    assert not (output/'features_17_test.npy').exists()
    original_load = np.load
    def guarded_load(path, *args, **kwargs):
        if 'test' in Path(path).relative_to(tmp_path).parts:
            raise AssertionError('Test opened during training/reporting')
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded_load)
    # Two tiny CPU epochs exercise the actual shared notebook train loop,
    # checkpoint, feature-width handling and export path, never formal results.
    manifest['fixed'] = {**manifest['fixed'], 'max_epochs':2, 'patience':2}
    atomic_json(output/'experiment.json', manifest)
    entry = next(r for r in manifest['runs'] if r['dimension']==21)
    path = Path(entry['path'])
    study.train_one(output, 21, entry['seed'], path, device='cpu')
    study.validate_run(entry, manifest)
    entry.update(status='completed', checkpoint_sha256=file_sha256(path/'best.pt'))
    atomic_json(output/'experiment.json', manifest)
    study.summarize(output)
    assert (output/'scenario_summary.csv').is_file()
    assert not (path/'test').exists()
    assert not (path/'validation/return_level_overall.csv').exists()
