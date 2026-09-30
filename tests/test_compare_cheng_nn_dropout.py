"""Comparison provenance and aggregation tests; no large training is run."""
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import compare_cheng_nn_dropout as comparison


@pytest.fixture
def experiment(tmp_path):
    reference = {field: 'fixed' for field in comparison.FIXED_FIELDS}
    reference.update(seed=20260929, split_sizes={'train': 8, 'validation': 4},
                     architecture=[17,512,512,512,128,128,5], trainable_parameters=617349,
                     dataset_files_sha256={s: {'inputs.npy': s} for s in ('train', 'validation')},
                     dataset_metadata_sha256={s: s for s in ('train', 'validation')})
    manifest = {'baseline_conditions': reference, 'runs': []}
    for p in comparison.PROBABILITIES:
        directory = tmp_path / f'p{p}'
        directory.mkdir()
        (directory / 'best.pt').write_bytes(f'model-{p}'.encode())
        metadata = dict(reference, status='completed', test_evaluation_requested=False,
                        test_used_for_training_or_selection=False, best_epoch=1,
                        best_validation_loss=1., training_seconds=3., total_seconds=4.,
                        epochs_completed=2, evaluation={'validation': {'valid_gev_fraction': 1.}})
        if p:  # Legacy p=0 checkpoints do not have this new field.
            metadata['dropout_p'] = p
        (directory / 'run_metadata.json').write_text(json.dumps(metadata))
        entry = {'dropout_p': p, 'directory': str(directory), 'reused': p == 0,
                 'checkpoint_sha256': comparison.file_sha256(directory / 'best.pt')}
        manifest['runs'].append(entry)
        for split, n in reference['split_sizes'].items():
            target = directory / split
            target.mkdir()
            marker = {'checkpoint_sha256': entry['checkpoint_sha256'], 'best_epoch': 1,
                      'dataset_metadata_sha256': split, 'n_samples': n}
            (target / 'prediction_metadata.json').write_text(json.dumps(marker))
            pd.DataFrame([{'scale': scale, 'split': split, 'coefficient': name,
                           'n_total': n, 'n_used': n, 'RMSE': 1.}
                          for scale in ('train_target_z', 'original')
                          for name in ('mu0','beta_mu','eta0','beta_sigma','xi0')]).to_csv(target / 'coefficient_metrics.csv', index=False)
            pd.DataFrame([{'subset': 'finite_RL', 'return_period': period, 'RMSE': 2.+p,
                           'n_total': n*50, 'n_used': n*50}
                          for period in (50, 100)]).to_csv(target / 'return_level_overall.csv', index=False)
            pd.DataFrame([{'subset':'finite_RL', 'return_period':50, 'year':2000, 'RMSE':2.+p}]).to_csv(target / 'return_level_by_year.csv', index=False)
        pd.DataFrame([{'epoch':1, 'train_loss':1., 'validation_loss':1.}]).to_csv(directory / 'training_history.csv', index=False)
    (tmp_path / 'experiment.json').write_text(json.dumps(manifest))
    return tmp_path


def test_summary_reuses_legacy_zero_and_compares_same_denominators(experiment):
    summary = comparison.summarize(experiment)
    assert summary.dropout_p.tolist() == list(comparison.PROBABILITIES)
    assert summary.baseline_reused.tolist() == [True,False,False,False]
    assert summary.RL100_n_used.tolist() == [200]*4
    assert summary.iloc[-1].RL50_RMSE_change_percent == pytest.approx(25.)


@pytest.mark.parametrize('change', [
    {'seed': 123}, {'batch_size': 256}, {'dropout_p': .3},
    {'status': 'running'}, {'test_evaluation_requested': True},
])
def test_rejects_mixed_conditions_partial_runs_and_test_use(experiment, change):
    path = experiment / 'p0.2/run_metadata.json'
    meta = json.loads(path.read_text())
    meta.update(change)
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        comparison.summarize(experiment)


def test_rejects_changed_baseline_checkpoint(experiment):
    (experiment / 'p0.0/best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint changed'):
        comparison.summarize(experiment)


def test_rejects_missing_probability(experiment):
    path = experiment / 'experiment.json'
    manifest = json.loads(path.read_text())
    manifest['runs'].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='four probabilities'):
        comparison.summarize(experiment)
