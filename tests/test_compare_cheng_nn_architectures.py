"""Aggregation safety checks using tiny synthetic run records (no training)."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import compare_cheng_nn_architectures as comparison


@pytest.fixture
def experiment(tmp_path):
    manifest = {'seeds': list(comparison.SEEDS), 'runs': []}
    for model, hidden in comparison.ARCHITECTURES.items():
        for seed in comparison.SEEDS:
            name = f'{model}_{seed}'
            path = tmp_path / name
            path.mkdir()
            manifest['runs'].append({'model': model, 'seed': seed, 'directory': name})
            (path / 'best.pt').write_bytes(b'checkpoint')
            hashes = {'train': 'train-hash', 'validation': 'validation-hash'}
            metadata = {key: 'fixed' for key in comparison.COMPARABLE_FIELDS}
            metadata.update(status='completed', seed=seed, architecture=[17, *hidden, 5],
                            trainable_parameters=10, test_evaluation_requested=False,
                            test_used_for_training_or_selection=False,
                            dataset_files_sha256={s: {'inputs.npy': s} for s in hashes},
                            dataset_metadata_sha256=hashes, best_epoch=2,
                            best_validation_loss=1., training_seconds=5., total_seconds=6.,
                            epochs_completed=3, evaluation={'validation': {'valid_gev_fraction': 1.}})
            (path / 'run_metadata.json').write_text(json.dumps(metadata))
            for split in ('train', 'validation'):
                target = path / split
                target.mkdir()
                marker = {'checkpoint_sha256': comparison.file_sha256(path / 'best.pt'),
                          'best_epoch': 2, 'dataset_metadata_sha256': hashes[split]}
                (target / 'prediction_metadata.json').write_text(json.dumps(marker))
                rows = [{'scale': scale, 'coefficient': f'c{j}', 'split': split,
                         'n_total': 4, 'n_used': 4, 'RMSE': 1., 'MAE': .8, 'bias': .1}
                        for scale in ('original', 'train_target_z') for j in range(5)]
                pd.DataFrame(rows).to_csv(target / 'coefficient_metrics.csv', index=False)
                pd.DataFrame([{'subset': 'finite_RL', 'return_period': period,
                               'RMSE': 2. if model == 'small' else 3., 'n_used': 200, 'n_total': 200}
                              for period in (50, 100)]).to_csv(target / 'return_level_overall.csv', index=False)
    (tmp_path / 'experiment.json').write_text(json.dumps(manifest))
    return tmp_path


def test_paired_seed_summary_uses_each_selected_checkpoint(experiment):
    _, _, per_run, summary, coefficients = comparison.summarize(experiment)
    assert len(per_run) == 6 and len(coefficients) == 60
    assert (summary['n_seeds'] == 3).all()
    small = summary.set_index('model').loc['small']
    assert small['validation_RL100_RMSE_mean'] == 2.
    assert small['validation_RL100_RMSE_std'] == 0.
    differences = pd.read_csv(experiment / 'paired_seed_differences.csv')
    np.testing.assert_array_equal(differences.validation_RL100_RMSE_small_minus_large, [-1., -1., -1.])
    assert not list(experiment.glob('*/test'))


@pytest.mark.parametrize('change,error', [
    ({'status': 'running'}, 'Incomplete'),
    ({'test_evaluation_requested': True}, 'test split'),
    ({'weight_decay': .02}, 'conditions'),
    ({'architecture': [17, 32, 5]}, 'architecture/seed'),
    ({'seed': -1}, 'architecture/seed'),
])
def test_rejects_incomparable_or_incomplete_runs(experiment, change, error):
    path = experiment / f'small_{comparison.SEEDS[-1]}' / 'run_metadata.json'
    metadata = json.loads(path.read_text())
    metadata.update(change)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=error):
        comparison.validate_runs(experiment)


def test_rejects_missing_seed(experiment):
    path = experiment / 'experiment.json'
    manifest = json.loads(path.read_text())
    manifest['runs'].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='every planned seed'):
        comparison.validate_runs(experiment)


def test_rejects_changed_checkpoint(experiment):
    path = experiment / f'large_{comparison.SEEDS[0]}' / 'best.pt'
    path.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='provenance'):
        comparison.validate_runs(experiment)
