"""Completed GP factorizations must not overlap the next independent fit."""
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'tools/thesis_run_bundle')]
import spatial_predictor_selection as selection
from taiwan_simulation import cheng_spatial_cv as cv
from portable import comparison, simulation
from portable.common import read, sha, write
from research_workflows import COEFFICIENTS


class AlternatingFolds:
    def __init__(self, n_clusters, **kwargs):
        self.n_clusters = n_clusters

    def fit_predict(self, values):
        return np.arange(len(values)) % self.n_clusters


def tracked_fit(refs, *, include_kernel=False):
    def fit(xy, covariates, response, *args, **kwargs):
        assert all(ref() is None for ref in refs), 'Previous GP factorization is still resident'
        chol = np.eye(len(xy))
        refs.append(weakref.ref(chol))
        model = dict(chol=(chol, True), optimizer_success=True,
                     optimizer_message='test fit', mean=float(np.mean(response)))
        if include_kernel:
            model['kernel'] = 'test kernel'
        return model
    return fit


def predict_mean(model, xy, covariates):
    # Prediction owns its memory and does not retain a view into the model.
    return np.full(len(xy), model['mean'])


def test_inner_fold_factorization_released_before_next_fit(monkeypatch):
    refs = []
    monkeypatch.setattr(selection, 'fit_gp_with_covariates', tracked_fit(refs))
    monkeypatch.setattr(selection, 'predict_gp_with_covariates', predict_mean)
    data = pd.DataFrame(dict(station=['G0', 'G1', 'G2', 'G3'],
        x_km=[0., 1., 2., 3.], y_km=[0., 1., 0., 1.], mu0=[10., 12., 20., 22.]))
    contexts = [dict(fold=0, train_indices=np.array([0, 1]), test_indices=np.array([2, 3]),
                     base_indices=np.array([0, 1])),
                dict(fold=1, train_indices=np.array([2, 3]), test_indices=np.array([0, 1]),
                     base_indices=np.array([2, 3]))]
    result = selection.evaluate_predictor_set(data, 'mu0', [], contexts, 'RBF', None,
                                              target_columns={'mu0': 'mu0'})
    assert len(refs) == 2 and all(ref() is None for ref in refs)
    np.testing.assert_array_equal(result['predictions'].y_pred, [11., 11., 21., 21.])
    np.testing.assert_array_equal(result['fold_metrics'].MSE, [101., 101.])
    assert result['RMSE'] == np.sqrt(101.) and result['optimizer_success']


def test_outer_refit_released_before_next_selection(monkeypatch):
    refs = []
    monkeypatch.setattr(cv, 'KMeans', AlternatingFolds)
    monkeypatch.setattr(cv, '_buffered_training_indices', lambda frame, train, test, buffer, **kw: train)
    fit = tracked_fit(refs)
    def fit_with_kernel(*args, **kwargs):
        model = fit(*args, **kwargs)
        model['kernel'] = SimpleNamespace(k1=SimpleNamespace(k2=SimpleNamespace(
            length_scale=3., length_scale_bounds=(1., 10.))))
        return model
    monkeypatch.setattr(cv, 'fit_gp_with_covariates', fit_with_kernel)
    monkeypatch.setattr(cv, 'predict_gp_with_covariates', predict_mean)
    selection_calls = []
    def select(data, target, **kwargs):
        assert all(ref() is None for ref in refs), 'Previous outer factorization entered the next FFS'
        selection_calls.append(target)
        return pd.DataFrame({'trial': [1]}), pd.DataFrame({'step': [0]}), dict(
            predictor_names=[], kernel='RBF', nu=None, RMSE=0.)
    monkeypatch.setattr(cv, 'spatial_forward_selection', select)
    data = pd.DataFrame(dict(grid_id=[f'G{i}' for i in range(8)],
                             x_km=np.arange(8, dtype=float), y_km=np.zeros(8)))
    for i, target in enumerate(COEFFICIENTS):
        data[target] = float(i+1)
    result = cv.nested_cv(data, candidate_groups={'dummy': ['dummy']},
        buffers_km={target: 0. for target in COEFFICIENTS}, outer_folds=2, inner_folds=2,
        min_train=2, max_train=8, n_jobs=1)
    assert len(refs) == len(selection_calls) == 2*len(COEFFICIENTS)
    assert all(ref() is None for ref in refs)
    np.testing.assert_array_equal(result['oof'].prediction, result['oof'].reference_nn)
    np.testing.assert_array_equal(result['metrics'].RMSE_vs_NN_reference, np.zeros(len(COEFFICIENTS)))
    assert result['selections'].optimizer_success.all()
    assert result['selections'].optimizer_message.eq('test fit').all()
    assert result['selections'].fitted_length_scale_km.eq(3.).all()
    assert not result['selections'].length_scale_near_bound.any()


def test_calibration_refit_released_before_next_coefficient_and_resume(tmp_path, monkeypatch):
    refs = []
    monkeypatch.setattr(comparison, 'validate_terrain_source', lambda *args: {})
    monkeypatch.setattr(simulation, 'memory_guard', lambda *args: None)
    monkeypatch.setattr(simulation, 'KMeans', AlternatingFolds)
    monkeypatch.setattr(simulation, 'fit_gp_with_covariates', tracked_fit(refs, include_kernel=True))
    monkeypatch.setattr(simulation, 'predict_gp_with_covariates', predict_mean)
    grids = pd.DataFrame({'grid_id': ['G0', 'G1', 'G2', 'G3']})
    frame = grids.assign(x_km=np.arange(4, dtype=float), y_km=0.)
    expected = np.array([20., .1, np.log(2.), .01, .1])
    for i, target in enumerate(COEFFICIENTS):
        frame[target] = expected[i]
    monkeypatch.setattr(simulation, 'join_predictors', lambda *args, **kwargs: frame.copy())
    def select(data, target, **kwargs):
        assert all(ref() is None for ref in refs), 'Previous full-data factorization entered the next FFS'
        return pd.DataFrame({'trial': [1]}), pd.DataFrame({'step': [0]}), dict(
            predictor_names=[], kernel='RBF', nu=None, RMSE=0.)
    monkeypatch.setattr(simulation, 'spatial_forward_selection', select)
    source = tmp_path/'real'; source.mkdir()
    frame.to_csv(source/'nn_coefficients.csv', index=False)
    write(source/'run_report.json', dict(workflow_completed=True, region='EAS', data_role='real_data',
        output_sha256={'nn_coefficients.csv': sha(source/'nn_coefficients.csv')}))
    config = dict(candidate_groups={'dummy': ['dummy']}, n_jobs=1, inner_folds=2,
        buffers_km={target: 0. for target in COEFFICIENTS}, min_train=2, n_restarts=0,
        min_relative_improvement=.01, maximum_allowed_vif=5., max_steps=None,
        kernel_candidates=[['RBF', None]], gp_options={}, buffer_distance_mode='projected')
    output = tmp_path/'calibration'
    truth, report = simulation.calibrate(source, grids, pd.DataFrame(), config, 'test', output, threads=1)
    assert len(refs) == len(COEFFICIENTS) and all(ref() is None for ref in refs)
    np.testing.assert_array_equal(truth, np.tile(expected, (4, 1)))
    assert report['status'] == 'completed' and not report['optimizer_warning_coefficients']
    for target in COEFFICIENTS:
        summary = read(output/target/'completed.json')['summary']
        assert summary['optimizer_success'] and summary['optimizer_message'] == 'test fit'
        assert summary['fitted_kernel'] == 'test kernel'
    # The completed calibration remains hash-validated and is reused unchanged.
    reused, reused_report = simulation.calibrate(source, grids, pd.DataFrame(), config, 'test', output, threads=1)
    np.testing.assert_array_equal(reused, truth)
    assert reused_report == report and len(refs) == len(COEFFICIENTS)
