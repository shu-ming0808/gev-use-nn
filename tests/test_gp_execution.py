import sys
from pathlib import Path

import numpy as np
import pytest
from joblib import delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from gp_execution import configure, describe, candidate_plan, run_task, refit_context
import spatial_predictor_selection as selection


@pytest.fixture(autouse=True)
def reset_policy():
    configure('fixed')
    yield
    configure('fixed')


def thread_counts():
    from threadpoolctl import threadpool_info
    np.ones((16, 16)) @ np.ones((16, 16))
    return [x['num_threads'] for x in threadpool_info() if x['user_api'] == 'blas']


def test_budget_adapts_to_task_count_without_oversubscription(monkeypatch):
    monkeypatch.setattr('gp_execution.cpu_count', lambda: 20)
    configure('staged', cpu_budget=19, max_worker_threads=4, refit_threads=4)
    for tasks, workers, threads in [(44, 19, 1), (4, 4, 4), (1, 1, 4)]:
        plan = candidate_plan(19, tasks)
        assert (plan['workers'], plan['native_threads']) == (workers, threads)
        assert workers * threads <= 19
    assert candidate_plan(2, 44)['workers'] == 2
    assert describe()['mode'] == 'staged'


def test_fixed_mode_and_bad_options():
    assert candidate_plan(-2, 4) is None
    for kwargs in [dict(cpu_budget=0), dict(max_worker_threads=0), dict(refit_threads=0)]:
        with pytest.raises(ValueError): configure('staged', **kwargs)
    with pytest.raises(ValueError): configure('bad')


def test_worker_limit_restored():
    before = thread_counts()
    assert all(n == 2 for n in run_task((thread_counts, (), {}), 2))
    assert thread_counts() == before


def test_real_dispatch_uses_planned_threads_and_preserves_order():
    configure('staged', cpu_budget=4, max_worker_threads=2, refit_threads=3)
    results = selection._run_candidate_jobs(2, [delayed(thread_counts)() for _ in range(2)], 2)
    assert len(results) == 2 and all(r and all(n == 2 for n in r) for r in results)
    with refit_context('test'):
        assert all(n == 3 for n in thread_counts())


def test_single_worker_preserves_real_gp_predictions():
    from land_cover_gp_analysis import fit_gp_with_covariates, predict_gp_with_covariates
    rng = np.random.default_rng(20260929)
    xy = rng.uniform(0, 50, (45, 2)); x = np.empty((45, 0)); y = np.sin(xy[:, 0]/20)
    def fit():
        model = fit_gp_with_covariates(xy, x, y, [], 'RBF', seed=42)
        return predict_gp_with_covariates(model, xy, x)
    baseline = selection._run_candidate_jobs(1, [delayed(fit)()], 1)[0]
    configure('staged', cpu_budget=2, max_worker_threads=1, refit_threads=2)
    staged = selection._run_candidate_jobs(1, [delayed(fit)()], 1)[0]
    np.testing.assert_allclose(staged, baseline, atol=1e-5, rtol=1e-5)
