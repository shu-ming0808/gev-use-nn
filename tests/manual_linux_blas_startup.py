"""Manual Linux CPU compatibility probe; no study data, output files or long run.

Run with the portable Linux Python. This is NOT a benchmark for another CPU.
It initializes native libraries at 8, exercises 8/4/1/8 transitions, then runs
four actual small GP fits through the production worker wrapper.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def fit_one(_index):
    import numpy as np
    from threadpoolctl import threadpool_info

    counts = [p['num_threads'] for p in threadpool_info() if p['user_api'] == 'blas']
    assert counts and all(n == 4 for n in counts), counts
    rng = np.random.default_rng(20260929)
    xy = rng.uniform(-200, 200, (384, 2))
    covariates = rng.normal(size=(384, 2))
    response = 30 + .01*xy[:, 0] + .5*covariates[:, 0] + rng.normal(scale=.3, size=384)
    model = fit_gp_with_covariates(
        xy, covariates, response, ['a', 'b'], 'RBF', seed=20260929,
        length_scale_bounds=(10, 1000), initial_length_scale=100,
    )
    predicted = predict_gp_with_covariates(model, xy[:8]+1, covariates[:8])
    assert np.isfinite(predicted).all()
    assert model['optimizer_success'], model['optimizer_message']
    return dict(pid=os.getpid(), native_threads=counts, prediction=predicted.tolist())


def main():
    global fit_gp_with_covariates, predict_gp_with_covariates
    if sys.platform != 'linux':
        raise SystemExit('This isolated compatibility probe requires Linux.')
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '8'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

    import numpy as np
    from scipy.linalg import cho_factor, cho_solve
    from threadpoolctl import threadpool_info, threadpool_limits
    from joblib import Parallel, delayed
    from gp_execution import run_task
    # Capture these module functions before dispatch, matching production tasks:
    # their SciPy library must load before run_task enters threadpool_limits.
    from land_cover_gp_analysis import fit_gp_with_covariates, predict_gp_with_covariates

    initial = [p['num_threads'] for p in threadpool_info() if p['user_api'] == 'blas']
    assert initial and all(n == 8 for n in initial), initial
    n = 4096
    x = np.arange(n, dtype=float)
    matrix = np.exp(-np.abs(x[:, None]-x[None, :])/100.)
    matrix.flat[::n+1] += .01
    rhs = np.cos(x/77.)
    checks = []
    for threads in (8, 4, 1, 8):
        with threadpool_limits(limits=threads):
            result = cho_solve(cho_factor(matrix, lower=True, check_finite=True), rhs)
            residual = float(np.max(np.abs(matrix@result-rhs)))
            assert residual < 1e-8 and np.isfinite(result).all()
            checks.append(dict(threads=threads, residual=residual, sum=float(result.sum())))
    assert max(row['sum'] for row in checks)-min(row['sum'] for row in checks) < 1e-8
    del x, matrix, rhs, result

    # The real production worker wrapper sets a per-job limit of four.
    with Parallel(n_jobs=4, backend='loky', pre_dispatch=4, batch_size=1) as parallel:
        fitted = parallel(delayed(run_task)((fit_one, (i,), {}), 4) for i in range(4))
    assert all(np.allclose(fitted[0]['prediction'], row['prediction'], atol=1e-8, rtol=1e-8)
               for row in fitted[1:])
    print(json.dumps(dict(status='passed', initial_blas_threads=initial,
                          matrix_size=n, thread_transition_checks=checks,
                          gp_fits=fitted, full_workflow_started=False,
                          hardware_optimality_claimed=False), indent=2))


if __name__ == '__main__':
    main()
