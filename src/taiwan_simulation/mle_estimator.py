"""Data-only five-coefficient MLE adapter for the paired spatial experiment."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits
from gp_execution import describe, adaptive_tasks, GIB

from cheng_nn_mle import fit_time_varying_gev, SUPPORT_EPS
from research_workflows import COEFFICIENTS, YEARS, REFERENCE_YEAR, TIME_SCALE_YEARS, validate_annual_sequences


class MLEFitFailure(RuntimeError):
    """Do not silently remove failed GRID before spatial CV."""


def fit_one(index, values, *, domain, maxiter, retry_maxiter, threads):
    t = (np.asarray(YEARS) - REFERENCE_YEAR) / TIME_SCALE_YEARS
    with threadpool_limits(limits=threads):
        first = fit_time_varying_gev(values, t, domain=domain, maxiter=maxiter)
        seconds = first['seconds']
        retry = not first['success'] or first['lower_nonconverged_nll_gap'] > 1e-4
        result = first
        if retry:
            second = fit_time_varying_gev(values, t, domain=domain, maxiter=retry_maxiter)
            seconds += second['seconds']
            if second['success'] and (not first['success'] or second['nll'] < first['nll']):
                result = second
    result = dict(result)
    coefficients = result.pop('coefficients')
    return dict(grid_index=index, **{**result, 'seconds': seconds}, retried=retry,
                **dict(zip(COEFFICIENTS, coefficients)))


def infer_mle(values, *, domain='wide_domain', jobs=1, threads=1,
              maxiter=600, retry_maxiter=1800):
    """No truth, NN predictions, predictors or scenario labels enter this API.

    Wide-domain bounded MLE is the default because calibrated GP surfaces need
    not lie inside the neural training box. It is NOT unconstrained/global MLE.
    All 50 years are used. MLE gets a serial estimator budget by default; GP
    candidate parallelism is a separate setting. No silent fallback or imputation.
    """
    if domain not in ('training_domain', 'wide_domain') or jobs == 0 or threads < 1:
        raise ValueError('Invalid MLE domain/jobs/threads')
    if maxiter < 1 or retry_maxiter < maxiter:
        raise ValueError('Require 1 <= maxiter <= retry_maxiter')
    start = time.perf_counter()
    values = validate_annual_sequences(np.asarray(YEARS), values)
    options = dict(domain=domain, threads=threads, maxiter=maxiter, retry_maxiter=retry_maxiter)
    adaptive = describe()['mode'] == 'adaptive'
    if adaptive:
        # Each 50-year MLE is small; keep numerical libraries single-threaded and
        # parallelize independent GRID. CPU and RAM caps are checked per wave.
        options['threads'] = 1
        rows = adaptive_tasks(jobs, (delayed(fit_one)(i, y, **options) for i,y in enumerate(values)),
                              len(values), per_worker_bytes=int(.5*GIB), stage='mle_estimation',
                              tasks_per_worker=4, native_limit=1)
    elif jobs == 1:
        rows = [fit_one(i, y, **options) for i, y in enumerate(values)]
    else:
        with parallel_config(backend='loky', inner_max_num_threads=threads):
            rows = Parallel(n_jobs=jobs)(delayed(fit_one)(i, y, **options) for i, y in enumerate(values))
    fits = pd.DataFrame(rows).sort_values('grid_index').reset_index(drop=True)
    prediction = fits[list(COEFFICIENTS)].to_numpy(float)
    return dict(prediction=prediction, checks=fits,
        timings=dict(fitting_wall_seconds=time.perf_counter()-start,
                     summed_grid_fit_seconds=float(fits.seconds.sum())),
        model=dict(kind='five_coefficient_bounded_mle', domain=domain,
            jobs=jobs, threads_per_worker=1 if adaptive else threads,
            execution_policy=describe(), maxiter=maxiter, retry_maxiter=retry_maxiter,
            starts=4, support_margin=SUPPORT_EPS, global_optimum_guaranteed=False,
            initialization='data_only_no_NN_no_truth_no_scenario',
            wide_bounds_standardized=dict(mu0=[-20,20], beta_mu_over_sigma0=[-3,3],
                eta0=[-7,4], beta_sigma=[-1,1], xi0=[-.49,.8]) if domain=='wide_domain' else None))
