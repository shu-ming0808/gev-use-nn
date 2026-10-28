"""Compare CPU worker/thread layouts on synthetic exact-GP fits, not formal data.

This diagnostic never modifies NN weights, research settings or existing results.
Run it on the target machine: a best layout on one CPU is not transferable proof.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import threading
import time


def parse_profiles(value):
    profiles = []
    for raw in value.split(','):
        match = re.fullmatch(r'\s*([1-9][0-9]*)[xX]([1-9][0-9]*)\s*', raw)
        if not match:
            raise argparse.ArgumentTypeError('Use worker x thread pairs, e.g. 4x4,6x2,8x2')
        profile = tuple(map(int, match.groups()))
        if profile in profiles:
            raise argparse.ArgumentTypeError('Duplicate CPU profile')
        profiles.append(profile)
    return profiles


def source_directory():
    here = Path(__file__).resolve().parent
    for root in (here / 'src', here.parent / 'src', here.parent.parent / 'src'):
        if (root / 'land_cover_gp_analysis.py').is_file():
            return root
    raise FileNotFoundError('Cannot locate the bundled/project src directory')


def validate_args(args):
    if args.train_grids < 24 or args.tasks < 1 or args.repeats < 1:
        raise ValueError('Use at least 24 grids and positive task/repeat counts')
    if args.cpu_budget is not None and args.cpu_budget < 1:
        raise ValueError('CPU budget must be positive')
    if not math.isfinite(args.memory_budget_gib) or args.memory_budget_gib <= 0:
        raise ValueError('Memory budget must be finite and positive')
    if not math.isfinite(args.prediction_atol) or args.prediction_atol <= 0:
        raise ValueError('Prediction tolerance must be finite and positive')
    if args.output is not None and args.output.suffix.lower() != '.json':
        raise ValueError('Output path must end in .json; a sibling .csv is also generated')
    if args.output is not None and (args.output.exists() or args.output.with_suffix('.csv').exists()):
        raise FileExistsError('Choose a new report path; existing benchmark results are preserved')


def available_budget(memory, memory_budget_gib):
    """Use actual free RAM and an application-RSS cap, with a 2 GiB reserve."""
    return max(0, min(memory_budget_gib*1024**3-memory['process_tree_rss_bytes'],
                      memory['available_bytes']-2*1024**3))


def profile_skip_reason(workers, threads, *, cpu_budget, tasks, worker_bytes, available_bytes):
    return ('CPU budget exceeded' if workers*threads > cpu_budget else
        'Fewer tasks than requested workers' if workers > tasks else
        'Conservative RAM estimate exceeds available budget' if workers*worker_bytes > available_bytes else None)


def completed_status(runs, skipped):
    if not runs:
        return 'no_trials_run'
    if any(not row['predictions_agree'] or row['optimizer_successes'] != row['task_count']
           for row in runs):
        return 'completed_with_warnings'
    return 'completed_with_skips' if skipped else 'completed'


def write_report(output, report):
    """Persist even a skipped benchmark so users can return its resource diagnosis."""
    if output is None:
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix('.csv').exists():
        raise FileExistsError('Existing benchmark reports are preserved')
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with output.with_suffix('.csv').open('x', encoding='utf-8', newline='') as handle:
        fields = ['profile', 'completed_repeats', 'median_seconds', 'max_sampled_rss_gib',
                  'all_predictions_agree', 'all_optimizers_succeeded']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report.get('summary', []))


def _fit_task(index, n_train, native_threads, source):
    # Import the numerical libraries BEFORE reducing the thread limit.
    # The parent initializes native runtimes to >= every requested thread count.
    sys.path.insert(0, source)
    import numpy as np
    from threadpoolctl import threadpool_limits, threadpool_info
    from land_cover_gp_analysis import fit_gp_with_covariates, predict_gp_with_covariates

    rng = np.random.default_rng(20260929 + index)
    coordinates = rng.uniform([0., 0.], [800., 600.], size=(n_train + 32, 2))
    covariates = np.column_stack((coordinates[:, 1] / 600.,
        np.sin(coordinates[:, 0] / 200.), rng.normal(size=n_train + 32)))
    response = (25. + covariates[:, 0] * 3. - covariates[:, 1] * 1.2
        + .5 * np.sin(coordinates[:, 0] / 85.)
        + .4 * np.cos(coordinates[:, 1] / 70.)
        + rng.normal(scale=.15, size=n_train + 32))
    kernel, nu = [('RBF', None), ('Matern', .5), ('Matern', 1.5), ('Matern', 2.5)][index % 4]
    started = time.perf_counter()
    cpu_started = time.process_time()
    with threadpool_limits(limits=native_threads):
        model = fit_gp_with_covariates(coordinates[:n_train], covariates[:n_train],
            response[:n_train], predictor_names=['north_gradient', 'wave', 'noise'],
            kernel_name=kernel, nu=nu, n_restarts=0, seed=20260929 + index,
            length_scale_bounds=(1., 6000.), initial_length_scale=300.)
        prediction = predict_gp_with_covariates(model, coordinates[n_train:], covariates[n_train:])
        pools = [{key: item.get(key) for key in ('internal_api', 'num_threads', 'version')}
                 for item in threadpool_info()]
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite benchmark prediction')
    return dict(task=index, seconds=time.perf_counter()-started,
        worker_cpu_seconds=time.process_time()-cpu_started,
        optimizer_success=bool(model['optimizer_success']), prediction=prediction.tolist(),
        log_likelihood=float(model['log_likelihood']), native_pools=pools)


class PeakRSS:
    """Sample parent and currently live descendants; shared pages may be counted twice."""
    def __init__(self):
        self.peak = 0
        self.stop = threading.Event()

    def sample(self):
        import psutil
        root = psutil.Process()
        while not self.stop.is_set():
            total = 0
            for process in [root, *root.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.peak = max(self.peak, total)
            self.stop.wait(.05)

    def __enter__(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', type=parse_profiles, default=parse_profiles('4x4,6x2,8x2'))
    parser.add_argument('--train-grids', type=int, default=512)
    parser.add_argument('--tasks', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--cpu-budget', type=int, default=18)
    parser.add_argument('--memory-budget-gib', type=float, default=20.)
    parser.add_argument('--prediction-atol', type=float, default=1e-3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    validate_args(args)
    source = source_directory()
    init_threads = max(8, *(threads for _, threads in args.profiles))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(init_threads)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path.insert(0, str(source))
    import numpy as np
    from joblib import cpu_count
    from joblib.externals.loky import ProcessPoolExecutor
    from gp_execution import gp_worker_bytes, memory_snapshot

    budget = min(args.cpu_budget or max(1, cpu_count()-1), cpu_count())
    memory = memory_snapshot()
    available = available_budget(memory, args.memory_budget_gib)
    estimate = gp_worker_bytes(args.train_grids, 32)
    approved, skipped = [], []
    for workers, threads in args.profiles:
        reason = profile_skip_reason(workers, threads, cpu_budget=budget, tasks=args.tasks,
                                     worker_bytes=estimate, available_bytes=available)
        if reason:
            skipped.append(dict(profile=f'{workers}x{threads}', reason=reason))
        else:
            approved.append((workers, threads))
    if not approved:
        report = dict(status='no_safe_profiles', benchmark='synthetic_exact_gp_cpu_layout_v1',
            cpu=platform.processor(), platform=platform.platform(), logical_cpus=cpu_count(),
            cpu_budget=budget, initialization_threads=init_threads, initial_memory=memory,
            n_train=args.train_grids, n_test=32, tasks=args.tasks, repeats=args.repeats,
            memory_worker_estimate_gib=estimate/1024**3, runs=[], summary=[], skipped=skipped,
            configuration_changed=False,
            message='No benchmark started. Close unrelated applications or use fewer workers; do not disable the RAM guard.')
        write_report(args.output, report)
        print(json.dumps(report, indent=2))
        return 2
    baseline = None
    runs = []
    # Rotate profile order across repeats to reduce a fixed warm-up/order advantage.
    for repeat in range(args.repeats):
        order = approved[repeat % len(approved):] + approved[:repeat % len(approved)]
        for workers, threads in order:
            snapshot = memory_snapshot()
            if workers*estimate > available_budget(snapshot, args.memory_budget_gib):
                skipped.append(dict(profile=f'{workers}x{threads}', repeat=repeat,
                                    reason='RAM decreased before the trial'))
                continue
            print(f'Benchmark {repeat+1}/{args.repeats}: {workers} workers x {threads} threads', flush=True)
            started = time.perf_counter()
            with PeakRSS() as rss:
                pool = ProcessPoolExecutor(max_workers=workers)
                try:
                    futures = [pool.submit(_fit_task, i, args.train_grids, threads, str(source))
                               for i in range(args.tasks)]
                    result = [future.result() for future in futures]
                except BaseException:
                    pool.shutdown(wait=True, kill_workers=True)
                    raise
                else:
                    pool.shutdown(wait=True)
            elapsed = time.perf_counter()-started
            predictions = np.array([item['prediction'] for item in result])
            if baseline is None:
                baseline = predictions
            delta = float(np.max(np.abs(predictions-baseline)))
            cpu_seconds = sum(item['worker_cpu_seconds'] for item in result)
            runs.append(dict(profile=f'{workers}x{threads}', repeat=repeat+1,
                elapsed_seconds=elapsed, worker_fit_cpu_seconds=cpu_seconds,
                sum_worker_fit_seconds=sum(item['seconds'] for item in result),
                median_worker_fit_seconds=statistics.median(item['seconds'] for item in result),
                average_worker_fit_core_equivalents=cpu_seconds/elapsed,
                sampled_peak_tree_rss_gib=rss.peak/1024**3,
                max_abs_prediction_difference_vs_first_run=delta,
                predictions_agree=bool(delta <= args.prediction_atol),
                optimizer_successes=sum(item['optimizer_success'] for item in result),
                task_count=args.tasks, native_pools=result[0]['native_pools']))
            print(json.dumps(runs[-1], ensure_ascii=False), flush=True)
    summary = []
    for workers, threads in approved:
        name = f'{workers}x{threads}'
        part = [row for row in runs if row['profile']==name]
        if part:
            summary.append(dict(profile=name, completed_repeats=len(part),
                median_seconds=statistics.median(row['elapsed_seconds'] for row in part),
                max_sampled_rss_gib=max(row['sampled_peak_tree_rss_gib'] for row in part),
                all_predictions_agree=all(row['predictions_agree'] for row in part),
                all_optimizers_succeeded=all(row['optimizer_successes']==args.tasks for row in part)))
    status = completed_status(runs, skipped)
    report = dict(status=status, benchmark='synthetic_exact_gp_cpu_layout_v1',
        cpu=platform.processor(), platform=platform.platform(), logical_cpus=cpu_count(),
        cpu_budget=budget, initialization_threads=init_threads, initial_memory=memory,
        n_train=args.train_grids, n_test=32, tasks=args.tasks, repeats=args.repeats,
        prediction_atol=args.prediction_atol, seed=20260929,
        memory_worker_estimate_gib=estimate/1024**3, summary=summary, runs=runs, skipped=skipped,
        configuration_changed=False,
        gp_source_sha256=hashlib.sha256((source/'land_cover_gp_analysis.py').read_bytes()).hexdigest(),
        notes=['Synthetic exact-GP fits only. No NN, CV, formal data or scientific settings modified.',
               'Wall time includes worker startup, imports and shutdown. Not full EAS runtime.',
               'CPU seconds include fit work only, not all startup/parent work.',
               'RSS is sampled parent-plus-child RSS; shared pages can be double counted.',
               'No automatic configuration changes. Validate selected layout on a representative full-size workload.'])
    write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if runs else 2


if __name__ == '__main__':
    raise SystemExit(main())
