"""Execution resources only: never changes GRID, folds, seeds or GP objective."""
from contextlib import contextmanager
import threading
import time

from joblib import cpu_count, effective_n_jobs
import psutil
from threadpoolctl import threadpool_limits

_policy = {'mode': 'fixed'}
_events = []


def configure(mode='fixed', *, cpu_budget=None, max_worker_threads=4, refit_threads=4):
    global _policy
    if mode not in ('fixed', 'staged'):
        raise ValueError('Execution mode must be fixed or staged')
    available = max(1, cpu_count())
    budget = max(1, available-1) if cpu_budget is None else int(cpu_budget)
    if budget < 1 or max_worker_threads < 1 or refit_threads < 1:
        raise ValueError('CPU budget and thread limits must be positive')
    _policy = dict(mode=mode, cpu_budget=min(budget, available),
                   max_worker_threads=int(max_worker_threads), refit_threads=int(refit_threads))
    _events.clear()
    return describe()


def describe():
    return dict(_policy)


def events(start=0):
    return [dict(row) for row in _events[start:]]


def candidate_plan(n_jobs, n_tasks):
    if _policy['mode'] == 'fixed':
        return None
    if n_tasks < 1 or n_jobs == 0:
        raise ValueError('Nonzero jobs and positive task count required')
    budget = _policy['cpu_budget']
    workers = min(effective_n_jobs(n_jobs), n_tasks, budget)
    native = min(_policy['max_worker_threads'], max(1, budget//workers))
    return dict(requested_jobs=n_jobs, tasks=n_tasks, workers=workers,
                native_threads=native, cpu_budget=budget)


def run_task(task, native_threads):
    function, args, kwargs = task
    # Explicit in EACH loky process: the parent's threadpoolctl is not inherited.
    with threadpool_limits(limits=native_threads):
        return function(*args, **kwargs)


@contextmanager
def measured_phase(stage, **details):
    start = time.perf_counter(); stopped = threading.Event()
    peak = {'sampled_process_tree_rss_bytes': 0, 'sampled_system_used_bytes': 0}
    process = psutil.Process()
    def sample():
        while True:
            rss = 0
            try:
                for child in [process, *process.children(recursive=True)]:
                    try: rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied): pass
                peak['sampled_process_tree_rss_bytes'] = max(peak['sampled_process_tree_rss_bytes'], rss)
                memory = psutil.virtual_memory()
                peak['sampled_system_used_bytes'] = max(peak['sampled_system_used_bytes'], memory.total-memory.available)
            except (psutil.NoSuchProcess, psutil.AccessDenied): pass
            if stopped.wait(.5): break
    monitor = threading.Thread(target=sample, daemon=True); monitor.start()
    success = False
    try:
        yield
        success = True
    finally:
        stopped.set(); monitor.join(timeout=2)
        seconds = time.perf_counter()-start
        _events.append(dict(stage=stage, seconds=seconds, success=success, **details, **peak))
        print(f'[GP resources] {stage} finished: {seconds:.2f}s, success={success}', flush=True)


@contextmanager
def refit_context(stage='final_gp_refit'):
    if _policy['mode'] == 'fixed':
        yield
        return
    count = min(_policy['refit_threads'], _policy['cpu_budget'])
    print(f'[GP resources] {stage}: 1 model x {count} matrix threads', flush=True)
    with measured_phase(stage, workers=1, native_threads=count), threadpool_limits(limits=count):
        yield
