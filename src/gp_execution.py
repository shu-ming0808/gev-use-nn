"""Execution resources only: never changes GRID, folds, seeds or GP objective."""
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, wait
from itertools import islice
from pathlib import Path
import math
import os
import threading
import time

from joblib import cpu_count, effective_n_jobs
from joblib.externals.loky import ProcessPoolExecutor
import psutil
from threadpoolctl import threadpool_limits

_policy = {'mode': 'fixed'}
_events = []


def configure(mode='fixed', *, cpu_budget=None, max_worker_threads=4, refit_threads=4,
              memory_budget_gib=26., memory_reserve_gib=2.):
    global _policy
    if mode not in ('fixed', 'staged', 'adaptive'):
        raise ValueError('Execution mode must be fixed, staged or adaptive')
    available = max(1, cpu_count())
    budget = max(1, available-1) if cpu_budget is None else int(cpu_budget)
    if budget < 1 or max_worker_threads < 1 or refit_threads < 1:
        raise ValueError('CPU budget and thread limits must be positive')
    if (not math.isfinite(memory_budget_gib) or not math.isfinite(memory_reserve_gib)
            or memory_budget_gib <= 0 or memory_reserve_gib < 0):
        raise ValueError('Memory budget must be positive and reserve nonnegative')
    _policy = dict(mode=mode, cpu_budget=min(budget, available),
                   max_worker_threads=int(max_worker_threads), refit_threads=int(refit_threads))
    if mode == 'adaptive':
        _policy.update(memory_budget_gib=float(memory_budget_gib), memory_reserve_gib=float(memory_reserve_gib),
                       memory_estimator_version=1, minimum_workers=1,
                       scheduler_version='rolling_loky_v1')
    _events.clear()
    return describe()


def describe():
    return dict(_policy)


def events(start=0):
    return [dict(row) for row in _events[start:]]


GIB = 1024**3


def _hierarchy_headrooms(root, relative, limit_name, current_name):
    """Respect finite ancestor cgroup caps; never treat swap as physical RAM."""
    root = Path(root).resolve()
    leaf = (root / relative.lstrip('/')).resolve()
    if leaf != root and root not in leaf.parents:
        leaf = root  # A cgroup namespace can expose an outside-relative membership.
    result = []
    for folder in [leaf, *leaf.parents]:
        if folder != root and root not in folder.parents:
            break
        try:
            raw = (folder / limit_name).read_text().strip()
            if raw != 'max':
                limit = int(raw); current = int((folder / current_name).read_text().strip())
                if 0 < limit < 2**60:
                    result.append((limit, max(0, limit-current)))
        except (OSError, ValueError):
            pass
        if folder == root:
            break
    return result


def cgroup_memory_limits():
    if os.name != 'posix':
        return []
    try:
        memberships = Path('/proc/self/cgroup').read_text().splitlines()
    except OSError:
        return []
    limits = []
    for row in memberships:
        _, controllers, relative = row.split(':', 2)
        if not controllers:
            limits.extend(_hierarchy_headrooms('/sys/fs/cgroup', relative, 'memory.max', 'memory.current'))
        elif 'memory' in controllers.split(','):
            limits.extend(_hierarchy_headrooms('/sys/fs/cgroup/memory', relative,
                                               'memory.limit_in_bytes', 'memory.usage_in_bytes'))
    return limits


def memory_snapshot():
    memory = psutil.virtual_memory()
    total, available = int(memory.total), int(memory.available)
    limits = cgroup_memory_limits()
    if limits:
        total = min(total, *(v[0] for v in limits))
        available = min(available, *(v[1] for v in limits))
    rss = 0
    process = psutil.Process()
    for p in [process, *process.children(recursive=True)]:
        try:
            rss += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return dict(visible_total_bytes=total, available_bytes=available,
                process_tree_rss_bytes=int(rss), cgroup_limited=bool(limits))


def memory_headroom():
    snap = memory_snapshot()
    reserve = int(_policy['memory_reserve_gib'] * GIB)
    cap = min(int(_policy['memory_budget_gib'] * GIB), snap['visible_total_bytes'])
    headroom = max(0, min(snap['available_bytes']-reserve, cap-snap['process_tree_rss_bytes']))
    return dict(**snap, budget_bytes=cap, reserve_bytes=reserve, schedulable_bytes=headroom)


def gp_worker_bytes(n_train, n_test=0, data_bytes=0):
    """Conservative working-set estimate, NOT a measured peak or allocation limit."""
    if n_train < 1 or n_test < 0 or data_bytes < 0:
        raise ValueError('Invalid GP memory dimensions')
    return int(.5*GIB + 8*(10*n_train*n_train + 4*n_train*n_test) + 3*data_bytes)


def candidate_plan(n_jobs, n_tasks, *, per_worker_bytes=None, native_limit=None):
    if _policy['mode'] == 'fixed':
        return None
    if n_tasks < 1 or n_jobs == 0:
        raise ValueError('Nonzero jobs and positive task count required')
    budget = _policy['cpu_budget']
    workers = min(effective_n_jobs(n_jobs), n_tasks, budget)
    info = {}
    if _policy['mode'] == 'adaptive':
        estimate = int(per_worker_bytes or .5*GIB)
        if estimate <= 0:
            raise ValueError('Worker memory estimate must be positive')
        snap = memory_headroom()
        fits = snap['schedulable_bytes'] // estimate
        workers = min(workers, max(1, fits))
        info = dict(**snap, per_worker_estimate_bytes=estimate, memory_worker_limit=int(fits),
                    estimated_single_worker_fits=bool(fits >= 1))
    native = min(native_limit or _policy['max_worker_threads'], max(1, budget//workers))
    return dict(requested_jobs=n_jobs, tasks=n_tasks, workers=int(workers),
                native_threads=native, cpu_budget=budget, **info)


def adaptive_tasks(n_jobs, tasks, n_tasks, *, per_worker_bytes, stage='candidate_models',
                   tasks_per_worker=1, native_limit=None):
    """Refill finished slots, preserve input order, and recheck RAM per submission.

    One future holds at most ``tasks_per_worker`` sequential small tasks (MLE
    uses four); GP uses one. There are never more outstanding futures than the
    CPU/RAM-approved initial pool size. Native threads stay fixed for that pool,
    so completing one task cannot overbook cores used by another running task.
    """
    if _policy['mode'] != 'adaptive':
        raise ValueError('adaptive_tasks requires adaptive policy')
    if not isinstance(tasks_per_worker, int) or isinstance(tasks_per_worker, bool) or tasks_per_worker < 1:
        raise ValueError('tasks_per_worker must be a positive integer')
    if not isinstance(n_tasks, int) or isinstance(n_tasks, bool) or n_tasks < 0:
        raise ValueError('n_tasks must be a nonnegative integer')
    if n_jobs == 0 or not math.isfinite(per_worker_bytes) or per_worker_bytes <= 0:
        raise ValueError('Nonzero jobs and positive worker estimate required')
    if native_limit is not None and (not isinstance(native_limit, int) or native_limit < 1):
        raise ValueError('native_limit must be a positive integer')
    estimate = int(per_worker_bytes)
    if estimate < 1:
        raise ValueError('Worker estimate must be at least one byte')
    iterator = iter(tasks); results = [None] * n_tasks
    submitted = completed = 0

    def take_batch():
        count = min(tasks_per_worker, n_tasks-submitted)
        batch = list(islice(iterator, count))
        if len(batch) != count:
            raise ValueError('Fewer tasks than declared')
        return batch

    while submitted < n_tasks:
        chunks_remaining = math.ceil((n_tasks-submitted)/tasks_per_worker)
        plan = candidate_plan(n_jobs, chunks_remaining, per_worker_bytes=estimate,
                              native_limit=native_limit)
        print(f"[resources] {stage} {completed}/{n_tasks}: {plan['workers']} workers x "
              f"{plan['native_threads']} threads; RAM headroom {plan['schedulable_bytes']/GIB:.2f} GiB; "
              f"estimate/worker {estimate/GIB:.2f} GiB; rolling refill", flush=True)
        if plan['workers'] == 1:
            if not plan['estimated_single_worker_fits']:
                _ram_advisory()
            batch = take_batch()
            with measured_phase(stage, completed_before=completed, wave_tasks=len(batch),
                                scheduler='rolling_loky_v1', **plan):
                part = _run_batch(batch, plan['native_threads'])
            results[submitted:submitted+len(batch)] = part
            submitted += len(batch); completed += len(batch)
            continue

        # Dedicated loky processes use spawn/cloudpickle semantics, not a fork
        # of a live BLAS runtime. Close the pool before returning to final refit.
        pool = ProcessPoolExecutor(max_workers=plan['workers'])
        pending = {}
        rolling = dict(dispatches=[], completed_tasks=0, peak_in_flight=0)
        success = False
        last_printed = completed
        try:
            with measured_phase(stage, completed_before=completed,
                                scheduler='rolling_loky_v1', rolling=rolling, **plan):
                while submitted < n_tasks or pending:
                    while submitted < n_tasks and len(pending) < plan['workers']:
                        snapshot = memory_headroom()
                        admission = _rolling_admission(snapshot, len(pending), estimate,
                            _active_worker_rss_credit(pool, len(pending)))
                        if not admission['fits_new_worker'] and pending:
                            break  # Wait for a completion; never force a second worker.
                        if not admission['fits_new_worker']:
                            _ram_advisory()  # Existing user-approved one-worker fallback.
                        batch = take_batch()
                        future = pool.submit(_run_batch, batch, plan['native_threads'])
                        pending[future] = (submitted, len(batch))
                        rolling['dispatches'].append(dict(task_start=submitted, task_count=len(batch),
                            active_workers=len(pending), native_threads=plan['native_threads'],
                            **snapshot, **admission))
                        rolling['peak_in_flight'] = max(rolling['peak_in_flight'], len(pending))
                        submitted += len(batch)
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    # Completion order must never change deterministic tie-breaking
                    # in model selection: all results go back to their input slots.
                    for future in sorted(done, key=lambda item: pending[item][0]):
                        start, count = pending.pop(future)
                        part = future.result()
                        if len(part) != count:
                            raise RuntimeError('Worker returned the wrong number of task results')
                        results[start:start+count] = part
                        completed += count; rolling['completed_tasks'] += count
                    if completed-last_printed >= max(1, n_tasks//20) or completed == n_tasks:
                        print(f'[resources] {stage}: completed {completed}/{n_tasks}; '
                              f'active workers {len(pending)}', flush=True)
                        last_printed = completed
                success = True
        finally:
            # On exception/KeyboardInterrupt, terminate running tasks and join
            # every owned process. Never leave GP work alive during final refit.
            pool.shutdown(wait=True, kill_workers=not success)
    sentinel = object()
    if next(iterator, sentinel) is not sentinel:
        raise ValueError('More tasks than declared')
    return results


def _ram_advisory():
    print('[RAM advisory] Estimate exceeds current headroom. Continuing ONE worker; '
          'no GRID cap. Actual allocation failure remains possible.', flush=True)


def _run_batch(tasks, native_threads):
    return [run_task(task, native_threads) for task in tasks]


def _active_worker_rss_credit(pool, active):
    """Conservative lower bound on RAM already allocated by active workers.

    Future objects do not expose their worker PID. Of all pool workers, the
    smallest ``active`` RSS values cannot exceed the actual active-worker RSS.
    Guard the loky implementation detail; missing PIDs get zero credit. Idle
    workers and parent/result memory stay fully charged by memory_headroom().
    """
    if not active:
        return 0
    try:
        processes = list(pool._processes.values())
    except (AttributeError, RuntimeError):
        return 0
    sizes = []
    for process in processes:
        try:
            sizes.append(int(psutil.Process(process.pid).memory_info().rss))
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            sizes.append(0)
    sizes.extend([0] * max(0, active-len(sizes)))
    return sum(sorted(sizes)[:active])


def _rolling_admission(snapshot, active, estimate, active_rss_credit):
    # memory_headroom already subtracts observed RSS. Reserve only the portion
    # of running-task estimates not allocated yet, not the full estimates again.
    growth = max(0, active*estimate-active_rss_credit)
    room = max(0, snapshot['schedulable_bytes']-growth)
    return dict(reserved_growth_bytes=int(growth), admission_headroom_bytes=int(room),
                fits_new_worker=bool(room >= estimate))


def nn_plan(batch_size):
    if _policy['mode'] != 'adaptive':
        return None
    if batch_size < 1:
        raise ValueError('Positive NN batch size required')
    snap = memory_headroom()
    # Small MLP scratch estimate; full input/feature arrays remain separately resident.
    size = min(batch_size, max(1, snap['schedulable_bytes']//(64*1024)))
    return dict(batch_size=int(size), workers=1,
                native_threads=min(4, _policy['cpu_budget']), **snap)


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
def refit_context(stage='final_gp_refit', *, n_train=None, n_test=0):
    if _policy['mode'] == 'fixed':
        yield
        return
    count = min(_policy['refit_threads'], _policy['cpu_budget'])
    info = {}
    if _policy['mode'] == 'adaptive':
        info = memory_headroom()
        if n_train is not None:
            info['per_worker_estimate_bytes'] = gp_worker_bytes(n_train, n_test)
    print(f'[GP resources] {stage}: 1 model x {count} matrix threads', flush=True)
    with measured_phase(stage, workers=1, native_threads=count, **info), threadpool_limits(limits=count):
        yield
