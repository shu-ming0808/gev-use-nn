"""Rolling scheduling keeps task identity/order and does not overbook RAM."""
from pathlib import Path
import os
import sys
import time

import pytest
import psutil
from joblib import delayed

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
import gp_execution as resources


@pytest.fixture(autouse=True)
def reset_policy():
    resources.configure('fixed')
    yield
    resources.configure('fixed')


def _snapshot(available=24, rss=2):
    return dict(visible_total_bytes=32*resources.GIB,
                available_bytes=int(available*resources.GIB),
                process_tree_rss_bytes=int(rss*resources.GIB), cgroup_limited=False)


def _dependency_probe(index, folder):
    marker = Path(folder) / 'third-task-started'
    if index == 0:
        limit = time.monotonic()+5
        while not marker.exists() and time.monotonic() < limit:
            time.sleep(.01)
        return index, marker.exists()
    if index == 2:
        marker.touch()
    return index, True


def test_refills_free_slot_before_slow_first_task_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, 'cpu_count', lambda: 2)
    monkeypatch.setattr(resources, 'memory_snapshot', _snapshot)
    resources.configure('adaptive', cpu_budget=2, max_worker_threads=1)
    result = resources.adaptive_tasks(2,
        (delayed(_dependency_probe)(i, str(tmp_path)) for i in range(3)), 3,
        per_worker_bytes=resources.GIB)
    assert [row[0] for row in result] == [0, 1, 2]
    assert result[0][1], 'Task 2 was blocked behind the slow task 0 wave barrier'


def _report_native_threads(index):
    import numpy as np
    from threadpoolctl import threadpool_info
    np.eye(2) @ np.eye(2)
    return index, os.getpid(), [p['num_threads'] for p in threadpool_info()
                               if p['user_api'] == 'blas']


def test_loky_threads_and_chunk_order(monkeypatch):
    monkeypatch.setattr(resources, 'cpu_count', lambda: 2)
    monkeypatch.setattr(resources, 'memory_snapshot', _snapshot)
    resources.configure('adaptive', cpu_budget=2, max_worker_threads=1)
    rows = resources.adaptive_tasks(2,
        (delayed(_report_native_threads)(i) for i in range(11)), 11,
        per_worker_bytes=resources.GIB, tasks_per_worker=4, native_limit=1)
    assert [row[0] for row in rows] == list(range(11))
    assert all(row[1] != os.getpid() for row in rows)
    assert all(row[2] and set(row[2]) == {1} for row in rows)


@pytest.mark.parametrize('bad', [0, -1, 1.5])
def test_invalid_chunk_size_rejected(bad):
    resources.configure('adaptive')
    with pytest.raises(ValueError, match='tasks_per_worker'):
        resources.adaptive_tasks(2, [], 0, per_worker_bytes=resources.GIB,
                                 tasks_per_worker=bad)


def test_memory_admission_reserves_growth_not_already_allocated_rss():
    g = resources.GIB
    snapshot = {'schedulable_bytes': 4*g}
    plan = resources._rolling_admission(snapshot, active=2, estimate=3*g,
                                        active_rss_credit=4*g)
    assert plan['reserved_growth_bytes'] == 2*g
    assert plan['admission_headroom_bytes'] == 2*g
    assert not plan['fits_new_worker']
    plan = resources._rolling_admission(snapshot, active=2, estimate=3*g,
                                        active_rss_credit=6*g)
    assert plan['reserved_growth_bytes'] == 0
    assert plan['admission_headroom_bytes'] == 4*g and plan['fits_new_worker']


def test_active_rss_credit_uses_conservative_smallest_worker_values(monkeypatch):
    from types import SimpleNamespace
    pool = SimpleNamespace(_processes={i: SimpleNamespace(pid=i) for i in (1, 2, 3)})
    monkeypatch.setattr(resources.psutil, 'Process', lambda pid: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss={1: 100, 2: 200, 3: 300}[pid])))
    assert resources._active_worker_rss_credit(pool, 2) == 300
    assert resources._active_worker_rss_credit(pool, 4) == 600  # Missing worker gets zero.
    assert resources._active_worker_rss_credit(object(), 2) == 0


def _brief_task(index):
    start = time.monotonic()
    time.sleep(.04)
    return index, start, time.monotonic()


def test_ram_rechecked_before_refill_and_pending_growth_is_reserved(monkeypatch):
    monkeypatch.setattr(resources, 'cpu_count', lambda: 2)
    calls = []
    def snapshot():
        calls.append(1)
        return _snapshot(available=6 if len(calls) == 1 else 5)
    monkeypatch.setattr(resources, 'memory_snapshot', snapshot)
    monkeypatch.setattr(resources, '_active_worker_rss_credit', lambda pool, active: 0)
    resources.configure('adaptive', cpu_budget=2, max_worker_threads=1)
    rows = resources.adaptive_tasks(2,
        (delayed(_brief_task)(i) for i in range(4)), 4, per_worker_bytes=2*resources.GIB)
    # Initial cap is two, but the new snapshot allows only one live reservation.
    assert all(rows[i+1][1] >= rows[i][2] for i in range(3))
    phase = resources.events()[0]
    assert phase['workers'] == 2 and phase['rolling']['peak_in_flight'] == 1
    assert len(calls) >= 5


def _fail_while_another_worker_runs(index, folder):
    path = Path(folder)
    if index == 1:
        (path/'active.pid').write_text(str(os.getpid()))
        time.sleep(5)
        (path/'unexpected-completion').touch()
        return index
    deadline = time.monotonic()+5
    while not (path/'active.pid').exists() and time.monotonic() < deadline:
        time.sleep(.01)
    raise RuntimeError('intentional-worker-error')


def test_worker_error_kills_and_joins_other_owned_work(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, 'cpu_count', lambda: 2)
    monkeypatch.setattr(resources, 'memory_snapshot', _snapshot)
    resources.configure('adaptive', cpu_budget=2, max_worker_threads=1)
    with pytest.raises(RuntimeError, match='intentional-worker-error'):
        resources.adaptive_tasks(2,
            (delayed(_fail_while_another_worker_runs)(i, str(tmp_path)) for i in range(2)),
            2, per_worker_bytes=resources.GIB)
    assert not (tmp_path/'unexpected-completion').exists()
    pid = int((tmp_path/'active.pid').read_text())
    assert not psutil.pid_exists(pid)
    assert resources.events()[-1]['success'] is False


def test_task_count_and_mode_validation(monkeypatch):
    monkeypatch.setattr(resources, 'memory_snapshot', _snapshot)
    with pytest.raises(ValueError, match='adaptive policy'):
        resources.adaptive_tasks(1, [], 0, per_worker_bytes=resources.GIB)
    resources.configure('adaptive', cpu_budget=1)
    assert resources.adaptive_tasks(1, [], 0, per_worker_bytes=resources.GIB) == []
    with pytest.raises(ValueError, match='More tasks'):
        resources.adaptive_tasks(1, [None], 0, per_worker_bytes=resources.GIB)
    with pytest.raises(ValueError, match='Fewer tasks'):
        resources.adaptive_tasks(1, [delayed(abs)(-1)], 2, per_worker_bytes=resources.GIB,
                                 tasks_per_worker=2)
