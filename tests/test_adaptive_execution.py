"""Execution-only changes: deterministic tasks, no GRID reduction."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from joblib import delayed

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
import gp_execution as resources


@pytest.fixture(autouse=True)
def reset():
    resources.configure('fixed')
    yield
    resources.configure('fixed')


def snapshot(total=32, available=24, rss=2):
    return dict(visible_total_bytes=int(total*resources.GIB), available_bytes=int(available*resources.GIB),
                process_tree_rss_bytes=int(rss*resources.GIB), cgroup_limited=False)


def test_ram_worker_cap_updates_and_cpu_threads_do_not_oversubscribe(monkeypatch):
    monkeypatch.setattr(resources, 'cpu_count', lambda: 20)
    ram = snapshot()
    monkeypatch.setattr(resources, 'memory_snapshot', lambda: ram)
    resources.configure('adaptive',cpu_budget=19,memory_budget_gib=26,memory_reserve_gib=2)
    first = resources.candidate_plan(19,44,per_worker_bytes=4*resources.GIB)
    assert first['workers'] == 5 and first['native_threads'] == 3
    ram.update(snapshot(available=7))
    second = resources.candidate_plan(19,44,per_worker_bytes=4*resources.GIB)
    assert second['workers'] == 1 and second['native_threads'] == 4
    assert second['workers']*second['native_threads'] <= 19
    assert resources.candidate_plan(1,44,per_worker_bytes=resources.GIB)['workers'] == 1


def test_16gb_cpu20_redistributes_budget_at_plan_boundary_without_extra_gp_workers(monkeypatch):
    monkeypatch.setattr(resources, 'cpu_count', lambda: 20)
    ram = snapshot(total=7.75, available=6.9, rss=.47)
    monkeypatch.setattr(resources, 'memory_snapshot', lambda: ram)
    # Reproduce the captured calibration: headroom4.53GiB, estimate2.98GiB.
    resources.configure('adaptive', cpu_budget=8, max_worker_threads=4,
                        memory_budget_gib=5, memory_reserve_gib=1.5)
    old = resources.candidate_plan(2, 52, per_worker_bytes=int(2.98*resources.GIB))
    assert (old['workers'], old['native_threads']) == (1, 4)
    resources.configure('adaptive', cpu_budget=20, max_worker_threads=20, refit_threads=20,
                        memory_budget_gib=5, memory_reserve_gib=1.5)
    new = resources.candidate_plan(2, 52, per_worker_bytes=int(2.98*resources.GIB))
    assert (new['workers'], new['native_threads']) == (1, 20)
    small = resources.candidate_plan(2, 52, per_worker_bytes=resources.GIB)
    assert (small['workers'], small['native_threads']) == (2, 10)
    mle = resources.candidate_plan(20, 7822, per_worker_bytes=int(.5*resources.GIB), native_limit=1)
    assert mle['workers'] < 20 and mle['native_threads'] == 1
    # Respect a genuinely restricted Linux CPU count rather than pretend20 exist.
    monkeypatch.setattr(resources, 'cpu_count', lambda: 8)
    resources.configure('adaptive', cpu_budget=20, max_worker_threads=20,
                        memory_budget_gib=5, memory_reserve_gib=1.5)
    restricted = resources.candidate_plan(2, 52, per_worker_bytes=int(2.98*resources.GIB))
    assert restricted['cpu_budget'] == 8 and restricted['native_threads'] == 8


def test_budget_includes_current_process_and_respects_wsl_smaller_ram(monkeypatch):
    monkeypatch.setattr(resources,'memory_snapshot',lambda: snapshot(total=16,available=12,rss=4))
    resources.configure('adaptive',memory_budget_gib=26,memory_reserve_gib=2)
    assert resources.memory_headroom()['schedulable_bytes'] == 10*resources.GIB
    monkeypatch.setattr(resources,'memory_snapshot',lambda: snapshot(available=25,rss=24))
    assert resources.memory_headroom()['schedulable_bytes'] == 2*resources.GIB
    monkeypatch.setattr(resources,'memory_snapshot',lambda: snapshot(available=1,rss=26))
    plan = resources.candidate_plan(-2,5,per_worker_bytes=resources.GIB)
    assert plan['workers'] == 1 and not plan['estimated_single_worker_fits']


def test_cgroup_finite_ancestor_and_unlimited_child(tmp_path):
    (tmp_path/'memory.max').write_text(str(8*resources.GIB))
    (tmp_path/'memory.current').write_text(str(6*resources.GIB))
    child=tmp_path/'group';child.mkdir()
    (child/'memory.max').write_text('max')
    limits=resources._hierarchy_headrooms(tmp_path,'/group','memory.max','memory.current')
    assert limits == [(8*resources.GIB,2*resources.GIB)]
    (child/'memory.max').write_text(str(4*resources.GIB))
    (child/'memory.current').write_text(str(resources.GIB))
    limits=resources._hierarchy_headrooms(tmp_path,'/group','memory.max','memory.current')
    assert min(v[1] for v in limits)==2*resources.GIB


def test_wave_scheduling_retains_order_and_rechecks_memory(monkeypatch):
    resources.configure('adaptive',cpu_budget=2)
    calls=[]
    def ram():
        calls.append(1)
        return snapshot(available=2.1)
    monkeypatch.setattr(resources,'memory_snapshot',ram)
    result=resources.adaptive_tasks(2,(delayed(abs)(-i) for i in range(5)),5,
        per_worker_bytes=resources.GIB)
    assert result == list(range(5)) and len(calls)==5
    assert len(resources.events())==5
    assert all(row['wave_tasks']==1 for row in resources.events())
    with pytest.raises(ValueError,match='Fewer tasks'):
        resources.adaptive_tasks(1,[],1,per_worker_bytes=resources.GIB)


def test_memory_estimate_scales_with_actual_fold_not_full_domain():
    assert resources.gp_worker_bytes(3000,1000) < resources.gp_worker_bytes(7822,1000)


def test_adaptive_nn_keeps_max_batch_or_reduces_when_memory_tight(monkeypatch):
    resources.configure('adaptive',cpu_budget=3)
    monkeypatch.setattr(resources,'memory_snapshot',lambda:snapshot())
    assert resources.nn_plan(2048)['batch_size']==2048
    monkeypatch.setattr(resources,'memory_snapshot',lambda:snapshot(available=2+1/1024))
    assert resources.nn_plan(2048)['batch_size']==16
    assert resources.nn_plan(2048)['native_threads']<=3
    for kw in [{'memory_budget_gib':0},{'memory_reserve_gib':-1},{'memory_budget_gib':float('nan')}]:
        with pytest.raises(ValueError): resources.configure('adaptive',**kw)


def test_adaptive_real_gp_dispatch_matches_fixed(monkeypatch):
    import spatial_predictor_selection as selection
    from test_simulation_mle_workflow import fixture
    inputs,_ = fixture(24)
    from taiwan_simulation import cheng_spatial_cv as cv
    import pandas as pd
    from research_workflows import COEFFICIENTS
    _,grids,predictors,config,crs=inputs
    config.update(kernel_candidates=[['RBF',None],['Matern',1.5]],n_jobs=2)
    responses=pd.concat([grids,pd.DataFrame(np.tile([25,.1,0,.01,-.1],(24,1)),columns=COEFFICIENTS)],axis=1)
    joined=cv.join_predictors(responses,predictors,config['candidate_groups'],coordinate_crs=crs)
    before=cv.nested_cv(joined,**config)
    resources.configure('adaptive',cpu_budget=2,refit_threads=1,max_worker_threads=1)
    monkeypatch.setattr(resources,'memory_snapshot',lambda:snapshot())
    after=cv.nested_cv(joined,**config)
    np.testing.assert_allclose(after['oof'].prediction,before['oof'].prediction,rtol=1e-5,atol=1e-5)
    assert after['oof'][['grid_id','outer_fold','coefficient']].equals(before['oof'][['grid_id','outer_fold','coefficient']])
    assert any(e['stage']=='candidate_models' and 'per_worker_estimate_bytes' in e for e in resources.events())
    assert any(e['stage']=='candidate_models' and e['workers']==2 for e in resources.events())


def test_mle_first_real_tiny_pair_and_resume(tmp_path,monkeypatch):
    sys.path.insert(0,str(REPO/'tools/thesis_run_bundle'))
    from portable import comparison,workflow,inference
    from portable.common import read
    from test_simulation_mle_workflow import fixture
    monkeypatch.setattr(inference,'ROOT',REPO)
    monkeypatch.setattr(inference,'CHECKPOINT',REPO/'results/cheng_nn_21d/reparameterization_100k_20261012/reparameterized/best.pt')
    resources.configure('adaptive',cpu_budget=2,max_worker_threads=1,refit_threads=1)
    monkeypatch.setattr(resources,'memory_snapshot',lambda:snapshot())
    inputs,truth=fixture(24)
    order=[];execute=workflow.execute
    def track(*args,**kw):
        order.append(kw['estimator'])
        return execute(*args,**kw)
    monkeypatch.setattr(workflow,'execute',track)
    root=tmp_path/'pair'
    report=comparison.run_replicates(inputs,truth,root,1,calibration_id='qa',method_order='mle-first',
        mle_options={'jobs':2},threads=1)
    assert report['completed_pairs']==1 and order==['mle','nn']
    for method in order:
        saved=read(root/f'replicates/replicate_001/{method}/attempt_001/run_report.json')
        assert saved['method_order']==['mle','nn'] and saved['gp_execution_policy']['mode']=='adaptive'
        phases=read(root/f'replicates/replicate_001/{method}/attempt_001/execution_phases.json')['phases']
        assert phases and any('schedulable_bytes' in phase for phase in phases)
    comparison.run_replicates(inputs,truth,root,1,calibration_id='qa',method_order='mle-first',mle_options={'jobs':2})
    assert order==['mle','nn']
    with pytest.raises(ValueError,match='protocol changed'):
        comparison.run_replicates(inputs,truth,root,1,calibration_id='qa',method_order='nn-first',mle_options={'jobs':2})


@pytest.mark.skipif(sys.platform!='win32',reason='Windows launcher')
def test_linux_entry_plan_is_mle_first_and_does_not_launch():
    proc=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',
        str(REPO/'tools/thesis_run_bundle/run-linux.ps1'),'-PlanOnly'],capture_output=True,text=True,encoding='utf-8',errors='replace')
    assert proc.returncode==0,proc.stderr
    plan=json.loads(proc.stdout)
    assert plan['order']==['MLE + GP','NN + GP'] and not plan['starts_workflow']
    assert plan['jobs']==4 and plan['mle_jobs']==16 and plan['memory_budget_gib']==20
