"""Benchmark safety, portable source discovery and CLI validation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'tools/thesis_run_bundle/benchmark_gp_execution.py'


def module():
    spec = importlib.util.spec_from_file_location('benchmark_gp_execution', SCRIPT)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_profile_parsing():
    assert module().parse_profiles('4x4, 6X2,8x2') == [(4,4),(6,2),(8,2)]


@pytest.mark.parametrize('value', ['0x4', '4x0', '-2x1', '4,4', '4x4,4x4', ''])
def test_invalid_profiles(value):
    with pytest.raises(Exception):
        module().parse_profiles(value)


def test_import_keeps_numeric_libraries_uninitialized():
    code = f'import runpy,sys; runpy.run_path({str(SCRIPT)!r}); assert not ({{"numpy","scipy","torch"}} & set(sys.modules))'
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_source_discovery():
    assert module().source_directory() == ROOT/'src'


def test_actual_ram_reserve_and_application_rss_cap():
    gib = 1024**3
    memory = dict(available_bytes=24*gib, process_tree_rss_bytes=3*gib)
    assert module().available_budget(memory,20) == 17*gib
    memory['available_bytes'] = 10*gib
    assert module().available_budget(memory,20) == 8*gib
    memory['available_bytes'] = gib
    assert module().available_budget(memory,20) == 0


@pytest.mark.parametrize('workers,threads,cpu,tasks,available,expected', [
    (4,4,18,8,20,None), (6,2,18,8,20,None), (8,2,18,8,20,None),
    (8,4,18,8,20,'CPU budget exceeded'),
    (8,2,18,4,20,'Fewer tasks than requested workers'),
    (8,2,18,8,15,'Conservative RAM estimate exceeds available budget'),
])
def test_profiles_only_admitted_with_cpu_tasks_and_memory(workers,threads,cpu,tasks,available,expected):
    assert module().profile_skip_reason(workers,threads,cpu_budget=cpu,tasks=tasks,
        worker_bytes=2,available_bytes=available) == expected


def test_no_overwrite_existing_report(tmp_path):
    report = tmp_path/'benchmark.json'
    report.write_text('old')
    args = SimpleNamespace(train_grids=24,tasks=1,repeats=1,cpu_budget=1,
        memory_budget_gib=20.,prediction_atol=.001,output=report)
    with pytest.raises(FileExistsError):
        module().validate_args(args)
    assert report.read_text() == 'old'


def test_requires_json_output_for_distinct_csv_sibling(tmp_path):
    args = SimpleNamespace(train_grids=24,tasks=1,repeats=1,cpu_budget=1,
        memory_budget_gib=20.,prediction_atol=.001,output=tmp_path/'benchmark.csv')
    with pytest.raises(ValueError, match='end in .json'):
        module().validate_args(args)


@pytest.mark.parametrize('agree,successes,skipped,expected', [
    (True,4,[],'completed'),
    (True,4,[{'profile':'8x2'}],'completed_with_skips'),
    (False,4,[],'completed_with_warnings'),
    (True,3,[],'completed_with_warnings'),
    (False,3,[{'profile':'8x2'}],'completed_with_warnings'),
])
def test_report_status_exposes_numerical_or_optimizer_warnings(agree,successes,skipped,expected):
    runs=[dict(predictions_agree=agree,optimizer_successes=successes,task_count=4)]
    assert module().completed_status(runs,skipped) == expected


def test_empty_run_status_is_not_success():
    assert module().completed_status([],[]) == 'no_trials_run'


def test_all_skipped_main_still_writes_resource_report(tmp_path,monkeypatch):
    app=module()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                 'PYTHONDONTWRITEBYTECODE'):
        monkeypatch.setenv(name,os.environ.get(name,'1'))
    def no_gp(*args,**kwargs):
        raise AssertionError('A skipped benchmark must not fit a GP')
    monkeypatch.setattr(app,'_fit_task',no_gp)
    output=tmp_path/'skipped.json'
    code=app.main(['--profiles','999x1','--cpu-budget','1','--tasks','1',
                  '--repeats','1','--output',str(output)])
    saved=json.loads(output.read_text(encoding='utf-8'))
    assert code==2 and saved['status']=='no_safe_profiles'
    assert saved['runs']==[] and saved['summary']==[]
    assert saved['skipped'][0]['reason']=='CPU budget exceeded'
    assert saved['configuration_changed'] is False
    assert output.with_suffix('.csv').read_text(encoding='utf-8').startswith('profile,completed_repeats')


def test_warning_report_is_saved_without_configuration_change(tmp_path):
    output=tmp_path/'warning.json'
    report=dict(status='completed_with_warnings',configuration_changed=False,summary=[],
                runs=[dict(predictions_agree=False,optimizer_successes=3,task_count=4)])
    module().write_report(output,report)
    assert json.loads(output.read_text(encoding='utf-8'))==report
    with pytest.raises(FileExistsError):
        module().write_report(output,dict(status='completed'))
    assert json.loads(output.read_text(encoding='utf-8'))==report


@pytest.mark.parametrize('field,value', [('train_grids',23),('tasks',0),('repeats',0),
    ('cpu_budget',0),('memory_budget_gib',float('inf')),('prediction_atol',0.)])
def test_invalid_workload(field, value):
    args = SimpleNamespace(train_grids=24,tasks=1,repeats=1,cpu_budget=1,
        memory_budget_gib=20.,prediction_atol=.001,output=None)
    setattr(args,field,value)
    with pytest.raises(ValueError):
        module().validate_args(args)
