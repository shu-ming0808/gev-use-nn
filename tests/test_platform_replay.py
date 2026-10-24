from pathlib import Path
import sys
import json
import subprocess

import pandas as pd
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'tools/thesis_run_bundle'))
from portable import platform_replay as replay, comparison, inference, workflow
from portable.common import read
from test_simulation_mle_workflow import fixture
from gp_execution import configure


def test_actual_paired_simulation_replay_and_three_way_summary(tmp_path,monkeypatch):
    monkeypatch.setattr(inference,'ROOT',REPO)
    monkeypatch.setattr(inference,'CHECKPOINT',REPO/'results/cheng_nn_21d/reparameterization_100k_20261012/reparameterized/best.pt')
    configure('staged',cpu_budget=2,max_worker_threads=2,refit_threads=2)
    try:
        inputs,truth=fixture()
        comparison.run_replicates(inputs,truth,tmp_path/'pair',1,calibration_id='tiny_shared_truth')
        target,report=replay.replay(tmp_path/'pair',inputs,tmp_path/'replay')
        assert report['same_fit_truth_evaluation_arrays'] and report['outer_fold_membership_matched']
        replay.summarize(tmp_path/'pair',tmp_path/'replay',inputs,tmp_path/'summary')
        summary=read(tmp_path/'summary/run_report.json')
        assert summary['same_execution_policy'] and summary['replicates']==1
        assert len(pd.read_csv(tmp_path/'summary/parameter_rmse.csv'))==30
        assert len(pd.read_csv(tmp_path/'summary/timing_and_support.csv'))==3
        saved=report['timings']['total_from_estimator_start_until_outputs_seconds']
        monkeypatch.setattr(workflow,'execute',lambda *a,**k:pytest.fail('Completed replay reran'))
        # The one-command entry reruns the paired aggregator before replay.
        # This must not invalidate a completed Linux run or overwrite timings.
        comparison.run_replicates(inputs,truth,tmp_path/'pair',1,calibration_id='tiny_shared_truth')
        again,report=replay.replay(tmp_path/'pair',inputs,tmp_path/'replay')
        assert again==target and report['timings']['total_from_estimator_start_until_outputs_seconds']==saved
        bad=list(inputs);bad[2]=inputs[2].copy();bad[2].loc[0,'elevation_m']+=1
        with pytest.raises(ValueError,match='settings differ'):
            replay.replay(tmp_path/'pair',tuple(bad),tmp_path/'bad')
        with (target/'truth.npy').open('ab') as f:f.write(b'changed')
        with pytest.raises(ValueError,match='output changed'):
            replay.summarize(tmp_path/'pair',tmp_path/'replay',inputs,tmp_path/'bad_summary')
    finally:
        configure('fixed')


@pytest.mark.skipif(sys.platform!='win32',reason='Windows PowerShell entry')
def test_one_command_plan_runs_no_workflow():
    proc=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',
        str(REPO/'tools/thesis_run_bundle/run-three.ps1'),'-PlanOnly','-Experiment','qa_three'],
        capture_output=True,text=True,encoding='utf-8',errors='replace')
    assert proc.returncode==0,proc.stderr
    plan=json.loads(proc.stdout)
    assert plan['sequential'] and not plan['starts_workflow']
    assert [r['method'] for r in plan['workflows']]==['Windows NN + GP','Windows MLE + GP','WSL NN + GP']
