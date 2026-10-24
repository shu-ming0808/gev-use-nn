from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from global_pipeline import compare_global_cheng_bound as study
from global_pipeline import infer_global_cheng_nn as base
from cheng_nn_evaluation import check_gev_predictions
from cheng_nn_simulation import ParameterRanges
from dataclasses import asdict


def test_independent_endpoints_cover_both_shape_signs_and_gumbel():
    c = np.array([[0,0,0,0,-.2],[0,0,0,0,.2],[0,0,0,0,0]], dtype=float)
    y = np.tile(np.linspace(-6,6,50), (3,1))
    audit = study.independent_audit(c,y,chunk_size=2)
    direct = check_gev_predictions(c,y,np.array(base.YEARS),reference_year=2000.5)
    np.testing.assert_array_equal(audit.endpoint_violation_years,direct.n_support_violations)
    assert audit.max_endpoint_excess_C.tolist() == [1.,1.,0.]
    assert (audit.nonfinite_logpdf_years == audit.endpoint_violation_years).all()


def test_transitions_and_coefficient_changes_are_not_rmse():
    y = np.full((4,50), 10.)
    original = np.array([[0,0,0,0,-.4],[0,0,0,0,0],
                         [0,0,0,0,-.4],[0,0,0,0,0]], dtype=float)
    candidate = original[[1,0,2,3]]
    x=np.zeros((4,21))
    checks=[base.diagnostic_table(c,y,x,np.full(21,-1),np.ones(21),asdict(ParameterRanges()))
            for c in (original,candidate)]
    audits=[study.independent_audit(c,y) for c in (original,candidate)]
    grids=pd.DataFrame({'grid_id':['a','b','c','d']})
    table,changes=study.compare_tables(grids,original,candidate,*checks,*audits)
    assert table.support_transition.tolist()==['resolved','new_violation','unresolved','valid_both']
    assert not any('RMSE' in col for col in changes.columns)
    assert len(changes)==15
    with pytest.raises(ValueError,match='identity'):
        study.compare_tables(pd.DataFrame({'grid_id':['a']*4}),original,candidate,*checks,*audits)


def test_cpu_end_to_end_never_overwrites_formal_results(tmp_path,monkeypatch):
    from test_infer_global_cheng_nn import annual_file,fixture_checkpoint
    official,train=fixture_checkpoint(tmp_path)
    monkeypatch.setattr(base,'CHECKPOINT_HASH',base.file_sha256(official))
    monkeypatch.setattr(base,'CHECKPOINT',official)
    annual=tmp_path/'annual.nc'
    annual_file(annual)
    baseline=tmp_path/'baseline'
    base.run(annual,baseline,official,train,device='cpu',batch_size=2)
    before={p:base.file_sha256(p) for p in baseline.iterdir() if p.is_file()}
    payload=torch.load(official,weights_only=False)
    payload.update(decoder=study.DECODER_ID,support_margin=study.MARGIN,
                   requires_observed_sequence_at_inference=True,
                   latent_coefficient_names=['mu0','beta_mu','delta','beta_sigma','xi0'])
    # Keep saved scaler equal to the baseline: encode shape in the output bias.
    bias_keys=[k for k in payload['model_state'] if k.endswith('bias')]
    payload['model_state'][bias_keys[-1]][4]=-.2
    candidate=tmp_path/'candidate.pt'
    torch.save(payload,candidate)
    result=study.run(baseline,tmp_path/'out',candidate,base.file_sha256(candidate),device='cpu',batch_size=2)
    assert result['decoder_audit_passed'] and result['official_model_replaced'] is False
    assert result['exact_input_and_feature_match']
    assert result['support_transitions']=={'valid_both':4}
    assert all(base.file_sha256(p)==h for p,h in before.items())
    assert result['summary'][1]['failed_grids']==0
    for name,h in result['output_sha256'].items():
        assert base.file_sha256(tmp_path/'out'/name)==h
    with pytest.raises(FileExistsError):
        study.run(baseline,tmp_path/'out',candidate,base.file_sha256(candidate),device='cpu')


def test_reject_wrong_candidate_hash_before_inference(tmp_path,monkeypatch):
    from test_infer_global_cheng_nn import annual_file,fixture_checkpoint
    official,train=fixture_checkpoint(tmp_path)
    monkeypatch.setattr(base,'CHECKPOINT_HASH',base.file_sha256(official))
    monkeypatch.setattr(base,'CHECKPOINT',official)
    annual=tmp_path/'annual.nc'
    annual_file(annual)
    baseline=tmp_path/'baseline'
    base.run(annual,baseline,official,train,device='cpu')
    with pytest.raises(ValueError,match='hash mismatch'):
        study.run(baseline,tmp_path/'bad',official,'wrong',device='cpu')
