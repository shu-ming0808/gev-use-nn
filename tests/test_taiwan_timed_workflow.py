"""Offline orchestration tests: no real NN, GP fitting, downloads or training."""
import json

import numpy as np
import pandas as pd
import pytest

from taiwan_simulation import run_timed_workflow as w


def fixture(tmp_path, monkeypatch):
    prepared = tmp_path/'pilot'; prepared.mkdir()
    years = np.asarray(w.YEARS)
    values = np.tile(np.linspace(25, 35, 50), (4, 1))
    lon, lat = np.array([120, 120.25, 121, 121.25]), np.array([23, 23.25, 24, 24.25])
    xy = w.cv.project_lonlat_to_twd97_km(lon, lat)
    grids = pd.DataFrame(dict(grid_id=['a','b','c','d'], longitude_180=lon, latitude=lat,
                             x_km=xy[:,0], y_km=xy[:,1]))
    np.save(prepared/'annual_maxima.npy', values); np.save(prepared/'years.npy', years)
    grids.to_csv(prepared/'grid_index.csv', index=False)
    raw = tmp_path/'annual.nc'; raw.write_bytes(b'protected test source')
    report = dict(status='completed', completed_stage='taiwan_preparation_and_nn_only',
        contract=w.CONTRACT, data_role='real_data_pilot_not_simulated_truth',
        coordinate_crs='EPSG:3826', coordinate_units='km', temporal_policy=w.TEMPORAL_POLICY,
        years=years.tolist(), n_grids=len(grids), input_path=str(raw.resolve()),
        source_sha256={str(raw.resolve()):w.file_sha256(raw)},
        output_sha256={p.name:w.file_sha256(p) for p in prepared.iterdir()})
    (prepared/'run_report.json').write_text(json.dumps(report), encoding='utf-8')
    predictors = tmp_path/'predictors.csv'
    pd.DataFrame(dict(grid_id=grids.grid_id[::-1], elevation_m=[400,300,200,100],
                     candidate_data_complete=True)).to_csv(predictors,index=False)
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(temporal_policy=w.TEMPORAL_POLICY,
        candidate_groups={'elevation':['elevation_m']}, buffers_km={c:0 for c in w.COEFFICIENTS},
        outer_folds=2, inner_folds=2, min_train=2, max_train=4)),encoding='utf-8')
    calls=[]
    def infer(a, **kwargs):
        calls.append(('nn',kwargs['model_kind']))
        np.testing.assert_array_equal(a,values)
        prediction=np.tile([30,.1,1.,.01,0.],(len(a),1))
        return dict(prediction=prediction,model={'kind':kwargs['model_kind']},timings={'nn_and_decoder_seconds':.001})
    def spatial(data, **kwargs):
        calls.append(('cv',data.grid_id.tolist()))
        assert data.elevation_m.tolist()==[100,200,300,400]
        rows=[dict(grid_id=r.grid_id,outer_fold=i%2,coefficient=c,
                   reference_nn=getattr(r,c),prediction=getattr(r,c))
              for i,r in enumerate(data.itertuples()) for c in w.COEFFICIENTS]
        return {'oof':pd.DataFrame(rows),'metrics':pd.DataFrame({'test':[1]})}
    monkeypatch.setattr(w.tw,'infer_sequences',infer)
    monkeypatch.setattr(w.cv,'nested_cv',spatial)
    return prepared,predictors,config,calls


def test_workflow_has_continuous_timing_and_full_oof_outputs(tmp_path,monkeypatch):
    prepared,predictors,config,calls=fixture(tmp_path,monkeypatch)
    output=tmp_path/'run'
    report=w.run(prepared,predictors,config,output,model_kind='bound',device='cpu')
    assert calls==[('nn','bound'),('cv',['a','b','c','d'])]
    assert report['status']=='completed' and report['workflow_completed']
    assert not report['full_research_pipeline_completed']
    assert report['timings']['total_from_nn_start_until_outputs_seconds']>0
    stage_sum=sum(v for k,v in report['timings'].items() if not k.startswith('total_'))
    assert report['timings']['total_from_nn_start_until_outputs_seconds']>=stage_sum
    assert 'report serialization' in report['timing_scope']
    levels=pd.read_csv(output/'return_levels.csv')
    assert len(levels)==4*50*2 and set(levels.return_period)=={50,100}
    assert set(levels.year)==set(w.YEARS)
    assert np.allclose(levels.gp_oof,levels.nn_reference)
    assert 'truth' not in levels.columns
    assert pd.read_csv(output/'oof_support.csv').n_support_violations.eq(0).all()
    assert 'bootstrap' in report['not_run']
    with pytest.raises(FileExistsError):w.run(prepared,predictors,config,output,model_kind='bound')


@pytest.mark.parametrize('problem',['changed_array','changed_source','wrong_years','null_buffer','missing_predictor','not_complete'])
def test_preflight_rejects_invalid_inputs_before_nn(tmp_path,monkeypatch,problem):
    prepared,predictors,config,calls=fixture(tmp_path,monkeypatch)
    if problem=='changed_array':np.save(prepared/'annual_maxima.npy',np.zeros((4,50)))
    if problem=='changed_source':(tmp_path/'annual.nc').write_bytes(b'changed')
    if problem=='wrong_years':
        np.save(prepared/'years.npy',np.arange(1975,2025))
        r=json.loads((prepared/'run_report.json').read_text())
        r['output_sha256']['years.npy']=w.file_sha256(prepared/'years.npy')
        (prepared/'run_report.json').write_text(json.dumps(r))
    if problem=='null_buffer':
        c=json.loads(config.read_text());c['buffers_km']['mu0']=None;config.write_text(json.dumps(c))
    if problem=='missing_predictor':pd.read_csv(predictors).iloc[:3].to_csv(predictors,index=False)
    if problem=='not_complete':
        f=pd.read_csv(predictors);f.loc[0,'candidate_data_complete']=False;f.to_csv(predictors,index=False)
    with pytest.raises((ValueError,FileNotFoundError)):
        w.run(prepared,predictors,config,tmp_path/'run',model_kind='bound',device='cpu')
    assert not calls


def test_cv_failure_is_not_marked_completed(tmp_path,monkeypatch):
    prepared,predictors,config,calls=fixture(tmp_path,monkeypatch)
    def fail(*args,**kwargs):raise ValueError('insufficient retained GRID')
    monkeypatch.setattr(w.cv,'nested_cv',fail)
    output=tmp_path/'run'
    with pytest.raises(ValueError,match='retained'):
        w.run(prepared,predictors,config,output,model_kind='original',device='cpu')
    status=json.loads((output/'run_status.json').read_text())
    assert status['status']=='failed' and not status['workflow_completed']
    assert not (output/'run_report.json').exists()


def test_legacy_pilot_without_temporal_policy_still_uses_new_config(tmp_path,monkeypatch):
    prepared,predictors,config,calls=fixture(tmp_path,monkeypatch)
    report_path=prepared/'run_report.json'
    report=json.loads(report_path.read_text());report.pop('temporal_policy')
    report_path.write_text(json.dumps(report))
    values,*_=w.load_inputs(prepared,predictors,config)
    assert values.shape==(4,50) and not calls
    current=json.loads(config.read_text());current['temporal_policy']='aic_selection'
    config.write_text(json.dumps(current))
    with pytest.raises(ValueError,match='Require explicit full_no_selection'):
        w.load_inputs(prepared,predictors,config)
    current.pop('temporal_policy');config.write_text(json.dumps(current))
    with pytest.raises(ValueError,match='Require explicit full_no_selection'):
        w.load_inputs(prepared,predictors,config)
    report['temporal_policy']='aic_selection';report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Pilot explicitly'):
        w.load_inputs(prepared,predictors,config)


def test_incomplete_oof_is_rejected(tmp_path,monkeypatch):
    prepared,predictors,config,calls=fixture(tmp_path,monkeypatch)
    spatial=w.cv.nested_cv
    def partial(*args,**kwargs):
        result=spatial(*args,**kwargs);result['oof']=result['oof'].iloc[:-1];return result
    monkeypatch.setattr(w.cv,'nested_cv',partial)
    with pytest.raises(ValueError,match='OOF'):
        w.run(prepared,predictors,config,tmp_path/'run',model_kind='bound',device='cpu')


def test_support_failures_are_reported_without_repairing_gp(tmp_path,monkeypatch):
    prepared,predictors,config,calls=fixture(tmp_path,monkeypatch)
    spatial=w.cv.nested_cv
    def invalid_support(*args,**kwargs):
        result=spatial(*args,**kwargs)
        result['oof'].loc[result['oof'].coefficient.eq('xi0'),'prediction']=-1.
        return result
    monkeypatch.setattr(w.cv,'nested_cv',invalid_support)
    output=tmp_path/'run'
    report=w.run(prepared,predictors,config,output,model_kind='bound',device='cpu')
    assert report['status']=='completed_with_support_or_rl_warnings'
    assert report['invalid_support_grids']=={'nn':0,'oof':4}
    assert pd.read_csv(output/'oof_coefficients.csv').xi0.eq(-1).all()
    assert pd.read_csv(output/'nn_coefficients.csv').xi0.eq(0).all()
