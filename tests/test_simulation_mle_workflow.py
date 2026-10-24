"""Tiny CPU scientific integration checks; never launch the full EAS experiment."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools/thesis_run_bundle'))
from portable import comparison, workflow, inference, simulation
from portable.common import EAS_CRS, read, write, sha
from taiwan_simulation import mle_estimator
from research_workflows import COEFFICIENTS
from spatial_coordinates import project_lonlat_to_km


def fixture(n=24):
    lon=120.+np.linspace(0,2,n);lat=23.+.2*np.sin(np.arange(n))
    grids=pd.DataFrame(dict(grid_id=[f'X{i}' for i in range(n)],longitude_180=lon,
        latitude=lat,lsm=np.ones(n)))
    grids[['x_km','y_km']]=project_lonlat_to_km(lon,lat,target_crs=EAS_CRS)
    predictors=pd.DataFrame(dict(grid_id=grids.grid_id,elevation_m=np.linspace(0,1000,n),
        candidate_data_complete=True))
    truth=np.column_stack([25.+np.sin(np.arange(n)/5),np.full(n,.12),np.full(n,np.log(1.2)),
        np.full(n,.02),np.full(n,-.12)])
    config=workflow.settings({'elevation':['elevation_m']},n,outer=2,inner=2,buffer=0,min_train=4)
    config['kernel_candidates']=[['RBF',None]];config['max_steps']=0
    return (simulation.draw_annual(truth,0),grids,predictors,config,EAS_CRS),truth


def test_mle_adapter_data_only_and_serial_parallel_agree():
    inputs,_=fixture(3)
    serial=mle_estimator.infer_mle(inputs[0])
    parallel=mle_estimator.infer_mle(inputs[0],jobs=2)
    assert serial['checks'].success.all()
    np.testing.assert_allclose(serial['prediction'],parallel['prediction'],rtol=1e-8,atol=1e-8)
    assert serial['prediction'].shape==(3,5)
    assert serial['model']['initialization']=='data_only_no_NN_no_truth_no_scenario'
    assert serial['timings']['fitting_wall_seconds']>0
    assert (serial['checks'].min_support_margin>0).all()
    for options in ({'jobs':0},{'domain':'bogus'},{'threads':0},{'maxiter':2,'retry_maxiter':1}):
        with pytest.raises(ValueError): mle_estimator.infer_mle(inputs[0],**options)


def test_mle_retries_and_keeps_successful_lower_nll(monkeypatch):
    calls=[]
    def fit(y,t,**kwargs):
        calls.append(kwargs['maxiter'])
        return dict(success=True,seconds=2.,nll=10.-len(calls),
            lower_nonconverged_nll_gap=.1 if len(calls)==1 else 0.,coefficients=[len(calls)]*5)
    monkeypatch.setattr(mle_estimator,'fit_time_varying_gev',fit)
    row=mle_estimator.fit_one(0,np.arange(50),domain='wide_domain',maxiter=600,retry_maxiter=1800,threads=1)
    assert calls==[600,1800] and row['retried'] and row['seconds']==4 and row['mu0']==2


def test_failed_mle_does_not_silently_drop_grid_or_run_gp(tmp_path,monkeypatch):
    inputs,truth=fixture()
    bad=dict(prediction=np.full_like(truth,np.nan),checks=pd.DataFrame({'success':[False]*len(truth)}),
        model={},timings={})
    monkeypatch.setattr(mle_estimator,'infer_mle',lambda *a,**kw:bad)
    def forbidden(*a,**kw): pytest.fail('GP/NN must not run after failed MLE')
    monkeypatch.setattr(workflow.cv,'nested_cv',forbidden)
    monkeypatch.setattr(workflow,'infer',forbidden)
    with pytest.raises(mle_estimator.MLEFitFailure):
        workflow.execute(*inputs,tmp_path/'failed',estimator='mle',truth=truth,threads=1,check_memory=False)
    assert read(tmp_path/'failed/mle_fit_report.json')['n_failed']==24
    assert read(tmp_path/'failed/run_status.json')['status']=='failed'
    assert len(pd.read_csv(tmp_path/'failed/mle_coefficients.csv'))==24
    assert not (tmp_path/'failed/spatial_oof.csv').exists()


def test_accuracy_is_against_truth_not_nn():
    truth=np.zeros((3,5))
    a=pd.concat([workflow.errors(truth+2,truth,'nn'),workflow.errors(truth+3,truth,'gp_oof')])
    b=pd.concat([workflow.errors(truth+4,truth,'mle'),workflow.errors(truth+6,truth,'gp_oof')])
    result=comparison.paired_accuracy(a,b)
    assert len(result)==10 and result.NN_improvement_percent.eq(50).all()
    with pytest.raises(ValueError): comparison.paired_accuracy(a,b.iloc[:-1])


def test_actual_bound_and_mle_complete_paired_pipeline_and_resume(tmp_path,monkeypatch):
    """Real frozen NN, actual SLSQP MLE, both actual nested exact GP workflows."""
    monkeypatch.setattr(inference,'ROOT',REPO)
    checkpoint=REPO/'results/cheng_nn_21d/reparameterization_100k_20261012/reparameterized/best.pt'
    monkeypatch.setattr(inference,'CHECKPOINT',checkpoint)
    inputs,truth=fixture()
    root=tmp_path/'comparison'
    report=comparison.run_replicates(inputs,truth,root,1,calibration_id='tiny_known_truth',threads=1)
    assert report['completed_pairs']==1 and report['all_grids_retained']
    accuracy=pd.read_csv(root/'parameter_comparison.csv')
    timing=pd.read_csv(root/'timing_comparison.csv')
    assert len(accuracy)==10 and accuracy.n_grids.eq(24).all()
    assert set(accuracy.stage)=={'estimation','gp_oof'}
    assert timing.timing_comparable.all()
    scores=pd.read_csv(root/'crps_comparison.csv')
    assert len(scores)==2 and scores.n_total.eq(24*50).all()
    assert scores[['mean_CRPS_C_NN','mean_CRPS_C_MLE','oracle_CRPS_C']].notna().all().all()
    assert len(pd.read_csv(root/'crps_yearly_comparison.csv'))==100
    assert len(pd.read_csv(root/'crps_summary.csv'))==2
    folders={m:Path(timing.iloc[0][f'{m}_output']) for m in ('nn','mle')}
    for m,folder in folders.items():
        r=read(folder/'run_report.json')
        assert r['metric_reference']=='known_simulated_truth' and r['estimator']==m
        assert r['support_failure_grids']['truth']==0
        pred=pd.read_csv(folder/'oof_coefficients.csv')[list(COEFFICIENTS)].to_numpy()
        got=accuracy.set_index(['stage','coefficient'])
        for i,c in enumerate(COEFFICIENTS):
            assert got.loc[('gp_oof',c),f'RMSE_{m.upper()}']==pytest.approx(np.sqrt(np.mean((pred[:,i]-truth[:,i])**2)))
        assert len(pd.read_csv(folder/'spatial_oof.csv'))==120
        b=np.load(folder/'crps_evaluation_maxima.npy')
        np.testing.assert_array_equal(b,simulation.draw_annual(truth,0,purpose='crps_evaluation'))
        assert not np.array_equal(b,inputs[0])
        assert r['crps_evaluated'] and r['timings']['crps_scoring_and_writing_seconds']>0
    assert 'reference_mle' in pd.read_csv(folders['mle']/'spatial_oof.csv').columns
    assert 'reference_nn' not in pd.read_csv(folders['mle']/'spatial_oof.csv').columns
    saved=timing.iloc[0].workflow_nn_seconds
    # Resume must preserve original timings and not re-execute either workflow.
    monkeypatch.setattr(workflow,'execute',lambda *a,**k:pytest.fail('Completed arm reran'))
    again=comparison.run_replicates(inputs,truth,root,1,calibration_id='tiny_known_truth',threads=1)
    assert again['completed_pairs']==1
    assert pd.read_csv(root/'timing_comparison.csv').iloc[0].workflow_nn_seconds==saved
    # Run every notebook analysis cell against actual integration outputs.
    import nbformat
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    import IPython.display
    monkeypatch.setattr(IPython.display,'display',lambda *a,**k:None)
    monkeypatch.setattr(plt,'show',lambda:None)
    nb=nbformat.read(REPO/'simulation/taiwan_validation/nn_mle_full_workflow_comparison.ipynb',as_version=4)
    namespace={}
    for cell in nb.cells:
        if cell.cell_type=='code':
            source=cell.source.replace('RESULT_ROOT = None',f'RESULT_ROOT = Path({str(root)!r})')
            source=source.replace('SAVE_YEAR_FIGURES = True','SAVE_YEAR_FIGURES = False')
            exec(compile(source,'comparison_notebook','exec'),namespace)
    assert len(namespace['summary'])==10
    assert len(namespace['crps_summary'])==2
    assert len(namespace['year_figures'])==5
    assert len(namespace['year_metrics'])==50
    for figure in namespace['year_figures'].values():
        assert len(figure.axes)==16  # 3x5 maps + one shared colorbar
        figure.canvas.draw()
    for number in plt.get_fignums():
        plt.figure(number).canvas.draw()
    plt.close('all')
    # Explicitly detect a hardware change; no cross-machine speed ratio.
    report_path=folders['mle']/'run_report.json';r=read(report_path)
    r['run_environment']['host']='different_machine';write(report_path,r)
    _,t,_=comparison.summarize_pair(0,folders,inputs[0],truth,inputs[1])
    assert not t['timing_comparable'] and np.isnan(t['workflow_speedup_MLE_over_NN'])
    # Reject mismatched observations rather than compare unpaired samples.
    with pytest.raises(ValueError,match='observations differ'):
        comparison.summarize_pair(0,folders,inputs[0]+1,truth,inputs[1])
    with pytest.raises(ValueError,match='evaluation observations differ'):
        comparison.paired_crps(0,folders,b+1)


def test_notebook_is_valid_read_only_analysis():
    import nbformat
    path=REPO/'simulation/taiwan_validation/nn_mle_full_workflow_comparison.ipynb'
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':
            compile(cell.source,str(path),'exec')
            assert 'run_replicates(' not in cell.source
            assert 'compare_simulations(' not in cell.source


def test_distinct_100_replication_streams():
    _,truth=fixture(2)
    samples=[simulation.draw_annual(truth,i) for i in range(100)]
    assert len({comparison.array_hash(y) for y in samples})==100
    np.testing.assert_array_equal(samples[0],simulation.draw_annual(truth,0))
    evaluations=[simulation.draw_annual(truth,i,purpose='crps_evaluation') for i in range(100)]
    assert len({comparison.array_hash(y) for y in samples+evaluations})==200
    np.testing.assert_array_equal(evaluations[0],simulation.draw_annual(truth,0,purpose='crps_evaluation'))


def test_crps_rejects_in_sample_observations_before_work(tmp_path):
    inputs,truth=fixture()
    with pytest.raises(ValueError,match='separate simulated'):
        workflow.execute(*inputs,tmp_path/'bad',truth=truth,evaluation_values=inputs[0])
    assert not (tmp_path/'bad').exists()


def test_pair_runner_one_then_100_only_runs_missing_arms(tmp_path,monkeypatch):
    """Orchestration regression only; fake timings are never research outputs."""
    from train_cheng_nn import atomic_csv, atomic_array
    inputs,truth=fixture(8);seen=[]
    def fake(values,grids,predictors,config,crs,folder,**kwargs):
        method=kwargs['estimator'];seen.append((method,comparison.array_hash(values)))
        folder.mkdir(parents=True)
        atomic_array(folder/'annual_maxima.npy',values);atomic_array(folder/'truth.npy',truth)
        b=kwargs['evaluation_values']
        atomic_array(folder/'crps_evaluation_maxima.npy',b)
        crps=[];yearly=[]
        for stage in (method,'gp_oof','truth'):
            _,summary,years=workflow.score_temperature_distribution(truth,b,stage=stage)
            crps.append(summary);yearly.append(years)
        atomic_csv(folder/'crps_metrics.csv',pd.concat(crps,ignore_index=True))
        atomic_csv(folder/'crps_yearly_metrics.csv',pd.concat(yearly,ignore_index=True))
        atomic_csv(folder/f'{method}_coefficients.csv',pd.concat([grids,pd.DataFrame(truth,columns=COEFFICIENTS)],axis=1))
        metric=pd.concat([workflow.errors(truth+.1,truth,method),workflow.errors(truth+.2,truth,'gp_oof')])
        atomic_csv(folder/'truth_parameter_metrics.csv',metric)
        membership=pd.MultiIndex.from_product([grids.grid_id,COEFFICIENTS],names=['grid_id','coefficient']).to_frame(index=False)
        membership['outer_fold']=1
        atomic_csv(folder/'spatial_oof.csv',membership)
        atomic_csv(folder/'spatial_selections.csv',pd.DataFrame({'optimizer_success':[True]*5}))
        atomic_csv(folder/f'{method}_domain_checks.csv',pd.DataFrame(dict(
            on_boundary=[False]*len(grids),retried=[False]*len(grids),lower_nonconverged_nll_gap=[0.]*len(grids))))
        report=dict(workflow_completed=True,protocol_id=kwargs['protocol_id'],model={'jobs':1},
            timings=dict(estimator_seconds=1.,join_preflight_nested_cv_seconds=2.,total_from_estimator_start_until_outputs_seconds=3.,
                         crps_scoring_and_writing_seconds=.01),
            support_failure_grids={method:0,'gp_oof':0},
            output_sha256={p.name:sha(p) for p in folder.iterdir()})
        write(folder/'run_report.json',report);return report
    monkeypatch.setattr(workflow,'execute',fake)
    root=tmp_path/'paired'
    comparison.run_replicates(inputs,truth,root,1,calibration_id='mock')
    comparison.run_replicates(inputs,truth,root,100,calibration_id='mock')
    assert len(seen)==200 and len({s for _,s in seen})==100
    for i in range(100):
        pair=seen[2*i:2*i+2]
        assert pair[0][1]==pair[1][1]
        assert [m for m,_ in pair]==(['nn','mle'] if i%2==0 else ['mle','nn'])
    assert len(pd.read_csv(root/'parameter_comparison.csv'))==1000
    assert len(pd.read_csv(root/'timing_comparison.csv'))==100
    assert len(pd.read_csv(root/'crps_comparison.csv'))==200
    assert len(pd.read_csv(root/'crps_yearly_comparison.csv'))==10000
    with pytest.raises(ValueError,match='truncate'):
        comparison.run_replicates(inputs,truth,root,1,calibration_id='mock')
