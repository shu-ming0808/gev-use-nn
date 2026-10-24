import json
import numpy as np
import pandas as pd
import pytest
from pyproj import Geod
from global_pipeline import run_eas_workflow as eas
from elevation_gp_analysis import _buffered_training_indices
from spatial_predictor_selection import _fold_contexts
from taiwan_simulation import cheng_spatial_cv as cv
from taiwan_simulation import run_timed_workflow as w


def test_buffer_uses_wgs84_not_projected_xy():
    data=pd.DataFrame({'longitude_180':[120.,120.1,130.], 'latitude':[30.,30.,30.],
                       'x_km':[0.,1000.,1.], 'y_km':[0.,0.,0.]})
    np.testing.assert_array_equal(_buffered_training_indices(data,np.array([1,2]),np.array([0]),50,
        distance_mode='wgs84'),[2])
    np.testing.assert_array_equal(_buffered_training_indices(data,np.array([1,2]),np.array([0]),50),[1])


def test_inner_contexts_use_geodesic_and_all_training_rows():
    lon=np.tile(np.linspace(100,140,6),4); lat=np.repeat(np.linspace(20,40,4),6)
    data=pd.DataFrame({'longitude_180':lon,'latitude':lat,'x_km':lon,'y_km':lat,
                       'spatial_fold':np.arange(len(lon))%3})
    contexts=_fold_contexts(data,'mu0',3,24,2,20260929,buffer_km=500,buffer_distance_mode='wgs84')
    geod=Geod(ellps='WGS84')
    for context in contexts:
        expected=[]
        for i in context['base_indices']:
            if min(geod.inv(lon[i],lat[i],lon[j],lat[j])[2]/1000 for j in context['test_indices'])>500:
                expected.append(i)
        assert context['train_indices'].tolist()==expected


def test_eas_custom_coordinates_and_no_training_cap():
    config=eas.make_config({'elevation':['elevation_m']},7822)
    assert config['max_train']==7822 and config['buffer_distance_mode']=='wgs84'
    assert config['gp_options']['length_scale_bounds']==[1.,6000.]
    lon=np.array([101.,120.]); lat=np.array([22.,35.])
    xy=cv.project_lonlat_to_km(lon,lat,target_crs=eas.eas.EAS_CRS)
    frame=pd.DataFrame({'grid_id':['a','b'],'longitude_180':lon,'latitude':lat,'x_km':xy[:,0],'y_km':xy[:,1]})
    for name in cv.COEFFICIENTS:frame[name]=0.
    predictors=pd.DataFrame({'grid_id':['b','a'],'elevation_m':[200.,100.]})
    paired=cv.join_predictors(frame,predictors,{'elevation':['elevation_m']},coordinate_crs=eas.eas.EAS_CRS)
    assert paired.elevation_m.tolist()==[100.,200.]
    with pytest.raises(ValueError,match='EPSG:3826'):
        cv.join_predictors(frame,predictors,{'elevation':['elevation_m']})


def test_preprocessing_failure_never_starts_nn(tmp_path):
    (tmp_path/'run_status.json').write_text(json.dumps({'status':'failed','error':'missing file'}),encoding='utf-8')
    with pytest.raises(RuntimeError,match='Preprocessing failed'):
        eas.wait_predictors(tmp_path,tmp_path/'launch.json',wait=True)


def test_eas_shared_executor_produces_timing_rl_and_case_checkpoints(tmp_path,monkeypatch):
    n=4
    lon=np.array([105.,115.,125.,135.]);lat=np.array([25.,30.,35.,40.])
    xy=cv.project_lonlat_to_km(lon,lat,target_crs=eas.eas.EAS_CRS)
    grids=pd.DataFrame({'grid_id':list('abcd'),'longitude_180':lon,'latitude':lat,'x_km':xy[:,0],'y_km':xy[:,1]})
    values=np.tile(np.linspace(25,35,50),(n,1))
    predictors=pd.DataFrame({'grid_id':list('abcd'),'elevation_m':[1.,2.,3.,4.]})
    config={'candidate_groups':{'elevation':['elevation_m']}}
    def load(*args):return values,grids,predictors,config,{}
    def infer(*args,**kwargs):
        return {'prediction':np.tile([30,.1,1.,.01,0],(n,1)),'model':{'kind':'bound'}}
    def spatial(frame,progress,**kwargs):
        oof=pd.DataFrame([dict(grid_id=r.grid_id,outer_fold=i%2,coefficient=c,
            reference_nn=getattr(r,c),prediction=getattr(r,c))
            for i,r in enumerate(frame.itertuples()) for c in cv.COEFFICIENTS])
        progress('case_completed',dict(outer_fold=0,coefficient='mu0',completed_cases=1,total_cases=10),{'oof':oof.iloc[:1]})
        return {'oof':oof,'metrics':pd.DataFrame({'n':[n]})}
    monkeypatch.setattr(w.tw,'infer_sequences',infer)
    monkeypatch.setattr(w.cv,'nested_cv',spatial)
    report=w.run(tmp_path,tmp_path/'x.csv',tmp_path/'c.json',tmp_path/'run',model_kind='bound',
        device='cpu',input_loader=load,coordinate_crs=eas.eas.EAS_CRS,region='EAS')
    assert report['completed_stage']=='eas_nn_nested_cv_support_and_conditional_rl'
    assert report['timings']['total_from_nn_start_until_outputs_seconds']>0
    assert report['workflow_completed'] and not report['full_research_pipeline_completed']
    assert (tmp_path/'run/checkpoints/outer_0_mu0/oof.csv').is_file()
    assert len(pd.read_csv(tmp_path/'run/return_levels.csv'))==n*50*2


def test_regional_gp_options_propagate_through_actual_ffs(monkeypatch):
    import spatial_predictor_selection as ffs
    from land_cover_gp_analysis import fit_gp_with_covariates as actual_fit
    lon=np.tile(np.linspace(100,140,4),3);lat=np.repeat(np.linspace(20,40,3),4)
    xy=cv.project_lonlat_to_km(lon,lat,target_crs=eas.eas.EAS_CRS)
    data=pd.DataFrame({'station':[str(i) for i in range(12)],'longitude_180':lon,'latitude':lat,
        'x_km':xy[:,0],'y_km':xy[:,1],'spatial_fold':np.arange(12)%2,
        'height':np.arange(12,dtype=float),'beta_mu':np.sin(np.arange(12))})
    seen=[]
    def fit(*args,**kwargs):
        assert kwargs['length_scale_bounds']==[1.,6000.]
        result=actual_fit(*args,**kwargs)
        assert tuple(result['kernel'].k1.k2.length_scale_bounds)==(1.,6000.)
        seen.append(result)
        return result
    monkeypatch.setattr(ffs,'fit_gp_with_covariates',fit)
    _,_,winner=ffs.spatial_forward_selection(data,'beta_mu',target_columns={'beta_mu':'beta_mu'},
        candidate_groups={'height':['height']},buffer_km=0,buffer_distance_mode='wgs84',
        n_folds=2,min_train=2,max_train=12,max_steps=1,n_jobs=1,kernel_candidates=(('RBF',None),),
        gp_options=dict(length_scale_bounds=[1.,6000.],initial_length_scale=300.))
    assert len(seen)==4 and np.isfinite(winner['RMSE'])
