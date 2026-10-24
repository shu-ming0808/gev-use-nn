import calendar
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from global_pipeline import infer_global_cheng_nn as g
from cheng_nn_simulation import ParameterRanges
from dataclasses import asdict


def annual_file(path, units='degrees_Celsius', missing=False, year_shift=0):
    values = (np.arange(50)[:,None,None]*.03 + 25 + np.sin(np.arange(50))[:,None,None]
              + np.arange(6).reshape(1,2,3)*.4).astype('f4')
    with g.open_nc(path,'w') as ds:
        for name,array in [('year',np.arange(1976,2026)+year_shift),('latitude',[20,-90]),('longitude',[0,120,240])]:
            ds.createDimension(name,len(array))
            ds.createVariable(name,'f8',(name,))[:] = array
        ds.createVariable('lsm','f4',('latitude','longitude'))[:] = [[1,.5,.7],[.9,0,1]]
        dims=('year','latitude','longitude')
        v=ds.createVariable('annual_maximum_t2m','f4',dims)
        v.units=units
        v[:]=values
        counts=ds.createVariable('valid_hour_count','u2',dims)
        counts[:]=np.broadcast_to(np.array([(366 if calendar.isleap(y) else 365)*24 for y in g.YEARS])[:,None,None],values.shape)
        if missing:
            counts[0,0,0]=1
    return values


def fixture_checkpoint(tmp_path):
    with redirect_stdout(io.StringIO()):
        ns=g.load_notebook_components()
    ns['INPUT_FEATURE_NAMES']=tuple(g.FEATURE_NAMES)
    model=ns['ChengGEVNet'](hidden_sizes=(128,128,64),dropout_p=0.)
    for p in model.parameters():
        p.data.zero_()
    checkpoint=tmp_path/'fake.pt'
    train=tmp_path/'train'
    train.mkdir()
    maxima=np.stack([np.arange(50)+i for i in range(4)]).astype('f4')
    _,scaling=g.transform_sequences(maxima)
    np.save(train/'annual_maxima.npy',maxima)
    np.save(train/'sample_location_scale.npy',scaling)
    meta=dict(years=list(g.YEARS),reference_year=g.REFERENCE_YEAR,time_scale_years=10.,
              input_quantile_levels=list(g.QUANTILE_LEVELS),parameter_ranges=asdict(ParameterRanges()),
              files_sha256={p.name:g.file_sha256(p) for p in train.iterdir()})
    sources={str(g.PROJECT_ROOT/folder/name):g.file_sha256(g.PROJECT_ROOT/folder/name)
             for folder,name in [('notebooks','cheng_NN.ipynb'),('src','train_cheng_nn.py'),
                                 ('src','compare_cheng_nn_inputs.py'),('src','cheng_nn_simulation.py')]}
    torch.save(dict(architecture=[21,128,128,64,5],input_shape=[21],dropout_p=0.,
                    coefficient_names=list(g.COEFFICIENTS),simulation_metadata=meta,
                    source_hashes=sources,model_state=model.state_dict(),
                    target_scaler=dict(mean=np.zeros(5,dtype='f4'),std=np.ones(5,dtype='f4')),best_epoch=1),checkpoint)
    return checkpoint,train


def test_reader_preserves_ids_celsius_order_and_land_threshold(tmp_path):
    p=tmp_path/'annual.nc'
    original=annual_file(p)
    values,grids=g.read_annual_land(p)
    assert values.shape==(4,50)
    assert grids.grid_id.tolist()==['G000_0000','G000_0002','G001_0000','G001_0002']
    np.testing.assert_array_equal(values[0],original[:,0,0])
    assert grids.longitude_180.tolist()==[0,-120,0,-120]
    assert grids.is_pole.sum()==2


@pytest.mark.parametrize('options,match',[({'units':'K'},'Celsius'),({'missing':True},'coverage'),({'year_shift':1},'1976')])
def test_reader_rejects_incompatible_data(tmp_path,options,match):
    p=tmp_path/'bad.nc'
    annual_file(p,**options)
    with pytest.raises(ValueError,match=match):
        g.read_annual_land(p)


def test_five_temporal_segments_share_one_scaling():
    y=np.arange(50,dtype=float)[None,:]
    features,scaling=g.transform_sequences(y)
    np.testing.assert_allclose(scaling,[[24.5,24.5]])
    np.testing.assert_allclose(features[0,11::2],(np.array([4.5,14.5,24.5,34.5,44.5])-24.5)/24.5,rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(features[0,12::2],np.full(5,4.5/24.5),rtol=1e-6)
    assert len(g.FEATURE_NAMES)==21
    with pytest.raises(ValueError,match='IQR'):
        g.transform_sequences(np.ones((1,50)))


def test_windows_checkpoint_source_names_are_portable(tmp_path):
    checkpoint,_=fixture_checkpoint(tmp_path)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    from pathlib import PureWindowsPath
    payload['source_hashes']={str(PureWindowsPath(p)):h for p,h in payload['source_hashes'].items()}
    torch.save(payload,checkpoint)
    _,_,_,loaded=g.load_frozen_model(checkpoint,g.file_sha256(checkpoint),'cpu')
    assert loaded['architecture']==[21,128,128,64,5]


def test_wrong_hash_never_deserializes(tmp_path,monkeypatch):
    p=tmp_path/'bad.pt'
    p.write_bytes(b'untrusted')
    def forbidden(*a,**kw):
        raise AssertionError('Must not deserialize')
    monkeypatch.setattr(torch,'load',forbidden)
    with pytest.raises(ValueError,match='SHA256'):
        g.load_frozen_model(p,'wrong','cpu')


def test_support_failure_is_preserved_not_clipped():
    pred=np.array([[0,0,0,0,-.4],[0,0,0,0,0]])
    data=np.full((2,50),10.)
    diag=g.diagnostic_table(pred,data,np.zeros((2,21)),np.full(21,-1),np.ones(21),asdict(ParameterRanges()))
    assert len(diag)==2 and diag.n_support_violations.tolist()==[50,0]
    assert diag.needs_review.tolist()==[True,False]


def test_cpu_end_to_end_keeps_inputs_outputs_and_timer(tmp_path,monkeypatch):
    checkpoint,train=fixture_checkpoint(tmp_path)
    monkeypatch.setattr(g,'CHECKPOINT_HASH',g.file_sha256(checkpoint))
    input_path=tmp_path/'annual.nc'
    annual_file(input_path)
    before=g.file_sha256(input_path)
    output=tmp_path/'out'
    report=g.run(input_path,output,checkpoint,train,device='cpu',batch_size=2)
    assert report['status']=='completed' and report['summary']['n_grids']==4
    assert report['model']['retrained'] is False
    assert g.file_sha256(input_path)==before
    x=np.load(output/'inputs_21d.npy')
    scale=np.load(output/'sample_location_scale.npy')
    pred=np.load(output/'coefficients_original.npy')
    assert x.shape==(4,21) and pred.shape==(4,5)
    np.testing.assert_allclose(pred[:,0],scale[:,0])
    np.testing.assert_allclose(pred[:,2],np.log(scale[:,1]))
    np.testing.assert_array_equal(pred[:,[1,3,4]],0)
    assert len(pd.read_csv(output/'coefficients_and_checks.csv'))==4
    t=report['timings']
    assert t['total_wall_seconds_until_report']>=t['nn_pipeline_seconds']>=t['nn_inference_seconds']>0
    for name,h in report['output_sha256'].items():
        assert g.file_sha256(output/name)==h
    with pytest.raises(FileExistsError):
        g.run(input_path,output,checkpoint,train,device='cpu')


def test_cuda_timer_synchronizes_both_boundaries(monkeypatch):
    calls=[]
    monkeypatch.setattr(torch.cuda,'synchronize',lambda device:calls.append(device))
    t={}
    with g.timed(t,'forward','cuda'):
        assert calls==['cuda']
    assert calls==['cuda','cuda'] and t['forward']>=0
