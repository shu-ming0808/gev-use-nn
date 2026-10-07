from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compare_cheng_nn_inputs import make_features, binned_metrics
from cheng_nn_simulation import build_input_features


def test_original_features_exact_and_whole_sample_scaling():
    y=np.random.default_rng(11).normal(size=(12,50))+np.linspace(-2,2,50)
    expected,scaling=build_input_features(y)
    np.testing.assert_allclose(make_features(y,scaling,17),expected,rtol=1e-6,atol=1e-6)
    z=(y-scaling[:,:1])/scaling[:,1:2]
    np.testing.assert_allclose(make_features(y,scaling,50),z,rtol=1e-6,atol=1e-6)
    x=make_features(y,scaling,21)
    assert x.shape==(12,21)
    for j in range(5):
        q=np.quantile(z[:,j*10:(j+1)*10],[.25,.5,.75],axis=1)
        np.testing.assert_allclose(x[:,11+j*2],q[1],rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(x[:,12+j*2],q[2]-q[0],rtol=1e-6,atol=1e-6)


def test_order_information_not_erased():
    y=np.arange(50,dtype=float)[None,:]
    _,scale=build_input_features(y)
    a=make_features(y,scale,21); b=make_features(y[:,::-1],scale,21)
    np.testing.assert_allclose(a[:,:11],b[:,:11])
    assert a[0,11]<a[0,19] and b[0,11]>b[0,19]


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):make_features(np.zeros((2,49)),np.ones((2,2)),50)
    with pytest.raises(ValueError):make_features(np.zeros((2,50)),np.zeros((2,2)),21)
    with pytest.raises(ValueError):make_features(np.zeros((2,50)),np.ones((2,2)),18)


@pytest.mark.parametrize('dimension',[17,21,50])
def test_affine_standardization_preserves_inputs(dimension):
    y=np.random.default_rng(9).normal(size=(7,50))
    _,scale=build_input_features(y)
    transformed=8*y-35
    new_scale=np.column_stack([8*scale[:,0]-35,8*scale[:,1]])
    np.testing.assert_allclose(make_features(y,scale,dimension),
        make_features(transformed,new_scale,dimension),rtol=1e-6,atol=1e-6)


def test_notebook_network_uses_actual_input_width_without_architecture_change():
    import torch
    from train_cheng_nn import load_notebook_components
    ns=load_notebook_components()
    for dim,count in [(17,27397),(21,27909),(50,31621)]:
        ns['INPUT_FEATURE_NAMES']=tuple(range(dim))
        model=ns['ChengGEVNet'](hidden_sizes=(128,128,64),dropout_p=0.)
        assert model(torch.zeros(2,dim)).shape==(2,5)
        assert sum(p.numel() for p in model.parameters())==count


def test_bins_preserve_all_cases_and_units():
    truth=np.array([[20,-.1,np.log(.2),-.2,-.4],[40,4,np.log(8),.2,.4],
                    [30,0,0,0,0]],dtype=float)
    prediction=truth+np.array([2,3,4,5,6])
    rows=binned_metrics(truth,prediction,np.ones_like(truth),17,1)
    for axis in {r['axis'] for r in rows}:
        for j,name in enumerate(['mu0','beta_mu','eta0','beta_sigma','xi0']):
            part=[r for r in rows if r['axis']==axis and r['coefficient']==name]
            assert sum(r['n_cases'] for r in part)==3
            for row in part:
                if row['n_cases']:
                    assert row['RMSE']==j+2 and row['z_RMSE']==1
