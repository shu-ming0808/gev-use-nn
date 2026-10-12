"""Guarantees and scientific-protocol checks for the five-coefficient decoder."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compare_cheng_nn_reparameterization import decode_coefficients, predict_sample_coefficients, MARGIN
from compare_cheng_nn_support_penalty import support_margin


def test_both_shape_signs_and_gumbel_and_positive_scale():
    c = torch.tensor([[0.,.4,-2.,-.1,-.3],[0.,-.4,-2.,.1,.3],
                      [0.,10.,-2.,.2,0.]],dtype=torch.float64)
    z = torch.tensor([[1.,20.],[-20.,1.],[-1000.,1000.]],dtype=torch.float64)
    t = torch.tensor([-2.45,2.45],dtype=torch.float64)
    decoded = decode_coefficients(c,z,t)
    assert torch.all(support_margin(decoded,z,t) > MARGIN)
    assert torch.all(torch.exp(decoded[:,2:3]+decoded[:,3:4]*t) > 0)
    torch.testing.assert_close(decoded[:,[0,1,3,4]],c[:,[0,1,3,4]])
    assert decoded[2,2] == c[2,2]


def test_observation_beyond_original_support_becomes_feasible():
    c = torch.tensor([[30.,.2,np.log(2.),-.03,-.2]],dtype=torch.float64)
    z = torch.tensor([[45.,32.]],dtype=torch.float64)
    t = torch.tensor([-2.45,2.45],dtype=torch.float64)
    assert support_margin(c,z,t).min() < 0
    out = decode_coefficients(c,z,t)
    assert support_margin(out,z,t).min() > MARGIN
    upper = out[:,0:1]+out[:,1:2]*t-torch.exp(out[:,2:3]+out[:,3:4]*t)/out[:,4:5]
    assert torch.all(z < upper)


def test_random_50_year_stress_and_linear_log_scale():
    gen = torch.Generator().manual_seed(731)
    c = torch.randn(400,5,generator=gen,dtype=torch.float64)
    c[:,4] *= .4
    c[:,2] -= 5
    z = torch.randn(400,50,generator=gen,dtype=torch.float64)*8
    t = torch.linspace(-2.45,2.45,50,dtype=torch.float64)
    out = decode_coefficients(c,z,t)
    assert torch.all(support_margin(out,z,t) > MARGIN-1e-12)
    eta = out[:,2:3]+out[:,3:4]*t
    assert torch.max(torch.abs(torch.diff(eta,n=2,dim=1))) < 1e-12


def test_affine_equivariance_of_delta_and_decoded_coefficients():
    c = torch.tensor([[30.,.2,np.log(2.),-.03,-.2]],dtype=torch.float64)
    y = torch.tensor([[44.,32.]],dtype=torch.float64)
    t = torch.tensor([-2.45,2.45],dtype=torch.float64)
    m,r = 32.,3.
    def standardized(v):
        return torch.stack(((v[:,0]-m)/r,v[:,1]/r,v[:,2]-np.log(r),v[:,3],v[:,4]),dim=1)
    torch.testing.assert_close(standardized(decode_coefficients(c,y,t)),
        decode_coefficients(standardized(c),(y-m)/r,t))


def test_gradient_through_all_five_outputs_and_zero_bound():
    c = torch.tensor([[0.,.2,-.5,.03,-.2]],dtype=torch.float64,requires_grad=True)
    y = torch.tensor([[8.,0.]],dtype=torch.float64)
    t = torch.tensor([-2.45,2.45],dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x:decode_coefficients(x,y,t), (c,))
    decode_coefficients(c,y,t)[:,2].sum().backward()
    assert torch.isfinite(c.grad).all() and torch.count_nonzero(c.grad)==5
    c2 = torch.tensor([[0.,0.,-1.,0.,.2]],dtype=torch.float64,requires_grad=True)
    out = decode_coefficients(c2,torch.ones((1,2)),t)
    out.sum().backward()
    assert torch.isfinite(c2.grad).all()
    assert out[0,2] == c2[0,2]


def test_recover_any_strictly_feasible_known_five_coefficient_curve():
    truth = torch.tensor([[0.,.2,np.log(2.),.1,-.2]],dtype=torch.float64)
    t = torch.linspace(-2.45,2.45,50,dtype=torch.float64)
    y = torch.sin(t)[None,:]
    a = (truth[:,4:5]*(truth[:,:1]+truth[:,1:2]*t-y)*torch.exp(-truth[:,3:4]*t)).amax(dim=1).clamp_min(0)/(1-MARGIN)
    latent = truth.clone()
    latent[:,2] = torch.log(torch.exp(truth[:,2])-a)
    torch.testing.assert_close(decode_coefficients(latent,y,t),truth)


def test_inference_uses_observations_but_never_labels_or_expands_mlp_input():
    from train_cheng_nn import load_notebook_components
    ns = load_notebook_components()
    model = torch.nn.Linear(21,5)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    scaler = ns['TargetScaler'](np.array([0.,0.,0.,0.,-.2]),np.ones(5))
    x = np.zeros((2,21),dtype=np.float32)
    y = np.array([[1.,2.],[10.,20.]])
    out = predict_sample_coefficients(model,x,scaler,y,[-1.,1.],'cpu',True)
    assert out[1,2] > out[0,2]  # disclosure: same 21D can decode differently
    assert model.in_features == 21


@pytest.mark.parametrize('margin',[0.,1.,-1.,np.nan])
def test_reject_invalid_margin(margin):
    with pytest.raises(ValueError):
        decode_coefficients(torch.zeros(1,5),torch.zeros(1,2),torch.arange(2),margin)


def test_cpu_training_checkpoint_reload_and_original_scale_audit(tmp_path,monkeypatch):
    import compare_cheng_nn_reparameterization as study
    import compare_cheng_nn_support_penalty as old
    from compare_cheng_nn_mixed_sampling import generate_split
    from cheng_nn_simulation import SimulationConfig,ParameterRanges
    from train_cheng_nn import atomic_json
    monkeypatch.setattr(old,'SIZES',{'train':40,'validation':8,'test':8})
    monkeypatch.setattr(study,'FIXED',{**study.FIXED,'max_epochs':2,'patience':2,'batch_size':8})
    torch.set_num_threads(1)
    for i,(s,n) in enumerate([('train',40),('validation',8),('test',8)]):
        generate_split(tmp_path/'data',s,n,101+i,SimulationConfig(),ParameterRanges())
    load = np.load
    def guarded(path,*args,**kwargs):
        if Path(path).parent.name=='test':
            raise AssertionError('Test split accessed before checkpoint lock.')
        return load(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',guarded)
    data = {s:study.load_split(tmp_path/'data',s) for s in ('train','validation')}
    protocol = {'source_hashes':{}}
    atomic_json(tmp_path/'protocol.json',protocol)
    meta = study.train_candidate(tmp_path/'candidate',data,'cpu',protocol)
    assert meta['status']=='completed'
    payload = torch.load(tmp_path/'candidate/best.pt',map_location='cpu',weights_only=False)
    assert payload['requires_observed_sequence_at_inference'] is True
    assert payload['latent_coefficient_names'][2]=='delta'
    rows,summary = study.evaluate(tmp_path/'candidate/best.pt',tmp_path/'candidate',data,'reparameterized','cpu')
    assert len(rows)==50 and len(summary)==10
    assert all(r['combined_failed_sequences']==0 for r in summary)
    assert all(r['minimum_support_margin']>=MARGIN-1e-10 for r in summary)
    assert all(np.isfinite(r['RMSE']) for r in rows)
    assert not (tmp_path/'candidate/test').exists()
