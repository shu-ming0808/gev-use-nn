"""Math and paired-training safeguards for the independent support experiment."""
from pathlib import Path
import copy
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compare_cheng_nn_support_penalty import (SupportObjective, support_margin, support_penalty)
from train_cheng_nn import load_notebook_components


def test_upper_and_lower_endpoint_and_gumbel():
    c=torch.tensor([[30.,0.,np.log(2.),0.,-.2], [30.,0.,np.log(2.),0.,.2], [30.,0.,np.log(2.),0.,0.]])
    y=torch.tensor([[39.,41.],[19.,21.],[-1000.,1000.]])
    g=support_margin(c,y,torch.tensor([0.,1.]))
    torch.testing.assert_close(g,torch.tensor([[.1,-.1],[-.1,.1],[1.,1.]],dtype=torch.float64))
    assert support_penalty(c,y,torch.tensor([0.,1.])).item()>0


def test_zero_penalty_for_valid_observations():
    c=torch.tensor([[0.,0.,0.,0.,-.2]],dtype=torch.float64)
    y=torch.tensor([[0.,2.]],dtype=torch.float64)
    assert support_penalty(c,y,torch.tensor([-1.,1.])).item()==0.


def test_penalty_uses_worst_year_not_mean_year():
    c=torch.tensor([[0.,0.,0.,0.,-.2]],dtype=torch.float64)
    y=torch.tensor([[0.,6.]],dtype=torch.float64)
    assert support_penalty(c,y,torch.tensor([0.,1.]),margin=0.).item()==pytest.approx(.04)


def test_affine_invariance_and_gradient():
    c=torch.tensor([[30.,.2,np.log(2.),-.03,-.2]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[44.,32.]],dtype=torch.float64)
    t=torch.tensor([-2.45,2.45],dtype=torch.float64)
    m,r=32.,3.
    cz=torch.stack(((c[:,0]-m)/r,c[:,1]/r,c[:,2]-np.log(r),c[:,3],c[:,4]),dim=1)
    torch.testing.assert_close(support_margin(c,y,t),support_margin(cz,(y-m)/r,t))
    assert torch.autograd.gradcheck(lambda x:support_penalty(x,y,t), (c,))
    support_penalty(c,y,t).backward()
    assert torch.isfinite(c.grad).all() and torch.count_nonzero(c.grad)==5


def test_inverse_target_scaling_and_loss_gradient():
    ns=load_notebook_components()
    scaler=ns['TargetScaler'](np.array([0,0,0,0,-.2]),np.array([1,1,1,1,.2]))
    objective=SupportObjective(scaler,[0.,1.],1.,'cpu')
    x=torch.ones((1,1)); target=torch.zeros((1,5)); y=torch.tensor([[6.,6.]],dtype=torch.float64)
    model=torch.nn.Linear(1,5,bias=False)
    torch.nn.init.zeros_(model.weight)
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    mse=objective(model,DataLoader(TensorDataset(x,target,y)), 'cpu',optimizer)
    assert mse==0 and objective.last_training['train_support_penalty_online']>0
    assert model.weight[4].item()>0  # moves negative xi toward a wider upper support


def test_zero_weight_exactly_matches_original_training_step():
    ns=load_notebook_components()
    torch.manual_seed(12)
    a=torch.nn.Linear(3,5); b=copy.deepcopy(a)
    x=torch.randn(8,3); y=torch.randn(8,5); obs=torch.randn(8,50,dtype=torch.float64)
    scaler=ns['TargetScaler'](np.zeros(5),np.ones(5))
    opt_a=torch.optim.Adam(a.parameters(),lr=1e-3,weight_decay=1e-4)
    opt_b=torch.optim.Adam(b.parameters(),lr=1e-3,weight_decay=1e-4)
    loss_a=ns['mean_batch_loss'](a,DataLoader(TensorDataset(x,y),batch_size=4),'cpu',opt_a)
    loss_b=SupportObjective(scaler,np.arange(50),0.,'cpu')(b,DataLoader(TensorDataset(x,y,obs),batch_size=4),'cpu',opt_b)
    assert loss_a==loss_b
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa,pb,rtol=0,atol=0)


def test_validation_is_mse_only_even_with_large_penalty():
    ns=load_notebook_components()
    model=torch.nn.Linear(1,5,bias=False)
    torch.nn.init.zeros_(model.weight)
    scaler=ns['TargetScaler']([0,0,0,0,-.2],[1,1,1,1,1])
    objective=SupportObjective(scaler,[0.,1.],100.,'cpu')
    loader=DataLoader(TensorDataset(torch.ones(2,1),torch.ones(2,5),torch.full((2,2),100.)))
    assert objective(model,loader,'cpu')==1.


def test_nonfinite_penalty_stops_training_instead_of_clipping():
    ns=load_notebook_components()
    scaler=ns['TargetScaler']([0,0,-1000,0,-.2],[1]*5)
    model=torch.nn.Linear(1,5,bias=False)
    torch.nn.init.zeros_(model.weight)
    objective=SupportObjective(scaler,[0.,1.],1.,'cpu')
    loader=DataLoader(TensorDataset(torch.ones(2,1),torch.zeros(2,5),torch.ones(2,2)))
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    with pytest.raises(FloatingPointError):
        objective(model,loader,'cpu',optimizer)


def test_cpu_train_checkpoint_and_evaluation_without_test_access(tmp_path, monkeypatch):
    import compare_cheng_nn_support_penalty as study
    from compare_cheng_nn_mixed_sampling import generate_split
    from cheng_nn_simulation import SimulationConfig, ParameterRanges
    from train_cheng_nn import atomic_json
    sizes={'train':40,'validation':8,'test':8}
    monkeypatch.setattr(study,'SIZES',sizes)
    monkeypatch.setattr(study,'FIXED',{**study.FIXED,'max_epochs':2,'patience':2,'batch_size':8})
    torch.set_num_threads(1)
    for i,(split,n) in enumerate(sizes.items()):
        generate_split(tmp_path/'data',split,n,55+i,SimulationConfig(),ParameterRanges())
    old_load=np.load
    def guarded_load(path,*args,**kwargs):
        if Path(path).parent.name=='test':
            raise AssertionError('Training/evaluation requested the sealed test split.')
        return old_load(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',guarded_load)
    data={s:study.load_split(tmp_path/'data',s) for s in ('train','validation')}
    protocol={'source_hashes':{}}
    atomic_json(tmp_path/'protocol.json',protocol)
    runs=[study.train_one(tmp_path/f'run_{w}',data,w,'cpu',protocol) for w in (0.,1.)]
    assert all(r['status']=='completed' for r in runs)
    assert runs[0]['initial_model_sha256']==runs[1]['initial_model_sha256']
    rows,support=study.evaluate(tmp_path/'run_1.0',data,1.,'cpu')
    assert len(rows)==50 and len(support)==10
    assert {r['split'] for r in rows}=={'train','validation'}
    assert all(np.isfinite(r['RMSE']) for r in rows)
    assert not (tmp_path/'run_1.0/test').exists()
