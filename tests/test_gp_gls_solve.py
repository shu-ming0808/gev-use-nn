"""Exact-GP solve reuse preserves the GLS likelihood and predictions."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import land_cover_gp_analysis as gp


def original_gls(chol, design, response):
    inverse_design = cho_solve(chol, design, check_finite=False)
    inverse_response = cho_solve(chol, response, check_finite=False)
    beta = np.linalg.solve(design.T @ inverse_design, design.T @ inverse_response)
    residual = response-design @ beta
    alpha = cho_solve(chol, residual, check_finite=False)
    return beta, residual, alpha


@pytest.mark.parametrize('columns', [1,4])
@pytest.mark.parametrize('condition', [10.,1e8])
def test_gls_objective_coefficients_and_heldout_prediction_match(columns,condition):
    rng = np.random.default_rng(20260929)
    n = 48
    basis, _ = np.linalg.qr(rng.normal(size=(n,n)))
    covariance = (basis*np.geomspace(1.,1./condition,n)) @ basis.T
    design = np.column_stack([np.ones(n),rng.normal(size=(n,columns-1))])
    response = 25.+design @ rng.normal(size=columns)+rng.normal(size=n)
    chol = cho_factor(covariance, lower=True, check_finite=False)
    reference = original_gls(chol,design,response)
    result = gp._profiled_gls_components(chol,design,response)
    for actual, expected in zip(result,reference):
        np.testing.assert_allclose(actual,expected,rtol=2e-8,atol=2e-8)
    logdet = 2.*np.log(np.diag(chol[0])).sum()
    objective = lambda value: .5*(value[1] @ value[2]+logdet+n*np.log(2*np.pi))
    np.testing.assert_allclose(objective(result),objective(reference),rtol=2e-10,atol=2e-8)
    test_design = np.column_stack([np.ones(5),rng.normal(size=(5,columns-1))])
    cross_covariance = covariance[:,:5]*.5
    prediction = lambda value: test_design @ value[0]+cross_covariance.T @ value[2]
    np.testing.assert_allclose(prediction(result),prediction(reference),rtol=2e-8,atol=2e-7)


def test_regular_case_uses_one_multi_rhs_solve(monkeypatch):
    calls = []
    real = gp.cho_solve
    def track(chol, rhs, **kwargs):
        calls.append(rhs.shape)
        return real(chol,rhs,**kwargs)
    monkeypatch.setattr(gp,'cho_solve',track)
    rng=np.random.default_rng(25)
    n=24
    design=np.column_stack([np.ones(n),rng.normal(size=n)])
    covariance=.3*np.ones((n,n))+.7*np.eye(n)
    gp._profiled_gls_components(cho_factor(covariance),design,rng.normal(size=n))
    assert calls==[(n,3)]


def test_nearly_exact_large_mean_uses_direct_residual_fallback(monkeypatch):
    calls=[]
    real=gp.cho_solve
    def track(chol,rhs,**kwargs):
        calls.append(rhs.shape)
        return real(chol,rhs,**kwargs)
    monkeypatch.setattr(gp,'cho_solve',track)
    rng=np.random.default_rng(37)
    n=32
    design=np.column_stack([np.ones(n),np.linspace(-1,1,n)])
    covariance=.3*np.ones((n,n))+.7*np.eye(n)
    response=design @ np.array([1e8,1e6])+rng.normal(scale=1e-5,size=n)
    chol=cho_factor(covariance)
    beta,residual,alpha=gp._profiled_gls_components(chol,design,response)
    assert calls==[(n,3),(n,)]
    np.testing.assert_array_equal(alpha,cho_solve(chol,residual,check_finite=False))
    assert np.isfinite(beta).all()


@pytest.mark.parametrize('kernel,nu', [('RBF',None),('Matern',.5),('Matern',1.5),('Matern',2.5)])
@pytest.mark.parametrize('columns', [0,2])
def test_actual_optimized_fit_matches_original_solve_path(monkeypatch,kernel,nu,columns):
    rng=np.random.default_rng(20260929)
    xy=rng.uniform([0.,0.],[180.,120.],size=(64,2))
    covariates=rng.normal(size=(64,columns))
    response=25.+.8*np.sin(xy[:,0]/35.)+.3*np.cos(xy[:,1]/40.)+rng.normal(scale=.2,size=64)
    if columns:
        response+=covariates @ np.array([.7,-.4])
    kwargs=dict(predictor_names=[f'x{i}' for i in range(columns)],kernel_name=kernel,nu=nu,
                n_restarts=0,seed=20260929,length_scale_bounds=(1.,6000.),initial_length_scale=300.)
    with threadpool_limits(limits=1):
        optimized=gp.fit_gp_with_covariates(xy[:48],covariates[:48],response[:48],**kwargs)
        monkeypatch.setattr(gp,'_profiled_gls_components',original_gls)
        original=gp.fit_gp_with_covariates(xy[:48],covariates[:48],response[:48],**kwargs)
        actual=gp.predict_gp_with_covariates(optimized,xy[48:],covariates[48:])
        expected=gp.predict_gp_with_covariates(original,xy[48:],covariates[48:])
    assert optimized['optimizer_success'] and original['optimizer_success']
    np.testing.assert_allclose(optimized['log_likelihood'],original['log_likelihood'],rtol=1e-6,atol=1e-5)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=2e-4)
