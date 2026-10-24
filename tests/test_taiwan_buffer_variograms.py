"""Scientific/numerical regression tests; no actual NN or spatial CV run."""
import numpy as np
import pandas as pd
import pytest

from taiwan_simulation import buffer_variograms as v


def grid():
    x, y = np.meshgrid(np.arange(5)*25., np.arange(10)*25.)
    return np.column_stack([x.ravel(), y.ravel()])


def test_effective_range_definitions():
    for model, (fn, factor) in v.MODELS.items():
        expected = 3. if model == 'spherical' else 2.9
        assert fn(np.array([factor*11.]), 1., 2., 11.)[0] == pytest.approx(expected)


def test_five_targets_and_exact_pair_accounting():
    xy = grid()
    rng = np.random.default_rng(4)
    names = ['mu0','beta_mu','eta0','beta_sigma','xi0']
    frame = pd.DataFrame(rng.normal(size=(50,5)), columns=names)
    result = v.diagnose_variograms(xy, frame)
    assert result['summary'].coefficient.tolist() == names
    assert len(result['fits']) == 15
    assert len(result['bins']) == 40
    ii, jj = np.triu_indices(50,1)
    distance = np.linalg.norm(xy[ii]-xy[jj],axis=1)
    expected = int((distance <= .6*distance.max()).sum())
    assert result['bins'].groupby('coefficient').n_pairs.sum().eq(expected).all()
    assert result['fits'].fit_success.any()
    assert result['summary'].n_points.eq(50).all()


def test_normalized_fitting_is_equivariant_to_scale_and_location():
    xy = grid()
    values = np.random.default_rng(6).normal(size=50)
    a = v.diagnose_variograms(xy,pd.DataFrame({'a':values}))
    b = v.diagnose_variograms(xy,pd.DataFrame({'b':100.+values*.01}))
    # A near-pure-nugget random surface does not identify a unique range. Its
    # fit error must transform correctly, but unsupported fitted ranges need
    # not be numerically identical under minute roundoff perturbations.
    np.testing.assert_allclose(a['fits'].weighted_rmse*.0001,b['fits'].weighted_rmse,rtol=1e-5)


def test_constant_surface_never_supplies_artificial_range():
    result = v.diagnose_variograms(grid(),pd.DataFrame({'mu0':np.repeat(30.,50)}))
    assert result['summary'].status.tolist() == ['constant_surface']
    assert not result['fits'].fit_success.any()
    assert result['fits'].effective_range_km.isna().all()
    assert not result['summary'].range_supported.any()


def test_small_pair_support_is_flagged_not_silently_fitted():
    result = v.diagnose_variograms(grid(),pd.DataFrame({'a':np.arange(50.)}),min_pairs=500)
    assert result['summary'].status.tolist() == ['insufficient_pair_bins']
    assert not result['bins'].fit_used.any()


def test_linear_spatial_drift_does_not_claim_supported_finite_range():
    xy = grid()
    result = v.diagnose_variograms(xy,pd.DataFrame({'mu0':xy[:,1]}))
    assert result['fits'].fit_success.all()
    best = result['summary'].iloc[0]
    assert not best.range_supported
    assert 'range_beyond_observed_lags' in best.warning or 'no_empirical_sill_support' in best.warning


@pytest.mark.parametrize('kind',['nan_values','nan_xy','duplicate_xy','bad_count','bad_names'])
def test_invalid_data_is_rejected(kind):
    xy=grid(); frame=pd.DataFrame({'a':np.arange(50.)})
    if kind=='nan_values': frame.iloc[2,0]=np.nan
    if kind=='nan_xy': xy[0,0]=np.nan
    if kind=='duplicate_xy': xy[1]=xy[0]
    if kind=='bad_count': frame=frame.iloc[:-1]
    if kind=='bad_names': frame.columns=[0]
    with pytest.raises(ValueError): v.diagnose_variograms(xy,frame)


@pytest.mark.parametrize('kwargs',[{'n_bins':3},{'n_bins':4.5},{'min_pairs':0},
    {'max_distance_fraction':0},{'max_distance_fraction':1.1},{'max_distance_fraction':np.nan}])
def test_invalid_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        v.diagnose_variograms(grid(),pd.DataFrame({'a':np.arange(50.)}),**kwargs)


def test_optimizer_failure_is_reported(monkeypatch):
    def failed(*args,**kwargs): raise RuntimeError('synthetic optimizer failure')
    monkeypatch.setattr(v,'curve_fit',failed)
    result=v.diagnose_variograms(grid(),pd.DataFrame({'a':np.arange(50.)}))
    assert result['summary'].status.tolist()==['fit_failed']
    assert not result['fits'].fit_success.any()
    assert result['fits'].warning.str.contains('all_optimizer_starts_failed').all()


def test_subgrid_range_is_explicitly_unresolved(monkeypatch):
    def short(*args,**kwargs): return np.array([0.,1.,5.]),np.eye(3)
    monkeypatch.setattr(v,'curve_fit',short)
    result=v.diagnose_variograms(grid(),pd.DataFrame({'a':np.arange(50.)}))
    assert result['fits'].fit_success.all()
    assert result['fits'].warning.str.contains('unresolved_short_range').all()
    assert not result['fits'].range_supported.any()
