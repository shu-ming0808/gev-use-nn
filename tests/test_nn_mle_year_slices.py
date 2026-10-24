"""Scientific transformation, fair color scales, and 5 x (3-by-5) figure checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg',force=True)
import matplotlib.pyplot as plt

from taiwan_simulation import plot_nn_mle_year_slices as plots


def sample_pair():
    x,y=np.meshgrid(np.linspace(-700,700,8),np.linspace(-900,900,7))
    x=x.ravel();y=y.ravel()
    truth=np.column_stack([25.+x/700+y/900,.2+x/5000,
        np.log(1.5)+y/8000,.02+x/50000,-.15+y/10000])
    return dict(grids=pd.DataFrame({'grid_id':[f'X{i}' for i in range(len(x))],
        'x_km':x,'y_km':y}),
        coefficients=dict(truth=truth,mle=truth+np.array([.1,.02,.03,.005,.01]),
                          nn=truth+np.array([-.12,-.01,-.02,-.003,-.008])),
        metadata=dict(stage='estimation',replicate=1,coordinate_crs='illustrative projected km'))


def test_evaluation_years_are_not_refits_and_log_scale_is_not_scale():
    theta=np.array([[30.,2.,np.log(3.),.1,-.2]])
    v,years=plots.parameters_at_years(theta)
    np.testing.assert_array_equal(years,[1985,1995,2005,2015,2025])
    t=np.array([-1.55,-.55,.45,1.45,2.45])
    np.testing.assert_allclose(v[0,:,0],30.+2*t)
    np.testing.assert_allclose(v[0,:,2],np.log(3.)+.1*t)
    assert v.shape==(1,5,5)
    for i in (1,3,4): np.testing.assert_array_equal(v[0,:,i],np.repeat(theta[0,i],5))
    np.testing.assert_array_equal(theta,[[30.,2.,np.log(3.),.1,-.2]])


@pytest.mark.parametrize('numbers',[(0,20,30,40,50),(10,20,30,40,51),(10,10,30,40,50),
    (10.,20.,30.,40.,50.),(10,20,30)])
def test_reject_invalid_years(numbers):
    with pytest.raises(ValueError):plots.parameters_at_years(np.ones((3,5)),numbers)


def test_never_silently_remove_nonfinite_grid():
    theta=np.ones((3,5));theta[0,2]=np.nan
    with pytest.raises(ValueError):plots.parameters_at_years(theta)


def test_five_figures_fifteen_maps_each_common_limits_and_correct_rmse(tmp_path):
    pair=sample_pair();figures,metrics,metadata=plots.make_figures(pair)
    try:
        assert len(figures)==5 and len(metrics)==50
        assert metadata['layout']=='3 rows x 5 columns per quantity'
        assert metadata['time_constant_quantities']==['beta_mu','beta_sigma','xi0']
        for q,(name,figure) in enumerate(figures.items()):
            assert len(figure.axes)==16  # 15 maps + shared colorbar
            maps=figure.axes[:15]
            assert len({ax.collections[0].get_clim() for ax in maps})==1
            assert len({ax.get_xlim() for ax in maps})==1
            assert len({ax.get_ylim() for ax in maps})==1
            assert all(len(ax.collections[0].get_array())==56 for ax in maps)
            if q in (1,3,4):
                for r in range(3):
                    for col in range(1,5):
                        np.testing.assert_array_equal(maps[5*r].collections[0].get_array(),
                                                      maps[5*r+col].collections[0].get_array())
            figure.canvas.draw()
        mu=metrics[(metrics.quantity=='mu_t')&(metrics.method=='mle')]
        np.testing.assert_allclose(mu.RMSE,np.abs(.1+.02*np.array([-1.55,-.55,.45,1.45,2.45])))
        eta=metrics[(metrics.quantity=='eta_t')&(metrics.method=='mle')]
        np.testing.assert_allclose(eta.RMSE,np.abs(.03+.005*np.array([-1.55,-.55,.45,1.45,2.45])))
        report=plots.save_figures(figures,metrics,metadata,tmp_path/'figures',dpi=60)
        assert len(list((tmp_path/'figures').glob('*.png')))==5
        assert len(list((tmp_path/'figures').glob('*.svg')))==5
        for name,digest in report['output_sha256'].items():assert plots._sha(tmp_path/'figures'/name)==digest
        assert json.loads((tmp_path/'figures/figure_report.json').read_text())['full_50_year_fit']
    finally:
        for figure in figures.values():plt.close(figure)


def test_relocate_recorded_drive_and_reject_other_replicate_or_traversal(tmp_path):
    expected=tmp_path/'replicates/replicate_001/nn/attempt_002'
    assert plots._attempt(tmp_path.resolve(),r'F:\論文跑\outputs\replicates\replicate_001\nn\attempt_002',1,'nn')==expected
    with pytest.raises(ValueError):plots._attempt(tmp_path.resolve(),'F:/other/attempt_002',1,'nn')
    with pytest.raises(ValueError):plots._attempt(tmp_path.resolve(),'F:/replicates/replicate_002/nn/attempt_002',1,'nn')
    with pytest.raises(ValueError):plots._attempt(tmp_path.resolve(),'F:/replicates/replicate_001/nn/../escape',1,'nn')


def test_gp_stage_is_explicitly_labelled():
    pair=sample_pair();pair['metadata']['stage']='gp_oof'
    figures,_,_=plots.make_figures(pair)
    try:
        labels=[ax.get_ylabel() for ax in figures['mu_t'].axes]
        assert any('MLE + GP OOF' in label for label in labels)
        assert any('BOUND NN + GP OOF' in label for label in labels)
    finally:
        for figure in figures.values():plt.close(figure)
