import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import benchmark_cheng_bound_timing as timing


def test_scaling_recomputed_from_observations():
    y=np.arange(100,dtype=float).reshape(2,50)
    scaling=timing.sample_scaling(y)
    np.testing.assert_array_equal(scaling,[[24.5,24.5],[74.5,24.5]])
    np.testing.assert_allclose(timing.make_features(y,scaling,21)[0],
                               timing.make_features(y,scaling,21)[1])


@pytest.mark.parametrize('values',[np.ones((2,50)),np.ones((2,49)),
                                  np.full((2,50),np.nan)])
def test_reject_invalid_raw_sequences(values):
    with pytest.raises(ValueError):
        timing.sample_scaling(values)


def test_nn_scope_includes_scaling_features_decoder_and_inverse(monkeypatch):
    y=np.arange(100,dtype=float).reshape(2,50)
    t=np.linspace(-2.45,2.45,50)
    seen=[]
    def predict(model,features,scaler,z,time,device,constrained,**kwargs):
        seen.append('decode')
        assert features.shape==(2,21) and z.shape==(2,50)
        assert constrained and device=='cpu'
        np.testing.assert_allclose(z[0],z[1])
        return np.zeros((2,5))
    def inverse(c,scaling):
        seen.append('inverse')
        np.testing.assert_array_equal(scaling,timing.sample_scaling(y))
        return c+1
    monkeypatch.setattr(timing,'predict_sample_coefficients',predict)
    assert timing.nn_estimate(y,t,None,None,inverse,.001).shape==(2,5)
    assert seen==['decode','inverse']


def test_summary_uses_median_and_amortized_not_single_request_latency():
    rows=[dict(method=m,seconds=s) for m,values in
          [('MLE',[20,30,100]),('constrained_NN',[.1,.2,.3])] for s in values]
    result=timing.summarize_times(rows,10000)
    assert result['speedup_ratio']==150
    assert result['methods']['MLE']['median_seconds']==30
    assert result['methods']['constrained_NN']['amortized_ms_per_sequence']==.02
