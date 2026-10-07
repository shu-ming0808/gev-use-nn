from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from compare_cheng_nn_mle_models import compare_errors, holm_adjust, beamer_source
import compare_cheng_nn_mle_models as comparison


def test_holm_and_invalid_values():
    np.testing.assert_allclose(holm_adjust([.01, .04, .03]), [.03, .06, .06])
    with pytest.raises(ValueError):
        holm_adjust([float('nan')])


def test_paired_comparison_percentage_and_null():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(100, 5))
    errors = {'MLE': 2*base, 'Small': base, 'Large': .5*base}
    result = compare_errors(errors, repeats=200)
    np.testing.assert_allclose(result.Small_improvement_percent, 50.)
    np.testing.assert_allclose(result.Large_improvement_percent, 75.)
    assert result.Large_significantly_better.all()
    assert (result.Small_minus_Large_RMSE_CI_low > 0).all()
    assert 'RL' not in beamer_source(result).split(r'\begin{frame}')[1]
    errors['Large'] = errors['Small'].copy()
    result = compare_errors(errors, repeats=100)
    assert (result.p_Holm == 1).all() and not result.Large_significantly_better.any()


def test_no_silent_nonfinite_exclusions():
    values = np.ones((10, 5))
    bad = values.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='silently'):
        compare_errors({'MLE': values, 'Small': values, 'Large': bad})


def test_boxplots_have_five_shared_scale_rows_and_three_model_columns(tmp_path, monkeypatch):
    values = np.random.default_rng(17).normal(size=(30, 5))
    errors = {'MLE': values, 'Small': 2*values, 'Large': 3*values}
    captured = []
    original_close = comparison.plt.close
    monkeypatch.setattr(comparison.plt, 'close', lambda fig: captured.append(fig))
    try:
        path = comparison.make_figure(errors, tmp_path)
        assert path.name == 'coefficient_error_boxplots_5x3.png'
        assert path.is_file()
        fig = captured[-1]
        assert len(fig.axes) == 15
        assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (5, 3)
        for j in range(5):
            assert fig.axes[3*j].get_ylim() == fig.axes[3*j+1].get_ylim() == fig.axes[3*j+2].get_ylim()
        assert all(len(ax.patches) == 1 for ax in fig.axes)
        assert [ax.get_title() for ax in fig.axes[:3]] == ['MLE\n(constrained)', 'Small NN', 'Large NN']
        assert all(comparison.SYMBOLS[j] in fig.axes[3*j].get_ylabel() for j in range(5))
    finally:
        for fig in captured:
            original_close(fig)
