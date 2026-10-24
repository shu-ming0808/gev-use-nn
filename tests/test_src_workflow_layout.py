"""Regression checks for workflow folders without real data or network jobs."""
import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
MODULES = [
    "taiwan_simulation.prepare_taiwan_cheng",
    "taiwan_simulation.cheng_spatial_cv",
    "taiwan_simulation.prepare_daily_tmax_block_maxima",
    "taiwan_simulation.export_spatial_selection_figures",
    "taiwan_simulation.plot_selected_oof_parameter_maps",
    "taiwan_simulation.plot_variograms",
    "taiwan_simulation.k_sensitivity_experiment",
    "taiwan_simulation.return_level_sensitivity",
    "global_pipeline.infer_global_cheng_nn",
    "global_pipeline.prepare_era5_annual_maxima",
    "global_pipeline.global_climate_regions",
    "global_pipeline.compare_global_cheng_bound",
    "parameters.download_global_predictors",
    "parameters.atmospheric_predictors",
    "parameters.coast_distance_predictor",
    "parameters.land_cover_predictors",
    "parameters.rainfall_predictors",
    "parameters.terrain_predictors"
]


@pytest.mark.parametrize("name", MODULES)
def test_moved_module_import_and_source_location(name):
    module = importlib.import_module(name)
    expected = SOURCE_ROOT / (name.replace(".", "/") + ".py")
    assert Path(module.__file__).resolve() == expected.resolve()
    assert module._SOURCE_ROOT == SOURCE_ROOT
    assert not (SOURCE_ROOT / expected.name).exists()
    ast.parse(expected.read_text(encoding="utf-8"), filename=str(expected))


def test_nested_workflows_still_resolve_original_repository():
    from global_pipeline import global_climate_regions
    from taiwan_simulation import k_sensitivity_experiment, plot_variograms
    from taiwan_simulation import prepare_daily_tmax_block_maxima
    assert global_climate_regions.ROOT == ROOT
    assert k_sensitivity_experiment.PROJECT_ROOT == ROOT
    assert Path(plot_variograms.PROJECT_ROOT) == ROOT
    assert prepare_daily_tmax_block_maxima.PROJECT_SRC == SOURCE_ROOT


@pytest.mark.parametrize("script", [
    "taiwan_simulation/prepare_taiwan_cheng.py",
    "taiwan_simulation/cheng_spatial_cv.py",
    "global_pipeline/infer_global_cheng_nn.py",
    "global_pipeline/global_climate_regions.py",
    "parameters/download_global_predictors.py",
])
def test_direct_cli_help_from_unrelated_directory(tmp_path, script):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-B", str(SOURCE_ROOT / script), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert list(tmp_path.iterdir()) == []

