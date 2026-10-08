"""Workspace contracts: no real datasets, training, bootstrap or downloads."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import research_workflows as w


@pytest.mark.parametrize("workspace", ["simulation", "global"])
def test_catalogue_references_exist_and_no_execution(tmp_path, monkeypatch, workspace):
    monkeypatch.chdir(tmp_path)
    report = w.describe_workspace(workspace)
    assert report["period"] == [1976, 2025]
    assert report["coefficient_names"] == list(w.COEFFICIENTS)
    assert len(report["stages"]) == len(w.STAGES[workspace])
    assert list(tmp_path.iterdir()) == []


def test_only_tools_not_unimplemented_pipeline_claimed_ready():
    for workspace, stages in w.STAGES.items():
        for stage in stages:
            actual = w.load_stage(workspace, stage)["status"]
            expected = "available_tools" if stage in {"nn_training", "preparation"} else "planned"
            assert actual == expected


def test_same_final_checkpoint_for_simulation_and_real_data():
    training = w.load_stage("simulation", "nn_training")
    inference = w.load_stage("global", "nn_inference")
    assert training["checkpoint"] == inference["checkpoint"]
    assert len(training["checkpoint_sha256"]) == 64
    assert w.load_stage("simulation", "taiwan_validation")["coordinate_crs"] == "EPSG:3826"


def test_four_structures_use_only_two_slopes():
    assert {key: 3 + len(slopes) for key, slopes in w.TIME_STRUCTURES.items()} == {
        "M0": 3, "M_mu": 4, "M_sigma": 4, "M_mu_sigma": 5,
    }


def test_valid_annual_contract_preserves_grid_values():
    values = np.arange(150, dtype=float).reshape(3, 50)
    np.testing.assert_array_equal(w.validate_annual_sequences(w.YEARS, values), values)


@pytest.mark.parametrize("years", [list(range(1980, 2025)), list(range(1975, 2025)),
                                   list(range(2025, 1975, -1)), [1976] * 50])
def test_reject_old_length_shifted_reordered_or_duplicate_years(years):
    with pytest.raises(ValueError, match="1976--2025"):
        w.validate_annual_sequences(years, np.arange(len(years))[None, :])


@pytest.mark.parametrize("values", [np.ones((2, 50)), np.zeros((0, 50)),
                                    np.arange(50), np.ones((50, 2)),
                                    np.full((2, 50), np.nan)])
def test_reject_bad_input_shape_missing_data_or_constant_sequences(values):
    with pytest.raises(ValueError):
        w.validate_annual_sequences(w.YEARS, values)


@pytest.mark.parametrize("path", ["../outside", "simulation/../../outside"])
def test_references_cannot_escape_project(tmp_path, path):
    with pytest.raises(ValueError):
        w.repository_path(path, tmp_path)


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError):
        w.load_stage("global", "training")


def test_cli_reports_inventory_only(capsys):
    assert w.main(["--workspace", "global"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "not a data-completeness" in report["note"]


@pytest.mark.parametrize("path,marker,variable", [
    ("global/preparation/global_region_partition.ipynb", "src/global_climate_regions.py", "ROOT"),
    ("simulation/nn_training/cheng_NN_diagnostics.ipynb", "src/cheng_nn_simulation.py", "PROJECT_ROOT"),
])
def test_moved_notebook_root_lookup_from_nested_directory(tmp_path, monkeypatch, path, marker, variable):
    project = tmp_path / "project"
    (project / marker).parent.mkdir(parents=True)
    (project / marker).touch()
    location = project / Path(path).parent
    location.mkdir(parents=True)
    monkeypatch.chdir(location)
    nb = json.loads((ROOT / path).read_text(encoding="utf-8"))
    source = "".join(next(cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code"))
    # Execute only root discovery, never later imports, data loading or prepare().
    start = source.index("CURRENT =")
    end = source.index("if str(", start)
    namespace = {"Path": Path}
    exec(compile(source[start:end], path, "exec"), namespace)
    assert namespace[variable] == project


def test_all_workflow_notebook_cells_are_valid_python():
    for folder in ("simulation", "global"):
        for path in (ROOT / folder).rglob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), f"{path}:{index}", "exec")
