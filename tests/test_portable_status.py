"""Read-only progress helper: saved state is not a live process check."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/thesis_run_bundle/status.ps1"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(not POWERSHELL, reason="PowerShell is unavailable")


def _write(path, value, timestamp):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    os.utime(path, (timestamp, timestamp))


def _snapshot(root):
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*") if p.is_file()
    }


def _invoke(root, *extra):
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(SCRIPT), "-Root", str(root), *extra],
        capture_output=True, text=True, encoding="utf-8", timeout=40,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_saved_progress_completion_and_latest_log_are_read_only(tmp_path):
    root = tmp_path / "portable bundle"
    old = root / "outputs_wsl/eas/attempt_old"
    current = root / "outputs_wsl/simulations/experiment/calibration"
    _write(old / "run_status.json", {"status": "completed_with_warnings", "workflow_completed": True}, 1000)
    _write(old / "run_report.json", {
        "status": "completed_with_warnings", "workflow_completed": True, "n_grids": 7822,
        "timings": {"total_from_nn_start_until_outputs_seconds": 93789.1},
    }, 1000)
    _write(current / "run_status.json", {
        "status": "running", "stage": "full_data_calibration", "coefficient": "beta_mu",
        "completed_coefficients": 1, "total_coefficients": 5,
    }, 2000)
    _write(root / "outputs_wsl/preprocessed/run_report.json", {"status": "completed"}, 3000)
    _write(root / "outputs_wsl/false_completion/run_report.json", {
        "status": "failed", "workflow_completed": True,
    }, 3000)
    logs = root / "outputs_wsl/logs"
    logs.mkdir()
    (logs / "old.log").write_text("old log", encoding="utf-8")
    os.utime(logs / "old.log", (1000, 1000))
    (logs / "new.log").write_text("\n".join(f"line {i}" for i in range(30)), encoding="utf-8")
    before = _snapshot(root)

    result = json.loads(_invoke(root, "-Json"))

    assert _snapshot(root) == before
    assert result["live_processes_checked"] is False
    assert result["live_wsl"] is None
    assert "NOT live" in result["notice"] and "stale" in result["notice"]
    wsl = result["surfaces"][0]
    assert Path(wsl["runs"][0]["path"]) == current / "run_status.json"
    assert wsl["runs"][0]["completion_confirmed_by_report"] is False
    assert wsl["runs"][0]["completed_coefficients"] == 1
    assert wsl["runs"][1]["completion_confirmed_by_report"] is True
    assert len(wsl["completed_reports"]) == 1
    assert wsl["completed_reports"][0]["elapsed_seconds"] == 93789.1
    assert wsl["latest_log"]["tail"] == [f"line {i}" for i in range(5, 30)]
    assert result["surfaces"][1]["exists"] is False


def test_malformed_json_does_not_hide_other_statuses(tmp_path):
    bad = tmp_path / "outputs/eas/latest/run_status.json"
    _write(bad, {}, 2000)
    bad.write_text("{broken json", encoding="utf-8")
    _write(tmp_path / "outputs/eas/older/run_status.json", {
        "status": "running", "outer_fold": 2, "completed_cases": 10, "total_cases": 25,
    }, 1000)
    before = _snapshot(tmp_path)
    result = json.loads(_invoke(tmp_path, "-Json"))
    runs = result["surfaces"][1]["runs"]
    assert "Cannot read JSON" in runs[0]["error"]
    assert runs[1]["outer_fold"] == 2
    assert runs[1]["completed_cases"] == 10
    assert result["surfaces"][1]["completed_reports"] == []
    assert _snapshot(tmp_path) == before
    text = _invoke(tmp_path)
    assert "Saved statuses are NOT live process checks" in text
    assert "Cannot read JSON" in text


def test_script_is_ascii_and_default_root_is_bundle_location():
    source = SCRIPT.read_text(encoding="ascii")
    assert "[string]$Root = $PSScriptRoot" in source
    assert "if ($LiveWSL)" in source
    assert "can exceed 100 per process" in source
