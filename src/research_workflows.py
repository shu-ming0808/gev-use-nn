"""Read-only catalogue and input contract for the two research workspaces.

Stage JSON files describe available tools and planned work; they are NOT an
executor and their presence does not assert that a scientific stage is ready.
Scientific implementations and hash-locked training sources remain shared.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "cheng_21d_annual_1976_2025_v1"
YEARS = tuple(range(1976, 2026))
REFERENCE_YEAR = 2000.5
TIME_SCALE_YEARS = 10.0
COEFFICIENTS = ("mu0", "beta_mu", "eta0", "beta_sigma", "xi0")
# Active slopes, not a fitted classifier or a hypothesis-test implementation.
TIME_STRUCTURES = {
    "M0": (),
    "M_mu": ("beta_mu",),
    "M_sigma": ("beta_sigma",),
    "M_mu_sigma": ("beta_mu", "beta_sigma"),
}
STAGES = {
    "simulation": ("nn_training", "taiwan_validation"),
    "global": (
        "preparation", "nn_inference", "temporal_selection", "spatial_cv",
        "return_levels",
    ),
}


def repository_path(relative: str, root: Path = ROOT) -> Path:
    """Resolve a repository reference, rejecting absolute or escaping paths."""
    root = Path(root).resolve()
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError("Repository references must be relative paths")
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Repository reference escapes the project")
    return resolved


def load_stage(workspace: str, stage: str, root: Path = ROOT) -> dict:
    if workspace not in STAGES or stage not in STAGES[workspace]:
        raise ValueError(f"Unknown workspace/stage: {workspace}/{stage}")
    path = repository_path(f"{workspace}/{stage}/stage.json", root)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("contract") != CONTRACT:
        raise ValueError(f"Unsupported input contract in {path}")
    if config.get("status") not in {"available_tools", "planned"}:
        raise ValueError(f"Unsupported stage status in {path}")
    if "time_structures" in config and config["time_structures"] != list(TIME_STRUCTURES):
        raise ValueError(f"Unexpected temporal candidates in {path}")
    for ref in config.get("shared_sources", []):
        if not repository_path(ref, root).is_file():
            raise FileNotFoundError(f"Missing shared source: {ref}")
    for key in ("output_directory", "checkpoint", "existing_data_directory"):
        if key in config:
            repository_path(config[key], root)
    for key in ("notebook", "diagnostics_notebook"):
        if key in config and not repository_path(config[key], root).is_file():
            raise FileNotFoundError(f"Missing notebook: {config[key]}")
    return config


def validate_annual_sequences(years, values) -> np.ndarray:
    """Validate complete, ordered Celsius sequences before future 21D inference.

    Values must be grid x year. Do not fill, reorder, pad, interpolate or accept
    a 45-year Taiwan series silently. The caller must convert Kelvin explicitly.
    This contract does not certify hourly source completeness or geographic
    selection; those checks belong to ERA5 preparation.
    """
    year_array = np.asarray(years)
    if year_array.shape != (50,) or not np.array_equal(year_array, YEARS):
        raise ValueError("Expected exactly the ordered years 1976--2025 (50 years)")
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 50 or array.shape[0] == 0:
        raise ValueError("Annual maxima must have shape (n_grid > 0, 50)")
    if not np.isfinite(array).all():
        raise ValueError("Missing/nonfinite annual maxima; do not impute silently")
    q25, q75 = np.quantile(array, [0.25, 0.75], axis=1)
    if np.any(q75 - q25 <= 1e-12):
        raise ValueError("Every sequence must have a positive IQR")
    return array


def describe_workspace(workspace: str, root: Path = ROOT) -> dict:
    """Return only configuration status. No data, model or network execution."""
    if workspace not in STAGES:
        raise ValueError(f"Unknown workspace: {workspace}")
    return {
        "workspace": workspace,
        "contract": CONTRACT,
        "period": [YEARS[0], YEARS[-1]],
        "coefficient_names": list(COEFFICIENTS),
        "note": "Configuration inventory only; not a data-completeness or pipeline-readiness audit.",
        "stages": [
            {"stage": stage, **load_stage(workspace, stage, root)}
            for stage in STAGES[workspace]
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, choices=tuple(STAGES))
    args = parser.parse_args(argv)
    print(json.dumps(describe_workspace(args.workspace), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
