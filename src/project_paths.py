"""Canonical repository paths shared by scripts and notebooks.

All generated tables, figures, histories, and spatial predictors should be
resolved through this module instead of rebuilding paths from the current
working directory.
"""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPOSITORY_ROOT / "data"
ORIGINAL_DATA_DIR = DATA_DIR / "original_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SIMULATED_DATA_DIR = DATA_DIR / "simulated"
SHAPEFILE_DIR = DATA_DIR / "shapefile"

SPATIAL_PREDICTOR_DIR = DATA_DIR / "spatial_predictors"
SPATIAL_PREDICTOR_RAW_DIR = SPATIAL_PREDICTOR_DIR / "raw"
SPATIAL_PREDICTOR_PROCESSED_DIR = SPATIAL_PREDICTOR_DIR / "processed"

MODEL_DIR = REPOSITORY_ROOT / "models"
NOTEBOOK_DIR = REPOSITORY_ROOT / "notebooks"
REPORT_DIR = REPOSITORY_ROOT / "reports"
RESULT_DIR = REPOSITORY_ROOT / "results"
FIGURE_DIR = RESULT_DIR / "figures"
TABLE_DIR = RESULT_DIR / "tables"
HISTORY_DIR = RESULT_DIR / "histories"


def ensure_project_directories() -> None:
    """Create the standard writable directories used by the project."""
    for directory in (
        PROCESSED_DATA_DIR,
        SPATIAL_PREDICTOR_RAW_DIR,
        SPATIAL_PREDICTOR_PROCESSED_DIR,
        MODEL_DIR,
        REPORT_DIR,
        FIGURE_DIR,
        TABLE_DIR,
        HISTORY_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
