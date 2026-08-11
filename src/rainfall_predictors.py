"""Construct rainfall predictors aligned with the TCCIP temperature GRID.

Two predictor families are kept separate:

1. outcome-independent rainfall climatology, calculated from every valid day;
2. event-conditioned rainfall, looked up at each GRID's monthly Tmax date.

The second family follows the requested event-matching definition, but it must
only be used when monthly-Tmax occurrence dates are available at prediction
time.  Otherwise it would leak response-derived timing into validation.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_daily_tmax_block_maxima import read_one_daily_file
from project_paths import (
    ORIGINAL_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPATIAL_PREDICTOR_PROCESSED_DIR,
    ensure_project_directories,
)


DEFAULT_RAINFALL_DIR = ORIGINAL_DATA_DIR / "觀測_日資料_臺灣_降雨量"
DEFAULT_EVENT_PATH = (
    PROCESSED_DATA_DIR / "daily_tmax_monthly_max_occurrences.csv"
)
DEFAULT_EVENT_OUTPUT_PATH = (
    SPATIAL_PREDICTOR_PROCESSED_DIR
    / "tccip_grid_tmax_event_day_rainfall.csv"
)
DEFAULT_PREDICTOR_OUTPUT_PATH = (
    SPATIAL_PREDICTOR_PROCESSED_DIR
    / "tccip_grid_rainfall_predictors.csv"
)


def _year_from_path(path: Path) -> int:
    match = re.search(r"(\d{4})\.csv$", path.name)
    if match is None:
        raise ValueError(f"Cannot infer year from {path.name}")
    return int(match.group(1))


def _lookup_event_rainfall(
    events: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Look up one rainfall value per GRID/month without expanding all days."""
    result = events.copy()
    result["max_date"] = pd.to_datetime(result["max_date"], errors="coerce")
    station_position = daily.index.get_indexer(result["station"])
    date_position = daily.columns.get_indexer(result["max_date"])
    valid_key = (station_position >= 0) & (date_position >= 0)
    values = np.full(len(result), np.nan, dtype=np.float64)
    values[valid_key] = daily.to_numpy()[
        station_position[valid_key],
        date_position[valid_key],
    ]
    result["rainfall_on_tmax_date_mm"] = values
    result["rainfall_key_matched"] = valid_key
    return result


def build_rainfall_predictors(
    rainfall_directory: str | Path = DEFAULT_RAINFALL_DIR,
    event_path: str | Path = DEFAULT_EVENT_PATH,
    event_output_path: str | Path = DEFAULT_EVENT_OUTPUT_PATH,
    predictor_output_path: str | Path = DEFAULT_PREDICTOR_OUTPUT_PATH,
    start_year: int = 1980,
    wet_day_threshold_mm: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match event-day rainfall and aggregate fixed GRID-level predictors."""
    ensure_project_directories()
    rainfall_directory = Path(rainfall_directory)
    files = sorted(rainfall_directory.glob("*.csv"))
    files = [path for path in files if _year_from_path(path) >= start_year]
    if not files:
        raise FileNotFoundError(
            f"No rainfall files from {start_year} under {rainfall_directory}"
        )

    events = pd.read_csv(
        event_path,
        usecols=[
            "station",
            "year",
            "month",
            "monthly_max_tmax_c",
            "max_date",
            "n_tied_max_dates",
        ],
    )
    events = events.loc[
        (events["year"] >= start_year) & events["max_date"].notna()
    ].copy()

    event_parts: list[pd.DataFrame] = []
    climatology_parts: list[pd.DataFrame] = []
    for path in files:
        year = _year_from_path(path)
        _, daily = read_one_daily_file(path)
        event_year = events.loc[events["year"] == year]
        if not event_year.empty:
            matched = _lookup_event_rainfall(event_year, daily)
            matched["rainfall_source_file"] = path.name
            event_parts.append(matched)

        valid_days = daily.notna().sum(axis=1)
        annual_total = daily.sum(axis=1, min_count=1)
        wet_days = daily.ge(wet_day_threshold_mm).sum(axis=1)
        climatology_parts.append(
            pd.DataFrame(
                {
                    "station": daily.index,
                    "year": year,
                    "annual_precip_mm": annual_total.to_numpy(),
                    "valid_rain_days": valid_days.to_numpy(),
                    "wet_rain_days": wet_days.to_numpy(),
                }
            )
        )
        print(f"Rainfall {year}: event dates matched and annual summaries built")

    event_table = pd.concat(event_parts, ignore_index=True)
    climatology = pd.concat(climatology_parts, ignore_index=True)
    event_summary = (
        event_table.groupby("station", as_index=False)
        .agg(
            tmax_event_months=("max_date", "size"),
            tmax_event_rain_available=(
                "rainfall_on_tmax_date_mm",
                "count",
            ),
            tmax_event_rain_mean_mm=(
                "rainfall_on_tmax_date_mm",
                "mean",
            ),
            tmax_event_rain_p90_mm=(
                "rainfall_on_tmax_date_mm",
                lambda values: values.quantile(0.90),
            ),
            tmax_event_rain_wet_ratio=(
                "rainfall_on_tmax_date_mm",
                lambda values: values.ge(wet_day_threshold_mm).mean(),
            ),
        )
    )
    event_summary["tmax_event_rain_available_ratio"] = (
        event_summary["tmax_event_rain_available"]
        / event_summary["tmax_event_months"]
    )

    climatology_summary = (
        climatology.groupby("station", as_index=False)
        .agg(
            rainfall_years=("annual_precip_mm", "count"),
            mean_annual_precip_mm=("annual_precip_mm", "mean"),
            total_valid_rain_days=("valid_rain_days", "sum"),
            total_wet_rain_days=("wet_rain_days", "sum"),
        )
    )
    climatology_summary["rain_wet_day_ratio"] = (
        climatology_summary["total_wet_rain_days"]
        / climatology_summary["total_valid_rain_days"]
    )
    predictors = climatology_summary.merge(
        event_summary,
        on="station",
        how="outer",
        validate="one_to_one",
    )

    event_output_path = Path(event_output_path)
    predictor_output_path = Path(predictor_output_path)
    event_output_path.parent.mkdir(parents=True, exist_ok=True)
    predictor_output_path.parent.mkdir(parents=True, exist_ok=True)
    event_table.to_csv(event_output_path, index=False, encoding="utf-8-sig")
    predictors.to_csv(
        predictor_output_path,
        index=False,
        encoding="utf-8-sig",
    )
    return event_table, predictors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TCCIP rainfall predictors and Tmax-event matches."
    )
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--wet-threshold-mm", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    matched_events, grid_predictors = build_rainfall_predictors(
        start_year=arguments.start_year,
        wet_day_threshold_mm=arguments.wet_threshold_mm,
    )
    print(
        f"Matched event rows: {len(matched_events):,}; "
        f"GRID predictors: {len(grid_predictors):,}"
    )
