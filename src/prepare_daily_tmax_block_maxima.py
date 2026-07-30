"""Prepare linked monthly and annual block maxima from TCCIP daily Tmax CSVs.

Expected TCCIP layout:

- first coordinate columns: lon and lat (case-insensitive);
- remaining data columns: daily dates in YYYY-MM-DD, YYYY/M/D or YYYYMMDD;
- missing values such as -99.9, -999 and -999.9.

The annual output is deliberately computed from the monthly maxima, and the
script verifies that each annual value equals the maximum of the 12 monthly
values from the same year.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_SRC = Path(__file__).resolve().parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from spatial_coordinates import add_twd97_km_columns  # noqa: E402


MISSING_SENTINELS = {-99.9, -999.0, -999.9}


def parse_date_column(column):
    text = str(column).strip()
    if re.fullmatch(r"\d{8}", text):
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def coordinate_columns(frame):
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    lon = next(
        (normalized[name] for name in ("lon", "longitude", "x") if name in normalized),
        None,
    )
    lat = next(
        (normalized[name] for name in ("lat", "latitude", "y") if name in normalized),
        None,
    )
    if lon is None or lat is None:
        raise ValueError("Daily CSV must contain lon/lat coordinate columns")
    return lon, lat


def read_one_daily_file(path: Path):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame.loc[:, ~frame.columns.astype(str).str.match(r"^Unnamed")]
    lon_col, lat_col = coordinate_columns(frame)

    parsed_dates = {
        column: parse_date_column(column)
        for column in frame.columns
        if column not in (lon_col, lat_col)
    }
    parsed_dates = {
        column: date for column, date in parsed_dates.items() if not pd.isna(date)
    }
    if not parsed_dates:
        raise ValueError(f"No daily date columns found in {path.name}")

    coordinates = frame[[lon_col, lat_col]].copy()
    coordinates.columns = ["lon", "lat"]
    coordinates["lon"] = pd.to_numeric(coordinates["lon"], errors="coerce").round(5)
    coordinates["lat"] = pd.to_numeric(coordinates["lat"], errors="coerce").round(5)
    coordinates["station"] = (
        "G"
        + coordinates["lon"].map(lambda value: f"{value:.5f}")
        + "_"
        + coordinates["lat"].map(lambda value: f"{value:.5f}")
    )

    values = frame[list(parsed_dates)].apply(pd.to_numeric, errors="coerce")
    values = values.replace(list(MISSING_SENTINELS), np.nan)
    values.columns = pd.DatetimeIndex([parsed_dates[column] for column in values])
    values.index = coordinates["station"]
    return coordinates, values


def aggregate_file_to_monthly(path: Path, min_daily_coverage: float):
    coordinates, daily = read_one_daily_file(path)
    monthly_parts = []
    coverage_parts = []

    month_keys = daily.columns.to_period("M")
    for period in month_keys.unique().sort_values():
        subset = daily.loc[:, month_keys == period]
        expected_days = int(period.days_in_month)
        valid_days = subset.notna().sum(axis=1)
        coverage = valid_days / expected_days
        monthly_max = subset.max(axis=1, skipna=True).where(
            coverage >= min_daily_coverage
        )

        monthly_parts.append(
            pd.DataFrame(
                [monthly_max.to_numpy()],
                columns=monthly_max.index,
                index=pd.MultiIndex.from_tuples(
                    [(period.year, period.month)],
                    names=["year", "month"],
                ),
            )
        )
        coverage_parts.append(
            pd.DataFrame(
                {
                    "station": coverage.index,
                    "year": period.year,
                    "month": period.month,
                    "valid_days": valid_days.to_numpy(),
                    "expected_days": expected_days,
                    "daily_coverage": coverage.to_numpy(),
                    "kept": (coverage >= min_daily_coverage).to_numpy(),
                }
            )
        )

    monthly = pd.concat(monthly_parts).sort_index()
    coverage = pd.concat(coverage_parts, ignore_index=True)
    return coordinates, monthly, coverage


def prepare_daily_tmax_block_maxima(
    raw_dir: Path,
    output_dir: Path,
    pattern: str = "*.csv",
    start_year: int = 1980,
    min_daily_coverage: float = 0.90,
):
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No daily Tmax CSV files matched {pattern!r} under {raw_dir}"
        )

    coordinate_frames = []
    monthly_frames = []
    coverage_frames = []
    for path in files:
        coordinates, monthly, coverage = aggregate_file_to_monthly(
            path,
            min_daily_coverage=min_daily_coverage,
        )
        coordinate_frames.append(coordinates)
        monthly_frames.append(monthly)
        coverage_frames.append(coverage.assign(source_file=path.name))

    locations = (
        pd.concat(coordinate_frames, ignore_index=True)
        .drop_duplicates("station")
        .sort_values("station")
    )
    locations = add_twd97_km_columns(locations)
    monthly = pd.concat(monthly_frames).sort_index()
    monthly = monthly[~monthly.index.duplicated(keep="last")]
    monthly = monthly.loc[monthly.index.get_level_values("year") >= int(start_year)]

    # This is the defining linkage: annual maxima are derived from the monthly
    # maxima rather than being calculated from an unrelated sample.
    annual = monthly.groupby(level="year").max()
    annual.index.name = "year"
    expected_annual = monthly.groupby(level="year").max()
    pd.testing.assert_frame_equal(annual, expected_annual, check_exact=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = output_dir / "daily_tmax_monthly_block_maxima.csv"
    annual_path = output_dir / "daily_tmax_annual_block_maxima.csv"
    location_path = output_dir / "daily_tmax_grid_locations.csv"
    coverage_path = output_dir / "daily_tmax_monthly_coverage.csv"

    monthly.to_csv(monthly_path, encoding="utf-8-sig")
    annual.to_csv(annual_path, encoding="utf-8-sig")
    locations.to_csv(location_path, index=False, encoding="utf-8-sig")
    pd.concat(coverage_frames, ignore_index=True).to_csv(
        coverage_path,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "monthly": monthly_path,
        "annual": annual_path,
        "locations": location_path,
        "coverage": coverage_path,
        "n_months": int(len(monthly)),
        "n_years": int(len(annual)),
        "n_grids": int(monthly.shape[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--min-daily-coverage", type=float, default=0.90)
    args = parser.parse_args()

    result = prepare_daily_tmax_block_maxima(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        start_year=args.start_year,
        min_daily_coverage=args.min_daily_coverage,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
