"""Canonical preprocessing pipeline for the real TCCIP temperature GRID.

This module ends at model-ready NN-derived GEV parameters and spatial
predictors.  GP smoothing, kernel selection, spatial feature selection, and
return-level validation deliberately remain in separate analysis modules.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from coast_distance_predictor import build_coast_distance_table
from gev_nn import P_SET, estimate_one, load_baseline_model, make_input
from land_cover_predictors import build_land_cover_predictors
from prepare_daily_tmax_block_maxima import prepare_daily_tmax_block_maxima
from rainfall_predictors import (
    DEFAULT_PREDICTOR_OUTPUT_PATH,
    build_rainfall_predictors,
)
from project_paths import (
    FIGURE_DIR,
    MODEL_DIR,
    ORIGINAL_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPATIAL_PREDICTOR_PROCESSED_DIR,
)
from spatial_coordinates import add_twd97_km_columns, main_island_grid_mask
from spatial_diagnostics import plot_isotropy_stationarity_diagnostics
from terrain_predictors import build_terrain_predictor_tables
from atmospheric_predictors import DEFAULT_OUTPUT_PATH as ATMOSPHERIC_OUTPUT_PATH


RAW_MONTHLY_TMAX_DIR = ORIGINAL_DATA_DIR / "觀測_月資料_臺灣_最高溫"
RAW_DAILY_TMAX_DIR = ORIGINAL_DATA_DIR / "觀測_日資料_臺灣_最高溫"
# Backward-compatible alias used by older notebooks.
RAW_TMAX_DIR = RAW_DAILY_TMAX_DIR
BASELINE_MODEL_PATH = MODEL_DIR / "best_baseline_model.pth"
ANALYSIS_START = pd.Timestamp("1980-01-01")
MIN_MONTH_COVERAGE = 0.80
MIN_ANNUAL_OBSERVATIONS = 30


def _quantile_column(probability: float) -> str:
    return "q_" + f"{probability:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def read_one_year_tmax(path: str | Path) -> pd.DataFrame:
    """Read one TCCIP monthly maximum-temperature file in long format."""
    path = Path(path)
    year_match = re.search(r"_(\d{4})\.csv$", path.name)
    if year_match is None:
        raise ValueError(f"Cannot read year from filename: {path.name}")
    year = int(year_match.group(1))
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame.loc[:, ~frame.columns.astype(str).str.match(r"^Unnamed")]
    frame.columns = [str(column).strip() for column in frame.columns]
    if not {"LON", "LAT"}.issubset(frame.columns):
        raise ValueError(f"{path.name} does not contain LON and LAT.")
    frame["LON"] = pd.to_numeric(frame["LON"], errors="coerce").round(2)
    frame["LAT"] = pd.to_numeric(frame["LAT"], errors="coerce").round(2)
    month_columns = [
        column
        for column in frame.columns
        if re.fullmatch(rf"{year}\d{{2}}", str(column))
    ]
    if len(month_columns) != 12:
        raise ValueError(
            f"{path.name} contains {len(month_columns)} monthly columns, not 12."
        )
    long = frame.melt(
        id_vars=["LON", "LAT"],
        value_vars=month_columns,
        var_name="yyyymm",
        value_name="monthly_tmax",
    )
    long["monthly_tmax"] = pd.to_numeric(
        long["monthly_tmax"],
        errors="coerce",
    )
    long.loc[long["monthly_tmax"] <= -90.0, "monthly_tmax"] = np.nan
    long["date"] = pd.to_datetime(long["yyyymm"] + "01", format="%Y%m%d")
    long["station"] = (
        "G"
        + long["LON"].map(lambda value: f"{value:.2f}")
        + "_"
        + long["LAT"].map(lambda value: f"{value:.2f}")
    )
    return long.rename(columns={"LON": "lon", "LAT": "lat"})[
        ["date", "station", "lon", "lat", "monthly_tmax"]
    ]


def build_temperature_tables(
    raw_directory: str | Path = RAW_MONTHLY_TMAX_DIR,
    analysis_start: pd.Timestamp = ANALYSIS_START,
    minimum_month_coverage: float = MIN_MONTH_COVERAGE,
    minimum_annual_observations: int = MIN_ANNUAL_OBSERVATIONS,
    output_directory: str | Path = PROCESSED_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """Merge source files, audit coverage, and create annual block maxima."""
    raw_directory = Path(raw_directory)
    files = sorted(raw_directory.glob("觀測_月資料_臺灣_最高溫_*.csv"))
    if not files:
        raise FileNotFoundError(f"No monthly TMAX files found in {raw_directory}")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    monthly_long = pd.concat(
        [read_one_year_tmax(path) for path in files],
        ignore_index=True,
    )
    station_locations = (
        monthly_long[["station", "lat", "lon"]]
        .drop_duplicates("station")
        .sort_values("station")
        .reset_index(drop=True)
    )
    station_locations = add_twd97_km_columns(station_locations)
    pivot_all = monthly_long.pivot(
        index="date",
        columns="station",
        values="monthly_tmax",
    ).sort_index()
    pivot = pivot_all.loc[pivot_all.index >= analysis_start].copy()
    coverage = pivot.notna().mean().rename("valid_month_ratio").reset_index()
    coverage["valid_months"] = pivot.notna().sum().to_numpy()
    coverage = coverage.merge(
        station_locations,
        on="station",
        how="left",
        validate="one_to_one",
    )
    keep = coverage.loc[
        coverage["valid_month_ratio"] >= minimum_month_coverage,
        "station",
    ]
    pivot_clean = pivot[keep].copy()
    annual_max = pivot_clean.resample("YE").max()
    annual_max.index = annual_max.index.year
    annual_max.index.name = "year"
    valid_years = annual_max.notna().sum()
    annual_keep = valid_years.loc[
        valid_years >= minimum_annual_observations
    ].index
    annual_max = annual_max[annual_keep]
    annual_locations = station_locations.loc[
        station_locations["station"].isin(annual_keep)
    ].copy()

    monthly_long.to_csv(
        output_directory / "monthly_long_grid_temperature.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_locations.to_csv(
        output_directory / "grid_station_location.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pivot_all.to_csv(
        output_directory / "pivot_grid_monthly_max_temperature_all.csv",
        encoding="utf-8-sig",
    )
    coverage.to_csv(
        output_directory / "grid_station_coverage_after_1980.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pivot_clean.to_csv(
        output_directory
        / "pivot_grid_monthly_max_temperature_after_1980_clean.csv",
        encoding="utf-8-sig",
    )
    annual_max.to_csv(
        output_directory / "annual_max_grid_temperature.csv",
        encoding="utf-8-sig",
    )
    annual_locations.to_csv(
        output_directory / "annual_grid_station_location.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = pd.DataFrame(
        {
            "item": [
                "source_year_files",
                "source_grid_cells",
                "months_after_start",
                "coverage_threshold",
                "kept_monthly_grid_cells",
                "minimum_annual_observations",
                "kept_annual_grid_cells",
            ],
            "value": [
                len(files),
                pivot_all.shape[1],
                pivot.shape[0],
                minimum_month_coverage,
                pivot_clean.shape[1],
                minimum_annual_observations,
                annual_max.shape[1],
            ],
        }
    )
    summary.to_csv(
        output_directory / "eda_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "monthly_long": monthly_long,
        "station_locations": station_locations,
        "pivot_all": pivot_all,
        "coverage": coverage,
        "pivot_clean": pivot_clean,
        "annual_max": annual_max,
        "annual_locations": annual_locations,
        "summary": summary,
    }


def load_daily_temperature_tables(
    processed_directory: str | Path = PROCESSED_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """Load canonical block maxima derived from the daily Tmax files."""
    processed_directory = Path(processed_directory)
    annual_max = pd.read_csv(
        processed_directory / "daily_tmax_annual_block_maxima.csv",
        index_col="year",
    )
    annual_max.index = pd.to_numeric(annual_max.index, errors="raise")
    annual_max = annual_max.loc[
        annual_max.index >= ANALYSIS_START.year
    ].copy()
    station_locations = pd.read_csv(
        processed_directory / "daily_tmax_grid_locations.csv"
    )
    if not {"x_km", "y_km"}.issubset(station_locations.columns):
        station_locations = add_twd97_km_columns(station_locations)
    valid_years = annual_max.notna().sum()
    annual_keep = valid_years.loc[
        valid_years >= MIN_ANNUAL_OBSERVATIONS
    ].index
    annual_max = annual_max[annual_keep]
    annual_locations = station_locations.loc[
        station_locations["station"].isin(annual_keep)
    ].copy()
    coverage = pd.DataFrame(
        {
            "station": annual_max.columns,
            "valid_years": annual_max.notna().sum().to_numpy(),
            "valid_year_ratio": annual_max.notna().mean().to_numpy(),
        }
    ).merge(
        station_locations,
        on="station",
        how="left",
        validate="one_to_one",
    )
    summary = pd.DataFrame(
        {
            "item": [
                "source_scale",
                "analysis_start_year",
                "kept_annual_grid_cells",
                "minimum_annual_observations",
            ],
            "value": [
                "daily Tmax -> monthly maxima -> annual maxima",
                int(annual_max.index.min()),
                annual_max.shape[1],
                MIN_ANNUAL_OBSERVATIONS,
            ],
        }
    )
    return {
        "annual_max": annual_max,
        "annual_locations": annual_locations,
        "station_locations": station_locations,
        "coverage": coverage,
        "summary": summary,
    }


def load_existing_temperature_tables(
    processed_directory: str | Path = PROCESSED_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """Load existing annual maxima and locations without rebuilding raw CSVs."""
    processed_directory = Path(processed_directory)
    annual_max = pd.read_csv(
        processed_directory / "annual_max_grid_temperature.csv",
        index_col="year",
    )
    annual_locations = pd.read_csv(
        processed_directory / "annual_grid_station_location.csv"
    )
    station_locations = pd.read_csv(
        processed_directory / "grid_station_location.csv"
    )
    if not {"x_km", "y_km"}.issubset(annual_locations.columns):
        annual_locations = add_twd97_km_columns(annual_locations)
    if not {"x_km", "y_km"}.issubset(station_locations.columns):
        station_locations = add_twd97_km_columns(station_locations)
    coverage = pd.read_csv(
        processed_directory / "grid_station_coverage_after_1980.csv"
    )
    summary = pd.read_csv(processed_directory / "eda_summary.csv")
    return {
        "annual_max": annual_max,
        "annual_locations": annual_locations,
        "station_locations": station_locations,
        "coverage": coverage,
        "summary": summary,
    }


def estimate_grid_gev_parameters(
    annual_max: pd.DataFrame,
    locations: pd.DataFrame,
    model_path: str | Path = BASELINE_MODEL_PATH,
    output_directory: str | Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create 11 quantiles and estimate GEV parameters with correct sign."""
    model, device = load_baseline_model(model_path=model_path)
    parameter_rows = []
    quantile_rows = []
    for station in annual_max.columns:
        values = annual_max[station].dropna().to_numpy(float)
        if len(values) < MIN_ANNUAL_OBSERVATIONS:
            continue
        quantiles, median, iqr = make_input(values)
        mu, sigma, shape_c = estimate_one(model, values, device)
        quantile_row = {
            "station": station,
            "n_obs": len(values),
            "sample_median": median,
            "sample_iqr": iqr,
        }
        quantile_row.update(
            {
                _quantile_column(probability): float(value)
                for probability, value in zip(P_SET, quantiles)
            }
        )
        quantile_rows.append(quantile_row)
        parameter_rows.append(
            {
                "station": station,
                "n_obs": len(values),
                "mu_hat": float(mu),
                "sigma_hat": float(sigma),
                "log_sigma_hat": float(np.log(sigma)),
                "shape_c_hat": float(shape_c),
                "xi_hat": float(-shape_c),
                "model_weights": Path(model_path).name,
            }
        )
    parameters = pd.DataFrame(parameter_rows).merge(
        locations[["station", "lat", "lon", "x_km", "y_km"]],
        on="station",
        how="left",
        validate="one_to_one",
    )
    quantiles = pd.DataFrame(quantile_rows)
    if parameters[["lat", "lon", "x_km", "y_km"]].isna().any().any():
        raise ValueError("Some NN-derived parameter rows lack GRID locations.")
    if not np.allclose(
        parameters["xi_hat"],
        -parameters["shape_c_hat"],
    ):
        raise AssertionError("GEV shape sign conversion failed.")
    output_directory = Path(output_directory)
    parameters.to_csv(
        output_directory / "grid_station_gev_params_with_loc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    quantiles.to_csv(
        output_directory / "real_grid_11_quantiles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return parameters, quantiles


def build_model_ready_grid(
    parameters: pd.DataFrame,
    terrain: pd.DataFrame,
    land_cover: pd.DataFrame,
    coast: pd.DataFrame,
    rainfall: pd.DataFrame,
    atmosphere: pd.DataFrame | None = None,
    output_path: str | Path = (
        PROCESSED_DATA_DIR / "model_ready_grid_parameters.csv"
    ),
) -> pd.DataFrame:
    """One-to-one join parameters with all candidate spatial predictors."""
    parameters = parameters.loc[
        main_island_grid_mask(parameters)
    ].reset_index(drop=True)
    terrain_predictors = [
        "station",
        "elevation_m",
        "slope_deg",
        "aspect_deg",
        "northness",
        "eastness",
        "local_relief_m",
        "tpi_m",
        "terrain_ruggedness_m",
    ]
    land_cover_predictors = [
        "station",
        "land_cover_year",
        "urban_ratio",
        "forest_ratio",
        "agriculture_ratio",
        "water_ratio",
        "other_ratio",
    ]
    result = (
        parameters.merge(
            terrain[terrain_predictors],
            on="station",
            how="left",
            validate="one_to_one",
        )
        .merge(
            land_cover[land_cover_predictors],
            on="station",
            how="left",
            validate="one_to_one",
        )
        .merge(
            coast[["station", "coast_distance_km"]],
            on="station",
            how="left",
            validate="one_to_one",
        )
        .merge(
            rainfall,
            on="station",
            how="left",
            validate="one_to_one",
        )
    )
    atmospheric_predictors = [
        "tmax_event_wind_mean_mps",
        "tmax_event_solar_radiation_mean_mj_m2",
        "tmax_event_agera5_cloud_cover_mean_fraction",
    ]
    if atmosphere is not None:
        coverage_columns = [
            f"{column}_available_ratio"
            for column in atmospheric_predictors
        ]
        missing_coverage = set(coverage_columns).difference(atmosphere.columns)
        if missing_coverage:
            raise ValueError(
                "大氣候選表缺少事件涵蓋率欄位："
                f"{sorted(missing_coverage)}；請用新版 atmospheric_predictors.py 重建。"
            )
        minimum_coverage = atmosphere[coverage_columns].min().min()
        if minimum_coverage < 0.999:
            raise ValueError(
                "大氣事件資料尚未完整下載或配對；最小 GRID-event 涵蓋率為 "
                f"{minimum_coverage:.3f}。完成 1980--2024 後再建立 model-ready table。"
            )
        result = result.merge(
            atmosphere[["station", *atmospheric_predictors]],
            on="station",
            how="left",
            validate="one_to_one",
        )
    required = [
        "mu_hat",
        "log_sigma_hat",
        "xi_hat",
        "elevation_m",
        "slope_deg",
        "northness",
        "eastness",
        "local_relief_m",
        "tpi_m",
        "terrain_ruggedness_m",
        "urban_ratio",
        "forest_ratio",
        "agriculture_ratio",
        "water_ratio",
        "coast_distance_km",
        "mean_annual_precip_mm",
        "rain_wet_day_ratio",
        "tmax_event_rain_mean_mm",
        "tmax_event_rain_wet_ratio",
    ]
    if atmosphere is not None:
        required.extend(atmospheric_predictors)
    missing = result[required].isna().sum()
    if (missing > 0).any():
        raise ValueError(
            "Model-ready table contains missing values: "
            + ", ".join(
                f"{column}={int(count)}"
                for column, count in missing[missing > 0].items()
            )
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def run_preprocessing_pipeline(
    rebuild_temperature: bool = False,
    rebuild_rainfall: bool = False,
    force_land_cover_download: bool = False,
    force_coast_download: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the canonical preprocessing and raw spatial diagnostics."""
    if rebuild_temperature:
        prepare_daily_tmax_block_maxima(
            raw_dir=RAW_DAILY_TMAX_DIR,
            output_dir=PROCESSED_DATA_DIR,
            pattern="觀測_日資料_臺灣_最高溫_*.csv",
            start_year=ANALYSIS_START.year,
        )
    temperature = load_daily_temperature_tables()
    parameters, quantiles = estimate_grid_gev_parameters(
        temperature["annual_max"],
        temperature["annual_locations"],
    )
    analysis_domain = parameters[["station", "lon", "lat"]].copy()
    analysis_domain["is_main_island"] = main_island_grid_mask(
        analysis_domain
    )
    analysis_domain["domain_reason"] = np.where(
        analysis_domain["is_main_island"],
        "largest_connected_tccip_grid_component",
        "disconnected_offshore_grid_excluded",
    )
    analysis_domain.to_csv(
        PROCESSED_DATA_DIR / "tccip_grid_analysis_domain.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _, terrain = build_terrain_predictor_tables()
    land_cover = build_land_cover_predictors(
        force_download=force_land_cover_download
    )
    coast = build_coast_distance_table(
        force_download=force_coast_download
    )
    if rebuild_rainfall or not DEFAULT_PREDICTOR_OUTPUT_PATH.exists():
        _, rainfall = build_rainfall_predictors()
    else:
        rainfall = pd.read_csv(DEFAULT_PREDICTOR_OUTPUT_PATH)
    atmosphere = (
        pd.read_csv(ATMOSPHERIC_OUTPUT_PATH)
        if ATMOSPHERIC_OUTPUT_PATH.exists()
        else None
    )
    model_ready = build_model_ready_grid(
        parameters,
        terrain,
        land_cover,
        coast,
        rainfall,
        atmosphere=atmosphere,
    )
    observed_surface = temperature["annual_locations"].merge(
        temperature["annual_max"]
        .mean(axis=0)
        .rename("mean_annual_max")
        .reset_index()
        .rename(columns={"index": "station"}),
        on="station",
        how="inner",
        validate="one_to_one",
    )
    observed_surface = observed_surface.loc[
        observed_surface["station"].isin(
            analysis_domain.loc[
                analysis_domain["is_main_island"], "station"
            ]
        )
    ].reset_index(drop=True)
    observed_diagnostic_summary, _ = (
        plot_isotropy_stationarity_diagnostics(
            observed_surface,
            value_columns={"annual_max_mean": "mean_annual_max"},
            value_kind="observed mean annual-maximum temperature surface",
            output_figure_path=(
                FIGURE_DIR
                / "preprocessing_observed_grid_isotropy_stationarity.png"
            ),
            output_table_path=(
                PROCESSED_DATA_DIR
                / "observed_grid_isotropy_stationarity_summary.csv"
            ),
        )
    )
    diagnostic_summary, _ = plot_isotropy_stationarity_diagnostics(
        model_ready,
        value_columns={
            "mu": "mu_hat",
            "log_sigma": "log_sigma_hat",
            "xi": "xi_hat",
        },
        value_kind="raw NN-derived parameter surfaces",
        output_figure_path=(
            FIGURE_DIR
            / "preprocessing_raw_parameter_isotropy_stationarity.png"
        ),
        output_table_path=(
            PROCESSED_DATA_DIR
            / "raw_parameter_isotropy_stationarity_summary.csv"
        ),
    )
    return {
        **temperature,
        "parameters": parameters,
        "quantiles": quantiles,
        "terrain": terrain,
        "land_cover": land_cover,
        "coast": coast,
        "rainfall": rainfall,
        "atmosphere": atmosphere,
        "model_ready": model_ready,
        "analysis_domain": analysis_domain,
        "observed_surface": observed_surface,
        "observed_spatial_diagnostics": observed_diagnostic_summary,
        "raw_spatial_diagnostics": diagnostic_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical preprocessing for the real TCCIP GRID."
    )
    parser.add_argument("--rebuild-temperature", action="store_true")
    parser.add_argument("--rebuild-rainfall", action="store_true")
    parser.add_argument("--force-land-cover-download", action="store_true")
    parser.add_argument("--force-coast-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_preprocessing_pipeline(
        rebuild_temperature=args.rebuild_temperature,
        rebuild_rainfall=args.rebuild_rainfall,
        force_land_cover_download=args.force_land_cover_download,
        force_coast_download=args.force_coast_download,
    )
    print(outputs["parameters"].describe().to_string())
    print("\nRaw isotropy/stationarity diagnostics")
    print(outputs["raw_spatial_diagnostics"].to_string(index=False))


if __name__ == "__main__":
    main()
