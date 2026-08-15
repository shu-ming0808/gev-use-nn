"""Download and align Tmax-event wind, solar-radiation, and cloud predictors.

Daily wind, solar, and cloud-cover fields come from AgERA5
(``sis-agrometeorological-indicators``) at 0.1 degree.  For each
TCCIP cell-month, the fields are evaluated at the cell centre on that cell's
monthly-Tmax date.  The event values are then averaged within each GRID so
the resulting static predictors can enter the current spatial GP mean
structure.  This is event-conditioned aggregation, not an all-day climate
mean and not statistical downscaling.

The downloader requires a CDS account, accepted dataset licences, and a
working ``~/.cdsapirc`` file.  Processing can be run independently after the
NetCDF files have been downloaded through the CDS website or another tool.
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from project_paths import (
    PROCESSED_DATA_DIR,
    SPATIAL_PREDICTOR_PROCESSED_DIR,
    SPATIAL_PREDICTOR_RAW_DIR,
)
from spatial_coordinates import main_island_grid_mask


AGERA5_DATASET = "sis-agrometeorological-indicators"
AGERA5_RESOLUTION_DEG = 0.1
TAIWAN_AREA = [26.0, 118.0, 21.5, 123.0]  # north, west, south, east

DEFAULT_RAW_DIR = SPATIAL_PREDICTOR_RAW_DIR / "atmosphere"
DEFAULT_OUTPUT_PATH = (
    SPATIAL_PREDICTOR_PROCESSED_DIR
    / "tccip_grid_atmospheric_predictors.csv"
)
DEFAULT_EVENT_PATH = (
    PROCESSED_DATA_DIR / "daily_tmax_monthly_max_occurrences.csv"
)
DEFAULT_EVENT_OUTPUT_PATH = (
    SPATIAL_PREDICTOR_PROCESSED_DIR
    / "tccip_grid_tmax_event_day_atmosphere.csv"
)
DEFAULT_AUDIT_PATH = (
    SPATIAL_PREDICTOR_PROCESSED_DIR
    / "tccip_grid_atmospheric_alignment_audit.csv"
)

AGERA5_REQUESTS = {
    "wind_speed": {
        "variable": "10m_wind_speed",
        "statistic": ["24_hour_mean"],
    },
    "solar_radiation": {"variable": "solar_radiation_flux"},
    "cloud_frequency": {
        "variable": "cloud_cover",
        "statistic": ["24_hour_mean"],
    },
}

VARIABLE_ALIASES = {
    "wind_speed": (
        "Wind_Speed_10m_Mean_24h",
        "wind_speed",
        "10m_wind_speed",
        "wind_speed_10m",
        "si10",
    ),
    "solar_radiation": (
        "Solar_Radiation_Flux",
        "solar_radiation_flux",
        "surface_solar_radiation_downwards",
        "ssrd",
    ),
    "cloud_frequency": (
        "Cloud_Cover_Mean_24h",
        "cloud_cover",
        "cloud_frequency",
    ),
}


def _cds_client():
    try:
        import cdsapi
    except ImportError as error:
        raise RuntimeError(
            "缺少 cdsapi；請先執行 pip install -r requirements.txt。"
        ) from error
    return cdsapi.Client()


def _chunked(values: list[int], size: int) -> list[list[int]]:
    if size < 1:
        raise ValueError("batch_years must be at least 1.")
    return [values[index : index + size] for index in range(0, len(values), size)]


def _covered_year_months(
    output_directory: Path,
    label: str,
) -> set[tuple[int, int]]:
    """Return year-months represented by legacy full-year or monthly zips."""
    covered: set[tuple[int, int]] = set()
    full_year_pattern = re.compile(
        rf"^agera5_{re.escape(label)}_(\d{{4}})(?:_(\d{{4}}))?\.zip$"
    )
    monthly_pattern = re.compile(
        rf"^agera5_{re.escape(label)}_(\d{{4}})_(\d{{4}})_m(\d{{2}})\.zip$"
    )
    for path in output_directory.glob(f"agera5_{label}_*.zip"):
        if path.stat().st_size == 0 or not zipfile.is_zipfile(path):
            continue
        monthly_match = monthly_pattern.match(path.name)
        if monthly_match is not None:
            start = int(monthly_match.group(1))
            end = int(monthly_match.group(2))
            month = int(monthly_match.group(3))
            covered.update((year, month) for year in range(start, end + 1))
            continue
        full_year_match = full_year_pattern.match(path.name)
        if full_year_match is None:
            continue
        start = int(full_year_match.group(1))
        end = int(full_year_match.group(2) or start)
        covered.update(
            (year, month)
            for year in range(start, end + 1)
            for month in range(1, 13)
        )
    return covered


def atmospheric_download_coverage(
    raw_directory: str | Path = DEFAULT_RAW_DIR,
    start_year: int = 1980,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Audit valid archives for every required variable and year-month."""
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year.")
    raw_directory = Path(raw_directory)
    expected = {
        (year, month)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    }
    rows = []
    for label in AGERA5_REQUESTS:
        available = _covered_year_months(raw_directory, label) & expected
        missing = sorted(expected - available)
        rows.append(
            {
                "variable": label,
                "expected_year_months": len(expected),
                "available_year_months": len(available),
                "missing_year_months": len(missing),
                "complete": not missing,
                "first_missing": (
                    f"{missing[0][0]:04d}-{missing[0][1]:02d}"
                    if missing
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def require_complete_atmospheric_downloads(
    raw_directory: str | Path = DEFAULT_RAW_DIR,
    start_year: int = 1980,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Stop partial atmospheric archives from entering model selection."""
    coverage = atmospheric_download_coverage(
        raw_directory=raw_directory,
        start_year=start_year,
        end_year=end_year,
    )
    incomplete = coverage.loc[~coverage["complete"]]
    if not incomplete.empty:
        details = ", ".join(
            f"{row.variable}: missing {row.missing_year_months} "
            f"(first {row.first_missing})"
            for row in incomplete.itertuples()
        )
        raise RuntimeError(
            "AgERA5 下載尚未完整，停止建立候選變數：" + details
        )
    return coverage


def _consecutive_runs(years: list[int]) -> list[list[int]]:
    """Split sorted years at gaps so archive names never imply missing years."""
    if not years:
        return []
    runs = [[years[0]]]
    for year in years[1:]:
        if year == runs[-1][-1] + 1:
            runs[-1].append(year)
        else:
            runs.append([year])
    return runs


def download_atmospheric_data(
    start_year: int = 1980,
    end_year: int = 2024,
    output_directory: str | Path = DEFAULT_RAW_DIR,
    batch_years: int = 9,
) -> None:
    """Download AgERA5 in nine-year groups and cost-safe monthly requests."""
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year.")
    if batch_years < 1:
        raise ValueError("batch_years must be at least 1.")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    client = _cds_client()
    requested_years = list(range(start_year, end_year + 1))

    for label, variable_request in AGERA5_REQUESTS.items():
        covered = _covered_year_months(output_directory, label)
        incomplete_years = [
            year
            for year in requested_years
            if any((year, month) not in covered for month in range(1, 13))
        ]
        if not incomplete_years:
            print(f"skip {label}: all requested years already exist")
            continue
        batches = [
            batch
            for run in _consecutive_runs(incomplete_years)
            for batch in _chunked(run, batch_years)
        ]
        for years in batches:
            for month in range(1, 13):
                missing_month_years = [
                    year for year in years if (year, month) not in covered
                ]
                for run in _consecutive_runs(missing_month_years):
                    start, end = run[0], run[-1]
                    target = output_directory / (
                        f"agera5_{label}_{start}_{end}_m{month:02d}.zip"
                    )
                    request = {
                        **variable_request,
                        "year": [str(year) for year in run],
                        "month": [f"{month:02d}"],
                        "day": [f"{day:02d}" for day in range(1, 32)],
                        "version": "2_0",
                        "area": TAIWAN_AREA,
                    }
                    print(
                        f"download AgERA5 {label}: {start}-{end}, "
                        f"month {month:02d} ({len(run)} years)"
                    )
                    client.retrieve(AGERA5_DATASET, request, str(target))
                    covered.update((year, month) for year in run)


def extract_downloads(directory: str | Path = DEFAULT_RAW_DIR) -> None:
    """Extract CDS NetCDF members using short, date-preserving file names.

    CDS member names are long enough to exceed the traditional Windows
    ``MAX_PATH`` limit once they are placed below the project directory.
    Each AgERA5 archive contains one field per date, so the unambiguous
    ``YYYYMMDD.nc`` name preserves the event-date key needed downstream while
    keeping the full path short.  Legacy long-name extractions are renamed in
    place, and an existing short file is never overwritten.
    """
    directory = Path(directory)
    for archive in sorted(directory.glob("*.zip")):
        destination = directory / archive.stem
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            seen_dates: set[str] = set()
            for member in handle.infolist():
                if member.is_dir() or Path(member.filename).suffix.lower() != ".nc":
                    continue
                match = re.search(
                    r"(?<!\d)((?:19|20)\d{6})(?!\d)",
                    Path(member.filename).name,
                )
                if match is None:
                    raise ValueError(
                        "AgERA5 ZIP member 缺少 YYYYMMDD 日期："
                        f"{archive.name}::{member.filename}"
                    )
                date_key = match.group(1)
                if date_key in seen_dates:
                    raise ValueError(
                        "同一 AgERA5 ZIP 內出現重複日期："
                        f"{archive.name}::{date_key}"
                    )
                seen_dates.add(date_key)

                target = destination / f"{date_key}.nc"
                if target.exists():
                    continue

                legacy = destination / member.filename
                if legacy.is_file():
                    legacy.replace(target)
                    continue

                temporary = destination / f".{date_key}.nc.part"
                try:
                    with handle.open(member) as source, temporary.open("wb") as sink:
                        shutil.copyfileobj(source, sink)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)


def _coordinate_name(dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.coords or name in dataset.dims:
            return name
    raise ValueError(f"找不到座標欄位：{candidates}")


def _data_variable(dataset, aliases: tuple[str, ...]) -> str:
    variables = {
        str(name).casefold(): str(name)
        for name in dataset.data_vars
        if str(name).casefold() not in {"crs", "spatial_ref"}
    }
    for alias in aliases:
        match = variables.get(alias.casefold())
        if match is not None:
            return match
    if len(variables) == 1:
        return next(iter(variables.values()))
    raise ValueError(
        f"無法從 {list(dataset.data_vars)} 判斷變數；預期 {aliases}。"
    )


@contextmanager
def _open_xarray(path: Path):
    """Open NetCDF, staging non-ASCII Windows paths when netCDF4 rejects them."""
    try:
        import xarray as xr
    except ImportError as error:
        raise RuntimeError(
            "處理 NetCDF 需要 xarray 與 netCDF4；請執行 "
            "pip install -r requirements.txt。"
        ) from error

    dataset = None
    try:
        dataset = xr.open_dataset(path)
    except OSError as error:
        invalid_argument = (
            error.errno == 22 or "Invalid argument" in str(error)
        )
        if not invalid_argument or str(path).isascii():
            raise
        # The Windows netCDF4/HDF5 backend may reject otherwise valid paths
        # containing non-ASCII characters.  Stage only the file being read;
        # AgERA5 daily subsets are small and the source archive remains intact.
        with tempfile.TemporaryDirectory(prefix="agera5_netcdf_") as temporary:
            staged_path = Path(temporary) / path.name
            if not str(staged_path).isascii():
                raise RuntimeError(
                    "系統暫存路徑仍含非 ASCII 字元，netCDF4 無法安全讀取。"
                ) from error
            shutil.copy2(path, staged_path)
            staged_dataset = xr.open_dataset(staged_path)
            try:
                yield staged_dataset
            finally:
                staged_dataset.close()
        return

    try:
        yield dataset
    finally:
        dataset.close()


def _date_from_path(path: Path) -> pd.Timestamp | None:
    """Extract a YYYYMMDD date from a daily-file name when available."""
    matches = re.findall(r"(?<!\d)((?:19|20)\d{6})(?!\d)", path.name)
    if not matches:
        return None
    return pd.to_datetime(matches[-1], format="%Y%m%d").normalize()


def _time_coordinate(array, spatial_dims: set[str]) -> str | None:
    for name in ("valid_time", "time", "date"):
        if name in array.coords or name in array.dims:
            return name
    for name in array.dims:
        if name in spatial_dims or name not in array.coords:
            continue
        try:
            pd.to_datetime(array[name].values)
        except (TypeError, ValueError):
            continue
        return name
    return None


def _event_values(
    paths: list[Path],
    kind: str,
    events: pd.DataFrame,
    initial_values: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, float]:
    """Interpolate a daily field only for requested Tmax-event dates."""
    if not paths:
        raise FileNotFoundError(f"沒有可處理的 {kind} NetCDF 檔。")

    values = (
        np.full(len(events), np.nan, dtype=float)
        if initial_values is None
        else np.asarray(initial_values, dtype=float).copy()
    )
    if values.shape != (len(events),):
        raise ValueError("initial_values must have one value per event.")
    missing_indices = np.flatnonzero(~np.isfinite(values))
    requested: dict[pd.Timestamp, np.ndarray] = {
        date: indices.to_numpy(dtype=int)
        for date, indices in events.iloc[missing_indices].groupby("max_date").groups.items()
        if pd.notna(date)
    }
    reference_lat = None
    reference_lon = None
    opened_files = 0
    source_dates: set[pd.Timestamp] = set()
    fallback_count = 0
    maximum_fallback_distance_km = 0.0

    for path in paths:
        path_date = _date_from_path(path)
        if path_date is not None and path_date not in requested:
            continue
        with _open_xarray(path) as dataset:
            lat_name = _coordinate_name(dataset, ("latitude", "lat", "y"))
            lon_name = _coordinate_name(dataset, ("longitude", "lon", "x"))
            variable = _data_variable(dataset, VARIABLE_ALIASES[kind])
            array = dataset[variable]
            spatial_dims = {lat_name, lon_name}
            time_name = _time_coordinate(array, spatial_dims)
            lat = np.asarray(dataset[lat_name].values, dtype=float)
            lon = np.asarray(dataset[lon_name].values, dtype=float)
            if reference_lat is None:
                reference_lat, reference_lon = lat, lon
            elif not (
                np.array_equal(reference_lat, lat)
                and np.array_equal(reference_lon, lon)
            ):
                raise ValueError(f"來源網格不一致：{path}")

            if time_name is None:
                if path_date is None:
                    raise ValueError(f"無法判斷 {path} 的日期。")
                dated_arrays = [(path_date, array)]
            else:
                dates = pd.to_datetime(array[time_name].values).normalize()
                dated_arrays = [
                    (date, array.isel({time_name: position}))
                    for position, date in enumerate(dates)
                    if date in requested
                ]

            for date, daily in dated_arrays:
                indices = requested.get(date)
                if indices is None:
                    continue
                extra_dims = [
                    dim for dim in daily.dims if dim not in spatial_dims
                ]
                if extra_dims:
                    daily = daily.mean(dim=extra_dims, skipna=True)
                field = np.asarray(
                    daily.transpose(lat_name, lon_name).values,
                    dtype=float,
                )
                interpolated, fallback, fallback_distances = (
                    _interpolate_ag_land_field(
                    field,
                    lat,
                    lon,
                    events.loc[indices, "lat"].to_numpy(float),
                    events.loc[indices, "lon"].to_numpy(float),
                    maximum_fallback_distance_km=10.0,
                    )
                )
                fallback_count += int(fallback.sum())
                if fallback.any():
                    maximum_fallback_distance_km = max(
                        maximum_fallback_distance_km,
                        float(np.max(fallback_distances[fallback])),
                    )
                existing = values[indices]
                duplicate = np.isfinite(existing)
                if duplicate.any() and not np.allclose(
                    existing[duplicate], interpolated[duplicate], equal_nan=True
                ):
                    raise ValueError(f"{kind} 在 {date.date()} 有衝突的重複資料。")
                values[indices] = interpolated
                source_dates.add(date)
            opened_files += 1

    if reference_lat is None or reference_lon is None:
        raise FileNotFoundError(
            f"{kind} 檔案存在，但沒有涵蓋任何 Tmax event date。"
        )
    return (
        values,
        reference_lat,
        reference_lon,
        opened_files,
        len(source_dates),
        fallback_count,
        maximum_fallback_distance_km,
    )


def _regular_interpolate(
    field: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate within the source-grid convex rectangle."""
    lat_order = np.argsort(latitude)
    lon_order = np.argsort(longitude)
    ordered = np.asarray(field)[np.ix_(lat_order, lon_order)]
    interpolator = RegularGridInterpolator(
        (latitude[lat_order], longitude[lon_order]),
        ordered,
        method="linear",
        bounds_error=True,
    )
    points = np.column_stack([target_latitude, target_longitude])
    return np.asarray(interpolator(points), dtype=float)


def _interpolate_ag_land_field(
    field: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    maximum_fallback_distance_km: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use bilinear interpolation, with a bounded coastal land-mask fallback."""
    values = _regular_interpolate(
        field,
        latitude,
        longitude,
        target_latitude,
        target_longitude,
    )
    fallback = ~np.isfinite(values)
    distances = np.zeros(len(values), dtype=float)
    if not fallback.any():
        return values, fallback, distances

    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
    valid = np.isfinite(field)
    if not valid.any():
        raise ValueError("AgERA5 daily field contains no finite land cells.")
    valid_lon = longitude_grid[valid]
    valid_lat = latitude_grid[valid]
    valid_values = np.asarray(field, dtype=float)[valid]
    for target_index in np.flatnonzero(fallback):
        target_lat = float(target_latitude[target_index])
        target_lon = float(target_longitude[target_index])
        dy = (valid_lat - target_lat) * 111.32
        dx = (
            (valid_lon - target_lon)
            * 111.32
            * np.cos(np.deg2rad(target_lat))
        )
        candidate_distances = np.hypot(dx, dy)
        nearest = int(np.argmin(candidate_distances))
        distance = float(candidate_distances[nearest])
        if distance > maximum_fallback_distance_km:
            raise ValueError(
                "AgERA5 coastal land-mask fallback exceeds "
                f"{maximum_fallback_distance_km:.1f} km: "
                f"target=({target_lon:.2f}, {target_lat:.2f}), "
                f"nearest valid={distance:.2f} km."
            )
        values[target_index] = valid_values[nearest]
        distances[target_index] = distance
    return values, fallback, distances


def _nearest_source_distance_km(
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    source_latitude: np.ndarray,
    source_longitude: np.ndarray,
) -> np.ndarray:
    nearest_lat = source_latitude[
        np.abs(target_latitude[:, None] - source_latitude[None, :]).argmin(1)
    ]
    nearest_lon = source_longitude[
        np.abs(target_longitude[:, None] - source_longitude[None, :]).argmin(1)
    ]
    dy = (target_latitude - nearest_lat) * 111.32
    dx = (
        (target_longitude - nearest_lon)
        * 111.32
        * np.cos(np.deg2rad(target_latitude))
    )
    return np.sqrt(dx**2 + dy**2)


def _find_netcdf(directory: Path, prefix: str) -> list[Path]:
    matches = sorted(
        path
        for path in directory.rglob("*.nc")
        if prefix in str(path.relative_to(directory)).lower()
    )
    return matches


def build_atmospheric_predictors(
    raw_directory: str | Path = DEFAULT_RAW_DIR,
    grid_path: str | Path = (
        PROCESSED_DATA_DIR / "grid_station_gev_params_with_loc.csv"
    ),
    event_path: str | Path = DEFAULT_EVENT_PATH,
    event_output_path: str | Path = DEFAULT_EVENT_OUTPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    audit_path: str | Path = DEFAULT_AUDIT_PATH,
    analysis_start_year: int = 1980,
    analysis_end_year: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match daily fields to monthly-Tmax dates and average by GRID."""
    raw_directory = Path(raw_directory)
    extract_downloads(raw_directory)
    grid = pd.read_csv(grid_path)[["station", "lon", "lat"]].drop_duplicates()
    if grid["station"].duplicated().any():
        raise ValueError("TCCIP station key is not one-to-one.")
    main_island = main_island_grid_mask(grid)
    grid = grid.loc[main_island].reset_index(drop=True)
    events = pd.read_csv(event_path)
    required_event_columns = {
        "station",
        "year",
        "month",
        "monthly_max_tmax_c",
        "max_date",
    }
    missing_event_columns = required_event_columns.difference(events.columns)
    if missing_event_columns:
        raise ValueError(
            "Tmax event table 缺少欄位："
            f"{sorted(missing_event_columns)}"
        )
    if analysis_end_year < analysis_start_year:
        raise ValueError(
            "analysis_end_year must be greater than or equal to "
            "analysis_start_year."
        )
    events["year"] = pd.to_numeric(events["year"], errors="raise").astype(int)
    events["max_date"] = pd.to_datetime(events["max_date"], errors="coerce")
    events = events.loc[
        events["year"].between(analysis_start_year, analysis_end_year)
    ].copy()
    dated = events["max_date"].notna()
    if not (
        events.loc[dated, "year"].to_numpy()
        == events.loc[dated, "max_date"].dt.year.to_numpy()
    ).all():
        raise ValueError("Tmax event year 與 max_date 年份不一致。")
    events = (
        events.loc[
            events["max_date"].notna()
            & events["station"].isin(grid["station"])
        ]
        .merge(grid, on="station", how="left", validate="many_to_one")
        .sort_values(["max_date", "station"])
        .reset_index(drop=True)
    )
    if events[["lon", "lat"]].isna().any().any():
        raise ValueError("部分 Tmax event 無法對到 TCCIP GRID 座標。")

    event_source_columns = {
        "wind_speed_on_tmax_date_mps",
        "solar_radiation_on_tmax_date_mj_m2",
        "agera5_cloud_cover_on_tmax_date_fraction",
    }
    event_output_path = Path(event_output_path)
    if event_output_path.exists():
        previous = pd.read_csv(event_output_path)
        available_source_columns = sorted(
            event_source_columns.intersection(previous.columns)
        )
        resume_keys = ["station", "year", "month", "max_date"]
        if available_source_columns and set(resume_keys).issubset(previous.columns):
            previous["max_date"] = pd.to_datetime(
                previous["max_date"], errors="coerce"
            )
            previous = previous[
                [*resume_keys, *available_source_columns]
            ].drop_duplicates(resume_keys)
            events = events.merge(
                previous,
                on=resume_keys,
                how="left",
                validate="one_to_one",
            )
            restored = int(
                events[available_source_columns].notna().sum().sum()
            )
            print(
                f"Resume atmospheric processing: restored {restored:,} "
                "existing event values"
            )

    specifications = {
        "wind_speed": (
            "agera5_wind_speed",
            "wind_speed_on_tmax_date_mps",
            "tmax_event_wind_mean_mps",
            AGERA5_RESOLUTION_DEG,
        ),
        "solar_radiation": (
            "agera5_solar_radiation",
            "solar_radiation_on_tmax_date_mj_m2",
            "tmax_event_solar_radiation_mean_mj_m2",
            AGERA5_RESOLUTION_DEG,
        ),
        "cloud_frequency": (
            "agera5_cloud_frequency",
            "agera5_cloud_cover_on_tmax_date_fraction",
            "tmax_event_agera5_cloud_cover_mean_fraction",
            AGERA5_RESOLUTION_DEG,
        ),
    }
    audit_rows = []
    source_columns = []
    summary_columns = []
    for kind, (
        prefix,
        event_column,
        summary_column,
        nominal_resolution,
    ) in specifications.items():
        paths = _find_netcdf(raw_directory, prefix)
        initial_values = (
            events[event_column].to_numpy(float)
            if event_column in events.columns
            else None
        )
        if kind == "solar_radiation" and initial_values is not None:
            # The resume CSV stores MJ m-2; source NetCDF stores J m-2.
            initial_values = initial_values * 1_000_000.0
        (
            values,
            source_lat,
            source_lon,
            opened_files,
            source_date_count,
            fallback_count,
            maximum_fallback_distance_km,
        ) = _event_values(
            paths,
            kind,
            events,
            initial_values=initial_values,
        )
        if kind == "solar_radiation":
            # AgERA5 solar radiation flux is daily energy in J m-2 day-1.
            values = values / 1_000_000.0
        events[event_column] = values
        source_columns.append(event_column)
        summary_columns.append(summary_column)
        nearest = _nearest_source_distance_km(
            grid["lat"].to_numpy(float),
            grid["lon"].to_numpy(float),
            source_lat,
            source_lon,
        )
        audit_rows.append(
            {
                "predictor": summary_column,
                "source_file_count": len(paths),
                "opened_source_file_count": opened_files,
                "matched_source_date_count": source_date_count,
                "matched_event_count": int(np.isfinite(values).sum()),
                "requested_event_count": len(events),
                "source_nominal_resolution_deg": nominal_resolution,
                "analysis_start_year": analysis_start_year,
                "analysis_end_year": analysis_end_year,
                "source_lat_min": float(np.min(source_lat)),
                "source_lat_max": float(np.max(source_lat)),
                "source_lon_min": float(np.min(source_lon)),
                "source_lon_max": float(np.max(source_lon)),
                "interpolation": "bilinear_at_tccip_cell_centre",
                "coastal_nearest_valid_fallback_count": fallback_count,
                "max_coastal_fallback_distance_km": (
                    maximum_fallback_distance_km
                ),
                "extrapolation_count": 0,
                "max_nearest_source_centre_km": float(nearest.max()),
                "definition": {
                    "wind_speed": (
                        "monthly-Tmax-date daily mean scalar wind speed at 10 m"
                    ),
                    "solar_radiation": (
                        "monthly-Tmax-date daily surface solar energy"
                    ),
                    "cloud_frequency": (
                        "monthly-Tmax-date AgERA5 daily mean total-cloud cover"
                    ),
                }[kind],
            }
        )

    fraction_columns = [
        "agera5_cloud_cover_on_tmax_date_fraction",
    ]
    finite_fractions = events[fraction_columns].stack()
    if not finite_fractions.between(0, 1).all():
        raise ValueError("Cloud fractions must be inside [0, 1].")
    if (events["wind_speed_on_tmax_date_mps"].dropna() < 0).any():
        raise ValueError("Wind speed cannot be negative.")
    if (events["solar_radiation_on_tmax_date_mj_m2"].dropna() < 0).any():
        raise ValueError("Solar radiation cannot be negative.")

    summary_parts = []
    for event_column, summary_column in zip(source_columns, summary_columns):
        part = events.groupby("station", as_index=False).agg(
            **{
                summary_column: (event_column, "mean"),
                f"{summary_column}_available": (event_column, "count"),
            }
        )
        summary_parts.append(part)
    requested = (
        events.groupby("station", as_index=False)
        .size()
        .rename(columns={"size": "tmax_event_months"})
    )
    result = grid.copy().merge(
        requested, on="station", how="left", validate="one_to_one"
    )
    for part, summary_column in zip(summary_parts, summary_columns):
        result = result.merge(
            part, on="station", how="left", validate="one_to_one"
        )
        result[f"{summary_column}_available_ratio"] = (
            result[f"{summary_column}_available"]
            / result["tmax_event_months"]
        )

    audit = pd.DataFrame(audit_rows)
    event_output_path = Path(event_output_path)
    output_path = Path(output_path)
    audit_path = Path(audit_path)
    event_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(event_output_path, index=False, encoding="utf-8-sig")
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    return result, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/process AgERA5 atmospheric predictors."
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download/resume archives without extracting or processing them.",
    )
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--batch-years",
        type=int,
        default=9,
        help=(
            "Maximum consecutive years per group; each group is submitted "
            "as one cost-safe CDS request per calendar month."
        ),
    )
    parser.add_argument("--raw-directory", type=Path, default=DEFAULT_RAW_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download_only and not args.download:
        raise ValueError("--download-only 必須與 --download 一起使用。")
    if args.download:
        download_atmospheric_data(
            start_year=args.start_year,
            end_year=args.end_year,
            output_directory=args.raw_directory,
            batch_years=args.batch_years,
        )
        if args.download_only:
            coverage = atmospheric_download_coverage(
                raw_directory=args.raw_directory,
                start_year=args.start_year,
                end_year=args.end_year,
            )
            print("\nArchive coverage")
            print(coverage.to_string(index=False))
            return
    coverage = require_complete_atmospheric_downloads(
        raw_directory=args.raw_directory,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print("\nArchive coverage")
    print(coverage.to_string(index=False))
    result, audit = build_atmospheric_predictors(
        raw_directory=args.raw_directory,
        analysis_start_year=args.start_year,
        analysis_end_year=args.end_year,
    )
    print(result.describe().to_string())
    print("\nAlignment audit")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
