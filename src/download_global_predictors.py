"""Resumable downloads for the global annual-extreme predictor workflow.

This module keeps every global source used by the predictor-selection design in
one place.  It intentionally downloads *sources*, not the final GP columns:

* hourly ERA5 2 m temperature (the existing 1976--2025 workflow);
* ERA5 invariant land-sea mask and surface geopotential;
* ESA CCI year-2000 global land cover;
* the global GSHHG coastline archive;
* daily ERA5 wind, cloud, solar-radiation, and precipitation fields.

Terrain derivatives (slope, aspect/northness/eastness, local relief, TPI, and
ruggedness) are derived later from geopotential.  Coast distance is derived
later from GSHHG.  Files are downloaded through ``.part`` paths and
promoted only after an audit; valid completed files are audited and skipped on
reruns.

Nothing is downloaded unless ``--components`` is supplied.  Use ``--dry-run``
first for a request inventory.  The default analysis period is 1976--2025.
"""

from __future__ import annotations

import argparse
import calendar
import contextlib
import datetime as dt
import json
import os
import shutil
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import cdsapi
import netCDF4
import numpy as np


ERA5_HOURLY_DATASET = "reanalysis-era5-single-levels"
ERA5_DAILY_DATASET = "derived-era5-single-levels-daily-statistics"
GLOBAL_GRID = [0.25, 0.25]
ALL_HOURS = [f"{hour:02d}:00" for hour in range(24)]
DEFAULT_START_YEAR = 1976
DEFAULT_END_YEAR = 2025

# Keep this default compatible with the data already downloaded by the older
# script.  The directory name is historical; the plan records 1976--2025.
DEFAULT_ROOT = Path("D:/論文資料/ERA5/1975-2025_global_hourly")

PILOT_MONTH_BYTES = 1_111_236_798
PILOT_MONTH_DAYS = 31
GRAVITY_M_S2 = 9.80665

COMPONENTS = {
    "hourly-temperature",
    "static",
    "land-cover",
    "coastline",
    "daily-predictors",
}

# Daily groups are separated because CDS applies one statistic per request.
# Wind speed is subsequently derived from u10/v10.  These daily fields can be
# matched to the day of the annual Tmax event without retaining hourly copies
# of every atmospheric predictor.
DAILY_GROUPS: dict[str, dict[str, Any]] = {
    "daily_mean": {
        "variables": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "total_cloud_cover",
        ],
        "short_names": {"u10", "v10", "tcc"},
    },
    "daily_sum": {
        "variables": [
            "surface_solar_radiation_downwards",
            "total_precipitation",
        ],
        "short_names": {"ssrd", "tp"},
    },
}

STATIC_FIELDS = {
    "land_sea_mask": {
        "short_name": "lsm",
        "filename": "era5_land_sea_mask_global_025.nc",
    },
    "geopotential": {
        "short_name": "z",
        "filename": "era5_surface_geopotential_global_025.nc",
    },
}

LAND_COVER_COLLECTION = "esa-cci-lc-netcdf"
LAND_COVER_ITEM = "ESACCI-LC-L4-LCCS-Map-300m-P1Y-2000-v2.0.7cds"
LAND_COVER_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
LAND_COVER_SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
LAND_COVER_FILENAME = "esa_cci_land_cover_2000_global_300m.nc"

GSHHG_VERSION = "2.3.7"
GSHHG_URL = (
    "https://github.com/GenericMappingTools/gshhg-gmt/releases/download/"
    f"{GSHHG_VERSION}/gshhg-shp-{GSHHG_VERSION}.zip"
)
GSHHG_FILENAME = f"gshhg-shp-{GSHHG_VERSION}.zip"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def month_days(year: int, month: int) -> list[str]:
    return [
        f"{day:02d}"
        for day in range(1, calendar.monthrange(year, month)[1] + 1)
    ]


def iter_months(start_year: int, end_year: int) -> Iterator[tuple[int, int]]:
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield year, month


def hourly_temperature_request(year: int, month: int) -> dict[str, object]:
    return {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": [f"{year:04d}"],
        "month": [f"{month:02d}"],
        "day": month_days(year, month),
        "time": ALL_HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "grid": GLOBAL_GRID,
    }


def static_request(variable: str) -> dict[str, object]:
    return {
        "product_type": ["reanalysis"],
        "variable": [variable],
        # ERA5 LSM and surface geopotential are invariant.  CDS nevertheless
        # requires one valid timestamp, so 2000-01-01 00:00 is used.
        "year": ["2000"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "grid": GLOBAL_GRID,
    }


def daily_request(
    variables: list[str], year: int, month: int, statistic: str
) -> dict[str, object]:
    return {
        "product_type": "reanalysis",
        "variable": variables,
        "year": f"{year:04d}",
        "month": [f"{month:02d}"],
        "day": month_days(year, month),
        "daily_statistic": statistic,
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


@contextlib.contextmanager
def open_netcdf_windows_safe(path: Path) -> Iterator[netCDF4.Dataset]:
    """Open by ASCII basename to avoid non-ASCII Windows netCDF path bugs."""

    previous_directory = Path.cwd()
    try:
        os.chdir(path.parent)
        with netCDF4.Dataset(path.name) as dataset:
            yield dataset
    finally:
        os.chdir(previous_directory)


def coordinate_name(dataset: netCDF4.Dataset, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in dataset.variables:
            return candidate
    raise ValueError(f"Missing coordinate; expected one of {candidates}.")


def audit_global_grid(dataset: netCDF4.Dataset) -> tuple[np.ndarray, np.ndarray]:
    latitude_name = coordinate_name(dataset, ("latitude", "lat"))
    longitude_name = coordinate_name(dataset, ("longitude", "lon"))
    latitude = np.asarray(dataset.variables[latitude_name][:], dtype=np.float64)
    longitude = np.asarray(dataset.variables[longitude_name][:], dtype=np.float64)
    if latitude.shape != (721,) or longitude.shape != (1440,):
        raise ValueError(
            f"Unexpected global 0.25-degree grid: latitude={latitude.shape}, "
            f"longitude={longitude.shape}."
        )
    if not np.allclose(np.sort(latitude), np.arange(-90.0, 90.001, 0.25)):
        raise ValueError("Latitude is not the expected global 0.25-degree grid.")
    if not np.allclose(
        np.sort(longitude % 360.0), np.arange(0.0, 360.0, 0.25)
    ):
        raise ValueError("Longitude is not the expected global 0.25-degree grid.")
    return latitude, longitude


def decoded_times(dataset: netCDF4.Dataset) -> list[Any]:
    time_name = coordinate_name(dataset, ("valid_time", "time"))
    variable = dataset.variables[time_name]
    return list(
        netCDF4.num2date(
            variable[:],
            variable.units,
            getattr(variable, "calendar", "standard"),
        )
    )


def audit_hourly_temperature(path: Path, year: int, month: int) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {path}")
    with open_netcdf_windows_safe(path) as dataset:
        if "t2m" not in dataset.variables:
            raise ValueError(f"{path.name} does not contain t2m.")
        latitude, longitude = audit_global_grid(dataset)
        times = decoded_times(dataset)
        expected_count = calendar.monthrange(year, month)[1] * 24
        actual = {(item.year, item.month, item.day, item.hour) for item in times}
        expected = {
            (year, month, day, hour)
            for day in range(1, calendar.monthrange(year, month)[1] + 1)
            for hour in range(24)
        }
        if len(times) != expected_count or actual != expected:
            raise ValueError(f"Incomplete hourly timestamps for {year:04d}-{month:02d}.")
        variable = dataset.variables["t2m"]
        lat_index = int(np.argmin(np.abs(latitude - 25.0)))
        lon_index = int(np.argmin(np.abs(longitude - 121.5)))
        samples = np.asarray(
            variable[[0, len(times) // 2, len(times) - 1], lat_index, lon_index],
            dtype=np.float64,
        )
        if not np.isfinite(samples).all():
            raise ValueError(f"Non-finite t2m audit sample in {path.name}.")
    return {
        "status": "complete",
        "file": str(path),
        "file_bytes": path.stat().st_size,
        "year": year,
        "month": month,
        "stored_hours": len(times),
        "sample_minimum_K": float(samples.min()),
        "sample_maximum_K": float(samples.max()),
        "audited_utc": utc_now(),
    }


def audit_static(path: Path, short_name: str) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {path}")
    with open_netcdf_windows_safe(path) as dataset:
        audit_global_grid(dataset)
        if short_name not in dataset.variables:
            raise ValueError(f"{path.name} does not contain {short_name}.")
        values = np.ma.asarray(dataset.variables[short_name][:]).filled(np.nan)
        minimum = float(np.nanmin(values))
        maximum = float(np.nanmax(values))
        if short_name == "lsm" and (minimum < -1e-6 or maximum > 1.0 + 1e-6):
            raise ValueError(f"LSM outside [0, 1]: [{minimum}, {maximum}].")
    result: dict[str, object] = {
        "status": "complete",
        "file": str(path),
        "file_bytes": path.stat().st_size,
        "variable": short_name,
        "minimum": minimum,
        "maximum": maximum,
        "audited_utc": utc_now(),
    }
    if short_name == "z":
        result["minimum_elevation_m"] = minimum / GRAVITY_M_S2
        result["maximum_elevation_m"] = maximum / GRAVITY_M_S2
    return result


def audit_daily_group(
    path: Path,
    year: int,
    month: int,
    short_names: set[str],
) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {path}")
    with open_netcdf_windows_safe(path) as dataset:
        audit_global_grid(dataset)
        missing = short_names.difference(dataset.variables)
        if missing:
            raise ValueError(f"{path.name} is missing variables {sorted(missing)}.")
        times = decoded_times(dataset)
        expected_days = calendar.monthrange(year, month)[1]
        actual = {(item.year, item.month, item.day) for item in times}
        expected = {(year, month, day) for day in range(1, expected_days + 1)}
        if len(times) != expected_days or actual != expected:
            raise ValueError(f"Incomplete daily timestamps for {year:04d}-{month:02d}.")
        sample_summary: dict[str, list[float]] = {}
        for name in sorted(short_names):
            variable = dataset.variables[name]
            sample = np.ma.asarray(variable[0, ::180, ::360]).filled(np.nan)
            finite = np.asarray(sample, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                raise ValueError(f"No finite audit samples for {name} in {path.name}.")
            sample_summary[name] = [float(finite.min()), float(finite.max())]
    return {
        "status": "complete",
        "file": str(path),
        "file_bytes": path.stat().st_size,
        "year": year,
        "month": month,
        "stored_days": len(times),
        "variables": sorted(short_names),
        "sample_minimum_maximum": sample_summary,
        "audited_utc": utc_now(),
    }


def audit_land_cover(path: Path) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {path}")
    with open_netcdf_windows_safe(path) as dataset:
        if "lccs_class" not in dataset.variables:
            raise ValueError(f"{path.name} does not contain lccs_class.")
        variable = dataset.variables["lccs_class"]
        if len(variable.shape) != 3 or variable.shape[0] != 1:
            raise ValueError(f"Unexpected lccs_class shape: {variable.shape}.")
        latitude_name = coordinate_name(dataset, ("lat", "latitude"))
        longitude_name = coordinate_name(dataset, ("lon", "longitude"))
        latitude = dataset.variables[latitude_name]
        longitude = dataset.variables[longitude_name]
        variable_shape = tuple(int(value) for value in variable.shape)
        latitude_count = int(latitude.shape[0])
        longitude_count = int(longitude.shape[0])
        samples = np.ma.asarray(
            variable[
                0,
                :: max(1, variable.shape[1] // 18),
                :: max(1, variable.shape[2] // 36),
            ]
        ).filled(np.nan)
        finite = np.asarray(samples, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("No finite lccs_class audit samples.")
    return {
        "status": "complete",
        "file": str(path),
        "file_bytes": path.stat().st_size,
        "item": LAND_COVER_ITEM,
        "shape": list(variable_shape),
        "latitude_count": latitude_count,
        "longitude_count": longitude_count,
        "sample_classes": sorted({int(value) for value in finite}),
        "audited_utc": utc_now(),
    }


def quarantine_invalid_file(path: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    quarantined = path.with_name(f"{path.name}.invalid-{timestamp}")
    path.replace(quarantined)
    return quarantined


def cds_download_with_retries(
    client: cdsapi.Client,
    dataset: str,
    request: dict[str, object],
    target: Path,
    *,
    attempts: int,
) -> Path:
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            if partial.exists():
                partial.unlink()
            result = client.retrieve(dataset, request)
            result.download(str(partial))
            if not partial.exists() or partial.stat().st_size == 0:
                raise RuntimeError("CDS returned no downloadable file.")
            return partial
        except Exception:
            if attempt >= attempts:
                raise
            delay_seconds = min(60 * (2 ** (attempt - 1)), 900)
            print(
                f"CDS attempt {attempt}/{attempts} failed; retrying in "
                f"{delay_seconds} seconds.",
                flush=True,
            )
            time.sleep(delay_seconds)
    raise AssertionError("unreachable")


def download_hourly_temperature_month(
    client: cdsapi.Client,
    root: Path,
    year: int,
    month: int,
    *,
    attempts: int,
) -> dict[str, object]:
    directory = root / f"{year:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"era5_t2m_{year:04d}{month:02d}_global_hourly"
    target = directory / f"{stem}.nc"
    audit_path = directory / f"{stem}.audit.json"
    request = hourly_temperature_request(year, month)
    atomic_write_json(
        directory / f"{stem}.request.json",
        {"dataset": ERA5_HOURLY_DATASET, "request": request},
    )
    if target.exists():
        try:
            audit = audit_hourly_temperature(target, year, month)
            atomic_write_json(audit_path, audit)
            print(f"SKIP valid hourly temperature: {year:04d}-{month:02d}")
            return audit
        except Exception as error:
            quarantined = quarantine_invalid_file(target)
            print(f"Quarantined {quarantined.name}: {error}")
    print(f"REQUEST hourly temperature {year:04d}-{month:02d}", flush=True)
    partial = cds_download_with_retries(
        client,
        ERA5_HOURLY_DATASET,
        request,
        target,
        attempts=attempts,
    )
    audit = audit_hourly_temperature(partial, year, month)
    os.replace(partial, target)
    audit["file"] = str(target)
    atomic_write_json(audit_path, audit)
    return audit


def download_static_fields(
    client: cdsapi.Client, root: Path, *, attempts: int
) -> list[dict[str, object]]:
    directory = root / "static"
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for variable, specification in STATIC_FIELDS.items():
        target = directory / specification["filename"]
        short_name = specification["short_name"]
        audit_path = target.with_suffix(".audit.json")
        request_path = target.with_suffix(".request.json")
        request = static_request(variable)
        atomic_write_json(
            request_path,
            {"dataset": ERA5_HOURLY_DATASET, "request": request},
        )
        if target.exists():
            try:
                audit = audit_static(target, short_name)
                atomic_write_json(audit_path, audit)
                print(f"SKIP valid static field: {variable}")
                results.append(audit)
                continue
            except Exception as error:
                quarantined = quarantine_invalid_file(target)
                print(f"Quarantined {quarantined.name}: {error}")
        print(f"REQUEST static field: {variable}", flush=True)
        partial = cds_download_with_retries(
            client,
            ERA5_HOURLY_DATASET,
            request,
            target,
            attempts=attempts,
        )
        audit = audit_static(partial, short_name)
        os.replace(partial, target)
        audit["file"] = str(target)
        atomic_write_json(audit_path, audit)
        results.append(audit)
    return results


def download_daily_group_month(
    client: cdsapi.Client,
    root: Path,
    statistic: str,
    year: int,
    month: int,
    *,
    attempts: int,
) -> dict[str, object]:
    specification = DAILY_GROUPS[statistic]
    directory = root / "daily_predictors" / f"{year:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"era5_{statistic}_{year:04d}{month:02d}_global_025"
    target = directory / f"{stem}.nc"
    audit_path = directory / f"{stem}.audit.json"
    request = daily_request(
        specification["variables"], year, month, statistic
    )
    atomic_write_json(
        directory / f"{stem}.request.json",
        {"dataset": ERA5_DAILY_DATASET, "request": request},
    )
    if target.exists():
        try:
            audit = audit_daily_group(
                target, year, month, specification["short_names"]
            )
            atomic_write_json(audit_path, audit)
            print(f"SKIP valid {statistic}: {year:04d}-{month:02d}")
            return audit
        except Exception as error:
            quarantined = quarantine_invalid_file(target)
            print(f"Quarantined {quarantined.name}: {error}")
    print(f"REQUEST {statistic} {year:04d}-{month:02d}", flush=True)
    partial = cds_download_with_retries(
        client,
        ERA5_DAILY_DATASET,
        request,
        target,
        attempts=attempts,
    )
    audit = audit_daily_group(
        partial, year, month, specification["short_names"]
    )
    os.replace(partial, target)
    audit["file"] = str(target)
    atomic_write_json(audit_path, audit)
    return audit


def read_json_url(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "gev-global-predictors/1.0"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def land_cover_asset() -> tuple[str, int | None]:
    item_url = (
        f"{LAND_COVER_STAC}/collections/{LAND_COVER_COLLECTION}/items/"
        f"{LAND_COVER_ITEM}"
    )
    item = read_json_url(item_url)
    raw_url = str(item["assets"]["netcdf"]["href"])
    signed = read_json_url(f"{LAND_COVER_SIGN}?href={quote(raw_url, safe='')}")
    signed_url = str(signed["href"])
    size: int | None = None
    try:
        with urlopen(Request(signed_url, method="HEAD"), timeout=120) as response:
            header = response.headers.get("Content-Length")
            size = int(header) if header else None
    except Exception:
        # The download remains possible even if the server rejects HEAD.
        pass
    return signed_url, size


def http_range_download(
    target: Path,
    *,
    attempts: int,
) -> Path:
    """Download the ESA file with byte-range resume across reruns."""

    partial = target.with_suffix(target.suffix + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            signed_url, expected_size = land_cover_asset()
            offset = partial.stat().st_size if partial.exists() else 0
            if expected_size is not None and offset == expected_size:
                return partial
            if expected_size is not None and offset > expected_size:
                partial.unlink()
                offset = 0
            headers = {"User-Agent": "gev-global-predictors/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urlopen(Request(signed_url, headers=headers), timeout=300) as response:
                status = getattr(response, "status", response.getcode())
                append = offset > 0 and status == 206
                if offset > 0 and not append:
                    offset = 0
                mode = "ab" if append else "wb"
                with partial.open(mode) as output:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Land-cover size mismatch: {partial.stat().st_size} != "
                    f"{expected_size}."
                )
            return partial
        except Exception:
            if attempt >= attempts:
                raise
            delay_seconds = min(60 * (2 ** (attempt - 1)), 900)
            print(
                f"Land-cover attempt {attempt}/{attempts} failed; retrying in "
                f"{delay_seconds} seconds.",
                flush=True,
            )
            time.sleep(delay_seconds)
    raise AssertionError("unreachable")


def download_land_cover(root: Path, *, attempts: int) -> dict[str, object]:
    directory = root / "static"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / LAND_COVER_FILENAME
    audit_path = target.with_suffix(".audit.json")
    if target.exists():
        try:
            audit = audit_land_cover(target)
            atomic_write_json(audit_path, audit)
            print("SKIP valid ESA CCI 2000 global land cover")
            return audit
        except Exception as error:
            quarantined = quarantine_invalid_file(target)
            print(f"Quarantined {quarantined.name}: {error}")
    print("REQUEST ESA CCI 2000 global land cover", flush=True)
    partial = http_range_download(target, attempts=attempts)
    audit = audit_land_cover(partial)
    os.replace(partial, target)
    audit["file"] = str(target)
    atomic_write_json(audit_path, audit)
    return audit


def audit_coastline_archive(path: Path) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {path}")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Invalid GSHHG zip archive: {path}")
    required = {
        "GSHHS_i_L1.shp",
        "GSHHS_i_L1.shx",
        "GSHHS_i_L1.dbf",
        "GSHHS_i_L1.prj",
    }
    with zipfile.ZipFile(path) as archive:
        basenames = {Path(name).name for name in archive.namelist()}
    missing = required.difference(basenames)
    if missing:
        raise ValueError(f"GSHHG archive is missing {sorted(missing)}.")
    return {
        "status": "complete",
        "file": str(path),
        "file_bytes": path.stat().st_size,
        "version": GSHHG_VERSION,
        "required_intermediate_level_1_files": sorted(required),
        "audited_utc": utc_now(),
    }


def download_public_file_with_retries(
    url: str, target: Path, *, attempts: int
) -> Path:
    """Download a public file while retaining a useful partial for resume."""

    partial = target.with_suffix(target.suffix + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "gev-global-predictors/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urlopen(Request(url, headers=headers), timeout=300) as response:
                status = getattr(response, "status", response.getcode())
                append = offset > 0 and status == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as output:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            if not partial.exists() or partial.stat().st_size == 0:
                raise RuntimeError("HTTP response produced no file.")
            return partial
        except Exception:
            if attempt >= attempts:
                raise
            delay_seconds = min(60 * (2 ** (attempt - 1)), 900)
            print(
                f"HTTP attempt {attempt}/{attempts} failed; retrying in "
                f"{delay_seconds} seconds.",
                flush=True,
            )
            time.sleep(delay_seconds)
    raise AssertionError("unreachable")


def download_coastline(root: Path, *, attempts: int) -> dict[str, object]:
    directory = root / "static"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / GSHHG_FILENAME
    audit_path = target.with_suffix(".audit.json")
    if target.exists():
        try:
            audit = audit_coastline_archive(target)
            atomic_write_json(audit_path, audit)
            print(f"SKIP valid GSHHG {GSHHG_VERSION} coastline")
            return audit
        except Exception as error:
            quarantined = quarantine_invalid_file(target)
            print(f"Quarantined {quarantined.name}: {error}")
    print(f"REQUEST GSHHG {GSHHG_VERSION} coastline", flush=True)
    partial = download_public_file_with_retries(
        GSHHG_URL, target, attempts=attempts
    )
    audit = audit_coastline_archive(partial)
    os.replace(partial, target)
    audit["file"] = str(target)
    atomic_write_json(audit_path, audit)
    return audit


def parse_components(value: str) -> set[str]:
    requested = {item.strip().lower() for item in value.split(",") if item.strip()}
    if "all" in requested:
        requested.remove("all")
        requested.update(COMPONENTS)
    unknown = requested.difference(COMPONENTS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown components {sorted(unknown)}; choose from "
            f"{sorted(COMPONENTS)} or all."
        )
    if not requested:
        raise argparse.ArgumentTypeError("At least one component is required.")
    return requested


def estimate_hourly_temperature_bytes(start_year: int, end_year: int) -> int:
    total_days = sum(
        calendar.monthrange(year, month)[1]
        for year, month in iter_months(start_year, end_year)
    )
    return int(total_days * (PILOT_MONTH_BYTES / PILOT_MONTH_DAYS) * 1.05)


def build_plan(args: argparse.Namespace, root: Path) -> dict[str, object]:
    months = 12 * (args.end_year - args.start_year + 1)
    daily_requests = months * len(DAILY_GROUPS)
    disk = shutil.disk_usage(root)
    return {
        "analysis_period": [args.start_year, args.end_year],
        "components": sorted(args.components),
        "output_directory": str(root),
        "hourly_temperature_requests": (
            months if "hourly-temperature" in args.components else 0
        ),
        "daily_predictor_requests": (
            daily_requests if "daily-predictors" in args.components else 0
        ),
        "static_requests": 2 if "static" in args.components else 0,
        "land_cover_requests": 1 if "land-cover" in args.components else 0,
        "coastline_requests": 1 if "coastline" in args.components else 0,
        "estimated_hourly_temperature_GB": (
            estimate_hourly_temperature_bytes(args.start_year, args.end_year) / 1e9
            if "hourly-temperature" in args.components
            else 0.0
        ),
        "free_GB": disk.free / 1e9,
        "notes": [
            "Daily-predictor size is not guessed; CDS compression varies by field.",
            "LSM and geopotential are invariant; year 2000 is only a required CDS timestamp.",
            "ESA CCI land cover is the fixed year-2000 candidate used by the Taiwan workflow.",
            "Terrain derivatives and coast distance are derived from geopotential and GSHHG.",
        ],
        "created_utc": utc_now(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download resumable global sources for annual GEV predictors."
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--components",
        type=parse_components,
        required=True,
        help=(
            "Comma-separated: hourly-temperature, static, land-cover, "
            "coastline, daily-predictors, or all."
        ),
    )
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def run_monthly_component(
    labels: list[tuple[str, int, int]],
    worker: Any,
    *,
    fail_fast: bool,
    status_path: Path,
) -> None:
    completed: list[str] = []
    failed: list[dict[str, str]] = []
    for index, (prefix, year, month) in enumerate(labels, start=1):
        label = f"{prefix}:{year:04d}-{month:02d}"
        print(f"[{index}/{len(labels)}] {label}", flush=True)
        try:
            worker(prefix, year, month)
            completed.append(label)
        except Exception as error:
            failed.append(
                {
                    "item": label,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(f"FAILED {label}: {error}", flush=True)
            if fail_fast:
                atomic_write_json(
                    status_path,
                    {
                        "status": "failed",
                        "completed": completed,
                        "failed": failed,
                        "updated_utc": utc_now(),
                    },
                )
                raise
        atomic_write_json(
            status_path,
            {
                "status": "running" if index < len(labels) else "finished",
                "completed_count": len(completed),
                "failed_count": len(failed),
                "last_processed": label,
                "completed": completed,
                "failed": failed,
                "updated_utc": utc_now(),
            },
        )
    if failed:
        raise RuntimeError(
            f"{len(failed)} item(s) failed. Re-run the same command; valid files "
            "will be audited and skipped."
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start-year must not exceed end-year.")
    if args.start_year < 1940:
        raise ValueError("ERA5 is unavailable before 1940.")
    if args.attempts <= 0:
        raise ValueError("attempts must be positive.")

    root = args.output_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args, root)
    atomic_write_json(root / "global_predictor_download_plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; no remote data were requested.")
        return

    client: cdsapi.Client | None = None
    if args.components.intersection(
        {"hourly-temperature", "static", "daily-predictors"}
    ):
        client = cdsapi.Client(
            quiet=False,
            debug=False,
            progress=True,
            timeout=300,
            retry_max=10,
            sleep_max=120,
        )

    if "static" in args.components:
        assert client is not None
        download_static_fields(client, root, attempts=args.attempts)

    if "land-cover" in args.components:
        download_land_cover(root, attempts=args.attempts)

    if "coastline" in args.components:
        download_coastline(root, attempts=args.attempts)

    months = list(iter_months(args.start_year, args.end_year))
    if "hourly-temperature" in args.components:
        assert client is not None
        hourly_labels = [("hourly-temperature", year, month) for year, month in months]

        def hourly_worker(_prefix: str, year: int, month: int) -> None:
            download_hourly_temperature_month(
                client, root, year, month, attempts=args.attempts
            )

        run_monthly_component(
            hourly_labels,
            hourly_worker,
            fail_fast=args.fail_fast,
            status_path=root / "hourly_temperature_status.json",
        )

    if "daily-predictors" in args.components:
        assert client is not None
        daily_labels = [
            (statistic, year, month)
            for year, month in months
            for statistic in DAILY_GROUPS
        ]

        def daily_worker(statistic: str, year: int, month: int) -> None:
            download_daily_group_month(
                client,
                root,
                statistic,
                year,
                month,
                attempts=args.attempts,
            )

        run_monthly_component(
            daily_labels,
            daily_worker,
            fail_fast=args.fail_fast,
            status_path=root / "daily_predictor_status.json",
        )

    print("Requested global predictor components are complete and audited.")


if __name__ == "__main__":
    main()
