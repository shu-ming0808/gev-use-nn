"""Resumable monthly download of global hourly ERA5 2 m temperature.

The default period is 1975-2025 inclusive.  One CDS request is submitted per
calendar month, following Copernicus guidance for large ERA5 retrievals.  A
single invariant land-sea mask is downloaded separately.

Existing monthly files are audited and skipped.  Downloads first use a
``.part`` suffix and are moved into place only after a structural audit, so a
stopped run can be restarted safely with the same command.
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
from pathlib import Path
from typing import Iterator

import cdsapi
import netCDF4
import numpy as np


DATASET = "reanalysis-era5-single-levels"
ALL_HOURS = [f"{hour:02d}:00" for hour in range(24)]
GLOBAL_GRID = [0.25, 0.25]
DEFAULT_ROOT = Path("D:/論文資料/ERA5/1975-2025_global_hourly")
PILOT_MONTH_BYTES = 1_111_236_798
PILOT_MONTH_DAYS = 31


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def month_days(year: int, month: int) -> list[str]:
    return [
        f"{day:02d}"
        for day in range(1, calendar.monthrange(year, month)[1] + 1)
    ]


def temperature_request(year: int, month: int) -> dict[str, object]:
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


def land_sea_mask_request() -> dict[str, object]:
    """Request one copy because the ERA5 atmospheric LSM is invariant."""

    return {
        "product_type": ["reanalysis"],
        "variable": ["land_sea_mask"],
        "year": ["2000"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "grid": GLOBAL_GRID,
    }


def expected_month_datetimes(year: int, month: int) -> set[tuple[int, ...]]:
    days = calendar.monthrange(year, month)[1]
    return {
        (year, month, day, hour)
        for day in range(1, days + 1)
        for hour in range(24)
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


def audit_global_grid(dataset: netCDF4.Dataset) -> tuple[np.ndarray, np.ndarray]:
    latitude = np.asarray(dataset.variables["latitude"][:], dtype=np.float64)
    longitude = np.asarray(dataset.variables["longitude"][:], dtype=np.float64)
    if latitude.shape != (721,) or longitude.shape != (1440,):
        raise ValueError(
            f"Unexpected global grid: latitude={latitude.shape}, "
            f"longitude={longitude.shape}."
        )
    if not np.allclose(np.sort(latitude), np.arange(-90.0, 90.001, 0.25)):
        raise ValueError("The latitude coordinate is not the global 0.25 degree grid.")
    if not np.allclose(
        np.sort(longitude % 360.0), np.arange(0.0, 360.0, 0.25)
    ):
        raise ValueError("The longitude coordinate is not the global 0.25 degree grid.")
    return latitude, longitude


def audit_temperature_file(
    path: Path,
    year: int,
    month: int,
    *,
    full_value_audit: bool = False,
) -> dict[str, object]:
    """Validate timestamps, grid, dimensions, and representative values."""

    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {path}")

    with open_netcdf_windows_safe(path) as dataset:
        variable_name = "t2m"
        if variable_name not in dataset.variables:
            raise ValueError(f"{path.name} does not contain t2m.")
        variable = dataset.variables[variable_name]
        time_name = (
            "valid_time" if "valid_time" in dataset.variables else "time"
        )
        expected_dimensions = (time_name, "latitude", "longitude")
        if tuple(variable.dimensions) != expected_dimensions:
            raise ValueError(
                f"Unexpected t2m dimensions: {variable.dimensions}; "
                f"expected {expected_dimensions}."
            )

        latitude, longitude = audit_global_grid(dataset)
        time_variable = dataset.variables[time_name]
        decoded = netCDF4.num2date(
            time_variable[:],
            time_variable.units,
            getattr(time_variable, "calendar", "standard"),
        )
        actual = {
            (item.year, item.month, item.day, item.hour)
            for item in decoded
        }
        expected = expected_month_datetimes(year, month)
        if len(decoded) != len(expected) or actual != expected:
            missing = sorted(expected - actual)[:5]
            extra = sorted(actual - expected)[:5]
            raise ValueError(
                f"Timestamp audit failed for {year:04d}-{month:02d}; "
                f"missing examples={missing}, extra examples={extra}."
            )

        time_indices = sorted({0, len(decoded) // 2, len(decoded) - 1})
        coordinate_pairs = [
            (25.0, 121.5),
            (0.0, 0.0),
            (-30.0, 150.0),
            (0.0, 220.0),
        ]
        samples = []
        for time_index in time_indices:
            for sample_latitude, sample_longitude in coordinate_pairs:
                lat_index = int(np.argmin(np.abs(latitude - sample_latitude)))
                lon_distance = np.abs(
                    (longitude - sample_longitude + 180.0) % 360.0 - 180.0
                )
                lon_index = int(np.argmin(lon_distance))
                value = float(variable[time_index, lat_index, lon_index])
                if not np.isfinite(value):
                    raise ValueError(
                        f"Non-finite audit sample in {path.name} at "
                        f"time index {time_index}."
                    )
                samples.append(value)

        missing_values = None
        if full_value_audit:
            missing_values = 0
            for start in range(0, len(decoded), 24):
                stop = min(start + 24, len(decoded))
                raw = np.ma.asarray(variable[start:stop, :, :])
                values = np.asarray(raw.filled(np.nan), dtype=np.float32)
                valid = ~np.ma.getmaskarray(raw) & np.isfinite(values)
                missing_values += int(valid.size - valid.sum())

    return {
        "status": "complete",
        "file": str(path),
        "file_bytes": path.stat().st_size,
        "year": year,
        "month": month,
        "expected_hours": len(expected),
        "stored_hours": len(decoded),
        "sample_minimum_K": min(samples),
        "sample_maximum_K": max(samples),
        "full_value_audit": full_value_audit,
        "missing_values": missing_values,
        "audited_utc": utc_now(),
    }


def audit_land_sea_mask(path: Path) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {path}")
    with open_netcdf_windows_safe(path) as dataset:
        if "lsm" not in dataset.variables:
            raise ValueError(f"{path.name} does not contain lsm.")
        audit_global_grid(dataset)
        values = np.ma.asarray(dataset.variables["lsm"][:]).filled(np.nan)
        minimum = float(np.nanmin(values))
        maximum = float(np.nanmax(values))
        if minimum < -1e-6 or maximum > 1.0 + 1e-6:
            raise ValueError(
                f"LSM values must be in [0, 1], observed [{minimum}, {maximum}]."
            )
    return {
        "status": "complete",
        "file": str(path),
        "file_bytes": path.stat().st_size,
        "minimum": minimum,
        "maximum": maximum,
        "audited_utc": utc_now(),
    }


def quarantine_invalid_file(path: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    quarantined = path.with_name(f"{path.name}.invalid-{timestamp}")
    path.replace(quarantined)
    return quarantined


def download_with_retries(
    client: cdsapi.Client,
    request: dict[str, object],
    target: Path,
    *,
    attempts: int,
) -> None:
    """Download to a partial file and preserve it only for the active retry."""

    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            if partial.exists():
                partial.unlink()
            result = client.retrieve(DATASET, request)
            result.download(str(partial))
            if not partial.exists() or partial.stat().st_size == 0:
                raise RuntimeError("CDS returned no downloadable file.")
            return
        except Exception:
            if attempt >= attempts:
                raise
            delay_seconds = min(60 * (2 ** (attempt - 1)), 900)
            print(
                f"Download attempt {attempt}/{attempts} failed; "
                f"retrying in {delay_seconds} seconds.",
                flush=True,
            )
            time.sleep(delay_seconds)


def estimate_required_bytes(start_year: int, end_year: int) -> int:
    total_days = sum(
        calendar.monthrange(year, month)[1]
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    )
    bytes_per_day = PILOT_MONTH_BYTES / PILOT_MONTH_DAYS
    return int(total_days * bytes_per_day * 1.05)


def preflight_disk_space(
    root: Path,
    start_year: int,
    end_year: int,
    *,
    ignore_space_check: bool,
) -> dict[str, float]:
    root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(root)
    required = estimate_required_bytes(start_year, end_year)
    summary = {
        "estimated_required_GB": required / 1e9,
        "free_GB": disk.free / 1e9,
        "total_GB": disk.total / 1e9,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if disk.free < required and not ignore_space_check:
        raise RuntimeError(
            "Estimated ERA5 download is larger than free disk space. "
            "Free space or choose another output directory. Use "
            "--ignore-space-check only after verifying capacity yourself."
        )
    return summary


def download_land_sea_mask(
    client: cdsapi.Client,
    root: Path,
    attempts: int,
) -> dict[str, object]:
    static_directory = root / "static"
    static_directory.mkdir(parents=True, exist_ok=True)
    target = static_directory / "era5_land_sea_mask_global_025.nc"
    audit_path = static_directory / "land_sea_mask_audit.json"

    if target.exists():
        try:
            audit = audit_land_sea_mask(target)
            atomic_write_json(audit_path, audit)
            print(f"SKIP valid LSM: {target}", flush=True)
            return audit
        except Exception as error:
            quarantined = quarantine_invalid_file(target)
            print(
                f"Quarantined invalid LSM as {quarantined.name}: {error}",
                flush=True,
            )

    request = land_sea_mask_request()
    atomic_write_json(
        static_directory / "land_sea_mask_request.json",
        {"dataset": DATASET, "request": request},
    )
    partial = target.with_suffix(target.suffix + ".part")
    download_with_retries(client, request, target, attempts=attempts)
    audit = audit_land_sea_mask(partial)
    os.replace(partial, target)
    audit["file"] = str(target)
    atomic_write_json(audit_path, audit)
    print(f"DONE LSM: {target}", flush=True)
    return audit


def download_temperature_month(
    client: cdsapi.Client,
    root: Path,
    year: int,
    month: int,
    *,
    attempts: int,
    full_value_audit: bool,
) -> dict[str, object]:
    year_directory = root / f"{year:04d}"
    year_directory.mkdir(parents=True, exist_ok=True)
    stem = f"era5_t2m_{year:04d}{month:02d}_global_hourly"
    target = year_directory / f"{stem}.nc"
    audit_path = year_directory / f"{stem}.audit.json"
    request_path = year_directory / f"{stem}.request.json"

    if target.exists():
        try:
            audit = audit_temperature_file(
                target,
                year,
                month,
                full_value_audit=full_value_audit,
            )
            atomic_write_json(audit_path, audit)
            print(f"SKIP valid: {year:04d}-{month:02d}", flush=True)
            return audit
        except Exception as error:
            quarantined = quarantine_invalid_file(target)
            print(
                f"Quarantined invalid file as {quarantined.name}: {error}",
                flush=True,
            )

    request = temperature_request(year, month)
    atomic_write_json(
        request_path,
        {"dataset": DATASET, "request": request},
    )
    partial = target.with_suffix(target.suffix + ".part")
    print(f"REQUEST {year:04d}-{month:02d}", flush=True)
    download_with_retries(client, request, target, attempts=attempts)
    audit = audit_temperature_file(
        partial,
        year,
        month,
        full_value_audit=full_value_audit,
    )
    os.replace(partial, target)
    audit["file"] = str(target)
    atomic_write_json(audit_path, audit)
    print(
        f"DONE {year:04d}-{month:02d}: {target.stat().st_size / 1e9:.2f} GB",
        flush=True,
    )
    return audit


def iter_months(start_year: int, end_year: int) -> Iterator[tuple[int, int]]:
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield year, month


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download global hourly ERA5 2 m temperature as resumable monthly "
            "NetCDF files."
        )
    )
    parser.add_argument("--start-year", type=int, default=1975)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument(
        "--skip-land-sea-mask",
        action="store_true",
        help="Do not download the single invariant global land-sea mask.",
    )
    parser.add_argument(
        "--full-value-audit",
        action="store_true",
        help="Read every value after each download; much slower than structural audit.",
    )
    parser.add_argument(
        "--ignore-space-check",
        action="store_true",
        help="Continue even if estimated bytes exceed currently free space.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed month instead of recording it and continuing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and disk estimate without contacting CDS.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start-year must not exceed end-year.")
    if args.start_year < 1940:
        raise ValueError("ERA5 single-level data are unavailable before 1940.")
    if args.attempts <= 0:
        raise ValueError("attempts must be positive.")

    root = args.output_directory.resolve()
    disk_summary = preflight_disk_space(
        root,
        args.start_year,
        args.end_year,
        ignore_space_check=args.ignore_space_check,
    )
    months = list(iter_months(args.start_year, args.end_year))
    plan = {
        "dataset": DATASET,
        "variable": "2m_temperature",
        "start_year": args.start_year,
        "end_year": args.end_year,
        "month_requests": len(months),
        "resolution_degrees": 0.25,
        "scope": "global land and ocean",
        "output_directory": str(root),
        "include_land_sea_mask": not args.skip_land_sea_mask,
        "full_value_audit": args.full_value_audit,
        "disk": disk_summary,
        "created_utc": utc_now(),
    }
    atomic_write_json(root / "download_plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; CDS was not contacted.", flush=True)
        return

    client = cdsapi.Client(
        quiet=False,
        debug=False,
        progress=True,
        timeout=300,
        retry_max=10,
        sleep_max=120,
    )

    if not args.skip_land_sea_mask:
        download_land_sea_mask(client, root, args.attempts)

    status_path = root / "download_status.json"
    completed: list[str] = []
    failed: list[dict[str, str]] = []
    for index, (year, month) in enumerate(months, start=1):
        label = f"{year:04d}-{month:02d}"
        print(f"[{index}/{len(months)}] {label}", flush=True)
        try:
            download_temperature_month(
                client,
                root,
                year,
                month,
                attempts=args.attempts,
                full_value_audit=args.full_value_audit,
            )
            completed.append(label)
        except Exception as error:
            failure = {
                "month": label,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failed.append(failure)
            print(f"FAILED {label}: {error}", flush=True)
            if args.fail_fast:
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
                "status": "running" if index < len(months) else "finished",
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
            f"{len(failed)} month(s) failed. Re-run the same command; valid "
            "months will be skipped and failed months will be retried."
        )
    print("All requested ERA5 months are complete and audited.", flush=True)


if __name__ == "__main__":
    main()
