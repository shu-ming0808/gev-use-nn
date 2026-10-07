"""Stream hourly ERA5 t2m to audited annual grid maxima (UTC calendar years).

Keeps all land/ocean cells in NetCDF, and exports a land-only wide CSV using
LSM > threshold. Original hourly files are read-only. Completed years are
reused only when source fingerprints, processing settings and output hashes
match; an incomplete year is recomputed. No joblib temporary memmaps are used.
"""
from __future__ import annotations

import argparse
import calendar
import contextlib
import csv
import datetime as dt
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import netCDF4 as nc
import numpy as np

VERSION = 1
EPOCH = "seconds since 1970-01-01 00:00:00"
DEFAULT_INPUT = Path("D:/論文資料/ERA5/1975-2025_global_hourly")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    if not stat.st_size:
        raise ValueError(f"Empty input: {path}")
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@contextlib.contextmanager
def open_nc(path, mode="r"):
    # netCDF on Windows can fail on Chinese paths. Each worker is a separate
    # process; there are no threads that share this temporary cwd change.
    path = Path(path).resolve()
    previous = Path.cwd()
    try:
        os.chdir(path.parent)
        with nc.Dataset(path.name, mode) as dataset:
            yield dataset
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def run_lock(directory):
    """OS lock is automatically released even if the Python process dies."""
    path = Path(directory) / ".annual_maxima.lock"
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def coordinates(dataset, global_grid=True):
    lat = np.asarray(dataset["latitude"][:], dtype=np.float64)
    lon = np.asarray(dataset["longitude"][:], dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1 or not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Invalid latitude/longitude coordinates")
    if global_grid and not (np.array_equal(lat, np.arange(90, -90.001, -.25))
                            and np.array_equal(lon, np.arange(0, 360, .25))):
        raise ValueError("Expected north-to-south global 0.25 degree ERA5 grid")
    return lat, lon


def celsius_offset(units):
    if str(units).strip().lower() in {"k", "kelvin"}:
        return 273.15
    if str(units).strip().lower() in {"degrees_celsius", "degree_celsius", "degc", "celsius"}:
        return 0.0
    raise ValueError(f"Unknown temperature units: {units!r}")


def audit_month(path, year, month, expected_grid=None, global_grid=True):
    before = fingerprint(path)
    with open_nc(path) as ds:
        lat, lon = coordinates(ds, global_grid)
        if expected_grid is not None and not all(np.array_equal(a, b) for a, b in zip((lat, lon), expected_grid)):
            raise ValueError(f"Grid mismatch: {path}")
        tname = "valid_time" if "valid_time" in ds.variables else "time"
        variable = ds["t2m"]
        if variable.dimensions != (tname, "latitude", "longitude"):
            raise ValueError(f"Unexpected t2m dimensions in {path}: {variable.dimensions}")
        offset = celsius_offset(getattr(variable, "units", None))
        tv = ds[tname]
        cal = getattr(tv, "calendar", "standard")
        if cal not in {"standard", "gregorian", "proleptic_gregorian"}:
            raise ValueError(f"Unsupported calendar: {cal}")
        dates = nc.num2date(tv[:], tv.units, cal)
        seconds = np.asarray(nc.date2num(dates, EPOCH, "standard"), dtype=np.float64)
        start = (dt.datetime(year, month, 1) - dt.datetime(1970, 1, 1)).total_seconds()
        expected = start + 3600 * np.arange(calendar.monthrange(year, month)[1] * 24)
        if not np.array_equal(seconds, expected):
            raise ValueError(f"Missing, duplicated, unordered or non-hourly timestamps: {path}")
        if variable.shape != (len(expected), len(lat), len(lon)):
            raise ValueError(f"Unexpected array shape: {path}")
        chunks = variable.chunking()
        chunk_hours = int(chunks[0]) if isinstance(chunks, list) else 24
    if fingerprint(path) != before:
        raise ValueError(f"Input changed during audit: {path}")
    return {**before, "year": year, "month": month, "hours": len(expected),
            "offset": offset, "start_seconds": int(start), "chunk_hours": chunk_hours}, (lat, lon)


def reduce_block(values, seconds, maximum, peak_time, counts):
    """Accumulate finite hourly observations; tied maxima keep earliest UTC hour."""
    values = np.asarray(np.ma.filled(values, np.nan), dtype=np.float32)
    valid = np.isfinite(values)
    counts += valid.sum(axis=0, dtype=np.uint32)
    values[~valid] = -np.inf
    local = values.max(axis=0)
    improved = local > maximum
    if improved.any():
        # Argmax along time would copy the entire strided 3-D cube. Scan time
        # instead, so memory remains bounded and first-tie semantics are clear.
        unresolved = improved.copy()
        for i, second in enumerate(seconds):
            selected = unresolved & (values[i] == local)
            peak_time[selected] = second
            unresolved[selected] = False
            if not unresolved.any():
                break
        maximum[improved] = local[improved]


def write_annual(path, year, lat, lon, maximum, peak_time, counts, expected):
    tmp = path.with_suffix(".nc.part")
    complete = counts == expected
    with open_nc(tmp, "w") as ds:
        ds.createDimension("latitude", len(lat))
        ds.createDimension("longitude", len(lon))
        for name, array, units in (("latitude", lat, "degrees_north"), ("longitude", lon, "degrees_east")):
            var = ds.createVariable(name, "f8", (name,))
            var[:] = array
            var.units = units
        dims = ("latitude", "longitude")
        vmax = ds.createVariable("annual_maximum_t2m", "f4", dims, zlib=True, complevel=2, fill_value=np.nan)
        vmax[:] = np.where(complete, maximum, np.nan)
        vmax.units = "degrees_Celsius"
        vmax.long_name = "Maximum of all hourly 2 m temperatures in the UTC calendar year"
        vtime = ds.createVariable("maximum_time", "f8", dims, zlib=True, complevel=2, fill_value=np.nan)
        vtime[:] = np.where(complete, peak_time, np.nan)
        vtime.units, vtime.calendar = EPOCH, "standard"
        ds.createVariable("valid_hour_count", "u2", dims, zlib=True, complevel=2)[:] = counts
        ds.year, ds.expected_hours = year, expected
        ds.missing_policy = "Annual maximum is missing unless every expected hourly value is finite."
        ds.tie_policy = "Earliest UTC hour"
        ds.time_definition = "UTC calendar year; not local-time year or daily Tmax product"
        ds.processing_version = VERSION
    os.replace(tmp, path)
    return int(complete.sum())


def prepare_year(year, records, output, global_grid=True):
    output = Path(output).resolve()
    path = output / f"era5_t2m_annual_maximum_{year}.nc"
    marker = output / f"year_{year}.json"
    signature = {"version": VERSION, "year": year, "sources": records,
                 "missing_policy": "all_hours_required", "time_zone": "UTC"}
    if path.exists() and marker.exists():
        try:
            cached = json.loads(marker.read_text(encoding="utf-8"))
            if cached["signature"] == signature and digest(path) == cached["sha256"]:
                return {**cached, "reused": True}
        except (KeyError, ValueError, OSError):
            pass
    started = time.perf_counter()
    with open_nc(records[0]["path"]) as ds:
        lat, lon = coordinates(ds, global_grid)
    maximum = np.full((len(lat), len(lon)), -np.inf, dtype=np.float32)
    peak_time = np.full(maximum.shape, -1, dtype=np.int64)
    counts = np.zeros(maximum.shape, dtype=np.uint32)
    for item in records:
        source = Path(item["path"])
        wanted = {k: item[k] for k in ("path", "bytes", "mtime_ns")}
        if fingerprint(source) != wanted:
            raise ValueError(f"Input changed since preflight: {source}")
        with open_nc(source) as ds:
            var = ds["t2m"]
            var.set_var_chunk_cache(32 * 1024 * 1024, 1009, .75)
            # Align reads with physical compressed time chunks, avoiding
            # decompression of the same chunk for each 24-hour slice.
            step = min(item["chunk_hours"], 168)
            for first in range(0, item["hours"], step):
                last = min(first + step, item["hours"])
                values = var[first:last, :, :]
                values -= np.float32(item["offset"])
                seconds = item["start_seconds"] + 3600 * np.arange(first, last)
                reduce_block(values, seconds, maximum, peak_time, counts)
        if fingerprint(source) != wanted:
            raise ValueError(f"Input changed while reading: {source}")
        atomic_json(output / f"progress_{year}.json", {"year": year, "completed_month": item["month"], "updated_utc": now()})
        print(f"{year}: month {item['month']:02d}/12 complete", flush=True)
    expected = (366 if calendar.isleap(year) else 365) * 24
    complete = write_annual(path, year, lat, lon, maximum, peak_time, counts, expected)
    result = {"signature": signature, "sha256": digest(path), "file": str(path),
              "complete_cells": complete, "incomplete_cells": int(maximum.size - complete),
              "expected_hours": expected, "seconds": time.perf_counter() - started,
              "completed_utc": now(), "reused": False}
    atomic_json(marker, result)
    return result


def read_mask(path, lat, lon):
    with open_nc(path) as ds:
        mlat, mlon = coordinates(ds, global_grid=False)
        if not np.array_equal(mlat, lat) or not np.array_equal(mlon, lon):
            raise ValueError("LSM grid differs from temperature grid")
        var = ds["lsm"]
        if var.dimensions == ("latitude", "longitude"):
            mask = var[:]
        elif var.dimensions[-2:] == ("latitude", "longitude") and var.shape[0] == 1 and var.ndim == 3:
            mask = var[0]
        else:
            raise ValueError("LSM must have exactly one static latitude/longitude field")
        mask = np.asarray(np.ma.filled(mask, np.nan), dtype=np.float32)
    if not np.isfinite(mask).all() or mask.min() < -1e-6 or mask.max() > 1 + 1e-6:
        raise ValueError("Invalid LSM values")
    return mask


def combine_years(years, output, mask_path, threshold):
    output = Path(output).resolve()
    with open_nc(output / f"era5_t2m_annual_maximum_{years[0]}.nc") as ds:
        lat, lon = coordinates(ds, global_grid=False)
    mask = read_mask(mask_path, lat, lon)
    selected = np.flatnonzero(mask.ravel() > threshold)
    land_values = np.empty((len(years), len(selected)), dtype=np.float32)
    stem = f"era5_t2m_annual_maxima_{years[0]}_{years[-1]}"
    target = output / f"{stem}_global.nc"
    with open_nc(target.with_suffix(".nc.part"), "w") as ds:
        for name, array, dtype in (("year", years, "i4"), ("latitude", lat, "f8"), ("longitude", lon, "f8")):
            ds.createDimension(name, len(array))
            var = ds.createVariable(name, dtype, (name,))
            var[:] = array
            if name != "year":
                var.units = "degrees_north" if name == "latitude" else "degrees_east"
        dims = ("year", "latitude", "longitude")
        vmax = ds.createVariable("annual_maximum_t2m", "f4", dims, zlib=True, complevel=2, fill_value=np.nan)
        vmax.units = "degrees_Celsius"
        vtime = ds.createVariable("maximum_time", "f8", dims, zlib=True, complevel=2, fill_value=np.nan)
        vtime.units, vtime.calendar = EPOCH, "standard"
        vcount = ds.createVariable("valid_hour_count", "u2", dims, zlib=True, complevel=2)
        ds.createVariable("expected_hours", "u2", ("year",))[:] = [(366 if calendar.isleap(y) else 365) * 24 for y in years]
        ds.createVariable("lsm", "f4", ("latitude", "longitude"), zlib=True)[:] = mask
        for i, year in enumerate(years):
            with open_nc(output / f"era5_t2m_annual_maximum_{year}.nc") as annual:
                if not all(np.array_equal(a, b) for a, b in zip(coordinates(annual, False), (lat, lon))):
                    raise ValueError("Annual grid mismatch")
                values = np.asarray(np.ma.filled(annual["annual_maximum_t2m"][:], np.nan))
                vmax[i], vtime[i], vcount[i] = values, annual["maximum_time"][:], annual["valid_hour_count"][:]
                land_values[i] = values.ravel()[selected]
        ds.time_definition = "UTC calendar year; maximum of hourly t2m, not daily Tmax"
        ds.missing_policy = "All expected hours required per cell/year"
        ds.source_directory = str(output)
        ds.processing_version = VERSION
    os.replace(target.with_suffix(".nc.part"), target)
    table = output / f"{stem}_land.csv"
    with table.with_suffix(".csv.part").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["grid_id", "latitude", "longitude", "longitude_180", "lsm", *years])
        for j, index in enumerate(selected):
            row, col = divmod(int(index), len(lon))
            writer.writerow([f"G{row:03d}_{col:04d}", lat[row], lon[col], (lon[col]+180) % 360-180,
                             f"{mask[row,col]:.6f}", *[f"{v:.4f}" if np.isfinite(v) else "" for v in land_values[:, j]]])
    os.replace(table.with_suffix(".csv.part"), table)
    return {"global_netcdf": str(target), "land_csv": str(table), "global_cells": len(lat)*len(lon),
            "land_cells": len(selected), "land_threshold_strict_greater_than": threshold,
            "years": years, "land_cell_years_missing": int((~np.isfinite(land_values)).sum()),
            "global_sha256": digest(target), "land_csv_sha256": digest(table)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-directory", type=Path)
    p.add_argument("--start-year", type=int, default=1976)
    p.add_argument("--end-year", type=int, default=2004)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--land-sea-mask", type=Path)
    p.add_argument("--land-threshold", type=float, default=.5)
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    if args.start_year > args.end_year or args.workers < 1 or not 0 <= args.land_threshold <= 1:
        p.error("Invalid year range, workers or land threshold")
    root = args.input_directory.resolve()
    output = (args.output_directory or root.parent / f"{args.start_year}-{args.end_year}_global_annual_maxima").resolve()
    if output == root or root in output.parents:
        p.error("Keep derived output outside the raw hourly directory")
    output.mkdir(parents=True, exist_ok=True)
    mask_path = (args.land_sea_mask or root / "static/era5_land_sea_mask_global_025.nc").resolve()
    years = list(range(args.start_year, args.end_year + 1))
    started = time.perf_counter()
    with run_lock(output):
        state = {"status": "auditing", "started_utc": now(), "years": years, "workers": args.workers,
                 "input_directory": str(root), "output_directory": str(output), "completed_years": []}
        atomic_json(output / "run_status.json", state)
        try:
            records, grid = {}, None
            for year in years:
                records[year] = []
                for month in range(1, 13):
                    source = root / str(year) / f"era5_t2m_{year}{month:02d}_global_hourly.nc"
                    item, grid = audit_month(source, year, month, grid)
                    records[year].append(item)
                print(f"Audit {year}: all 12 months / exact hourly timestamps OK", flush=True)
            read_mask(mask_path, *grid)
            atomic_json(output / "input_audit.json", {"sources": records, "mask": fingerprint(mask_path), "audited_utc": now()})
            state.update(status="audit_complete" if args.audit_only else "running", input_months=12*len(years))
            atomic_json(output / "run_status.json", state)
            if args.audit_only:
                return
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(prepare_year, y, records[y], output): y for y in years}
                try:
                    for future in as_completed(futures):
                        result = future.result()
                        state["completed_years"].append(futures[future])
                        state["completed_years"].sort()
                        state.update(elapsed_seconds=time.perf_counter()-started, updated_utc=now())
                        atomic_json(output / "run_status.json", state)
                        print(f"YEAR {futures[future]} complete: {result['seconds']:.1f}s, reused={result['reused']}", flush=True)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
            state.update(status="combining")
            atomic_json(output / "run_status.json", state)
            outputs = combine_years(years, output, mask_path, args.land_threshold)
            state.update(status="completed", outputs=outputs, completed_utc=now(), elapsed_seconds=time.perf_counter()-started)
            atomic_json(output / "run_status.json", state)
            print(json.dumps(state, ensure_ascii=True, indent=2), flush=True)
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         error=f"{type(error).__name__}: {error}", elapsed_seconds=time.perf_counter()-started)
            atomic_json(output / "run_status.json", state)
            raise


if __name__ == "__main__":
    main()
