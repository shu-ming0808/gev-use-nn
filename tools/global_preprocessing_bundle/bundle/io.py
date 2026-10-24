"""Pinned audited NetCDF/date/unit helpers, copied for the portable bundle.
No original repository module is imported or modified.
"""
from __future__ import annotations

import argparse
import calendar
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import netCDF4 as nc
import numpy as np
import pandas as pd


VERSION = "taiwan_annual_event_matching_v1"
FIELDS = {
    "u10": ("mean", "m s-1"),
    "v10": ("mean", "m s-1"),
    "tcc": ("mean", "1"),
    "tp": ("sum", "mm"),
    "ssrd": ("sum", "MJ m-2"),
}
DERIVED_WIND = "wind_vector_magnitude"


class DataIssue(ValueError):
    """An input defect that should become an audit row, not disappear."""

    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")
    os.replace(temp, path)


def write_csv(path, frame):
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


@contextmanager
def open_nc(path):
    # netCDF-C on Windows may reject Chinese directory names. Single-process
    # reader only: temporarily use the ASCII basename, always restore cwd.
    path = Path(path).resolve()
    previous = Path.cwd()
    try:
        os.chdir(path.parent)
        with nc.Dataset(path.name, "r") as dataset:
            yield dataset
    finally:
        os.chdir(previous)


def numeric(value):
    return np.asarray(np.ma.filled(np.ma.asarray(value, dtype=float), np.nan))


def coordinate(ds, names):
    candidates = [name for name in names if name in ds.variables]
    if len(candidates) != 1:
        raise DataIssue("schema_error", f"Expected one coordinate from {names}.")
    name = candidates[0]
    var = ds[name]
    values = numeric(var[:])
    if var.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise DataIssue("grid_mismatch", f"Invalid {name} coordinate.")
    return var.dimensions[0], values


def spatial_indices(ds, grids):
    lat_dim, lat = coordinate(ds, ("latitude", "lat"))
    lon_dim, lon = coordinate(ds, ("longitude", "lon"))
    lon = (lon + 180) % 360 - 180
    indices = []
    for values, requested in ((lat, grids.latitude), (lon, grids.longitude_180)):
        if len(np.unique(values)) != len(values):
            raise DataIssue("grid_mismatch", "Duplicate coordinate, including longitude aliases.")
        # Vectorized exact lookup: global runs must not scan 1,440 coordinates
        # independently for each of ~350,000 GRID on every day.
        order = np.argsort(values)
        sorted_values = values[order]
        requested = np.asarray(requested, dtype=float)
        at = np.searchsorted(sorted_values, requested)
        at = np.clip(at, 0, len(values)-1)
        before = np.maximum(at-1, 0)
        at = np.where(abs(sorted_values[before]-requested) < abs(sorted_values[at]-requested), before, at)
        if not np.allclose(sorted_values[at], requested, atol=1e-6, rtol=0):
            raise DataIssue("grid_mismatch", "Missing exact GRID coordinate; no interpolation.")
        indices.append(order[at])
    return {lat_dim: indices[0], lon_dim: indices[1]}


def gather(var, index_by_dim, n):
    """Orthogonal netCDF subset, then paired indexing; never a full global read."""
    selectors, inverse = [], []
    for dimension, size in zip(var.dimensions, var.shape):
        if dimension in index_by_dim:
            idx = np.asarray(index_by_dim[dimension], dtype=int)
            if len(idx) != n:
                raise DataIssue("schema_error", "Paired index length mismatch.")
            unique, back = np.unique(idx, return_inverse=True)
            selectors.append(unique)
            inverse.append(back)
        elif size == 1:
            selectors.append(np.array([0]))
            inverse.append(np.zeros(n, dtype=int))
        else:
            raise DataIssue("schema_error", f"Unresolved dimension {dimension} in {var.name}.")
    return numeric(var[tuple(selectors)])[tuple(inverse)]


def require_dimensions(var, dimensions):
    if not set(dimensions).issubset(var.dimensions):
        raise DataIssue("schema_error", f"{var.name} lacks required dimensions {dimensions}.")


def decoded_time(var, raw):
    if getattr(var, "calendar", "standard") not in ("standard", "gregorian", "proleptic_gregorian"):
        raise DataIssue("time_mismatch", "Only Gregorian ERA5 calendars are supported.")
    try:
        dates = nc.num2date(raw, units=var.units,
                           calendar=getattr(var, "calendar", "standard"),
                           only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        return pd.DatetimeIndex(pd.to_datetime(np.asarray(dates).ravel(), utc=True))
    except (AttributeError, ValueError, TypeError, OverflowError) as error:
        raise DataIssue("time_mismatch", f"Cannot decode {var.name}: {error}") from error




def daily_dates(ds, spec):
    name = next((key for key in ("valid_time", "time") if key in ds.variables), None)
    if name is None or ds[name].ndim != 1:
        raise DataIssue("time_mismatch", "Expected a 1D time/valid_time coordinate.")
    var = ds[name]
    times = decoded_time(var, var[:])
    if times.isna().any() or not times.equals(times.normalize()):
        raise DataIssue("time_mismatch", "Daily labels must be midnight UTC; hourly/noon labels are not guessed.")
    dates = times - pd.Timedelta(days=int(spec["day_label"] == "end"))
    if dates.duplicated().any():
        raise DataIssue("duplicate_date", "Multiple records for the same UTC day.")
    # CF time bounds, if present, must describe exactly that UTC calendar day.
    bounds_name = getattr(var, "bounds", None)
    if bounds_name:
        if bounds_name not in ds.variables:
            raise DataIssue("time_mismatch", "Missing declared time bounds.")
        bounds = numeric(ds[bounds_name][:])
        if bounds.shape != (len(times), 2):
            raise DataIssue("time_mismatch", "Expected time bounds with shape (time, 2).")
        starts = decoded_time(var, bounds[:, 0])
        ends = decoded_time(var, bounds[:, 1])
        if not starts.equals(dates) or not ends.equals(dates + pd.Timedelta(days=1)):
            raise DataIssue("time_mismatch", "Time bounds are not 00:00--24:00 UTC.")
    for key in ("time_zone", "timezone"):
        declared = getattr(ds, key, None)
        if declared is not None and str(declared).lower() not in ("utc", "utc+00:00", "utc+0", "+00:00"):
            raise DataIssue("time_mismatch", f"Non-UTC source attribute {key}={declared}.")
    return var.dimensions[0], dates.strftime("%Y-%m-%d").tolist()


def conversion(name, units):
    """Explicit units only; do not infer sum-vs-flux or metres-vs-mm from magnitudes."""
    unit = str(units).strip().lower().replace("**", "").replace("^", "").replace(" ", "")
    options = {
        "u10": {"ms-1": 1., "m/s": 1.},
        "v10": {"ms-1": 1., "m/s": 1.},
        "tcc": {"1": 1., "0-1": 1., "(0-1)": 1., "%": .01, "percent": .01},
        "tp": {"m": 1000., "mm": 1., "kgm-2": 1., "kg/m2": 1.},
        "ssrd": {"jm-2": 1e-6, "j/m2": 1e-6, "mjm-2": 1., "mj/m2": 1.},
    }
    if unit not in options[name]:
        raise DataIssue("unit_mismatch", f"{name}: unsupported units {units!r}.")
    return options[name][unit]


def check_daily_contract(ds, var, spec, path):
    methods = str(getattr(var, "cell_methods", ""))
    for method in ("mean", "sum", "maximum", "minimum"):
        if f"time: {method}" in methods and method != spec["statistic"]:
            raise DataIssue("statistic_mismatch", f"{var.name}: {methods}.")
    declared = getattr(ds, "daily_statistic", None)
    if declared is not None and declared not in (spec["statistic"], f"daily_{spec['statistic']}"):
        raise DataIssue("statistic_mismatch", f"Conflicting daily_statistic={declared}.")
    if not spec.get("request_sidecar", False):
        return
    sidecar = path.with_suffix(".request.json")
    if not sidecar.is_file():
        raise DataIssue("missing_request_metadata", str(sidecar))
    try:
        request = json.loads(sidecar.read_text(encoding="utf-8"))["request"]
    except (ValueError, KeyError) as error:
        raise DataIssue("invalid_request_metadata", str(error)) from error
    if not isinstance(request, dict):
        raise DataIssue("invalid_request_metadata", "request must be a JSON object.")
    if (request.get("daily_statistic") != f"daily_{spec['statistic']}"
            or request.get("time_zone") != "utc+00:00"
            or request.get("frequency") != "1_hourly"):
        raise DataIssue("statistic_mismatch", "CDS request must specify this statistic, UTC and 1_hourly.")
    # A request proves the requested calculation, NOT 24 finite source values.



