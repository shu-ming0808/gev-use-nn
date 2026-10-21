"""Match Taiwan annual-maximum UTC dates to daily ERA5 covariates.

Read-only inputs; no NN imports, fitting, downloads, interpolation, or imputation.
The default scope is the existing 50-GRID registry, not a new geographic sample.
Daily files are opened once per path and only the requested time/spatial subset
is read. Missing events remain in every output. See event_matching.example.json.
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
        found = []
        for value in requested:
            match = np.flatnonzero(np.isclose(values, value, atol=1e-6, rtol=0))
            if len(match) != 1:
                raise DataIssue("grid_mismatch", f"No unique exact GRID centre for {value}.")
            found.append(int(match[0]))
        indices.append(np.array(found))
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


def validate_grids(path, expected_grids):
    if not isinstance(expected_grids, int) or not 1 <= expected_grids <= 50:
        raise ValueError("This pilot is limited to 1--50 GRID; global expansion is not enabled.")
    grids = pd.read_csv(path)
    required = {"grid_id", "latitude", "longitude", "longitude_180", "lsm"}
    if not required.issubset(grids):
        raise ValueError(f"GRID registry needs {sorted(required)}.")
    if len(grids) != expected_grids or grids.grid_id.isna().any() or grids.grid_id.duplicated().any():
        raise ValueError(f"Expected {expected_grids} unique, nonmissing GRID IDs.")
    cols = ["latitude", "longitude", "longitude_180", "lsm"]
    if not np.isfinite(grids[cols].to_numpy(dtype=float)).all():
        raise ValueError("Registry coordinates and LSM must be finite.")
    if not grids.lsm.between(0, 1).all() or not grids.lsm.gt(.5).all():
        raise ValueError("The Taiwan registry must already satisfy LSM > 0.5.")
    if not grids.latitude.between(-90, 90).all():
        raise ValueError("Invalid latitude.")
    expected_lon = (grids.longitude.to_numpy() + 180) % 360 - 180
    if not np.allclose(expected_lon, grids.longitude_180, atol=1e-6, rtol=0):
        raise ValueError("longitude and longitude_180 disagree.")
    if grids.duplicated(["latitude", "longitude_180"]).any():
        raise ValueError("Duplicate GRID centres.")
    # Preserve identity from the original regular ERA5 global 0.25-degree grid.
    rows = (90 - grids.latitude.to_numpy()) / .25
    cols = (grids.longitude.to_numpy() % 360) / .25
    if not (np.allclose(rows, np.rint(rows), atol=1e-6, rtol=0)
            and np.allclose(cols, np.rint(cols), atol=1e-6, rtol=0)):
        raise ValueError("Registry is not on the ERA5 0.25-degree grid.")
    ids = [f"G{r:03d}_{c:04d}" for r, c in zip(np.rint(rows).astype(int), np.rint(cols).astype(int))]
    if ids != grids.grid_id.tolist():
        raise ValueError("GRID IDs disagree with their ERA5 coordinates.")
    return grids.reset_index(drop=True)


def build_event_index(path, grids, years):
    """Merged annual NetCDF -> all registry GRID/year pairs, including failures."""
    events = pd.DataFrame([
        dict(grid_id=row.grid_id, year=year, latitude=row.latitude,
             longitude_180=row.longitude_180, annual_tmax_c=np.nan,
             maximum_time_utc="", event_date_utc="", temperature_valid_hours=np.nan,
             expected_hours=24 * (366 if calendar.isleap(year) else 365),
             event_status="missing_year", detail="")
        for row in grids.itertuples() for year in years
    ])
    with open_nc(path) as ds:
        for name in ("year", "annual_maximum_t2m", "maximum_time",
                     "valid_hour_count", "expected_hours", "lsm"):
            if name not in ds.variables:
                raise ValueError(f"Annual NetCDF is missing {name}.")
        year_var = ds["year"]
        stored = numeric(year_var[:])
        if year_var.ndim != 1 or not np.isfinite(stored).all() or not np.equal(stored, np.rint(stored)).all():
            raise ValueError("Invalid annual year coordinate.")
        if len(np.unique(stored)) != len(stored):
            raise ValueError("Duplicate annual years.")
        year_dim = year_var.dimensions[0]
        # Fail fast for a different grid, instead of silently selecting a neighbour.
        spatial = spatial_indices(ds, grids)
        require_dimensions(ds["lsm"], spatial)
        mask = gather(ds["lsm"], spatial, len(grids))
        if not np.allclose(mask, grids.lsm, atol=1e-6, rtol=0):
            raise ValueError("Annual LSM disagrees with the saved Taiwan registry.")
        units = str(getattr(ds["annual_maximum_t2m"], "units", "")).lower()
        if units not in ("k", "kelvin", "degrees_celsius", "degree_celsius", "degc", "celsius"):
            raise ValueError(f"Unknown annual temperature units: {units}.")
        for year in years:
            where = np.flatnonzero(stored == year)
            if not len(where):
                continue
            index = {**spatial, year_dim: np.full(len(grids), where[0])}
            for name in ("annual_maximum_t2m", "maximum_time", "valid_hour_count"):
                require_dimensions(ds[name], index)
            values = gather(ds["annual_maximum_t2m"], index, len(grids))
            if units in ("k", "kelvin"):
                values -= 273.15
            seconds = gather(ds["maximum_time"], index, len(grids))
            counts = gather(ds["valid_hour_count"], index, len(grids))
            required = 24 * (366 if calendar.isleap(year) else 365)
            expected = gather(ds["expected_hours"], {year_dim: np.array([where[0]])}, 1)[0]
            if expected != required:
                raise ValueError(f"Wrong expected_hours for {year}.")
            for j, target in enumerate(events.index[events.year.eq(year)]):
                status, detail = "ok", ""
                stamp = ""
                if not np.isfinite(counts[j]) or counts[j] != required:
                    status = "incomplete_temperature_year"
                elif not np.isfinite(values[j]):
                    status = "missing_temperature"
                elif not np.isfinite(seconds[j]):
                    status = "missing_event_time"
                else:
                    try:
                        stamp = decoded_time(ds["maximum_time"], [seconds[j]])[0]
                        if stamp.year != year or stamp != stamp.floor("h"):
                            status = "event_time_mismatch"
                    except DataIssue as error:
                        status, detail = error.status, str(error)
                events.loc[target, ["annual_tmax_c", "temperature_valid_hours"]] = [values[j], counts[j]]
                if isinstance(stamp, pd.Timestamp):
                    events.loc[target, ["maximum_time_utc", "event_date_utc"]] = [
                        stamp.isoformat(), stamp.strftime("%Y-%m-%d")]
                events.loc[target, ["event_status", "detail"]] = [status, detail]
    return events


def validate_config(config):
    if config.get("version") != 1:
        raise ValueError("Configuration version must be 1.")
    specs = config.get("variables", {})
    if set(specs) != set(FIELDS):
        raise ValueError(f"Configure exactly {sorted(FIELDS)}; no silently omitted predictor.")
    for name, spec in specs.items():
        if spec.get("statistic") != FIELDS[name][0] or spec.get("timezone") != "UTC":
            raise ValueError(f"{name}: require {FIELDS[name][0]} over UTC days.")
        if spec.get("day_label") not in ("start", "end"):
            raise ValueError(f"{name}: specify midnight start/end label explicitly.")
        for key in ("path_template", "variable", "provenance"):
            if not isinstance(spec.get(key), str) or not spec[key].strip():
                raise ValueError(f"{name}: missing {key}.")
        if not isinstance(spec.get("allow_unverified_coverage", False), bool):
            raise ValueError("allow_unverified_coverage must be a boolean.")
    return config


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


def coverage_counts(ds, spec, index, n, spatial_dims, time_dim):
    name = spec.get("valid_count_variable")
    if not name:
        return np.full(n, np.nan)
    if name not in ds.variables:
        raise DataIssue("missing_coverage_variable", name)
    var = ds[name]
    # A global/time-only count cannot certify finite observations at each GRID.
    require_dimensions(var, [time_dim, *spatial_dims])
    return gather(var, index, n)


def match_daily(events, config, daily_root):
    """Long event-variable table; group by physical path, preserving failures."""
    records, jobs, sources = [], {}, []
    for event in events.itertuples():
        for name, spec in config["variables"].items():
            row = dict(grid_id=event.grid_id, year=event.year,
                       event_date_utc=event.event_date_utc, variable=name,
                       value=np.nan, units=FIELDS[name][1], statistic=spec["statistic"],
                       source_file="", source_variable=spec["variable"],
                       status="event_unavailable", coverage_status="not_checked",
                       valid_hours=np.nan, provenance=spec["provenance"], detail="")
            records.append(row)
            if event.event_status != "ok":
                row["detail"] = event.event_status
                continue
            date = pd.Timestamp(event.event_date_utc)
            label = date + pd.Timedelta(days=int(spec["day_label"] == "end"))
            # Templates use the STORED label date, so end-labelled 31 Dec data
            # correctly routes to the next January file.
            relative = spec["path_template"].format(year=label.year, month=label.month,
                                                    day=label.day, date=label.strftime("%Y-%m-%d"))
            path = Path(relative)
            path = (daily_root / path).resolve() if not path.is_absolute() else path.resolve()
            row["source_file"] = str(path)
            jobs.setdefault(path, []).append((len(records)-1, event))
    for number, (path, jobs_here) in enumerate(jobs.items(), 1):
        if number == 1 or number % 50 == 0 or number == len(jobs):
            print(f"Daily files {number}/{len(jobs)}: {path.name}", flush=True)
        if not path.is_file():
            for i, _ in jobs_here:
                records[i].update(status="missing_file", detail=str(path))
            continue
        before = signature(path)
        sidecar = path.with_suffix(".request.json")
        sidecar_before = sha256(sidecar) if sidecar.is_file() else None
        try:
            with open_nc(path) as ds:
                for name, spec in config["variables"].items():
                    subset = [(i, event) for i, event in jobs_here if records[i]["variable"] == name]
                    if not subset:
                        continue
                    try:
                        if spec["variable"] not in ds.variables:
                            raise DataIssue("missing_variable", spec["variable"])
                        var = ds[spec["variable"]]
                        check_daily_contract(ds, var, spec, path)
                        factor = conversion(name, getattr(var, "units", ""))
                        time_dim, dates = daily_dates(ds, spec)
                        lookup = {date: j for j, date in enumerate(dates)}
                        present = []
                        for i, event in subset:
                            if event.event_date_utc not in lookup:
                                records[i].update(status="missing_date", detail=event.event_date_utc)
                            else:
                                present.append((i, event))
                        if not present:
                            continue
                        frame = pd.DataFrame([dict(latitude=e.latitude, longitude_180=e.longitude_180)
                                              for _, e in present])
                        spatial = spatial_indices(ds, frame)
                        index = {**spatial, time_dim: np.array([lookup[e.event_date_utc] for _, e in present])}
                        require_dimensions(var, index)
                        values = gather(var, index, len(present)) * factor
                        counts = coverage_counts(ds, spec, index, len(present), spatial, time_dim)
                        for j, (i, _) in enumerate(present):
                            row, value, count = records[i], values[j], counts[j]
                            row.update(value=value, valid_hours=count, status="ok")
                            if not np.isfinite(value):
                                row["status"] = "missing_value"
                            elif (name == "tcc" and not 0 <= value <= 1) or (name in ("tp", "ssrd") and value < 0):
                                row["status"] = "invalid_value"
                            if not np.isfinite(count):
                                row["coverage_status"] = "unverified"
                                if row["status"] == "ok":
                                    row["status"] = ("ok_unverified_coverage" if spec.get("allow_unverified_coverage", False)
                                                     else "coverage_unverified")
                            elif count != 24:
                                row["coverage_status"] = "incomplete"
                                if row["status"] == "ok":
                                    row["status"] = "incomplete_day"
                            else:
                                row["coverage_status"] = "verified_24"
                    except DataIssue as error:
                        for i, _ in subset:
                            records[i].update(status=error.status, detail=str(error))
        except (OSError, RuntimeError, ValueError) as error:
            for i, _ in jobs_here:
                records[i].update(value=np.nan, status="unreadable_file", detail=str(error))
        after = signature(path)
        sidecar_after = sha256(sidecar) if sidecar.is_file() else None
        unchanged = before == after and sidecar_before == sidecar_after
        if not unchanged:
            for i, _ in jobs_here:
                records[i].update(value=np.nan, status="source_changed", detail="File changed during extraction.")
        sources.append({**before, "unchanged_after_read": unchanged,
                        "request_sidecar_sha256": sidecar_before})
    return pd.DataFrame(records), sources


def make_tables(events, matched, grids, min_valid_years):
    """Wide paired rows and one row per GRID; only accepted values enter means."""
    paired = events.copy()
    accepted = ("ok", "ok_unverified_coverage")
    indexed = matched.set_index(["grid_id", "year", "variable"])
    keys = pd.MultiIndex.from_frame(events[["grid_id", "year"]])
    for name in FIELDS:
        part = indexed.xs(name, level="variable").reindex(keys)
        for column in ("value", "status", "coverage_status", "valid_hours"):
            paired[f"{name}_{column}"] = part[column].to_numpy()
    usable_wind = paired.u10_status.isin(accepted) & paired.v10_status.isin(accepted)
    paired[f"{DERIVED_WIND}_value"] = np.where(
        usable_wind, np.hypot(paired.u10_value, paired.v10_value), np.nan)
    paired[f"{DERIVED_WIND}_status"] = np.where(
        ~usable_wind, "component_unavailable",
        np.where(paired.u10_status.eq("ok") & paired.v10_status.eq("ok"),
                 "ok", "ok_unverified_coverage"))
    rows = []
    for grid in grids.itertuples(index=False):
        part = paired.loc[paired.grid_id.eq(grid.grid_id)]
        row = grid._asdict()
        row.update(n_expected_years=len(part), n_valid_events=int(part.event_status.eq("ok").sum()))
        eligible = part.event_status.eq("ok").to_numpy()
        verified = eligible.copy()
        for name in (*FIELDS, DERIVED_WIND):
            valid = part[f"{name}_status"].isin(accepted)
            values = part.loc[valid, f"{name}_value"].to_numpy()
            count = len(values)
            # The observed mean is descriptive even when coverage is insufficient.
            # The GP-ready feature remains NaN until the prespecified threshold.
            row[f"{name}_n_valid_years"] = count
            row[f"{name}_n_missing_years"] = len(part) - count
            row[f"{name}_n_verified_years"] = int(part[f"{name}_status"].eq("ok").sum())
            row[f"{name}_observed_mean"] = float(np.mean(values)) if count else np.nan
            row[f"{name}_event_mean"] = row[f"{name}_observed_mean"] if count >= min_valid_years else np.nan
            eligible &= valid.to_numpy()
            verified &= part[f"{name}_status"].eq("ok").to_numpy()
        row["n_joint_valid_years"] = int(eligible.sum())
        row["n_joint_verified_years"] = int(verified.sum())
        # Common-year gate avoids falsely declaring readiness when variables
        # happen to be available in different sets of years.
        row["ready_for_spatial_cv"] = bool(eligible.sum() >= min_valid_years)
        row["all_daily_coverage_verified"] = bool(verified.sum() == len(part))
        rows.append(row)
    summary = pd.DataFrame(rows)
    issues = matched.loc[~matched.status.eq("ok")].copy()
    issues["stage"] = "daily_matching"
    issues["severity"] = np.where(issues.status.eq("ok_unverified_coverage"), "warning", "error")
    bad_events = events.loc[~events.event_status.eq("ok")].copy()
    bad_events = bad_events.rename(columns={"event_status": "status"})
    bad_events["variable"], bad_events["stage"], bad_events["severity"] = "temperature", "event_index", "error"
    issues = pd.concat([bad_events, issues], ignore_index=True)
    issues = issues.reindex(columns=["stage", "severity", "grid_id", "year", "event_date_utc",
                                    "variable", "status", "coverage_status", "source_file", "detail"])
    missing_summary = issues.groupby(["stage", "severity", "year", "variable", "status"],
                                    dropna=False).size().reset_index(name="n_rows")
    return paired, summary, issues, missing_summary


def run(config_path, output, *, annual_path=None, daily_root=None, grid_registry=None):
    started = time.perf_counter()
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = validate_config(json.loads(config_path.read_text(encoding="utf-8-sig")))

    def resolve(value):
        path = Path(value)
        return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()

    annual_path = resolve(annual_path or config["annual_netcdf"])
    daily_root = resolve(daily_root or config["daily_root"])
    grid_registry = resolve(grid_registry or config["grid_registry"])
    start, end = config.get("start_year", 1976), config.get("end_year", 2025)
    if not isinstance(start, int) or not isinstance(end, int) or start > end:
        raise ValueError("Invalid year range.")
    years = list(range(start, end+1))
    min_years = config.get("min_valid_years", len(years))
    if not isinstance(min_years, int) or not 1 <= min_years <= len(years):
        raise ValueError("min_valid_years must be between 1 and the requested year count.")
    grids = validate_grids(grid_registry, config.get("expected_grids", 50))
    # Avoid generating an all-missing result just because the external drive
    # was not connected. Individual absent files are reported once root exists.
    if not annual_path.is_file() or not daily_root.is_dir():
        raise FileNotFoundError(f"Mount the data drive first. Annual: {annual_path}; daily root: {daily_root}")
    if output.exists():
        raise FileExistsError("Use a new output directory; previous outputs are never overwritten.")
    output.mkdir(parents=True)
    status_path = output / "run_status.json"
    write_json(status_path, {"status": "running", "started_utc": utc_now()})
    try:
        annual_before = signature(annual_path)
        registry_hash, config_hash = sha256(grid_registry), sha256(config_path)
        events = build_event_index(annual_path, grids, years)
        write_csv(output / "annual_event_index.csv", events)
        print(f"Event index: {len(grids)} GRID x {len(years)} years; {events.event_status.eq('ok').sum()} valid.", flush=True)
        matched, sources = match_daily(events, config, daily_root)
        paired, summary, missing, missing_summary = make_tables(events, matched, grids, min_years)
        if (signature(annual_path) != annual_before or sha256(grid_registry) != registry_hash
                or sha256(config_path) != config_hash):
            raise RuntimeError("An input/configuration changed during the run; do not use these outputs.")
        outputs = {
            "grid_registry.csv": grids, "annual_event_index.csv": events,
            "event_weather_long.csv": matched, "event_weather.csv": paired,
            "missing_report.csv": missing, "missing_summary.csv": missing_summary,
            "grid_predictors.csv": summary,
        }
        for filename, frame in outputs.items():
            write_csv(output / filename, frame)
        write_json(output / "effective_config.json", {**config, "annual_netcdf": str(annual_path),
                   "daily_root": str(daily_root), "grid_registry": str(grid_registry), "min_valid_years": min_years})
        report = dict(
            version=VERSION, status="completed" if missing.empty else "completed_with_issues",
            completed_utc=utc_now(), real_data=not config.get("test_fixture", False),
            n_grids=len(grids), years=years, n_event_rows=len(events),
            n_valid_events=int(events.event_status.eq("ok").sum()),
            n_matched_verified=int(matched.status.eq("ok").sum()),
            n_matched_unverified_accepted=int(matched.status.eq("ok_unverified_coverage").sum()),
            n_requested_weather_rows=len(matched), n_issue_rows=len(missing),
            n_ready_grids=int(summary.ready_for_spatial_cv.sum()), min_valid_years=min_years,
            annual_source=annual_before, grid_registry_sha256=registry_hash, config_sha256=config_hash,
            daily_sources=sources, elapsed_seconds=time.perf_counter()-started,
            output_sha256={name: sha256(output/name) for name in outputs},
            nn_loaded=False, nn_changed=False, global_run=False,
            variable_units={**{k: v[1] for k, v in FIELDS.items()}, DERIVED_WIND: "m s-1"},
            definitions={
                "event": "Maximum of hourly t2m in UTC year; existing maximum_time retained; upstream ties use earliest hour.",
                "weather": "Statistics over the event UTC day, not weather at the maximum-temperature hour.",
                "wind": "hypot(daily mean u10, daily mean v10); NOT daily mean scalar wind speed.",
                "summary": "Arithmetic mean of accepted annual-event-day values, NOT all-day climatology.",
                "coverage": "Only a per-cell valid_count_variable equal to 24 certifies coverage; requests alone do not.",
                "missing": "No interpolation, zero filling, or GRID/year deletion; partial descriptive means kept separately.",
                "cv": "ready_for_spatial_cv gates event-weather coverage only, not static predictors/folds/buffers; held-out event dates must match the intended prediction task.",
                "fingerprints": "Large NetCDF inputs use size/mtime before and after reading, not a full content hash.",
            })
        write_json(output/"run_report.json", report)
        write_json(status_path, {"status": report["status"], "n_ready_grids": report["n_ready_grids"],
                                "n_issue_rows": report["n_issue_rows"], "completed_utc": utc_now()})
        return report
    except BaseException as error:
        write_json(status_path, {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annual-netcdf", type=Path, help="Override the annual source (e.g. D: to F:).")
    parser.add_argument("--daily-root", type=Path, help="Override the daily source directory.")
    parser.add_argument("--grid-registry", type=Path, help="Override the saved Taiwan GRID registry.")
    args = parser.parse_args(argv)
    report = run(args.config, args.output, annual_path=args.annual_netcdf,
                 daily_root=args.daily_root, grid_registry=args.grid_registry)
    print(json.dumps({key: report[key] for key in
                      ("status", "n_grids", "n_event_rows", "n_ready_grids", "n_issue_rows", "elapsed_seconds")},
                     ensure_ascii=False, indent=2))
    # Nonzero code for issues prevents treating a partial run as a ready GP table.
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
