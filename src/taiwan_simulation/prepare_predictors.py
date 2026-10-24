"""Taiwan-only, read-only ERA5 covariate extraction; no NN/GP fitting.

Unify audited ARCO daily files and CDS monthly daily-statistic files. Keep all
50 years and source/quality flags, then make event-day and annual summaries.
Monthly NPZ caches are reusable only after input fingerprints and cache hashes
match. Each process owns its NetCDF handles; threads must not share them.
"""
from __future__ import annotations

import sys
from pathlib import Path
SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from taiwan_simulation import match_annual_events as event

VERSION = "taiwan_predictors_mixed_sources_v1"
ASSEMBLY_VERSION = "taiwan_predictors_assembly_v2"
ARCO_CONTRACT = "arco_era5_daily_utc_v1"
ARCO_STORE = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
NAMES = tuple(event.FIELDS)
QUALITY = {0: "ok", 1: "missing_file", 2: "invalid_source", 3: "missing_date",
           4: "missing_value", 5: "invalid_value", 6: "source_changed"}
COVERAGE = {0: "unknown", 1: "upstream_24_hour_all_grid_audit", 2: "cds_daily_contract_hourly_count_unavailable"}
ROOT = SOURCE_ROOT.parent
DEFAULT_REGISTRY = ROOT/"results/simulation/taiwan_validation_era5_1976_2025/pilot_20261013_original/grid_index.csv"
DEFAULT_ANNUAL = Path("D:/論文資料/ERA5/1976-2004_global_annual_maxima/era5_t2m_annual_maxima_1976_2025_global.nc")
DEFAULT_DATA = Path("D:/論文資料/ERA5/global_predictors_1976_2025")


def arco_path(root, day):
    return Path(root)/"arco_daily_predictors"/f"{day.year}"/f"{day.month:02d}"/f"era5_arco_daily_{day:%Y%m%d}_global_025.nc"


def cds_path(root, year, month, name):
    statistic = event.FIELDS[name][0]
    return Path(root)/"daily_predictors"/str(year)/f"era5_daily_{statistic}_{year}{month:02d}_global_025.nc"


def stat_or_missing(path):
    path = Path(path)
    return event.signature(path) if path.is_file() else {"path": str(path.resolve()), "missing": True}


def check_arco(ds, path, dates):
    """Trust only this inspected producer's fail-closed all-grid 24-hour contract.

Its aggregate_day rejects any nonfinite hourly grid; write_day audits every
stored daily value before publishing. A bare hourly_samples=24 is insufficient.
The source arrays are not re-downloaded/reconstructed by this extractor.
"""
    if (getattr(ds, "processing_contract", None) != ARCO_CONTRACT
            or getattr(ds, "complete", None) != 1 or getattr(ds, "source", None) != ARCO_STORE
            or len(dates) != 1 or getattr(ds, "date_utc", None) != dates[0]):
        raise ValueError("ARCO completed source/date/processing contract mismatch")
    audit = json.loads(path.with_suffix(".audit.json").read_text(encoding="utf-8"))
    metrics = json.loads(ds.processing_metrics)
    if (audit.get("check") != "all_values" or audit.get("date_utc") != dates[0]
            or audit.get("source") != ARCO_STORE or audit.get("file_bytes") != path.stat().st_size
            or audit.get("hourly_chunks") != 120 or metrics.get("hourly_chunks") != 120):
        raise ValueError("ARCO full-value audit/120 hourly chunks mismatch")
    for name, (statistic, _) in event.FIELDS.items():
        var = ds[name]
        window = "01:00 through next-day 00:00" if statistic == "sum" else "00:00 through 23:00"
        if (getattr(var, "hourly_samples", None) != 24
                or getattr(var, "source_hour_window_utc", None) != window
                or getattr(var, "cell_methods", None) != "time: " + statistic):
            raise ValueError(f"ARCO UTC window/statistic mismatch: {name}")


def read_fields(path, grids, wanted_dates, names, kind):
    """Return date x GRID x variable arrays in analysis units, with quality."""
    nday, ngrid = len(wanted_dates), len(grids)
    values = np.full((nday, ngrid, len(names)), np.nan)
    quality = np.full(values.shape, 3, dtype=np.uint8)
    coverage = np.zeros(values.shape, dtype=np.uint8)
    errors = []
    with event.open_nc(path) as ds:
        time_dim, dates = event.daily_dates(ds, {"day_label": "start"})
        if kind == "arco":
            check_arco(ds, path, dates)
        spatial = event.spatial_indices(ds, grids)
        lookup = {date: i for i, date in enumerate(dates)}
        selected = [i for i, date in enumerate(wanted_dates) if date in lookup]
        if not selected:
            return values, quality, coverage, errors
        index = {key: np.tile(val, len(selected)) for key, val in spatial.items()}
        index[time_dim] = np.repeat([lookup[wanted_dates[i]] for i in selected], ngrid)
        for k, name in enumerate(names):
            try:
                var = ds[name]
                event.require_dimensions(var, index)
                spec = {"statistic": event.FIELDS[name][0], "request_sidecar": kind == "cds"}
                event.check_daily_contract(ds, var, spec, path)
                factor = event.conversion(name, getattr(var, "units", ""))
                data = event.gather(var, index, len(selected)*ngrid).reshape(len(selected), ngrid)*factor
                q = np.zeros(data.shape, dtype=np.uint8)
                q[~np.isfinite(data)] = 4
                if name == "tcc":
                    q[np.isfinite(data) & ((data < 0) | (data > 1))] = 5
                if name in ("tp", "ssrd"):
                    q[np.isfinite(data) & (data < 0)] = 5
                values[selected, :, k] = data
                quality[selected, :, k] = q
                coverage[selected, :, k] = 1 if kind == "arco" else 2
            except (ValueError, KeyError, IndexError, AttributeError, OSError) as error:
                quality[:, :, k] = 2
                errors.append({"variable": name, "error": str(error)})
    return values, quality, coverage, errors


def extraction_identity(grids):
    return hashlib.sha256(grids.to_csv(index=False).encode()).hexdigest()


def extract_month(root, grids, year, month, cache):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    days = pd.date_range(f"{year}-{month:02d}-01", periods=event.calendar.monthrange(year, month)[1])
    dates = days.strftime("%Y-%m-%d").tolist()
    target = cache/f"{year}{month:02d}.npz"
    metadata = cache/f"{year}{month:02d}.json"
    groups = {}
    # Prefer an ARCO day when present. A present but invalid ARCO file is NOT
    # silently replaced by a different source; report it for review.
    for d, day in enumerate(days):
        arco = arco_path(root, day)
        for k, name in enumerate(NAMES):
            kind, path = ("arco", arco) if arco.is_file() else ("cds", cds_path(root, year, month, name))
            groups.setdefault((kind, path), []).append((d, k))
    watched = set()
    # Watch absent ARCO days too, so later newly supplied days invalidate cache.
    for day in days:
        watched.add(arco_path(root, day))
    for _, path in groups:
        watched.update((path, path.with_suffix(".audit.json"), path.with_suffix(".request.json")))
    fingerprints = [stat_or_missing(p) for p in sorted(watched)]
    identity = {"version": VERSION, "grid_sha256": extraction_identity(grids), "year": year,
                "month": month, "sources": fingerprints}
    if target.is_file() and metadata.is_file():
        old = json.loads(metadata.read_text(encoding="utf-8"))
        if old.get("identity") == identity and old.get("cache_sha256") == event.sha256(target):
            return {"year": year, "month": month, "reused": True, **old["summary"]}
    values = np.full((len(days), len(grids), len(NAMES)), np.nan)
    quality = np.ones(values.shape, dtype=np.uint8)
    coverage = np.zeros(values.shape, dtype=np.uint8)
    source_kind = np.zeros((len(days), len(NAMES)), dtype=np.uint8)
    source_files, errors = [], []
    for (kind, path), pairs in groups.items():
        selected_days = sorted({d for d, _ in pairs})
        selected_vars = sorted({k for _, k in pairs})
        source_files.append({"kind": kind, "path": str(path), "dates": [dates[d] for d in selected_days],
                             "variables": [NAMES[k] for k in selected_vars]})
        if not path.is_file():
            continue
        for d, k in pairs:
            source_kind[d, k] = 1 if kind == "arco" else 2
        try:
            a, q, c, problems = read_fields(path, grids, [dates[d] for d in selected_days],
                                          [NAMES[k] for k in selected_vars], kind)
            for i, d in enumerate(selected_days):
                for j, k in enumerate(selected_vars):
                    values[d, :, k], quality[d, :, k], coverage[d, :, k] = a[i, :, j], q[i, :, j], c[i, :, j]
                    source_kind[d, k] = 1 if kind == "arco" else 2
            errors.extend({"file": str(path), **p} for p in problems)
        except (ValueError, KeyError, IndexError, AttributeError, OSError, RuntimeError) as error:
            for d, k in pairs:
                quality[d, :, k] = 2
            errors.append({"file": str(path), "error": str(error)})
    if fingerprints != [stat_or_missing(p) for p in sorted(watched)]:
        quality[:] = 6
        errors.append({"error": "Source changed while reading month; rerun after download stops."})
    cache.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".npz.part")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, dates=np.asarray(dates), grid_id=grids.grid_id.to_numpy(dtype=str),
                            variables=np.asarray(NAMES), values=values, quality=quality,
                            coverage=coverage, source_kind=source_kind)
    os.replace(partial, target)
    summary = {"n_days": len(days), "n_invalid_values": int((quality != 0).sum()),
               "n_arco_variable_days": int((source_kind == 1).sum()),
               "n_cds_variable_days": int((source_kind == 2).sum())}
    event.write_json(metadata, {"identity": identity, "source_files": source_files, "errors": errors,
                               "summary": summary, "cache_sha256": event.sha256(target)})
    return {"year": year, "month": month, "reused": False, **summary}


def slope(values, years):
    """All requested years required; per-decade OLS slope, descriptive not causal."""
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all() or len(a) < 2:
        return np.nan
    t = (np.asarray(years, dtype=float)-2000.5)/10
    t -= t.mean()
    return float(t @ (a-a.mean()) / (t @ t))


def verified_cached_month(cache, grids, year, month):
    """Read-only reuse: reject stale/incomplete caches; never re-extract here."""
    target = Path(cache)/f"{year}{month:02d}.npz"
    meta_path = target.with_suffix(".json")
    old = json.loads(meta_path.read_text(encoding="utf-8"))
    identity = old["identity"]
    if (identity["version"] != VERSION or identity["grid_sha256"] != extraction_identity(grids)
            or (identity["year"], identity["month"]) != (year, month)
            or old["cache_sha256"] != event.sha256(target)):
        raise ValueError(f"Cached identity/hash mismatch: {year}-{month:02d}")
    if identity["sources"] != [stat_or_missing(s["path"]) for s in identity["sources"]]:
        raise ValueError(f"Cached source changed: {year}-{month:02d}")
    return {"year": year, "month": month, "reused": True, **old["summary"],
            "cache_sha256": old["cache_sha256"], "metadata_sha256": event.sha256(meta_path)}


def assemble(cache, grids, events, years, output, root, *, precipitation_zero_tolerance_mm=0.0):
    if not np.isfinite(precipitation_zero_tolerance_mm) or precipitation_zero_tolerance_mm < 0:
        raise ValueError("Precipitation tolerance must be finite and nonnegative (mm/day)")
    annual_rows, event_rows, issues, daily_files, corrections = [], [], [], [], []
    event_map = {(r.grid_id, r.year): r for r in events.itertuples()}
    for year in years:
        months = []
        for month in range(1, 13):
            with np.load(cache/f"{year}{month:02d}.npz", allow_pickle=False) as data:
                if data["grid_id"].tolist() != grids.grid_id.tolist() or tuple(data["variables"]) != NAMES:
                    raise ValueError("Cached GRID/variable order mismatch")
                months.append({key: data[key] for key in data.files})
        dates = np.concatenate([m["dates"] for m in months])
        values = np.concatenate([m["values"] for m in months])
        quality = np.concatenate([m["quality"] for m in months])
        coverage = np.concatenate([m["coverage"] for m in months])
        kinds = np.concatenate([m["source_kind"] for m in months])
        tp_index = NAMES.index("tp")
        raw_tp = values[:, :, tp_index].copy()
        raw_tp_quality = quality[:, :, tp_index].copy()
        corrected = ((raw_tp_quality == 5) & np.isfinite(raw_tp) & (raw_tp < 0)
                     & (raw_tp >= -precipitation_zero_tolerance_mm))
        # Derived arrays only: original NetCDF and NPZ values/flags stay untouched.
        values[:, :, tp_index][corrected] = 0.0
        quality[:, :, tp_index][corrected] = 0
        day_lookup = {d: i for i, d in enumerate(dates)}
        day_frames = []
        for j, grid in enumerate(grids.itertuples()):
            row = {"grid_id": grid.grid_id, "year": year, "expected_days": len(dates)}
            daily = pd.DataFrame({"grid_id": grid.grid_id, "date_utc": dates})
            daily["tp_raw"] = raw_tp[:, j]
            daily["tp_quality_raw"] = raw_tp_quality[:, j]
            daily["tp_zero_clipped"] = corrected[:, j]
            row["tp_zero_clipped_days"] = int(corrected[:, j].sum())
            row["tp_zero_correction_total_mm"] = float(-raw_tp[corrected[:, j], j].sum())
            for d in np.flatnonzero(corrected[:, j]):
                stamp = pd.Timestamp(str(dates[d]))
                source = arco_path(root, stamp) if kinds[d, tp_index] == 1 else cds_path(root, year, stamp.month, "tp")
                corrections.append({"grid_id": grid.grid_id, "date_utc": str(dates[d]), "variable": "tp",
                    "raw_value_mm": raw_tp[d, j], "cleaned_value_mm": 0.0, "raw_quality": 5,
                    "rule": "small_negative_to_zero_v1", "tolerance_mm_day": precipitation_zero_tolerance_mm,
                    "source": str(source)})
            for k, name in enumerate(NAMES):
                valid = quality[:, j, k] == 0
                a = values[:, j, k]
                row[f"{name}_valid_days"] = int(valid.sum())
                row[f"{name}_arco_audited_days"] = int(((coverage[:, j, k] == 1) & valid).sum())
                row[f"{name}_cds_contract_days"] = int(((coverage[:, j, k] == 2) & valid).sum())
                # Totals only for precipitation. Daily radiation mean avoids a
                # leap-year day-count artefact in its per-decade trend.
                annual_name = f"{name}_annual_total" if name == "tp" else f"{name}_annual_mean"
                row[annual_name] = float(a.sum() if name == "tp" else a.mean()) if valid.all() else np.nan
                daily[name], daily[f"{name}_quality"], daily[f"{name}_coverage"] = a, quality[:, j, k], coverage[:, j, k]
                for d in np.flatnonzero(~valid):
                    source = (arco_path(root, pd.Timestamp(str(dates[d]))) if kinds[d, k] == 1
                              else cds_path(root, year, pd.Timestamp(str(dates[d])).month, name))
                    issues.append({"grid_id": grid.grid_id, "date_utc": dates[d], "variable": name,
                                   "status": QUALITY[int(quality[d, j, k])], "raw_converted_value": a[d], "source": str(source)})
                e = event_map[(grid.grid_id, year)]
                d = day_lookup.get(e.event_date_utc) if e.event_status == "ok" else None
                q = int(quality[d, j, k]) if d is not None else None
                cov = int(coverage[d, j, k]) if d is not None else 0
                status = ("event_unavailable" if d is None else QUALITY[q])
                if status == "ok" and cov == 2:
                    status = "ok_unverified_coverage"
                source = (arco_path(root, pd.Timestamp(e.event_date_utc)) if d is not None and kinds[d, k] == 1
                          else cds_path(root, year, pd.Timestamp(e.event_date_utc).month, name) if d is not None else "")
                event_rows.append({"grid_id": grid.grid_id, "year": year, "event_date_utc": e.event_date_utc,
                                   "variable": name, "value": values[d, j, k] if d is not None else np.nan,
                                   "raw_converted_value": (raw_tp[d, j] if name == "tp" else values[d, j, k]) if d is not None else np.nan,
                                   "zero_clipped": bool(d is not None and name == "tp" and corrected[d, j]),
                                   "units": event.FIELDS[name][1], "statistic": event.FIELDS[name][0],
                                   "source_file": str(source), "source_variable": name,
                                   "status": status, "coverage_status": COVERAGE[cov],
                                   "valid_hours": 24 if cov == 1 else np.nan,
                                   "provenance": "ARCO audited hourly reduction" if cov == 1 else "CDS UTC daily product; source-hour count unavailable",
                                   "detail": e.event_status if d is None else ""})
            wind_valid = (quality[:, j, 0] == 0) & (quality[:, j, 1] == 0)
            row["wind_vector_magnitude_annual_mean"] = float(np.hypot(values[:, j, 0], values[:, j, 1]).mean()) if wind_valid.all() else np.nan
            row["complete_daily_values"] = bool((quality[:, j] == 0).all())
            row["all_hourly_coverage_audited"] = bool((coverage[:, j] == 1).all())
            annual_rows.append(row)
            day_frames.append(daily)
        daily_path = output/"daily_taiwan"/f"taiwan_daily_{year}.csv.gz"
        daily_path.parent.mkdir(exist_ok=True)
        pd.concat(day_frames, ignore_index=True).to_csv(daily_path, index=False, compression="gzip")
        daily_files.append(str(daily_path))
    annual = pd.DataFrame(annual_rows)
    matched = pd.DataFrame(event_rows)
    paired, event_summary, event_issues, event_issue_summary = event.make_tables(events, matched, grids, len(years))
    tp_events = matched.loc[matched.variable.eq("tp"), ["grid_id", "year", "raw_converted_value", "zero_clipped"]]
    paired = paired.merge(tp_events.rename(columns={"raw_converted_value": "tp_raw_value", "zero_clipped": "tp_zero_clipped"}),
                          on=["grid_id", "year"], validate="one_to_one", sort=False)
    summary = []
    features = [c for c in annual if c.endswith(("_annual_mean", "_annual_total"))]
    for grid in grids.itertuples():
        part = annual.loc[annual.grid_id.eq(grid.grid_id)].sort_values("year")
        row = {"grid_id": grid.grid_id}
        for name in features:
            a = part[name].to_numpy()
            row[f"{name}_valid_years"] = int(np.isfinite(a).sum())
            row[f"{name}_mean_50y"] = float(a.mean()) if np.isfinite(a).all() else np.nan
            row[f"{name}_trend_per_decade"] = slope(a, part.year)
        row["annual_weather_complete"] = bool(part.complete_daily_values.all())
        row["all_hourly_coverage_audited"] = bool(part.all_hourly_coverage_audited.all())
        summary.append(row)
    frames = {"annual_weather.csv": annual, "annual_weather_summary.csv": pd.DataFrame(summary),
              "event_weather_long.csv": matched, "event_weather.csv": paired,
              "event_grid_predictors.csv": event_summary, "event_missing_report.csv": event_issues,
              "event_missing_summary.csv": event_issue_summary,
              "daily_missing_report.csv": pd.DataFrame(issues, columns=["grid_id", "date_utc", "variable", "status", "raw_converted_value", "source"]),
              "precipitation_corrections.csv": pd.DataFrame(corrections, columns=["grid_id", "date_utc", "variable", "raw_value_mm",
                  "cleaned_value_mm", "raw_quality", "rule", "tolerance_mm_day", "source"])}
    for name, frame in frames.items():
        event.write_csv(output/name, frame)
    return frames, daily_files


def run(root, annual, registry, output, *, start_year=1976, end_year=2025, workers=2, resume=False, static=True,
        cache_source=None, precipitation_zero_tolerance_mm=0.0):
    started = time.perf_counter()
    root, annual, registry, output = [Path(p).resolve() for p in (root, annual, registry, output)]
    if not root.is_dir() or not annual.is_file():
        raise FileNotFoundError("Data drive or annual source is unavailable")
    if not 1 <= workers <= 4 or start_year > end_year:
        raise ValueError("Use 1--4 workers and an ordered year interval")
    if not np.isfinite(precipitation_zero_tolerance_mm) or precipitation_zero_tolerance_mm < 0:
        raise ValueError("Precipitation tolerance must be finite and nonnegative (mm/day)")
    grids = event.validate_grids(registry, 50)
    if output.exists() and not resume:
        raise FileExistsError("Use a new output directory, or explicit --resume for this pipeline")
    output.mkdir(parents=True, exist_ok=True)
    cache = output/"cache_months"
    protocol = {"version": VERSION, "data_root": str(root), "annual_source": event.signature(annual),
                "registry_sha256": event.sha256(registry), "grid_sha256": extraction_identity(grids),
                "years": [start_year, end_year], "static": static,
                "cds_policy": "Accept documented daily product with finite-day checks; retain missing hourly-coverage evidence flag",
                "arco_policy": "Require completed fail-closed 24-hour/all-grid producer contract and matching full-value audit",
                "source_code_sha256": event.sha256(__file__), "matching_code_sha256": event.sha256(event.__file__)}
    protocol.update(assembly_version=ASSEMBLY_VERSION, precipitation_cleaning={
        "rule": "small_negative_to_zero_v1", "tolerance_mm_day": precipitation_zero_tolerance_mm,
        "condition": "quality=5 AND finite AND -tolerance <= tp < 0; only tp, no missing-value imputation",
        "raw_sources_modified": False})
    if static:
        protocol["static_code_sha256"] = event.sha256(Path(__file__).with_name("prepare_static_predictors.py"))
    if cache_source is not None:
        cache_source = Path(cache_source).resolve()
        if cache_source == output or cache_source in output.parents or output in cache_source.parents:
            raise ValueError("Use a separate, non-nested output for the derived cache-based run")
        source_protocol_path = cache_source/"protocol.json"
        source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
        for key in ("version", "data_root", "annual_source", "registry_sha256", "grid_sha256", "years", "matching_code_sha256"):
            if source_protocol.get(key) != protocol[key]:
                raise ValueError(f"Cache source protocol mismatch: {key}")
        cache = cache_source/"cache_months"
        protocol["cache_source"] = str(cache_source)
        protocol["cache_source_protocol_sha256"] = event.sha256(source_protocol_path)
    protocol_path = output/"protocol.json"
    if resume and (not protocol_path.is_file() or json.loads(protocol_path.read_text(encoding="utf-8")) != protocol):
        raise ValueError("Resume protocol differs; use a new output directory")
    event.write_json(protocol_path, protocol)
    status_path = output/"run_status.json"
    event.write_json(status_path, {"status": "running", "stage": "annual_event_index", "started_utc": event.utc_now()})
    try:
        years = list(range(start_year, end_year+1))
        events = event.build_event_index(annual, grids, years)
        event.write_csv(output/"annual_event_index.csv", events)
        event.write_csv(output/"grid_registry.csv", grids)
        if static:
            from taiwan_simulation.prepare_static_predictors import prepare_static
            prepare_static(root/"static", grids, output)
        reports = []
        jobs = [(y, m) for y in years for m in range(1, 13)]
        def accept_month(info):
                reports.append(info)
                event.write_json(status_path, {"status": "running", "stage": "daily_extraction", "completed_months": len(reports),
                                              "expected_months": len(jobs), "last_month": [info["year"], info["month"]],
                                              "invalid_values_so_far": sum(r["n_invalid_values"] for r in reports),
                                              "elapsed_seconds": time.perf_counter()-started})
                if len(reports) == 1 or len(reports) % 12 == 0 or len(reports) == len(jobs):
                    print(f"Extracted {len(reports)}/{len(jobs)} months; {time.perf_counter()-started:.1f} s", flush=True)
        if cache_source is not None:
            for year, month in jobs:
                accept_month(verified_cached_month(cache, grids, year, month))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(extract_month, root, grids, y, m, cache): (y, m) for y, m in jobs}
                for future in as_completed(futures):
                    accept_month(future.result())
        event.write_csv(output/"source_month_inventory.csv", pd.DataFrame(reports).sort_values(["year", "month"]))
        event.write_json(status_path, {"status": "running", "stage": "summaries", "completed_months": len(jobs)})
        frames, daily_files = assemble(cache, grids, events, years, output, root,
                                      precipitation_zero_tolerance_mm=precipitation_zero_tolerance_mm)
        if cache_source is not None:
            if event.sha256(cache_source/"protocol.json") != protocol["cache_source_protocol_sha256"]:
                raise RuntimeError("Cache source protocol changed during assembly")
            for item in reports:
                target = cache/f"{item['year']}{item['month']:02d}.npz"
                if (event.sha256(target) != item["cache_sha256"]
                        or event.sha256(target.with_suffix(".json")) != item["metadata_sha256"]):
                    raise RuntimeError("Cache source changed during assembly")
        summary = frames["annual_weather_summary.csv"]
        combined = grids.merge(summary, on="grid_id", validate="one_to_one")
        if static:
            fixed = pd.read_csv(output/"static_predictors.csv")
            combined = combined.merge(fixed, on="grid_id", validate="one_to_one")
        combined["candidate_data_complete"] = combined.annual_weather_complete & (combined.static_complete if static else True)
        event.write_csv(output/"grid_predictors.csv", combined)
        baseline, trends = {}, {}
        for name in NAMES + ("wind_vector_magnitude",):
            metric = f"{name}_annual_total" if name == "tp" else f"{name}_annual_mean"
            baseline[name] = [metric+"_mean_50y"]
            trends[name] = baseline[name] + [metric+"_trend_per_decade"]
        if static:
            for groups in (baseline, trends):
                groups.update(elevation=["elevation_m"], coast=["coast_distance_km"],
                              land_cover=["agriculture_ratio", "forest_ratio", "urban_ratio", "water_ratio"],
                              terrain_position=["terrain_025_tpi_m"], terrain_relief=["terrain_025_local_relief_m"])
        event.write_json(output/"candidate_groups_comparison.json", {"baseline_means": baseline, "means_plus_trends": trends,
            "note": "Candidates only; not selected models. Omit one land-cover fraction with intercept. Wind group definitions are correlated; let training-only FFS/VIF handle this."})
        if event.signature(annual) != protocol["annual_source"] or event.sha256(registry) != protocol["registry_sha256"]:
            raise RuntimeError("Annual source or GRID registry changed during extraction")
        has_errors = (not frames["daily_missing_report.csv"].empty or not events.event_status.eq("ok").all()
                      or not combined.candidate_data_complete.all())
        cds_days = sum(r["n_cds_variable_days"] for r in reports)
        report = {"version": VERSION, "status": ("completed_with_issues" if has_errors else
                                                 "completed_with_coverage_notes" if cds_days else "completed"),
                  "completed_utc": event.utc_now(), "n_grids": len(grids), "years": years, "event_rows": len(events),
                  "annual_weather_rows": len(frames["annual_weather.csv"]), "n_complete_weather_grids": int(summary.annual_weather_complete.sum()),
                  "n_candidate_data_complete_grids": int(combined.candidate_data_complete.sum()),
                  "daily_error_rows": len(frames["daily_missing_report.csv"]), "daily_files": daily_files,
                  "cds_variable_days": cds_days,
                  "arco_variable_days": sum(r["n_arco_variable_days"] for r in reports),
                  "precipitation_cleaning": protocol["precipitation_cleaning"],
                  "n_precipitation_zero_clipped": len(frames["precipitation_corrections.csv"]),
                  "n_event_precipitation_zero_clipped": int(frames["event_weather_long.csv"].zero_clipped.sum()),
                  "max_grid_year_correction_mm": float(frames["annual_weather.csv"].tp_zero_correction_total_mm.max()),
                  "cache_source": str(cache_source) if cache_source is not None else None,
                  "nn_changed": False, "nn_loaded": False, "global_processed": False,
                  "elapsed_seconds": time.perf_counter()-started,
                  "definitions": {"tp": "Annual total mm; 50-year mean and OLS trend in mm/decade",
                    "ssrd": "Annual mean daily MJ/m2; 50-year mean and trend per decade",
                    "u10_v10_tcc": "Annual average of UTC daily means; mean and trend per decade",
                    "wind": "Annual average of hypot(daily mean u, daily mean v), NOT mean scalar hourly speed",
                    "trends": "OLS on t=(year-2000.5)/10, all requested years required; no significance or causality claim",
                    "events": "Separate event-conditioned features; not silently substituted for annual climatology",
                    "coverage": "CDS has validated daily values and request metadata, not independently certified finite hourly counts",
                    "static": "ERA5 coarse geopotential terrain, CCI year-2000 fixed land cover, GSHHG ocean coastline",
                    "cv": "No GP/FFS/CV executed. Feature scaling/selection remains inside training folds."}}
        event.write_json(output/"run_report.json", report)
        event.write_json(status_path, {"status": report["status"], "stage": "completed", "elapsed_seconds": report["elapsed_seconds"]})
        return report
    except BaseException as error:
        event.write_json(status_path, {"status": "failed", "error": str(error), "error_type": type(error).__name__})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--annual-netcdf", type=Path, default=DEFAULT_ANNUAL)
    p.add_argument("--grid-registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start-year", type=int, default=1976)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-static", action="store_true")
    p.add_argument("--cache-source", type=Path, help="Read-only validated cache reuse from a previous output; never re-extract or overwrite that run")
    p.add_argument("--precipitation-zero-tolerance-mm", type=float, default=0.0,
                   help="Explicit daily precipitation tolerance; default 0 retains strict negative-value rejection")
    a = p.parse_args()
    report = run(a.data_root, a.annual_netcdf, a.grid_registry, a.output,
                 start_year=a.start_year, end_year=a.end_year, workers=a.workers,
                 resume=a.resume, static=not a.no_static, cache_source=a.cache_source,
                 precipitation_zero_tolerance_mm=a.precipitation_zero_tolerance_mm)
    print(json.dumps({k: report[k] for k in ("status", "n_grids", "n_complete_weather_grids", "daily_error_rows", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
