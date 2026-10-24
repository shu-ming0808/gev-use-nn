"""Pinned ARCO/CDS daily-contract reader for the portable bundle."""
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

from bundle import io as event

VERSION = "taiwan_predictors_mixed_sources_v1"
ASSEMBLY_VERSION = "taiwan_predictors_assembly_v2"
ARCO_CONTRACT = "arco_era5_daily_utc_v1"
ARCO_STORE = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
NAMES = tuple(event.FIELDS)
QUALITY = {0: "ok", 1: "missing_file", 2: "invalid_source", 3: "missing_date",
           4: "missing_value", 5: "invalid_value", 6: "source_changed"}
COVERAGE = {0: "unknown", 1: "upstream_24_hour_all_grid_audit", 2: "cds_daily_contract_hourly_count_unavailable"}
ROOT = SOURCE_ROOT.parent


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



