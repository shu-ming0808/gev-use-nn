import calendar
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from src.prepare_era5_annual_maxima import (
    EPOCH, audit_month, combine_years, celsius_offset, open_nc,
    prepare_year, read_mask, reduce_block, run_lock,
)


def make_month(path, year=2000, month=1, missing=False, time_error=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = calendar.monthrange(year, month)[1] * 24
    start = (dt.datetime(year, month, 1) - dt.datetime(1970, 1, 1)).total_seconds()
    with open_nc(path, "w") as ds:
        for name, size in (("valid_time", n), ("latitude", 2), ("longitude", 3)):
            ds.createDimension(name, size)
        ds.createVariable("latitude", "f8", ("latitude",))[:] = [1, 0]
        ds.createVariable("longitude", "f8", ("longitude",))[:] = [0, 1, 2]
        tv = ds.createVariable("valid_time", "i8", ("valid_time",))
        times = start + 3600 * np.arange(n)
        if time_error:
            times[1] = times[0]
        tv[:] = times
        tv.units = EPOCH
        var = ds.createVariable("t2m", "f4", ("valid_time", "latitude", "longitude"), fill_value=-9999., chunksizes=(24, 2, 3))
        var.units = "K"
        values = np.full((n, 2, 3), 280., dtype=np.float32)
        values[0, 0, 0] = 300.
        values[-1, 1, 2] = 290. + month
        if missing:
            values[3, 0, 1] = -9999.
        var[:] = values


def make_mask(path):
    with open_nc(path, "w") as ds:
        ds.createDimension("latitude", 2)
        ds.createDimension("longitude", 3)
        ds.createDimension("valid_time", 1)
        ds.createVariable("latitude", "f8", ("latitude",))[:] = [1, 0]
        ds.createVariable("longitude", "f8", ("longitude",))[:] = [0, 1, 2]
        ds.createVariable("lsm", "f4", ("valid_time", "latitude", "longitude"))[:] = [[[.5, .51, 1], [0, .8, .2]]]


def test_block_reduction_preserves_first_tie_and_counts():
    maximum = np.full((1, 2), -np.inf, dtype=np.float32)
    peak = np.full((1, 2), -1, dtype=np.int64)
    counts = np.zeros((1, 2), dtype=np.uint32)
    reduce_block(np.array([[[2, np.nan]], [[3, 4]], [[3, 1]]]), [100, 200, 300], maximum, peak, counts)
    reduce_block(np.array([[[3, 5]]]), [400], maximum, peak, counts)
    np.testing.assert_array_equal(maximum, [[3, 5]])
    np.testing.assert_array_equal(peak, [[200, 400]])
    np.testing.assert_array_equal(counts, [[4, 3]])


def test_hourly_audit_rejects_duplicate_hour_and_wrong_grid(tmp_path):
    path = tmp_path / "bad.nc"
    make_month(path, time_error=True)
    with pytest.raises(ValueError, match="timestamps"):
        audit_month(path, 2000, 1, global_grid=False)
    make_month(path)
    with pytest.raises(ValueError, match="global 0.25"):
        audit_month(path, 2000, 1)
    with pytest.raises(ValueError, match="Grid mismatch"):
        audit_month(path, 2000, 1, (np.array([2, 1]), np.array([0, 1, 2])), False)


@pytest.mark.parametrize("year,hours", [(2000, 696), (2001, 672)])
def test_february_including_leap_day(tmp_path, year, hours):
    path = tmp_path / "feb.nc"
    make_month(path, year, 2)
    info, _ = audit_month(path, year, 2, global_grid=False)
    assert info["hours"] == hours


def test_unit_validation():
    assert celsius_offset("K") == 273.15
    assert celsius_offset("degrees_Celsius") == 0
    with pytest.raises(ValueError, match="units"):
        celsius_offset("F")


def test_annual_roundtrip_resume_corruption_and_land_export(tmp_path):
    records = []
    for month in range(1, 13):
        path = tmp_path / f"input_{month}.nc"
        make_month(path, month=month, missing=(month == 1))
        info, _ = audit_month(path, 2000, month, global_grid=False)
        records.append(info)
    out = tmp_path / "年度輸出"
    out.mkdir()
    first = prepare_year(2000, records, out, global_grid=False)
    assert first["expected_hours"] == 8784
    assert first["complete_cells"] == 5
    with open_nc(first["file"]) as ds:
        values = np.ma.filled(ds["annual_maximum_t2m"][:], np.nan)
        assert values[0, 0] == pytest.approx(26.85, abs=3e-5)
        assert values[1, 2] == pytest.approx(28.85, abs=3e-5)
        assert np.isnan(values[0, 1])
        assert ds["valid_hour_count"][0, 1] == 8783
        assert ds["maximum_time"][0, 0] == records[0]["start_seconds"]
        assert ds["maximum_time"][1, 2] == records[-1]["start_seconds"] + 3600 * (records[-1]["hours"]-1)
    assert prepare_year(2000, records, out, False)["reused"]
    # A present but corrupted result must not be trusted just because a marker exists.
    Path(first["file"]).write_bytes(b"broken")
    assert not prepare_year(2000, records, out, False)["reused"]
    # Changing the source creates a new signature and forces a recomputation.
    with open_nc(records[1]["path"], "a") as ds:
        ds["t2m"][0, 0, 0] = 310
    records[1], _ = audit_month(records[1]["path"], 2000, 2, global_grid=False)
    assert not prepare_year(2000, records, out, False)["reused"]
    mask = tmp_path / "lsm.nc"
    make_mask(mask)
    outputs = combine_years([2000], out, mask, .5)
    assert outputs["land_cells"] == 3
    assert outputs["land_cell_years_missing"] == 1
    with Path(outputs["land_csv"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(float(row["lsm"]) > .5 for row in rows)
    assert rows[0]["2000"] == ""
    with open_nc(outputs["global_netcdf"]) as ds:
        assert ds["annual_maximum_t2m"].shape == (1, 2, 3)
        assert ds["expected_hours"][0] == 8784


def test_misaligned_mask_and_concurrent_lock(tmp_path):
    path = tmp_path / "lsm.nc"
    make_mask(path)
    with pytest.raises(ValueError, match="grid differs"):
        read_mask(path, np.array([0, 1]), np.array([0, 1, 2]))
    with run_lock(tmp_path):
        with pytest.raises(OSError):
            with run_lock(tmp_path):
                pass
    with run_lock(tmp_path):
        pass
