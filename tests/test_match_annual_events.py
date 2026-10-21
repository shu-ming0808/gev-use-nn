"""Offline fixtures: exact joins, UTC semantics, missingness and NN isolation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from taiwan_simulation import match_annual_events as pipeline


EPOCH = "hours since 1970-01-01 00:00:00"
ROOT = Path(__file__).resolve().parents[1]


def timestamp(value):
    return pd.Timestamp(value, tz="UTC").timestamp() / 3600


def write_registry(path, lat=(25.,), lon=(121., 121.25), pairs=None):
    pairs = pairs or [(a, b) for a in lat for b in lon]
    rows = []
    for a, b in pairs:
        rows.append(dict(grid_id=f"G{round((90-a)/.25):03d}_{round((b%360)/.25):04d}",
                         latitude=a, longitude=b % 360, longitude_180=(b+180)%360-180, lsm=1.))
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def write_annual(path, grids, years=(2000, 2001), dates=None):
    lat = np.sort(grids.latitude.unique())[::-1]
    lon = np.sort(grids.longitude.unique())
    with nc.Dataset(path, "w") as ds:
        for name, values in (("year", years), ("latitude", lat), ("longitude", lon)):
            ds.createDimension(name, len(values))
            ds.createVariable(name, "f8", (name,))[:] = values
        dims = ("year", "latitude", "longitude")
        shape = (len(years), len(lat), len(lon))
        maximum = ds.createVariable("annual_maximum_t2m", "f8", dims, fill_value=np.nan)
        maximum.units = "degrees_Celsius"
        maximum[:] = 35 + np.arange(np.prod(shape)).reshape(shape) * .01
        occurrence = ds.createVariable("maximum_time", "f8", dims, fill_value=np.nan)
        occurrence.units = EPOCH
        counts = ds.createVariable("valid_hour_count", "i4", dims)
        expected = ds.createVariable("expected_hours", "i4", ("year",))
        ds.createVariable("lsm", "f4", ("latitude", "longitude"))[:] = 1
        for i, year in enumerate(years):
            date = dates[i] if dates else f"{year}-07-01"
            occurrence[i] = timestamp(date) + 12
            expected[i] = 8784 if pipeline.calendar.isleap(year) else 8760
            counts[i] = expected[i]
    return lat, lon


def write_daily(path, lat=(25.,), lon=(121., 121.25), dates=("2000-07-01", "2001-07-01"),
                reverse=False, day_end=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deliberately shuffle dimension/coordinate order to catch positional joins.
    lat, lon = np.array(lat), np.array(lon)
    if reverse:
        lat, lon = lat[::-1], lon[::-1]
    with nc.Dataset(path, "w") as ds:
        for name, values in (("time", dates), ("lat", lat), ("lon", lon)):
            ds.createDimension(name, len(values))
            var = ds.createVariable(name, "f8", (name,))
            var[:] = ([timestamp(d) + (24 if day_end else 0) for d in dates]
                      if name == "time" else values)
            if name == "time":
                var.units = EPOCH
        dims = ("lon", "time", "lat") if reverse else ("time", "lat", "lon")
        shape = tuple(len(ds.dimensions[d]) for d in dims)
        for name, value, units in (("u10", 3., "m s-1"), ("v10", 4., "m/s"),
                                   ("tcc", 50., "%"), ("tp", .002, "m"),
                                   ("ssrd", 2e6, "J m**-2")):
            var = ds.createVariable(name, "f8", dims, fill_value=np.nan)
            var.units = units
            var.cell_methods = f"time: {pipeline.FIELDS[name][0]}"
            # Coordinate-based perturbation proves paired indexing stays aligned.
            for coord in np.ndindex(shape):
                longitude = lon[coord[dims.index("lon")]]
                var[coord] = value + ((longitude % 360) - 121.) * (1 if name == "u10" else 0)
            ds.createVariable(f"{name}_n", "i4", dims)[:] = 24
        ds.time_zone = "UTC"


def config_for(tmp_path, n=2, years=(2000, 2001)):
    return dict(
        version=1, annual_netcdf="annual.nc", daily_root="daily", grid_registry="grids.csv",
        expected_grids=n, start_year=years[0], end_year=years[-1],
        min_valid_years=len(years), test_fixture=True,
        variables={name: dict(path_template="daily.nc", variable=name, statistic=stat,
                              timezone="UTC", day_label="start", valid_count_variable=f"{name}_n",
                              allow_unverified_coverage=False, provenance="Offline test fixture")
                   for name, (stat, _) in pipeline.FIELDS.items()})


def save_config(tmp_path, config):
    path = tmp_path/"config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.fixture
def setup(tmp_path):
    grids = write_registry(tmp_path/"grids.csv")
    write_annual(tmp_path/"annual.nc", grids)
    write_daily(tmp_path/"daily/daily.nc")
    config = config_for(tmp_path)
    return tmp_path, grids, config


def execute(setup, name="out"):
    path, _, config = setup
    return pipeline.run(save_config(path, config), path/name)


def test_end_to_end_units_pairing_summary_and_no_input_changes(setup):
    path, _, _ = setup
    before = {str(p): pipeline.sha256(p) for p in path.rglob("*") if p.is_file()}
    report = execute(setup)
    assert report["status"] == "completed"
    assert report["n_event_rows"] == 4
    assert report["n_requested_weather_rows"] == 20
    assert report["n_ready_grids"] == 2
    assert not report["nn_loaded"] and not report["nn_changed"] and not report["global_run"]
    events = pd.read_csv(path/"out/annual_event_index.csv")
    assert events.maximum_time_utc.iloc[0] == "2000-07-01T12:00:00+00:00"
    assert set(events.event_date_utc) == {"2000-07-01", "2001-07-01"}
    summary = pd.read_csv(path/"out/grid_predictors.csv")
    np.testing.assert_allclose(summary.u10_event_mean, [3, 3.25])
    np.testing.assert_allclose(summary.tcc_event_mean, [.5, .5])
    np.testing.assert_allclose(summary.tp_event_mean, [2, 2])
    np.testing.assert_allclose(summary.ssrd_event_mean, [2, 2])
    assert summary.wind_vector_magnitude_event_mean.iloc[0] == 5
    assert summary.all_daily_coverage_verified.all()
    assert pd.read_csv(path/"out/missing_report.csv").empty
    assert all(pipeline.sha256(p) == digest for p, digest in before.items())
    with pytest.raises(FileExistsError, match="new output"):
        execute(setup)


def test_reversed_coordinates_and_transposed_dimensions_are_exact(setup):
    path, _, _ = setup
    write_daily(path/"daily/daily.nc", reverse=True)
    report = execute(setup)
    assert report["status"] == "completed"
    summary = pd.read_csv(path/"out/grid_predictors.csv")
    np.testing.assert_allclose(summary.u10_event_mean, [3, 3.25])


def test_paired_indices_do_not_form_cartesian_output(tmp_path):
    grids = write_registry(tmp_path/"grids.csv", pairs=[(25., 121.), (24.75, 121.25)])
    write_annual(tmp_path/"annual.nc", grids)
    write_daily(tmp_path/"daily/daily.nc", lat=(25., 24.75), reverse=True)
    config = config_for(tmp_path)
    report = pipeline.run(save_config(tmp_path, config), tmp_path/"out")
    assert report["n_event_rows"] == 4
    summary = pd.read_csv(tmp_path/"out/grid_predictors.csv")
    np.testing.assert_allclose(summary.u10_event_mean, [3, 3.25])


@pytest.mark.parametrize("defect,expected", [
    ("missing_value", "missing_value"), ("count", "incomplete_day"),
    ("units", "unit_mismatch"), ("bad_cloud", "invalid_value"),
    ("duplicate_time", "duplicate_date"), ("noon", "time_mismatch"),
    ("grid", "grid_mismatch"), ("cell_methods", "statistic_mismatch"),
    ("timezone", "time_mismatch"), ("count_dimension", "schema_error"),
])
def test_data_defects_become_reports_without_dropping_events(setup, defect, expected):
    path, _, config = setup
    with nc.Dataset(path/"daily/daily.nc", "a") as ds:
        if defect == "missing_value":
            ds["tp"][0, 0, 0] = np.nan
        elif defect == "count":
            ds["tp_n"][0, 0, 0] = 23
        elif defect == "units":
            ds["tp"].units = "m s-1"
        elif defect == "bad_cloud":
            ds["tcc"][0, 0, 0] = 101
        elif defect == "duplicate_time":
            ds["time"][1] = ds["time"][0]
        elif defect == "noon":
            ds["time"][:] += 12
        elif defect == "grid":
            ds["lon"][:] += .01
        elif defect == "cell_methods":
            ds["tp"].cell_methods = "time: mean"
        elif defect == "timezone":
            ds.time_zone = "Asia/Taipei"
        elif defect == "count_dimension":
            ds.createVariable("global_count", "i4", ("time",))[:] = 24
            config["variables"]["tp"]["valid_count_variable"] = "global_count"
    report = execute(setup)
    assert report["status"] == "completed_with_issues"
    assert report["n_event_rows"] == 4
    issues = pd.read_csv(path/"out/missing_report.csv")
    assert expected in set(issues.status)
    assert len(pd.read_csv(path/"out/event_weather.csv")) == 4
    assert report["n_ready_grids"] < 2


@pytest.mark.parametrize("missing", ["file", "date", "variable"])
def test_missing_weather_is_explicit(setup, missing):
    path, _, config = setup
    if missing == "file":
        config["variables"]["tp"]["path_template"] = "absent.nc"
    elif missing == "date":
        write_daily(path/"daily/daily.nc", dates=("2000-07-01",))
    else:
        config["variables"]["tp"]["variable"] = "not_in_file"
    execute(setup)
    issues = pd.read_csv(path/"out/missing_report.csv")
    assert f"missing_{missing}" in set(issues.status)
    summary = pd.read_csv(path/"out/grid_predictors.csv")
    assert summary.tp_event_mean.isna().all()
    assert not summary.ready_for_spatial_cv.any()


def test_missing_temperature_year_and_timestamp_preserve_rows(setup):
    path, _, config = setup
    config["end_year"] = 2002
    config["min_valid_years"] = 3
    with nc.Dataset(path/"annual.nc", "a") as ds:
        ds["valid_hour_count"][0, 0, 0] = 8783
        ds["maximum_time"][1, 0, 1] = np.nan
    report = execute(setup)
    assert report["n_event_rows"] == 6
    events = pd.read_csv(path/"out/annual_event_index.csv")
    assert events.event_status.value_counts().to_dict() == {
        "ok": 2, "missing_year": 2, "incomplete_temperature_year": 1, "missing_event_time": 1}
    assert report["n_requested_weather_rows"] == 30
    assert report["n_ready_grids"] == 0


def test_invalid_event_date_is_not_matched(setup):
    path, _, _ = setup
    with nc.Dataset(path/"annual.nc", "a") as ds:
        ds["maximum_time"][0, 0, 0] = timestamp("2001-01-01")
    execute(setup)
    events = pd.read_csv(path/"out/annual_event_index.csv")
    assert events.event_status.iloc[0] == "event_time_mismatch"
    matched = pd.read_csv(path/"out/event_weather_long.csv")
    assert (matched.loc[(matched.grid_id == events.grid_id.iloc[0]) & (matched.year == 2000),
                        "status"] == "event_unavailable").all()


@pytest.mark.parametrize("allow", [False, True])
def test_unverified_coverage_requires_explicit_policy_and_stays_visible(setup, allow):
    path, _, config = setup
    config["variables"]["tp"].update(valid_count_variable=None, allow_unverified_coverage=allow)
    report = execute(setup)
    assert report["status"] == "completed_with_issues"
    assert report["n_ready_grids"] == (2 if allow else 0)
    rows = pd.read_csv(path/"out/event_weather_long.csv")
    assert set(rows.loc[rows.variable.eq("tp"), "status"]) == {
        "ok_unverified_coverage" if allow else "coverage_unverified"}
    # Even when explicitly allowed, do not certify observed-hour completeness.
    summary = pd.read_csv(path/"out/grid_predictors.csv")
    assert not summary.all_daily_coverage_verified.any()
    assert summary.tp_n_verified_years.eq(0).all()


def test_cds_request_is_checked_but_not_used_as_proof_of_valid_hours(setup):
    path, _, config = setup
    config["variables"]["tp"].update(request_sidecar=True, valid_count_variable=None)
    sidecar = path/"daily/daily.request.json"
    sidecar.write_text(json.dumps({"request": {"daily_statistic": "daily_sum",
                                              "time_zone": "utc+00:00", "frequency": "1_hourly"}}))
    execute(setup)
    issues = pd.read_csv(path/"out/missing_report.csv")
    assert "coverage_unverified" in set(issues.status)
    sidecar.write_text(json.dumps({"request": {"daily_statistic": "daily_sum",
                                              "time_zone": "utc+08:00", "frequency": "1_hourly"}}))
    execute(setup, "bad-request")
    issues = pd.read_csv(path/"bad-request/missing_report.csv")
    assert "statistic_mismatch" in set(issues.status)


def test_day_end_label_routes_to_next_year_file(tmp_path):
    grids = write_registry(tmp_path/"grids.csv")
    write_annual(tmp_path/"annual.nc", grids, years=(2000,), dates=("2000-12-31",))
    write_daily(tmp_path/"daily/2001/01.nc", dates=("2000-12-31",), day_end=True)
    config = config_for(tmp_path, years=(2000,))
    for spec in config["variables"].values():
        spec.update(path_template="{year}/{month:02d}.nc", day_label="end")
    report = pipeline.run(save_config(tmp_path, config), tmp_path/"out")
    assert report["status"] == "completed"
    events = pd.read_csv(tmp_path/"out/event_weather.csv")
    assert set(events.event_date_utc) == {"2000-12-31"}


def test_cf_bounds_reject_wrong_accumulation_day(setup):
    path, _, _ = setup
    with nc.Dataset(path/"daily/daily.nc", "a") as ds:
        ds.createDimension("bounds", 2)
        ds["time"].bounds = "time_bounds"
        bounds = ds.createVariable("time_bounds", "f8", ("time", "bounds"))
        times = ds["time"][:]
        bounds[:] = np.stack([times-24, times], axis=1)
    execute(setup)
    assert "time_mismatch" in set(pd.read_csv(path/"out/missing_report.csv").status)


def test_fifty_grids_fifty_years_contract(tmp_path):
    grids = write_registry(tmp_path/"grids.csv", lat=(25., 24.75, 24.5, 24.25, 24.),
                           lon=tuple(120 + .25*np.arange(10)))
    years = tuple(range(1976, 2026))
    lat, lon = write_annual(tmp_path/"annual.nc", grids, years)
    write_daily(tmp_path/"daily/daily.nc", lat, lon, dates=tuple(f"{y}-07-01" for y in years), reverse=True)
    report = pipeline.run(save_config(tmp_path, config_for(tmp_path, n=50, years=years)), tmp_path/"out")
    assert report["status"] == "completed"
    assert report["n_grids"] == report["n_ready_grids"] == 50
    assert report["n_event_rows"] == 2500
    assert report["n_requested_weather_rows"] == 12500
    assert len(report["daily_sources"]) == 1


def test_registry_coordinate_identity_and_unavailable_drive_fail_early(setup):
    path, _, config = setup
    frame = pd.read_csv(path/"grids.csv")
    frame.loc[0, "grid_id"] = "G000_0000"
    frame.to_csv(path/"grids.csv", index=False)
    with pytest.raises(ValueError, match="IDs disagree"):
        execute(setup)
    write_registry(path/"grids.csv")
    config["daily_root"] = "not_mounted"
    with pytest.raises(FileNotFoundError, match="Mount"):
        execute(setup)
    assert not (path/"out").exists()


def test_partial_means_are_separate_from_ready_features(setup):
    path, _, _ = setup
    with nc.Dataset(path/"daily/daily.nc", "a") as ds:
        ds["tp"][0] = np.nan
    execute(setup)
    summary = pd.read_csv(path/"out/grid_predictors.csv")
    assert summary.tp_observed_mean.eq(2).all()
    assert summary.tp_event_mean.isna().all()
    assert summary.tp_n_valid_years.eq(1).all()


def test_joint_year_gate_catches_disjoint_variable_coverage(setup):
    path, _, config = setup
    config["min_valid_years"] = 1
    with nc.Dataset(path/"daily/daily.nc", "a") as ds:
        ds["tp"][0] = np.nan
        ds["tcc"][1] = np.nan
    execute(setup)
    summary = pd.read_csv(path/"out/grid_predictors.csv")
    assert summary.tp_event_mean.notna().all() and summary.tcc_event_mean.notna().all()
    assert summary.n_joint_valid_years.eq(0).all()
    assert not summary.ready_for_spatial_cv.any()


def test_help_from_unrelated_directory_and_no_nn_import(tmp_path):
    script = ROOT/"src/taiwan_simulation/match_annual_events.py"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", str(script), "--help"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--daily-root" in result.stdout
    code = (
        "import importlib.util, sys;"
        f"s=importlib.util.spec_from_file_location('event_matcher', {str(script)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "assert 'torch' not in sys.modules;"
        "assert not any('cheng' in x for x in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())
