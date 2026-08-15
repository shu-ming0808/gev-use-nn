import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.transform import from_origin
from rasterio.windows import from_bounds


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atmospheric_predictors import (
    _interpolate_ag_land_field,
    _open_xarray,
    _regular_interpolate,
    atmospheric_download_coverage,
    build_atmospheric_predictors,
    download_atmospheric_data,
    extract_downloads,
)
import atmospheric_predictors
from land_cover_predictors import _integer_window
from spatial_coordinates import main_island_grid_mask


def test_atmospheric_archives_extract_to_short_date_filenames(tmp_path: Path):
    archive = tmp_path / "agera5_cloud_frequency_1981_1989_m01.zip"
    long_member = (
        "Cloud-Cover_Mean-24h_C3S-glob-agric_AgERA5_19810101_"
        "final-v2.0.0.area-subset.26.0.123.0.21.5.118.0.nc"
    )
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(long_member, b"netcdf fixture")

    extract_downloads(tmp_path)

    destination = tmp_path / archive.stem
    assert (destination / "19810101.nc").read_bytes() == b"netcdf fixture"
    assert not (destination / long_member).exists()


def test_open_xarray_stages_non_ascii_path_after_backend_error(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "資料" / "19810101.nc"
    source.parent.mkdir()
    source.write_bytes(b"netcdf fixture")
    calls = []

    def fake_open_dataset(path):
        candidate = Path(path)
        calls.append(candidate)
        if candidate == source:
            raise OSError(22, "Invalid argument", str(candidate))
        assert candidate.name == source.name
        assert candidate.read_bytes() == source.read_bytes()
        return xr.Dataset({"value": ("x", [1.0])})

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    with _open_xarray(source) as dataset:
        assert dataset["value"].item() == 1.0

    assert calls[0] == source
    assert calls[1] != source
    assert not calls[1].exists()


def test_land_cover_window_is_exactly_18_by_18_pixels():
    transform = from_origin(90.0, 30.0, 1 / 360, 1 / 360)
    bounds = (120.975, 23.975, 121.025, 24.025)
    window = _integer_window(from_bounds(*bounds, transform=transform))

    assert int(window.width) == 18
    assert int(window.height) == 18


def test_main_island_mask_keeps_largest_connected_grid_component():
    grid = pd.DataFrame(
        {
            "lon": [121.00, 121.05, 121.00, 121.05, 119.50, 122.10],
            "lat": [23.00, 23.00, 23.05, 23.05, 23.50, 25.65],
        }
    )

    mask = main_island_grid_mask(grid)

    assert mask.tolist() == [True, True, True, True, False, False]


def test_atmospheric_interpolation_uses_cell_centres_without_extrapolation():
    latitude = np.array([23.0, 24.0])
    longitude = np.array([120.0, 121.0])
    field = latitude[:, None] + 2 * longitude[None, :]

    interpolated = _regular_interpolate(
        field,
        latitude,
        longitude,
        target_latitude=np.array([23.5]),
        target_longitude=np.array([120.5]),
    )

    assert np.allclose(interpolated, [23.5 + 2 * 120.5])


def test_ag_land_interpolation_uses_nearby_valid_cell_at_coast():
    values, fallback, distances = _interpolate_ag_land_field(
        field=np.array([[1.0, np.nan], [3.0, 4.0]]),
        latitude=np.array([23.0, 23.1]),
        longitude=np.array([120.0, 120.1]),
        target_latitude=np.array([23.0]),
        target_longitude=np.array([120.0]),
        maximum_fallback_distance_km=10.0,
    )

    assert np.allclose(values, [1.0])
    assert fallback.tolist() == [True]
    assert np.allclose(distances, [0.0])


def test_atmospheric_download_batches_nine_years_and_skips_1980(
    tmp_path: Path, monkeypatch
):
    calls = []

    class FakeClient:
        def retrieve(self, dataset, request, target):
            calls.append((dataset, request, Path(target).name))
            with zipfile.ZipFile(target, "w"):
                pass

    monkeypatch.setattr(
        atmospheric_predictors,
        "_cds_client",
        lambda: FakeClient(),
    )
    for label in ("wind_speed", "solar_radiation", "cloud_frequency"):
        with zipfile.ZipFile(tmp_path / f"agera5_{label}_1980.zip", "w"):
            pass

    download_atmospheric_data(
        start_year=1980,
        end_year=2024,
        output_directory=tmp_path,
        batch_years=9,
    )

    # A nine-year all-month request exceeds the CDS cost limit.  Keep years
    # grouped in batches of nine, but submit one calendar month per request.
    assert len(calls) == 15 * 12
    for label in ("wind_speed", "solar_radiation", "cloud_frequency"):
        label_calls = [call for call in calls if label in call[2]]
        assert [call[1]["year"] for call in label_calls[::12]] == [
            [str(year) for year in range(1981, 1990)],
            [str(year) for year in range(1990, 1999)],
            [str(year) for year in range(1999, 2008)],
            [str(year) for year in range(2008, 2017)],
            [str(year) for year in range(2017, 2025)],
        ]
        assert [call[1]["month"] for call in label_calls[:12]] == [
            [f"{month:02d}"] for month in range(1, 13)
        ]
        assert all("_m" in call[2] for call in label_calls)

    coverage = atmospheric_download_coverage(tmp_path, 1980, 2024)
    assert coverage["complete"].all()
    assert (coverage["available_year_months"] == 45 * 12).all()


def test_atmospheric_processor_keeps_definitions_and_units(tmp_path: Path):
    latitude = [23.0, 24.0]
    longitude = [120.0, 121.0]
    time = pd.date_range("2000-01-01", periods=2)
    sources = {
        "agera5_wind_speed_2000": (
            "Wind_Speed_10m_Mean_24h",
            [1.0, 2.0],
        ),
        "agera5_solar_radiation_2000": (
            "Solar_Radiation_Flux",
            [5_000_000.0, 10_000_000.0],
        ),
        "agera5_cloud_frequency_2000": (
            "Cloud_Cover_Mean_24h",
            [0.10, 0.25],
        ),
    }
    for directory_name, (variable, daily_values) in sources.items():
        directory = tmp_path / directory_name
        directory.mkdir()
        values = np.stack(
            [np.full((2, 2), value) for value in daily_values]
        )
        dataset = xr.Dataset(
            {
                variable: (
                    ("time", "latitude", "longitude"),
                    values,
                )
            },
            coords={
                "time": time,
                "latitude": latitude,
                "longitude": longitude,
            },
        )
        dataset.to_netcdf(directory / "source.nc")

    grid_path = tmp_path / "grid.csv"
    pd.DataFrame(
        {"station": ["G"], "lon": [120.5], "lat": [23.5]}
    ).to_csv(grid_path, index=False)
    event_path = tmp_path / "events.csv"
    pd.DataFrame(
        {
            "station": ["G", "G"],
            "year": [1979, 2000],
            "month": [1, 1],
            "monthly_max_tmax_c": [34.0, 35.0],
            "max_date": ["1979-01-02", "2000-01-02"],
        }
    ).to_csv(event_path, index=False)
    event_output_path = tmp_path / "event_atmosphere.csv"
    result, audit = build_atmospheric_predictors(
        raw_directory=tmp_path,
        grid_path=grid_path,
        event_path=event_path,
        event_output_path=event_output_path,
        output_path=tmp_path / "atmosphere.csv",
        audit_path=tmp_path / "audit.csv",
        analysis_start_year=2000,
        analysis_end_year=2000,
    )

    assert np.isclose(result.loc[0, "tmax_event_wind_mean_mps"], 2.0)
    assert np.isclose(
        result.loc[0, "tmax_event_solar_radiation_mean_mj_m2"], 10.0
    )
    assert np.isclose(
        result.loc[0, "tmax_event_agera5_cloud_cover_mean_fraction"],
        0.25,
    )
    events = pd.read_csv(event_output_path)
    assert events.loc[0, "max_date"] == "2000-01-02"
    assert np.isclose(events.loc[0, "wind_speed_on_tmax_date_mps"], 2.0)
    assert (audit["extrapolation_count"] == 0).all()
    assert (audit["matched_event_count"] == 1).all()
