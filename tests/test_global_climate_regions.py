"""Offline regression tests; official-source integration uses the local cache."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box, Polygon, MultiPolygon
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import global_climate_regions as g


@pytest.fixture
def regions():
    return [g.Region(1, "A", "West", "Land", box(-10, -10, 0, 10)),
            g.Region(2, "B", "East", "Land-Ocean", box(0, -10, 10, 10))]


@pytest.mark.parametrize("longitude,expected", [(-180, -180), (180, -180), (360, 0), (270, -90), (121, 121)])
def test_longitude_conventions(longitude, expected):
    assert g.normalize_longitudes(longitude) == expected


@pytest.mark.parametrize("lon,lat", [(361, 0), (-181, 0), (np.nan, 0), (0, np.inf), (0, 91)])
def test_reject_invalid_points(regions, lon, lat):
    with pytest.raises(ValueError):
        g.assign_regions([lon], [lat], regions)


def test_shared_edges_resolve_once_and_do_not_depend_on_catalogue_order(regions):
    assigned, matches = g.assign_regions([-5, 0, 5, 50], [0, 0, 0, 0], regions, chunk_size=2)
    np.testing.assert_array_equal(assigned, [1, 1, 2, -1])
    np.testing.assert_array_equal(matches, [1, 2, 1, 0])
    np.testing.assert_array_equal(g.assign_regions([-5, 0, 5, 50], [0]*4, regions[::-1])[0], assigned)


def test_numerical_edge_tolerance_does_not_bridge_real_gaps():
    region = g.Region(1, "A", "A", "Land", box(0, 0, 1 - 1e-13, 1))
    ids, _ = g.assign_regions([1, 1 + 1e-7], [0.5, 0.5], [region])
    np.testing.assert_array_equal(ids, [1, -1])


def test_date_line_tests_both_sides_without_double_counting():
    regions = [g.Region(3, "DAT", "Date line", "Land", MultiPolygon([
        box(170, -10, 180, 10), box(-180, -10, -170, 10)])),
        g.Region(2, "EDGE", "East edge only", "Land", box(175, -5, 180, 5))]
    ids, counts = g.assign_regions([180, -180, 179, 181], [0]*4, regions)
    np.testing.assert_array_equal(ids, [2, 2, 2, 3])
    np.testing.assert_array_equal(counts, [2, 2, 2, 1])


@pytest.fixture(scope="module")
def canonical_grid():
    # One complete synthetic land polygon makes mask tests independent of downloads.
    region = g.Region(1, "ALL", "All", "Land", box(-180, -90, 180, 90))
    lat = np.linspace(90, -90, 721)
    lon = np.arange(1440) / 4
    values = np.ones((721, 1440), dtype=np.float32)
    values[360, :3] = [0.49, 0.5, 0.51]
    mask = xr.DataArray(values, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon})
    return g.build_grid([region], mask), g.catalogue([region])


def test_lsm_threshold_is_strict_and_poles_are_not_duplicate_cv_points(canonical_grid):
    grid, _ = canonical_grid
    np.testing.assert_array_equal(grid.land_mask.values[360, :3], [0, 0, 1])
    assert grid.spatial_cv_eligible.values[0].sum() == 1
    assert grid.spatial_cv_eligible.values[-1].sum() == 1


def test_missing_mask_is_unknown_not_false():
    grid = g.build_grid([g.Region(1, "ALL", "All", "Land", box(-180, -90, 180, 90))], None)
    assert (grid.land_mask == -1).all()
    assert (grid.analysis_land_mask == -1).all()
    assert (grid.spatial_cv_eligible == -1).all()


def test_land_outside_land_capable_polygon_is_flagged_not_forced(canonical_grid):
    grid, _ = canonical_grid
    mask = grid.lsm.copy()
    ocean = g.Region(47, "O", "Ocean", "Ocean", box(-180, -90, 180, 90))
    result = g.build_grid([ocean], mask)
    assert (result.analysis_land_mask == 0).all()
    assert (result.land_mask == 1).any()


def test_join_preserves_repeated_years_values_order_and_index(canonical_grid):
    grid, catalogue = canonical_grid
    original = pd.DataFrame({"lon": [121, -180, 121], "lat": [24, 70, 24],
                             "year": [2001, 2000, 2000], "temperature": [31.2, 15.8, 32.1]}, index=[9, 3, 9])
    result = g.attach_region_labels(original, grid, catalogue, lon_column="lon", lat_column="lat")
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert result.grid_id_025.iloc[0] == result.grid_id_025.iloc[2]
    assert (result.ar6_region == "ALL").all()


def test_join_rejects_off_grid_and_label_overwrite(canonical_grid):
    grid, catalogue = canonical_grid
    with pytest.raises(ValueError, match="0.25"):
        g.attach_region_labels(pd.DataFrame({"longitude": [121.01], "latitude": [24]}), grid, catalogue)
    with pytest.raises(ValueError, match="overwrite"):
        g.attach_region_labels(pd.DataFrame({"longitude": [121], "latitude": [24], "ar6_region": ["old"]}), grid, catalogue)


def test_geodesic_km_and_date_line():
    assert g.geodesic_distance_km(0, 0, 1, 0) == pytest.approx(111.31949, rel=1e-5)
    assert g.geodesic_distance_km(179, 0, -179, 0) == pytest.approx(222.63898, rel=1e-5)
    assert g.geodesic_distance_km(0, 60, 1, 60) < g.geodesic_distance_km(0, 0, 1, 0)


def test_netcdf_mask_axes_reordered_and_unicode_path(tmp_path):
    path = tmp_path / "遮罩.nc"
    ds = xr.Dataset({"lsm": (("valid_time", "longitude", "latitude"),
                            np.array([[[0.2, 0.4], [0.6, 0.8]]]))},
                    coords={"valid_time": [0], "longitude": [-0.25, 0], "latitude": [-90, 90]})
    ds.to_netcdf(path, engine="scipy")
    mask = g.read_land_sea_mask(path)
    np.testing.assert_array_equal(mask.latitude, [90, -90])
    np.testing.assert_array_equal(mask.longitude, [0, 359.75])
    np.testing.assert_allclose(mask.values, [[0.8, 0.4], [0.6, 0.2]])


def test_mask_rejects_multiple_times(tmp_path):
    path = tmp_path / "multiple.nc"
    xr.Dataset({"lsm": (("time", "latitude", "longitude"), np.ones((2, 1, 1)))},
               coords={"latitude": [0], "longitude": [0]}).to_netcdf(path, engine="scipy")
    with pytest.raises(ValueError, match="size 1"):
        g.read_land_sea_mask(path)


def test_build_grid_rejects_wrong_resolution():
    mask = xr.DataArray(np.ones((2, 2)), dims=("latitude", "longitude"),
                        coords={"latitude": [0, 1], "longitude": [0, 1]})
    with pytest.raises(ValueError, match="no interpolation"):
        g.build_grid([g.Region(1, "A", "A", "Land", box(-180, -90, 180, 90))], mask)


def test_verified_cache_works_offline(monkeypatch, tmp_path):
    path = tmp_path / "boundaries.geojson"
    path.write_bytes(b"test fixture")
    monkeypatch.setattr(g, "SOURCE_SHA256", g.sha256(path))
    monkeypatch.setattr(g, "urlopen", lambda *a, **k: pytest.fail("Should use verified cache"))
    assert g.ensure_boundaries(path) == path


def test_wrong_source_checksum_preserves_existing_file(monkeypatch, tmp_path):
    from io import BytesIO
    path = tmp_path / "boundaries.geojson"
    path.write_bytes(b"preserve existing")
    monkeypatch.setattr(g, "urlopen", lambda *a, **k: BytesIO(b"corrupt response"))
    with pytest.raises(ValueError, match="checksum"):
        g.ensure_boundaries(path)
    assert path.read_bytes() == b"preserve existing"


def test_stream_apply_and_failure_do_not_replace_data(canonical_grid, tmp_path):
    grid, catalogue = canonical_grid
    grid.to_netcdf(tmp_path / "era5_ar6_grid.nc", engine="scipy")
    catalogue.to_csv(tmp_path / "region_catalogue.csv", index=False)
    manifest = {"status": "complete", "output_sha256": {
        name: g.sha256(tmp_path / name) for name in ["era5_ar6_grid.nc", "region_catalogue.csv"]}}
    g.write_json(tmp_path / "region_manifest.json", manifest)
    input_path = tmp_path / "annual.csv"
    original = pd.DataFrame({"longitude": [0, 0.5, 1], "latitude": [0, 0, 0], "year": [2000]*3,
                             "annual_maximum": [21, 22, 23]})
    original.to_csv(input_path, index=False)
    output_path = tmp_path / "labelled.csv"
    result = g.apply_csv(input_path, output_path, tmp_path, land_only=True, chunksize=1)
    assert result == {"input_rows": 3, "output_rows": 2}
    assert pd.read_csv(output_path).annual_maximum.tolist() == [22, 23]
    before = output_path.read_bytes()
    with pytest.raises(ValueError, match="never overwritten"):
        g.apply_csv(input_path, output_path, tmp_path)
    assert output_path.read_bytes() == before
    original.loc[2, "longitude"] = 0.13
    original.to_csv(input_path, index=False)
    with pytest.raises(ValueError, match="GRID centres"):
        g.apply_csv(input_path, tmp_path / "bad.csv", tmp_path, chunksize=1)
    assert not (tmp_path / "bad.csv").exists()


@pytest.mark.skipif(not g.DEFAULT_BOUNDARIES.exists(), reason="Official integration cache not downloaded")
def test_official_geometry_counts_known_locations_and_dateline():
    regions = g.load_regions(g.DEFAULT_BOUNDARIES)
    assert sum("Land" in r.kind for r in regions) == 46
    assert sum("Ocean" in r.kind for r in regions) == 15
    labels = {r.id: r.acronym for r in regions}
    ids, _ = g.assign_regions([121, -74, 0, 180, -180, 0, 143.5], [24, 40.75, 51.5, 70, 70, -90, -2.5], regions)
    assert [labels[i] for i in ids[:6]] == ["EAS", "ENA", "NEU", "RAR", "RAR", "EAN"]
    assert ids[-1] >= 0  # a formerly floating-point edge gap in the official densified source
