"""Analytic terrain checks, coordinate orientation and source-read contracts."""
import hashlib

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest
from pyproj import Geod

from parameters import regional_terrain as terrain


def local_plane(lat, lon, east_slope=.003, north_slope=-.004):
    longitude, latitude = np.meshgrid(lon, lat)
    center_lon, center_lat = lon[len(lon)//2], lat[len(lat)//2]
    azimuth, _, distance = Geod(ellps="WGS84").inv(
        np.full(longitude.shape, center_lon), np.full(latitude.shape, center_lat), longitude, latitude)
    east = distance*np.sin(np.radians(azimuth))
    north = distance*np.cos(np.radians(azimuth))
    return 800.+east_slope*east+north_slope*north


@pytest.mark.parametrize("latitude", [12., 35., 60.])
def test_analytic_plane_recovers_physical_slope_and_compass_aspect(latitude):
    lat = latitude+np.arange(-2, 3)*.25
    lon = 130.+np.arange(-2, 3)*.25
    dem = local_plane(lat, lon)
    result = terrain.extra_terrain_from_grid(dem, lat, lon, np.array([2]), np.array([2]))
    assert result["terrain_025_slope_deg"][0] == pytest.approx(np.degrees(np.arctan(.005)), abs=1e-12)
    assert result["terrain_025_northness"][0] == pytest.approx(.8, abs=1e-12)
    assert result["terrain_025_eastness"][0] == pytest.approx(-.6, abs=1e-12)


@pytest.mark.parametrize("reverse_lat,reverse_lon", [(True, False), (False, True), (True, True)])
def test_reversing_source_axes_preserves_features(reverse_lat, reverse_lon):
    lat = 35.+np.arange(-2, 3)*.25
    lon = 100.+np.arange(-2, 3)*.25
    dem = local_plane(lat, lon)
    expected = terrain.extra_terrain_from_grid(dem, lat, lon, np.array([2]), np.array([2]))
    if reverse_lat:
        lat, dem = lat[::-1], dem[::-1]
    if reverse_lon:
        lon, dem = lon[::-1], dem[:, ::-1]
    actual = terrain.extra_terrain_from_grid(dem, lat, lon, np.array([2]), np.array([2]))
    for column in expected:
        np.testing.assert_allclose(actual[column], expected[column], atol=1e-12, rtol=1e-12)


def test_wrapped_longitude_axis_preserves_true_compass_directions():
    lat = 35.+np.arange(-2, 3)*.25
    lon = np.array([179.5, 179.75, -180., -179.75, -179.5])
    result = terrain.extra_terrain_from_grid(local_plane(lat, lon), lat, lon, np.array([2]), np.array([2]))
    assert result["terrain_025_northness"][0] == pytest.approx(.8, abs=1e-12)
    assert result["terrain_025_eastness"][0] == pytest.approx(-.6, abs=1e-12)


def test_flat_terrain_has_zero_slope_aspect_and_ruggedness():
    result = terrain.extra_terrain_from_grid(np.full((5, 5), 200.),
        np.arange(30., 35.), np.arange(120., 125.), np.array([1, 2, 3]), np.array([3, 2, 1]))
    for column in terrain.EXTRA_TERRAIN_COLUMNS:
        np.testing.assert_array_equal(result[column], np.zeros(3))


def test_ruggedness_matches_original_rms_not_root_sum_of_squares():
    dem = np.full((5, 5), 110.)
    dem[2, 2] = 100.
    result = terrain.extra_terrain_from_grid(dem, np.arange(30., 35.), np.arange(120., 125.),
                                           np.array([2]), np.array([2]))
    assert result["terrain_025_ruggedness_m"][0] == 10.
    assert result["terrain_025_slope_deg"][0] == 0.


@pytest.mark.parametrize("position", [(2, 2), (1, 1), (2, 3)])
def test_missing_target_or_any_eight_neighbours_fails(position):
    dem = np.ones((5, 5))
    dem[position] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        terrain.extra_terrain_from_grid(dem, np.arange(30., 35.), np.arange(120., 125.),
                                       np.array([2]), np.array([2]))


def test_target_at_source_edge_is_not_padded():
    with pytest.raises(ValueError, match="one-cell"):
        terrain.extra_terrain_from_grid(np.ones((5, 5)), np.arange(30., 35.), np.arange(120., 125.),
                                       np.array([0]), np.array([2]))


@pytest.mark.parametrize("lat,lon", [([30., 31., 31.], [120., 121., 122.]),
    ([30., 31., 32.], [120., 121., 120.]), ([89., 90., 91.], [120., 121., 122.])])
def test_invalid_axes_fail(lat, lon):
    with pytest.raises(ValueError):
        terrain.extra_terrain_from_grid(np.ones((3, 3)), lat, lon, np.array([1]), np.array([1]))


def test_schema_is_explicit_and_aspect_is_a_joint_group():
    assert terrain.extra_candidate_groups("legacy_v1") == {}
    groups = terrain.extra_candidate_groups("expanded_v2")
    assert len(groups) == 3
    assert groups["terrain_aspect"] == ["terrain_025_northness", "terrain_025_eastness"]
    assert sum(len(columns) for columns in groups.values()) == 4
    with pytest.raises(ValueError, match="Unknown terrain schema"):
        terrain.extra_candidate_groups("typo")


def write_geopotential(path, lat, lon, dem, *, units="m**2 s**-2", aliases=False):
    lat_name, lon_name = ("lat", "lon") if aliases else ("latitude", "longitude")
    with nc.Dataset(path, "w") as ds:
        ds.createDimension(lat_name, len(lat)); ds.createDimension(lon_name, len(lon))
        ds.createVariable(lat_name, "f8", (lat_name,))[:] = lat
        ds.createVariable(lon_name, "f8", (lon_name,))[:] = lon
        z = ds.createVariable("z", "f8", (lat_name, lon_name))
        z.units = units
        z[:] = dem*9.80665


@pytest.mark.parametrize("aliases", [False, True])
def test_netcdf_reader_preserves_grid_order_and_file(tmp_path, aliases):
    lat, lon = np.arange(34.5, 35.51, .25), np.arange(129.5, 130.51, .25)
    dem = local_plane(lat, lon)
    path = tmp_path/"terrain.nc"
    write_geopotential(path, lat[::-1], lon, dem[::-1], aliases=aliases)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    rows, cols = np.array([3, 1, 2]), np.array([1, 3, 2])
    grids = pd.DataFrame({"grid_id": ["Z", "A", "B"], "latitude": lat[rows], "longitude_180": lon[cols]})
    actual = terrain.read_extra_terrain(path, grids)
    assert actual.grid_id.tolist() == ["Z", "A", "B"]
    assert actual.columns.tolist() == ["grid_id", *terrain.EXTRA_TERRAIN_COLUMNS]
    expected = terrain.extra_terrain_from_grid(dem, lat, lon, rows, cols)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, rtol=1e-12)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_netcdf_reader_handles_global_longitude_seam(tmp_path):
    lat, lon = np.array([34., 35., 36.]), np.arange(360.)
    dem = np.tile(np.cos(np.radians(lon))*20.+200., (3, 1))
    path = tmp_path/"terrain.nc"
    write_geopotential(path, lat, lon, dem)
    grids = pd.DataFrame({"grid_id": ["zero", "last"], "latitude": [35., 35.], "longitude_180": [0., -1.]})
    actual = terrain.read_extra_terrain(path, grids)
    assert actual.grid_id.tolist() == ["zero", "last"]
    assert np.isfinite(actual[list(terrain.EXTRA_TERRAIN_COLUMNS)].to_numpy()).all()


def test_netcdf_reader_rejects_units_and_missing_exact_grid(tmp_path):
    lat, lon = np.arange(30., 35.), np.arange(120., 125.)
    path = tmp_path/"terrain.nc"
    write_geopotential(path, lat, lon, np.ones((5, 5)), units="m")
    grids = pd.DataFrame({"grid_id": ["a"], "latitude": [32.], "longitude_180": [122.]})
    with pytest.raises(ValueError, match="units"):
        terrain.read_extra_terrain(path, grids)
    write_geopotential(path, lat, lon, np.ones((5, 5)))
    grids.loc[0, "longitude_180"] = 122.1
    with pytest.raises(ValueError, match="unique exact GRID"):
        terrain.read_extra_terrain(path, grids)
