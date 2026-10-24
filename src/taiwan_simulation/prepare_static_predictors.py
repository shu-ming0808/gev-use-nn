"""Local static covariates for saved Taiwan ERA5 GRID, never a global projection."""
from __future__ import annotations

from pathlib import Path
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from parameters.land_cover_predictors import CLASS_GROUPS
from parameters.terrain_predictors import derive_terrain_predictors
from taiwan_simulation import match_annual_events as io


def terrain(path, grids):
    """ERA5 geopotential-height terrain at 0.25 degrees, not high-resolution DEM."""
    with io.open_nc(path) as ds:
        if str(getattr(ds["z"], "units", "")) not in ("m**2 s**-2", "m2 s-2", "m^2 s^-2"):
            raise ValueError("Unexpected geopotential units")
        lat = io.numeric(ds["latitude"][:])
        lon = io.numeric(ds["longitude"][:])
        # Two source-cell halo for the full 5x5 TPI/relief neighbourhood.
        rr = np.flatnonzero((lat >= grids.latitude.min()-.5-1e-6) & (lat <= grids.latitude.max()+.5+1e-6))
        cc = np.flatnonzero((lon >= grids.longitude.min()-.5-1e-6) & (lon <= grids.longitude.max()+.5+1e-6))
        if len(rr) < 5 or len(cc) < 5 or not np.allclose(abs(np.diff(lat[rr])), .25) or not np.allclose(np.diff(lon[cc]), .25):
            raise ValueError("Expected complete ERA5 0.25-degree terrain window")
        var = ds["z"]
        raw = io.gather(var, {"latitude": np.repeat(rr, len(cc)), "longitude": np.tile(cc, len(rr))}, len(rr)*len(cc))
        lon_grid, lat_grid = np.meshgrid(lon[cc], lat[rr])
        dem = pd.DataFrame({"lon": lon_grid.ravel(), "lat": lat_grid.ravel(), "elevation_m": raw/9.80665})
    result = derive_terrain_predictors(dem, neighbourhood_radius=2)
    selected = grids[["grid_id", "longitude", "latitude"]].merge(
        result, left_on=["longitude", "latitude"], right_on=["lon", "lat"], how="left", validate="one_to_one")
    names = ["slope_deg", "northness", "eastness", "local_relief_m", "tpi_m", "terrain_ruggedness_m"]
    selected = selected[["grid_id", "elevation_m", *names]]
    if not np.isfinite(selected.drop(columns="grid_id").to_numpy()).all():
        raise ValueError("Incomplete terrain extraction")
    return selected.rename(columns={name: "terrain_025_"+name for name in names})


def class_fractions(values, latitude_bounds, longitude_bounds, valid_codes):
    """Exact spherical pixel-area weights, common sphere radius cancels in ratios."""
    values = np.asarray(values)
    dy = np.abs(np.diff(np.sin(np.deg2rad(latitude_bounds)), axis=1)).ravel()
    dx = np.abs(np.diff(np.deg2rad(longitude_bounds), axis=1)).ravel()
    weight = dy[:, None]*dx[None, :]
    valid = np.isin(values, valid_codes) & (values != 0)
    total, covered = weight.sum(), weight[valid].sum()
    result = {"land_cover_valid_fraction": float(covered/total), "land_cover_pixel_count": int(valid.sum())}
    used = set()
    for name, codes in CLASS_GROUPS.items():
        result[f"{name}_ratio"] = float(weight[valid & np.isin(values, list(codes))].sum()/covered) if covered else np.nan
        used.update(codes)
    result["other_ratio"] = float(weight[valid & ~np.isin(values, list(used))].sum()/covered) if covered else np.nan
    return result


def land_cover(path, grids):
    rows = []
    with io.open_nc(path) as ds:
        if str(getattr(ds, "time_coverage_start", "")) != "20000101":
            raise ValueError("Expected fixed year-2000 CCI land cover")
        lat, lon = io.numeric(ds["lat"][:]), io.numeric(ds["lon"][:])
        var = ds["lccs_class"]
        known = np.asarray(var.flag_values)
        for grid in grids.itertuples():
            left, right = grid.longitude_180-.125, grid.longitude_180+.125
            bottom, top = grid.latitude-.125, grid.latitude+.125
            rr = np.flatnonzero((lat >= bottom) & (lat < top))
            cc = np.flatnonzero((lon >= left) & (lon < right))
            if len(rr) != 90 or len(cc) != 90:
                raise ValueError(f"CCI cell footprint must have 90x90 pixels: {grid.grid_id}")
            latb, lonb = io.numeric(ds["lat_bounds"][rr]), io.numeric(ds["lon_bounds"][cc])
            if not (np.isclose(latb.min(), bottom) and np.isclose(latb.max(), top)
                    and np.isclose(lonb.min(), left) and np.isclose(lonb.max(), right)):
                raise ValueError("CCI pixel bounds do not cover the exact ERA5 cell")
            data = io.numeric(var[0, rr, cc])
            rows.append({"grid_id": grid.grid_id, **class_fractions(data, latb, lonb, known)})
    return pd.DataFrame(rows)


def coast(path, grids):
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name.endswith("GSHHS_i_L1.shp")]
        if len(matches) != 1:
            raise ValueError("Expected one intermediate-resolution level-1 ocean coastline")
    bounds = (118., 20., 124., 28.)
    frame = gpd.read_file(f"zip://{path.as_posix()}!{matches[0]}", bbox=bounds)
    if frame.crs is None:
        raise ValueError("Coastline CRS missing")
    frame = frame.to_crs(4326)
    # Clip original boundaries, NOT boundaries of clipped polygons (which would
    # introduce artificial coastlines along the clipping rectangle).
    lines = frame.geometry.boundary.intersection(box(*bounds))
    lines = lines.loc[~lines.is_empty]
    if not len(lines):
        raise ValueError("No ocean coastline found")
    lines = gpd.GeoSeries(lines, crs=4326).to_crs(3826)
    geometry = lines.unary_union
    points = gpd.GeoSeries(gpd.points_from_xy(grids.longitude_180, grids.latitude), crs=4326).to_crs(3826)
    distances = points.distance(geometry).to_numpy()/1000
    edge = gpd.GeoSeries([box(*bounds).boundary], crs=4326).to_crs(3826).iloc[0]
    if not np.isfinite(distances).all() or np.any(distances*1000 >= points.distance(edge).to_numpy()):
        raise ValueError("Coastline window too small or invalid projected distance")
    return pd.DataFrame({"grid_id": grids.grid_id, "coast_distance_km": distances})


def prepare_static(root, grids, output):
    root, output = Path(root), Path(output)
    paths = {"lsm": root/"era5_land_sea_mask_global_025.nc",
             "geopotential": root/"era5_surface_geopotential_global_025.nc",
             "land_cover": root/"esa_cci_land_cover_2000_global_300m.nc",
             "coast": root/"gshhg-shp-2.3.7.zip"}
    sources = {key: io.signature(path) for key, path in paths.items()}
    with io.open_nc(paths["lsm"]) as ds:
        idx = io.spatial_indices(ds, grids)
        lsm = io.gather(ds["lsm"], idx, len(grids))
        if not np.allclose(lsm, grids.lsm, rtol=0, atol=1e-6):
            raise ValueError("Static LSM differs from Taiwan registry")
    fixed = terrain(paths["geopotential"], grids)
    fixed = fixed.merge(land_cover(paths["land_cover"], grids), on="grid_id", validate="one_to_one")
    fixed = fixed.merge(coast(paths["coast"], grids), on="grid_id", validate="one_to_one")
    fixed["static_complete"] = np.isfinite(fixed.drop(columns="grid_id").to_numpy()).all(axis=1) & fixed.land_cover_valid_fraction.ge(1-1e-10)
    if sources != {key: io.signature(path) for key, path in paths.items()}:
        raise RuntimeError("Static source changed during extraction")
    io.write_csv(output/"static_predictors.csv", fixed)
    io.write_json(output/"static_report.json", {"n_grids": len(grids), "complete_grids": int(fixed.static_complete.sum()), "sources": sources,
        "definitions": {"elevation": "ERA5 geopotential / 9.80665; geopotential height in m, coarse 0.25-degree field, not high-resolution DEM",
            "terrain": "Metric finite differences via EPSG:3826; TPI/relief 5x5 source-cell window (1.25 degrees); terrain_025 prefix",
            "land_cover": "Year 2000 fixed CCI map; exact 0.25-degree footprint, 90x90 pixels, spherical area weighted; water included in denominator",
            "coast": "GSHHG 2.3.7 intermediate L1 ocean boundary; EPSG:3826 km; excludes lake shores, clips original lines before projection"}})
    return fixed
