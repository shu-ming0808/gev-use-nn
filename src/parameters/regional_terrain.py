"""Optional coarse-grid terrain features in local physical east/north metres.

These features do not replace EAS's existing TPI, relief or coastline metric.
ERA5 0.25-degree geopotential represents coarse terrain, not a high-resolution
DEM. Slopes use WGS84 geodesic displacements at each cell, avoiding a fixed
Taiwan projection and the distinction between projected grid north and true
north. Ruggedness retains the original Taiwan implementation's RMS definition.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod


TERRAIN_SCHEMAS = ("legacy_v1", "expanded_v2")
EXTRA_TERRAIN_COLUMNS = (
    "terrain_025_slope_deg",
    "terrain_025_northness",
    "terrain_025_eastness",
    "terrain_025_ruggedness_m",
)
EXTRA_TERRAIN_DEFINITIONS = {
    "source": "ERA5 0.25-degree geopotential / 9.80665 in metres; coarse terrain, not a high-resolution DEM",
    "slope": "Central finite differences solved in local WGS84 east/north metres; degrees above horizontal",
    "aspect": "Steepest descent clockwise from true north; northness=cos(aspect), eastness=sin(aspect); flat cells use (0,0)",
    "ruggedness": "Root-mean-square elevation difference to all eight adjacent source cells, in metres; not root-sum-of-squares",
    "halo": "Every target requires eight finite adjacent source cells; no padding, imputation or silent GRID removal",
    "unchanged": "Existing EAS TPI, relief and projected coastline distances retain their original definitions",
}


def validate_terrain_schema(schema: str) -> str:
    if schema not in TERRAIN_SCHEMAS:
        raise ValueError(f"Unknown terrain schema {schema!r}; choose one of {TERRAIN_SCHEMAS}.")
    return schema


def extra_candidate_groups(schema: str) -> dict[str, list[str]]:
    """The aspect sine/cosine pair is selected together, never separately."""
    validate_terrain_schema(schema)
    if schema == "legacy_v1":
        return {}
    return {
        "terrain_slope": [EXTRA_TERRAIN_COLUMNS[0]],
        "terrain_aspect": list(EXTRA_TERRAIN_COLUMNS[1:3]),
        "terrain_ruggedness": [EXTRA_TERRAIN_COLUMNS[3]],
    }


def _axis(values, name, *, latitude=False):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite one-dimensional coordinate with at least three cells.")
    if latitude and np.any(np.abs(values) > 90):
        raise ValueError("Latitude outside [-90,90].")
    ordered = values if latitude else np.degrees(np.unwrap(np.radians(values)))
    diff = np.diff(ordered)
    if not (np.all(diff > 0) or np.all(diff < 0)):
        raise ValueError(f"{name} must be strictly monotone after longitude unwrapping.")
    if not latitude and (np.ptp(ordered) >= 360 or np.any(np.abs(diff) >= 180)):
        raise ValueError("Longitude coordinate contains duplicates or ambiguous steps.")
    return values


def extra_terrain_from_grid(elevation_m, latitudes, longitudes, row_indices, column_indices):
    """Return four feature arrays for paired targets with an exact 1-cell halo.

    Ascending/descending source axes and wrapped longitudes are supported.
    Points at a pole, with degenerate geometry, or with missing neighbours fail
    explicitly. No neighbouring value is replaced by the target elevation.
    """
    dem = np.asarray(elevation_m, dtype=float)
    lat = _axis(latitudes, "Latitude", latitude=True)
    lon = _axis(longitudes, "Longitude")
    if dem.shape != (len(lat), len(lon)):
        raise ValueError("Elevation shape must match latitude-by-longitude coordinates.")
    raw_row, raw_col = np.asarray(row_indices), np.asarray(column_indices)
    if raw_row.ndim != 1 or raw_col.shape != raw_row.shape or not len(raw_row):
        raise ValueError("Paired, nonempty one-dimensional target indices are required.")
    if (not np.issubdtype(raw_row.dtype, np.integer)
            or not np.issubdtype(raw_col.dtype, np.integer)):
        raise ValueError("Target indices must be integers.")
    row, col = raw_row.astype(int), raw_col.astype(int)
    if np.any((row < 1) | (row >= len(lat)-1) | (col < 1) | (col >= len(lon)-1)):
        raise ValueError("Targets require a complete one-cell source halo.")
    center = dem[row, col]
    neighbours = np.stack([dem[row+dr, col+dc]
        for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)])
    if not np.isfinite(center).all() or not np.isfinite(neighbours).all():
        raise ValueError("Terrain halo contains nonfinite elevations.")

    geod = Geod(ellps="WGS84")
    target_lon, target_lat = lon[col], lat[row]

    def displacement(dr, dc):
        coordinates = (target_lon, target_lat, lon[col+dc], lat[row+dr])
        if len(row) == 1:
            # pyproj's one-point path expects scalars, not length-one arrays.
            # Retain the vector return contract without deprecated coercion.
            azimuth, _, distance = geod.inv(*(float(value[0]) for value in coordinates))
            azimuth, distance = np.atleast_1d(azimuth), np.atleast_1d(distance)
        else:
            azimuth, _, distance = geod.inv(*coordinates)
        radians = np.radians(azimuth)
        return distance*np.sin(radians), distance*np.cos(radians)

    east_plus, north_plus = displacement(0, 1)
    east_minus, north_minus = displacement(0, -1)
    dx_dc, dy_dc = (east_plus-east_minus)/2, (north_plus-north_minus)/2
    east_plus, north_plus = displacement(1, 0)
    east_minus, north_minus = displacement(-1, 0)
    dx_dr, dy_dr = (east_plus-east_minus)/2, (north_plus-north_minus)/2
    determinant = dx_dc*dy_dr-dy_dc*dx_dr
    geometry_scale = np.hypot(dx_dc, dy_dc)*np.hypot(dx_dr, dy_dr)
    if (not np.isfinite(determinant).all() or np.any(geometry_scale <= 0)
            or np.any(np.abs(determinant) <= 1e-10*geometry_scale)):
        raise ValueError("Degenerate terrain geometry, including a pole.")
    dz_dc = (dem[row, col+1]-dem[row, col-1])/2
    dz_dr = (dem[row+1, col]-dem[row-1, col])/2
    dz_deast = (dz_dc*dy_dr-dy_dc*dz_dr)/determinant
    dz_dnorth = (dx_dc*dz_dr-dz_dc*dx_dr)/determinant
    gradient = np.hypot(dz_deast, dz_dnorth)
    nonflat = gradient >= 1e-12
    northness = np.divide(-dz_dnorth, gradient, out=np.zeros_like(gradient), where=nonflat)
    eastness = np.divide(-dz_deast, gradient, out=np.zeros_like(gradient), where=nonflat)
    values = (
        np.degrees(np.arctan(gradient)),
        northness,
        eastness,
        np.sqrt(np.mean((neighbours-center)**2, axis=0)),
    )
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("Nonfinite derived terrain features.")
    return dict(zip(EXTRA_TERRAIN_COLUMNS, values))


def read_extra_terrain(path: str | Path, grids: pd.DataFrame) -> pd.DataFrame:
    """Read the exact ERA5 source halo and preserve GRID identities and order.

    Longitude halo wraps only for a proven complete, regular global axis.
    Latitude cannot wrap, so a pole/edge requires an explicit alternative
    definition and raises rather than silently producing a partial feature.
    """
    from taiwan_simulation import match_annual_events as io

    required = {"grid_id", "latitude", "longitude_180"}
    if not required.issubset(grids) or grids.grid_id.isna().any() or grids.grid_id.duplicated().any() or grids.empty:
        raise ValueError("Terrain registry needs unique GRID IDs and latitude/longitude_180.")
    with io.open_nc(Path(path)) as ds:
        if getattr(ds["z"], "units", "") not in ("m**2 s**-2", "m2 s-2", "m^2 s^-2"):
            raise ValueError("Unexpected geopotential units.")
        lat_dim, lat = io.coordinate(ds, ("latitude", "lat"))
        lon_dim, lon = io.coordinate(ds, ("longitude", "lon"))
        idx = io.spatial_indices(ds, grids)
        rows, cols = idx[lat_dim], idx[lon_dim]
        if np.any((rows < 1) | (rows >= len(lat)-1)):
            raise ValueError("Targets require a complete latitude halo; poles are unsupported.")
        rr = np.arange(rows.min()-1, rows.max()+2)
        cc = np.arange(cols.min()-1, cols.max()+2)
        if cc[0] < 0 or cc[-1] >= len(lon):
            unwrapped = np.degrees(np.unwrap(np.radians(lon)))
            delta = np.diff(unwrapped)
            if (len(delta) == 0 or not np.allclose(delta, delta[0], rtol=0, atol=1e-8)
                    or not np.isclose(abs(delta[0])*len(lon), 360, rtol=0, atol=1e-6)):
                raise ValueError("Longitude halo requires a complete regular global axis.")
            # A target set spanning the whole axis would repeat the seam in
            # a rectangle. Use at most three rectangular reads, preserving
            # source order without one disk read for every GRID.
            if len(cc) > len(lon):
                features = {key: np.empty(len(grids), dtype=float) for key in EXTRA_TERRAIN_COLUMNS}
                partitions = (cols == 0, cols == len(lon)-1, (cols > 0) & (cols < len(lon)-1))
                for selected in partitions:
                    if not selected.any():
                        continue
                    r = np.arange(rows[selected].min()-1, rows[selected].max()+2)
                    c = np.arange(cols[selected].min()-1, cols[selected].max()+2)
                    source_c = c % len(lon)
                    dem = io.gather(ds["z"], {lat_dim: np.repeat(r, len(c)), lon_dim: np.tile(source_c, len(r))},
                                    len(r)*len(c)).reshape(len(r), len(c))/9.80665
                    block = extra_terrain_from_grid(dem, lat[r], lon[source_c], rows[selected]-r[0], cols[selected]-c[0])
                    for key in features:
                        features[key][selected] = block[key]
                return pd.DataFrame({"grid_id": grids.grid_id.to_numpy(), **features})
        source_cc = cc % len(lon)
        dem = io.gather(ds["z"], {lat_dim: np.repeat(rr, len(cc)), lon_dim: np.tile(source_cc, len(rr))},
                        len(rr)*len(cc)).reshape(len(rr), len(cc))/9.80665
        values = extra_terrain_from_grid(dem, lat[rr], lon[source_cc], rows-rr[0], cols-cc[0])
    return pd.DataFrame({"grid_id": grids.grid_id.to_numpy(), **values})
