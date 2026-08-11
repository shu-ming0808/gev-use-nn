"""Derive reusable terrain predictors from the Taiwan elevation grid.

The raw elevation grid is kept at 0.01-degree resolution, but every gradient
uses projected TWD97 distances in metres. The resulting table contains:

- elevation in metres;
- slope in degrees;
- downslope aspect in degrees clockwise from north;
- northness and eastness;
- local relief (neighbourhood maximum minus minimum);
- topographic position index (centre elevation minus neighbourhood mean);
- terrain ruggedness (RMS elevation difference to eight neighbours).

The command-line entry point also joins these predictors to the TCCIP grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
from scipy.spatial import cKDTree

from project_paths import (
    PROCESSED_DATA_DIR,
    SPATIAL_PREDICTOR_PROCESSED_DIR,
    SPATIAL_PREDICTOR_RAW_DIR,
    ensure_project_directories,
)
from spatial_coordinates import (
    add_twd97_km_columns,
    project_lonlat_to_twd97_km,
)


DEFAULT_ELEVATION_PATH = (
    SPATIAL_PREDICTOR_RAW_DIR / "taiwan_grid_elevation_001deg.csv"
)
DEFAULT_TERRAIN_PATH = (
    SPATIAL_PREDICTOR_PROCESSED_DIR
    / "taiwan_terrain_predictors_001deg.csv"
)
DEFAULT_TCCIP_GRID_PATH = PROCESSED_DATA_DIR / "grid_station_location.csv"
DEFAULT_TCCIP_TERRAIN_PATH = (
    SPATIAL_PREDICTOR_PROCESSED_DIR
    / "tccip_grid_terrain_predictors.csv"
)

PREDICTOR_COLUMNS = (
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "northness",
    "eastness",
    "local_relief_m",
    "tpi_m",
    "terrain_ruggedness_m",
)


def load_elevation_grid(path: str | Path) -> pd.DataFrame:
    """Read and validate the first three columns of the elevation CSV."""
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=[0, 1, 2],
        skipinitialspace=True,
    )
    frame.columns = [
        str(column).strip().lstrip("\ufeff")
        for column in frame.columns
    ]
    normalized = {
        str(column).strip().lower().replace(" ", ""): column
        for column in frame.columns
    }
    lon_column = next(
        (
            normalized[name]
            for name in ("lon", "longitude")
            if name in normalized
        ),
        None,
    )
    lat_column = next(
        (
            normalized[name]
            for name in ("lat", "latitude")
            if name in normalized
        ),
        None,
    )
    elevation_column = next(
        (
            normalized[name]
            for name in ("height(m)", "elevation_m", "elevation", "height")
            if name in normalized
        ),
        None,
    )
    if lon_column is None or lat_column is None or elevation_column is None:
        raise ValueError(
            "Elevation CSV must contain longitude, latitude, and height."
        )

    frame = frame.rename(
        columns={
            lon_column: "lon",
            lat_column: "lat",
            elevation_column: "elevation_m",
        }
    )[["lon", "lat", "elevation_m"]]
    for column in ("lon", "lat", "elevation_m"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.isna().any().any():
        raise ValueError("Elevation CSV contains missing or non-numeric values.")
    if frame.duplicated(["lon", "lat"]).any():
        raise ValueError("Elevation CSV contains duplicate coordinates.")
    if not frame["lon"].between(118.0, 123.5).all():
        raise ValueError("Longitude values are outside the Taiwan domain.")
    if not frame["lat"].between(20.0, 27.0).all():
        raise ValueError("Latitude values are outside the Taiwan domain.")
    if (frame["elevation_m"] < 0.0).any():
        raise ValueError("Elevation must be non-negative in this dataset.")
    return frame.sort_values(["lat", "lon"]).reset_index(drop=True)


def _terrain_ruggedness(elevation: np.ndarray) -> np.ndarray:
    """Return RMS elevation difference from the eight adjacent cells."""
    padded = np.pad(
        elevation.astype(np.float64),
        pad_width=1,
        mode="constant",
        constant_values=np.nan,
    )
    squared_differences = []
    n_rows, n_columns = elevation.shape
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == 0 and column_offset == 0:
                continue
            neighbour = padded[
                1 + row_offset : 1 + row_offset + n_rows,
                1 + column_offset : 1 + column_offset + n_columns,
            ]
            squared_differences.append((neighbour - elevation) ** 2)
    return np.sqrt(np.nanmean(np.stack(squared_differences), axis=0))


def derive_terrain_predictors(
    elevation_grid: pd.DataFrame,
    neighbourhood_radius: int = 2,
) -> pd.DataFrame:
    """Derive terrain predictors using metric finite differences.

    ``neighbourhood_radius=2`` creates a 5-by-5 local-relief window for the
    current 0.01-degree source grid.
    """
    if neighbourhood_radius < 1:
        raise ValueError("neighbourhood_radius must be at least one.")

    longitudes = np.sort(elevation_grid["lon"].unique())
    latitudes = np.sort(elevation_grid["lat"].unique())
    expected_size = len(longitudes) * len(latitudes)
    if len(elevation_grid) != expected_size:
        raise ValueError("Elevation coordinates do not form a complete grid.")

    elevation = (
        elevation_grid
        .pivot(index="lat", columns="lon", values="elevation_m")
        .reindex(index=latitudes, columns=longitudes)
        .to_numpy(dtype=np.float64)
    )
    if not np.isfinite(elevation).all():
        raise ValueError("Elevation matrix contains gaps.")

    median_latitude = float(np.median(latitudes))
    median_longitude = float(np.median(longitudes))
    x_axis_m = (
        project_lonlat_to_twd97_km(
            longitudes,
            np.full(longitudes.shape, median_latitude),
        )[:, 0]
        * 1000.0
    )
    y_axis_m = (
        project_lonlat_to_twd97_km(
            np.full(latitudes.shape, median_longitude),
            latitudes,
        )[:, 1]
        * 1000.0
    )
    if not (np.all(np.diff(x_axis_m) > 0) and np.all(np.diff(y_axis_m) > 0)):
        raise ValueError("Projected terrain axes must be strictly increasing.")

    dz_dy, dz_dx = np.gradient(
        elevation,
        y_axis_m,
        x_axis_m,
        edge_order=2,
    )
    slope_radians = np.arctan(np.hypot(dz_dx, dz_dy))
    slope_degrees = np.degrees(slope_radians)

    # Aspect is the direction of steepest descent, clockwise from north.
    aspect_radians = np.mod(
        np.arctan2(-dz_dx, -dz_dy),
        2.0 * np.pi,
    )
    aspect_degrees = np.degrees(aspect_radians)
    flat = np.hypot(dz_dx, dz_dy) < 1e-12
    aspect_degrees[flat] = np.nan
    northness = np.cos(aspect_radians)
    eastness = np.sin(aspect_radians)
    northness[flat] = 0.0
    eastness[flat] = 0.0

    window_size = 2 * neighbourhood_radius + 1
    local_relief = (
        maximum_filter(elevation, size=window_size, mode="nearest")
        - minimum_filter(elevation, size=window_size, mode="nearest")
    )
    # TPI distinguishes locally elevated ridges (positive) from valleys
    # (negative). Excluding the centre cell avoids shrinking the contrast.
    neighbourhood_sum = (
        uniform_filter(
            elevation,
            size=window_size,
            mode="nearest",
        )
        * window_size**2
    )
    neighbourhood_mean = (
        neighbourhood_sum - elevation
    ) / (window_size**2 - 1)
    tpi = elevation - neighbourhood_mean
    ruggedness = _terrain_ruggedness(elevation)

    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    output = pd.DataFrame(
        {
            "lon": longitude_grid.ravel(),
            "lat": latitude_grid.ravel(),
            "elevation_m": elevation.ravel(),
            "slope_deg": slope_degrees.ravel(),
            "aspect_deg": aspect_degrees.ravel(),
            "northness": northness.ravel(),
            "eastness": eastness.ravel(),
            "local_relief_m": local_relief.ravel(),
            "tpi_m": tpi.ravel(),
            "terrain_ruggedness_m": ruggedness.ravel(),
        }
    )
    return add_twd97_km_columns(output)


def attach_terrain_to_grid(
    target_grid: pd.DataFrame,
    terrain_grid: pd.DataFrame,
    nearest_tolerance_km: float = 1.5,
) -> pd.DataFrame:
    """Attach terrain predictors by exact coordinate, then nearest fallback."""
    target = target_grid.copy()
    if "lon" not in target or "lat" not in target:
        raise KeyError("Target grid must contain lon and lat.")
    target["lon_key"] = pd.to_numeric(target["lon"]).round(5)
    target["lat_key"] = pd.to_numeric(target["lat"]).round(5)

    terrain = terrain_grid.copy()
    terrain["lon_key"] = terrain["lon"].round(5)
    terrain["lat_key"] = terrain["lat"].round(5)
    selected = terrain[
        ["lon_key", "lat_key", "x_km", "y_km", *PREDICTOR_COLUMNS]
    ]
    if selected.duplicated(["lon_key", "lat_key"]).any():
        raise ValueError("Terrain predictor coordinates are not unique.")

    joined = target.merge(
        selected,
        on=["lon_key", "lat_key"],
        how="left",
        validate="many_to_one",
    )
    missing = joined["elevation_m"].isna()
    if missing.any():
        target_projected = add_twd97_km_columns(
            joined.loc[missing, ["lon", "lat"]]
        )
        tree = cKDTree(terrain[["x_km", "y_km"]].to_numpy())
        distances, indices = tree.query(
            target_projected[["x_km", "y_km"]].to_numpy(),
            k=1,
        )
        if np.any(distances > nearest_tolerance_km):
            raise ValueError(
                "Some target cells have no terrain value within "
                f"{nearest_tolerance_km} km."
            )
        replacement = terrain.iloc[indices]
        joined.loc[missing, ["x_km", "y_km", *PREDICTOR_COLUMNS]] = (
            replacement[["x_km", "y_km", *PREDICTOR_COLUMNS]].to_numpy()
        )

    return joined.drop(columns=["lon_key", "lat_key"])


def build_terrain_predictor_tables(
    elevation_path: str | Path = DEFAULT_ELEVATION_PATH,
    terrain_output_path: str | Path = DEFAULT_TERRAIN_PATH,
    tccip_grid_path: str | Path = DEFAULT_TCCIP_GRID_PATH,
    tccip_output_path: str | Path = DEFAULT_TCCIP_TERRAIN_PATH,
    neighbourhood_radius: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the full-resolution and TCCIP-aligned predictor tables."""
    ensure_project_directories()
    elevation = load_elevation_grid(elevation_path)
    terrain = derive_terrain_predictors(
        elevation,
        neighbourhood_radius=neighbourhood_radius,
    )
    terrain.to_csv(terrain_output_path, index=False, encoding="utf-8-sig")

    target_grid = pd.read_csv(tccip_grid_path)
    tccip_terrain = attach_terrain_to_grid(target_grid, terrain)
    tccip_terrain.to_csv(
        tccip_output_path,
        index=False,
        encoding="utf-8-sig",
    )
    return terrain, tccip_terrain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive terrain predictors and attach them to TCCIP GRID."
    )
    parser.add_argument(
        "--elevation",
        type=Path,
        default=DEFAULT_ELEVATION_PATH,
    )
    parser.add_argument(
        "--tccip-grid",
        type=Path,
        default=DEFAULT_TCCIP_GRID_PATH,
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Local-relief neighbourhood radius in source-grid cells.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    full_grid, tccip_grid = build_terrain_predictor_tables(
        elevation_path=arguments.elevation,
        tccip_grid_path=arguments.tccip_grid,
        neighbourhood_radius=arguments.radius,
    )
    print(f"Full terrain grid: {len(full_grid):,} rows -> {DEFAULT_TERRAIN_PATH}")
    print(
        f"TCCIP terrain grid: {len(tccip_grid):,} rows "
        f"-> {DEFAULT_TCCIP_TERRAIN_PATH}"
    )
