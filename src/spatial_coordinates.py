"""Coordinate utilities shared by the spatial simulation and GP workflows.

Longitude and latitude are retained only for raw-data identifiers, joins, and
coordinate conversion.
Every operation that represents a physical distance uses TWD97 / TM2 zone 121
(EPSG:3826), converted from metres to kilometres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer


GEOGRAPHIC_CRS = "EPSG:4326"
TAIWAN_METRIC_CRS = "EPSG:3826"

_TO_TWD97 = Transformer.from_crs(
    GEOGRAPHIC_CRS,
    TAIWAN_METRIC_CRS,
    always_xy=True,
)


def project_lonlat_to_twd97_km(
    lon: np.ndarray | pd.Series,
    lat: np.ndarray | pd.Series,
) -> np.ndarray:
    """Return projected ``[x_km, y_km]`` coordinates for Taiwan."""
    lon_array = np.asarray(lon, dtype=np.float64)
    lat_array = np.asarray(lat, dtype=np.float64)
    if lon_array.shape != lat_array.shape:
        raise ValueError("lon and lat must have the same shape")
    if not np.all(np.isfinite(lon_array)) or not np.all(np.isfinite(lat_array)):
        raise ValueError("lon and lat must contain only finite values")

    x_m, y_m = _TO_TWD97.transform(lon_array, lat_array)
    return np.column_stack(
        (
            np.asarray(x_m, dtype=np.float64) / 1000.0,
            np.asarray(y_m, dtype=np.float64) / 1000.0,
        )
    )


def add_twd97_km_columns(
    frame: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    copy: bool = True,
) -> pd.DataFrame:
    """Add ``x_km`` and ``y_km`` while preserving longitude and latitude."""
    required = {lon_col, lat_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing coordinate columns: {sorted(missing)}")

    output = frame.copy() if copy else frame
    projected = project_lonlat_to_twd97_km(
        output[lon_col],
        output[lat_col],
    )
    output["x_km"] = projected[:, 0]
    output["y_km"] = projected[:, 1]
    return output


def center_train_test_coordinates(
    train_xy_km: np.ndarray,
    test_xy_km: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Center projected coordinates without changing kilometre distances.

    Subtracting the training-set origin improves numerical conditioning, while
    Euclidean distances and GP kernel length scales remain expressed in km.
    Unlike per-axis standardization, this operation does not distort isotropy.
    """
    train = np.asarray(train_xy_km, dtype=np.float64)
    if train.ndim != 2 or train.shape[1] != 2:
        raise ValueError("train_xy_km must have shape (n, 2)")

    origin = train.mean(axis=0)
    centered_train = train - origin

    centered_test = None
    if test_xy_km is not None:
        test = np.asarray(test_xy_km, dtype=np.float64)
        if test.ndim != 2 or test.shape[1] != 2:
            raise ValueError("test_xy_km must have shape (m, 2)")
        centered_test = test - origin

    return centered_train, centered_test, origin
