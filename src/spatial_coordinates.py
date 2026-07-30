"""Metric-coordinate utilities for country- or region-scale spatial analyses.

Longitude and latitude are retained as raw-data identifiers and transformation
inputs. Distance-based modelling uses a projected CRS that is appropriate for
the study country, with coordinates converted to kilometres.

The current registry contains the two documented examples:

- Taiwan: TWD97 / TM2 zone 121 (EPSG:3826);
- Iceland: ISN2016 / Lambert 2016 (EPSG:8088).

Callers working in another country must supply its official projected CRS
explicitly instead of silently applying a globally fixed projection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


GEOGRAPHIC_CRS = "EPSG:4326"
TAIWAN_METRIC_CRS = "EPSG:3826"
ICELAND_METRIC_CRS = "EPSG:8088"

COUNTRY_METRIC_CRS = {
    "taiwan": TAIWAN_METRIC_CRS,
    "tw": TAIWAN_METRIC_CRS,
    "iceland": ICELAND_METRIC_CRS,
    "is": ICELAND_METRIC_CRS,
}


def country_metric_crs(country: str) -> str:
    """Return the registered projected CRS for one study country.

    The registry is deliberately explicit. Choosing a national projection is
    a scientific design decision, so an unknown country raises an error rather
    than falling back to Web Mercator or an arbitrary UTM zone.
    """
    if not isinstance(country, str) or not country.strip():
        raise ValueError("country must be a non-empty string")

    key = country.strip().casefold()
    try:
        return COUNTRY_METRIC_CRS[key]
    except KeyError as exc:
        supported = ", ".join(sorted({"Taiwan", "Iceland"}))
        raise ValueError(
            f"No projected CRS is registered for {country!r}. "
            f"Supported countries: {supported}. "
            "Pass target_crs explicitly for another single-country study."
        ) from exc


def _resolve_projected_crs(
    country: str | None,
    target_crs: str | CRS | None,
) -> CRS:
    if country is None and target_crs is None:
        raise ValueError("Provide either country or target_crs")

    crs = CRS.from_user_input(
        target_crs if target_crs is not None else country_metric_crs(country)
    )
    if not crs.is_projected:
        raise ValueError(
            f"target_crs must be projected with linear units; got {crs.to_string()}"
        )
    if len(crs.axis_info) < 2:
        raise ValueError("target_crs must provide two projected coordinate axes")
    if any(axis.unit_conversion_factor is None for axis in crs.axis_info[:2]):
        raise ValueError("target_crs axes must use units convertible to metres")
    return crs


def project_lonlat_to_km(
    lon: np.ndarray | pd.Series,
    lat: np.ndarray | pd.Series,
    *,
    country: str | None = None,
    target_crs: str | CRS | None = None,
) -> np.ndarray:
    """Project WGS84 longitude/latitude to ``[x_km, y_km]``.

    Parameters
    ----------
    lon, lat:
        WGS84 longitude and latitude in decimal degrees.
    country:
        Registered single-country study area, currently ``"taiwan"`` or
        ``"iceland"``. Ignored when ``target_crs`` is supplied.
    target_crs:
        Explicit projected CRS for another country or for reproducible
        overrides, for example ``"EPSG:8088"``.
    """
    lon_array = np.asarray(lon, dtype=np.float64)
    lat_array = np.asarray(lat, dtype=np.float64)
    if lon_array.shape != lat_array.shape:
        raise ValueError("lon and lat must have the same shape")
    if not np.all(np.isfinite(lon_array)) or not np.all(np.isfinite(lat_array)):
        raise ValueError("lon and lat must contain only finite values")
    if np.any((lon_array < -180.0) | (lon_array > 180.0)):
        raise ValueError("longitude must lie within [-180, 180] degrees")
    if np.any((lat_array < -90.0) | (lat_array > 90.0)):
        raise ValueError("latitude must lie within [-90, 90] degrees")

    projected_crs = _resolve_projected_crs(country, target_crs)
    transformer = Transformer.from_crs(
        GEOGRAPHIC_CRS,
        projected_crs,
        always_xy=True,
    )
    x_native, y_native = transformer.transform(lon_array, lat_array)

    x_to_metres = float(projected_crs.axis_info[0].unit_conversion_factor)
    y_to_metres = float(projected_crs.axis_info[1].unit_conversion_factor)
    x_km = np.asarray(x_native, dtype=np.float64) * x_to_metres / 1000.0
    y_km = np.asarray(y_native, dtype=np.float64) * y_to_metres / 1000.0

    if not np.all(np.isfinite(x_km)) or not np.all(np.isfinite(y_km)):
        raise ValueError(
            "Coordinate transformation produced non-finite values; "
            "verify that the selected CRS covers the study country"
        )
    return np.column_stack((x_km, y_km))


def add_projected_km_columns(
    frame: pd.DataFrame,
    *,
    country: str | None = None,
    target_crs: str | CRS | None = None,
    lon_col: str = "lon",
    lat_col: str = "lat",
    copy: bool = True,
) -> pd.DataFrame:
    """Add projected ``x_km`` and ``y_km`` columns to a data frame."""
    required = {lon_col, lat_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing coordinate columns: {sorted(missing)}")

    output = frame.copy() if copy else frame
    projected = project_lonlat_to_km(
        output[lon_col],
        output[lat_col],
        country=country,
        target_crs=target_crs,
    )
    output["x_km"] = projected[:, 0]
    output["y_km"] = projected[:, 1]
    return output


def project_lonlat_to_twd97_km(
    lon: np.ndarray | pd.Series,
    lat: np.ndarray | pd.Series,
) -> np.ndarray:
    """Backward-compatible Taiwan wrapper returning ``[x_km, y_km]``."""
    return project_lonlat_to_km(
        lon,
        lat,
        target_crs=TAIWAN_METRIC_CRS,
    )


def add_twd97_km_columns(
    frame: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    copy: bool = True,
) -> pd.DataFrame:
    """Backward-compatible Taiwan wrapper for projected columns."""
    return add_projected_km_columns(
        frame,
        target_crs=TAIWAN_METRIC_CRS,
        lon_col=lon_col,
        lat_col=lat_col,
        copy=copy,
    )


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
