"""Build year-2000 land-cover proportions for the TCCIP GRID cells.

The source is ESA CCI Land Cover v2.0.7cds, hosted as Cloud Optimized
GeoTIFFs by Microsoft Planetary Computer.  Only the Taiwan window is read
from the remote 300 m raster.  Each TCCIP coordinate is treated as the
centre of a 0.05-degree cell, and pixel areas are weighted by cosine of
latitude before class proportions are calculated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.windows import Window, from_bounds

from project_paths import (
    PROCESSED_DATA_DIR,
    SPATIAL_PREDICTOR_PROCESSED_DIR,
    SPATIAL_PREDICTOR_RAW_DIR,
)


LAND_COVER_YEAR = 2000
SOURCE_NAME = "ESA CCI Land Cover v2.0.7cds"
SOURCE_ITEM = (
    "ESACCI-LC-L4-LCCS-Map-300m-P1Y-2000-v2.0.7cds-N00E090"
)
SOURCE_URL = (
    "https://landcoverdata.blob.core.windows.net/esa-cci-lc/"
    "cog/v2.0.7cds/N00E090/2000/"
    f"{SOURCE_ITEM}-lccs_class.tif"
)
SIGN_ENDPOINT = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"

# ESA CCI / FAO LCCS class mapping.  Mosaic natural vegetation (40) is kept
# in "other" because only "<50% cropland" is known, not its exact fraction.
CLASS_GROUPS = {
    "agriculture": {10, 11, 12, 20, 30},
    "forest": {
        50,
        60,
        61,
        62,
        70,
        71,
        72,
        80,
        81,
        82,
        90,
        160,
        170,
    },
    "urban": {190},
    "water": {210},
}


def _signed_href(href: str = SOURCE_URL) -> str:
    """Return a short-lived signed Planetary Computer asset URL."""
    request_url = f"{SIGN_ENDPOINT}?href={quote(href, safe='')}"
    with urlopen(request_url, timeout=60) as response:
        payload = json.load(response)
    return str(payload["href"])


def _integer_window(window: Window) -> Window:
    """Expand a floating raster window to integer pixel boundaries."""
    col_start = int(np.floor(window.col_off))
    row_start = int(np.floor(window.row_off))
    col_stop = int(np.ceil(window.col_off + window.width))
    row_stop = int(np.ceil(window.row_off + window.height))
    return Window(
        col_start,
        row_start,
        col_stop - col_start,
        row_stop - row_start,
    )


def download_taiwan_subset(
    grid: pd.DataFrame,
    destination: str | Path,
    cell_size_degrees: float = 0.05,
) -> Path:
    """Read and save only the raster window needed by the TCCIP GRID."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    margin = cell_size_degrees / 2.0
    bounds = (
        float(grid["lon"].min()) - margin,
        float(grid["lat"].min()) - margin,
        float(grid["lon"].max()) + margin,
        float(grid["lat"].max()) + margin,
    )

    with rasterio.open(_signed_href()) as source:
        window = _integer_window(from_bounds(*bounds, source.transform))
        values = source.read(1, window=window)
        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            height=values.shape[0],
            width=values.shape[1],
            transform=source.window_transform(window),
            count=1,
            compress="deflate",
            tiled=True,
        )
        tags = source.tags()

    with rasterio.open(destination, "w", **profile) as target:
        target.write(values, 1)
        target.update_tags(
            **tags,
            source=SOURCE_NAME,
            source_item=SOURCE_ITEM,
            reference_year=str(LAND_COVER_YEAR),
        )
    return destination


def _weighted_class_ratios(
    values: np.ndarray,
    latitudes: np.ndarray,
) -> dict[str, float | int]:
    """Calculate area-weighted class fractions for one GRID cell."""
    valid = values != 0
    weights = np.cos(np.deg2rad(latitudes))
    valid_weight = float(weights[valid].sum())
    if valid_weight <= 0:
        return {
            "valid_pixel_count": 0,
            "urban_ratio": np.nan,
            "forest_ratio": np.nan,
            "agriculture_ratio": np.nan,
            "water_ratio": np.nan,
            "other_ratio": np.nan,
        }

    ratios: dict[str, float | int] = {
        "valid_pixel_count": int(valid.sum())
    }
    selected_weight = 0.0
    for group, codes in CLASS_GROUPS.items():
        membership = valid & np.isin(values, tuple(codes))
        group_weight = float(weights[membership].sum())
        ratios[f"{group}_ratio"] = group_weight / valid_weight
        selected_weight += group_weight
    ratios["other_ratio"] = max(
        0.0,
        1.0 - selected_weight / valid_weight,
    )
    return ratios


def calculate_grid_land_cover_ratios(
    grid: pd.DataFrame,
    raster_path: str | Path,
    cell_size_degrees: float = 0.05,
) -> pd.DataFrame:
    """Calculate four land-cover ratios inside every TCCIP GRID cell."""
    required = {"station", "lon", "lat"}
    if not required.issubset(grid.columns):
        raise ValueError(
            f"GRID 資料必須包含 {sorted(required)}；"
            f"目前欄位為 {grid.columns.tolist()}。"
        )
    if grid.duplicated(["lon", "lat"]).any():
        raise ValueError("GRID 資料含有重複經緯度。")

    rows: list[dict] = []
    half_size = cell_size_degrees / 2.0
    with rasterio.open(raster_path) as source:
        for record in grid.itertuples(index=False):
            window = _integer_window(
                from_bounds(
                    float(record.lon) - half_size,
                    float(record.lat) - half_size,
                    float(record.lon) + half_size,
                    float(record.lat) + half_size,
                    source.transform,
                )
            )
            values = source.read(1, window=window)
            row_numbers = np.arange(
                int(window.row_off),
                int(window.row_off + window.height),
            )
            _, pixel_lats = xy(
                source.transform,
                row_numbers,
                np.full_like(row_numbers, int(window.col_off)),
                offset="center",
            )
            latitude_matrix = np.broadcast_to(
                np.asarray(pixel_lats, dtype=float)[:, None],
                values.shape,
            )
            ratios = _weighted_class_ratios(values, latitude_matrix)
            rows.append(
                {
                    "station": record.station,
                    "lat": float(record.lat),
                    "lon": float(record.lon),
                    "land_cover_year": LAND_COVER_YEAR,
                    "land_cover_source": SOURCE_NAME,
                    "cell_size_degrees": cell_size_degrees,
                    **ratios,
                }
            )

    result = pd.DataFrame(rows)
    ratio_columns = [
        "urban_ratio",
        "forest_ratio",
        "agriculture_ratio",
        "water_ratio",
        "other_ratio",
    ]
    result["ratio_sum"] = result[ratio_columns].sum(axis=1)
    return result


def build_land_cover_predictors(
    grid_path: str | Path = (
        PROCESSED_DATA_DIR / "grid_station_location.csv"
    ),
    raster_path: str | Path = (
        SPATIAL_PREDICTOR_RAW_DIR
        / "land_cover"
        / "esa_cci_land_cover_2000_taiwan.tif"
    ),
    output_path: str | Path = (
        SPATIAL_PREDICTOR_PROCESSED_DIR
        / "tccip_grid_land_cover_2000.csv"
    ),
    force_download: bool = False,
) -> pd.DataFrame:
    """Download, calculate, audit, and save the GRID predictors."""
    grid = pd.read_csv(grid_path)
    raster_path = Path(raster_path)
    output_path = Path(output_path)
    if force_download or not raster_path.exists():
        download_taiwan_subset(grid, raster_path)
    result = calculate_grid_land_cover_ratios(grid, raster_path)

    ratio_columns = [
        "urban_ratio",
        "forest_ratio",
        "agriculture_ratio",
        "water_ratio",
        "other_ratio",
    ]
    if result[ratio_columns].isna().any().any():
        raise ValueError("部分 GRID 沒有有效的 ESA CCI 像元。")
    if not np.allclose(result["ratio_sum"], 1.0, atol=1e-8):
        raise ValueError("土地覆蓋比例未加總為 1，請檢查分類或像元遮罩。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="建立 TCCIP GRID 的 2000 年土地覆蓋比例。"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="即使本地已有臺灣 raster，仍重新下載。",
    )
    args = parser.parse_args()
    result = build_land_cover_predictors(
        force_download=args.force_download
    )
    print(
        result[
            [
                "urban_ratio",
                "forest_ratio",
                "agriculture_ratio",
                "water_ratio",
                "other_ratio",
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()
