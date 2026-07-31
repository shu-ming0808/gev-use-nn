"""Derive distance from each TCCIP GRID centre to the nearest coastline.

The coastline source is GSHHG version 2.3.7.  Only the intermediate-resolution
level-1 polygons (land/ocean boundary) are extracted from the global archive.
Both GRID centres and coastlines are projected to TWD97 / TM2 zone 121
(EPSG:3826) before Euclidean distances are calculated in kilometres.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from project_paths import (
    PROCESSED_DATA_DIR,
    SPATIAL_PREDICTOR_PROCESSED_DIR,
    SPATIAL_PREDICTOR_RAW_DIR,
    ensure_project_directories,
)


GSHHG_VERSION = "2.3.7"
GSHHG_URL = (
    "https://github.com/GenericMappingTools/gshhg-gmt/releases/download/"
    f"{GSHHG_VERSION}/gshhg-shp-{GSHHG_VERSION}.zip"
)
COASTLINE_BASENAME = "GSHHS_i_L1"
TAIWAN_BBOX_WGS84 = (118.0, 20.0, 123.5, 27.0)
METRIC_CRS = "EPSG:3826"

DEFAULT_RAW_DIR = SPATIAL_PREDICTOR_RAW_DIR / "coastline"
DEFAULT_ARCHIVE_PATH = DEFAULT_RAW_DIR / f"gshhg-shp-{GSHHG_VERSION}.zip"
DEFAULT_COASTLINE_DIR = DEFAULT_RAW_DIR / COASTLINE_BASENAME
DEFAULT_GRID_PATH = PROCESSED_DATA_DIR / "grid_station_location.csv"
DEFAULT_OUTPUT_PATH = (
    SPATIAL_PREDICTOR_PROCESSED_DIR / "tccip_grid_coast_distance.csv"
)


def _download_file(url: str, destination: Path) -> None:
    """Download ``url`` atomically to ``destination``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "gev-spatial-predictor-pipeline/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def download_and_extract_coastline(
    archive_path: Path = DEFAULT_ARCHIVE_PATH,
    coastline_dir: Path = DEFAULT_COASTLINE_DIR,
    force_download: bool = False,
) -> Path:
    """Download GSHHG and extract the intermediate level-1 shapefile."""
    shapefile_path = coastline_dir / f"{COASTLINE_BASENAME}.shp"
    if force_download or not archive_path.exists():
        _download_file(GSHHG_URL, archive_path)

    required_suffixes = {".shp", ".shx", ".dbf", ".prj"}
    if force_download or not shapefile_path.exists():
        coastline_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            members = [
                member
                for member in archive.infolist()
                if Path(member.filename).stem == COASTLINE_BASENAME
                and Path(member.filename).suffix.lower() in required_suffixes
            ]
            found_suffixes = {
                Path(member.filename).suffix.lower() for member in members
            }
            if found_suffixes != required_suffixes:
                raise FileNotFoundError(
                    "The GSHHG archive does not contain all required "
                    f"{COASTLINE_BASENAME} shapefile components."
                )
            for member in members:
                destination = coastline_dir / Path(member.filename).name
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
    return shapefile_path


def load_tccip_grid(path: str | Path = DEFAULT_GRID_PATH) -> pd.DataFrame:
    """Load and validate TCCIP GRID identifiers and centre coordinates."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"station", "lon", "lat"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"TCCIP GRID file is missing columns: {sorted(missing)}")
    frame = frame[["station", "lon", "lat"]].copy()
    frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    if frame.isna().any().any():
        raise ValueError("TCCIP GRID identifiers or coordinates contain missing values.")
    if frame["station"].duplicated().any():
        raise ValueError("TCCIP GRID identifiers must be unique.")
    if not frame["lon"].between(118.0, 123.5).all():
        raise ValueError("TCCIP GRID longitudes are outside the Taiwan domain.")
    if not frame["lat"].between(20.0, 27.0).all():
        raise ValueError("TCCIP GRID latitudes are outside the Taiwan domain.")
    return frame.sort_values("station").reset_index(drop=True)


def derive_coast_distance(
    grid: pd.DataFrame,
    coastline_path: str | Path,
) -> pd.DataFrame:
    """Return nearest level-1 coastline distance for every GRID centre."""
    coast = gpd.read_file(
        coastline_path,
        bbox=TAIWAN_BBOX_WGS84,
    )
    if coast.empty:
        raise ValueError("No GSHHG coastline polygons intersect the Taiwan domain.")
    if coast.crs is None:
        raise ValueError("GSHHG coastline shapefile has no CRS definition.")

    # Level 1 represents the land/ocean boundary.  Its polygon boundaries are
    # therefore coastlines, without adding level-2 lake shorelines.
    coast = coast.to_crs(METRIC_CRS)
    coastline = coast.geometry.boundary.union_all()

    points = gpd.GeoDataFrame(
        grid.copy(),
        geometry=gpd.points_from_xy(grid["lon"], grid["lat"]),
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)
    distance_km = points.geometry.distance(coastline).to_numpy() / 1000.0
    if not np.isfinite(distance_km).all() or np.any(distance_km < 0.0):
        raise ValueError("Calculated coastline distances are invalid.")

    result = grid.copy()
    result["coast_distance_km"] = distance_km
    result["coastline_source"] = f"GSHHG_{GSHHG_VERSION}_intermediate_L1"
    result["distance_crs"] = METRIC_CRS
    return result


def build_coast_distance_table(
    grid_path: str | Path = DEFAULT_GRID_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    force_download: bool = False,
) -> pd.DataFrame:
    """Download the source if needed and write the processed predictor table."""
    ensure_project_directories()
    coastline_path = download_and_extract_coastline(
        force_download=force_download,
    )
    grid = load_tccip_grid(grid_path)
    result = derive_coast_distance(grid, coastline_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate TCCIP GRID-centre distance to GSHHG coastline."
    )
    parser.add_argument("--grid-path", type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the GSHHG archive and re-extract its shapefile.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_coast_distance_table(
        grid_path=args.grid_path,
        output_path=args.output_path,
        force_download=args.force_download,
    )
    print(f"Wrote {len(result):,} GRID rows to {args.output_path}")
    print(result["coast_distance_km"].describe().to_string())


if __name__ == "__main__":
    main()
