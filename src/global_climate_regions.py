"""Prepare IPCC AR6 reference-region labels before ERA5 temperatures arrive.

Iturbide et al. (2020), Fig. 1(b), doi:10.5194/essd-12-2959-2020.
Region membership uses grid CENTRES in WGS84, not cell-overlap area weights.
The 46 land-capable polygons are not a coastline: ERA5 LSM > 0.5 is a
separate filter. This module never downloads temperature or averages maxima.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import warnings
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from pyproj import Geod
import shapely
from shapely.geometry import shape
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "90bc1728e1a90298ba188b8810823e108724f354"
SOURCE_URL = (
    "https://raw.githubusercontent.com/SantanderMetGroup/ATLAS/"
    f"{SOURCE_COMMIT}/reference-regions/IPCC-WGI-reference-regions-v4.geojson"
)
SOURCE_SHA256 = "fa12b8134bd9f125cbc596821ab195194b114b93c1d171077a6689ed5c8a7e03"
DEFAULT_BOUNDARIES = ROOT / "data/spatial_predictors/raw/ar6/IPCC-WGI-reference-regions-v4.geojson"
DEFAULT_OUTPUT = ROOT / "data/processed/global_regions/ar6_025"
DEFAULT_LSM = Path("D:/論文資料/ERA5/1975-2025_global_hourly/static/era5_land_sea_mask_global_025.nc")
SCHEMA_VERSION = 1
BOUNDARY_TOLERANCE_DEGREES = 1e-9  # ~0.1 mm: repair floating-point edge gaps, not geographic buffers


@dataclass(frozen=True)
class Region:
    id: int
    acronym: str
    name: str
    kind: str
    geometry: object


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def ensure_boundaries(path: Path = DEFAULT_BOUNDARIES) -> Path:
    """Reuse only the exact pinned official file; verify before replacing cache."""
    path = Path(path)
    if path.is_file() and sha256(path) == SOURCE_SHA256:
        return path
    request = Request(SOURCE_URL, headers={"User-Agent": "gev-use-nn AR6 preparation"})
    with urlopen(request, timeout=60) as response:
        contents = response.read()
    if hashlib.sha256(contents).hexdigest() != SOURCE_SHA256:
        raise ValueError("Official AR6 boundary checksum mismatch; cache not replaced.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(contents)
    temporary.replace(path)
    return path


def load_regions(path: Path) -> list[Region]:
    if sha256(path) != SOURCE_SHA256:
        raise ValueError("Boundary file is not the pinned official AR6 v4 source.")
    document = json.loads(path.read_text(encoding="utf-8"))
    regions = []
    for feature in document["features"]:
        properties = feature["properties"]
        geometry = shape(feature["geometry"])
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"Invalid region geometry: {properties['Acronym']}")
        regions.append(Region(int(properties["id"]), properties["Acronym"],
                              properties["Name"], properties["Type"], geometry))
    regions.sort(key=lambda region: region.id)
    if (len(regions) != 58 or len({r.id for r in regions}) != 58
            or sum("Land" in r.kind for r in regions) != 46
            or sum("Ocean" in r.kind for r in regions) != 15):
        raise ValueError("Unexpected AR6 catalogue (expected 58 polygons, 46 land / 15 ocean).")
    return regions


def normalize_longitudes(longitude) -> np.ndarray:
    longitude = np.asarray(longitude, dtype=float)
    if not np.isfinite(longitude).all() or np.any((longitude < -180) | (longitude > 360)):
        raise ValueError("Longitude must be finite in [-180, 180] or [0, 360].")
    return (longitude + 180) % 360 - 180


def validate_points(longitude, latitude) -> tuple[np.ndarray, np.ndarray]:
    longitude, latitude = np.broadcast_arrays(normalize_longitudes(longitude),
                                               np.asarray(latitude, dtype=float))
    if not np.isfinite(latitude).all() or np.any(np.abs(latitude) > 90):
        raise ValueError("Latitude must be finite in [-90, 90].")
    return longitude, latitude


def assign_regions(longitude, latitude, regions: list[Region], chunk_size=100_000):
    """Return ID and number of covering polygons; -1 means outside all polygons.

    Shared edges belong to the lowest official region ID. Ties are reported,
    never duplicated. Date-line points are tested on BOTH -180 and +180 edges.
    This deterministic grid-centre convention is not an area-fraction mask.
    """
    longitude, latitude = validate_points(longitude, latitude)
    original_shape = longitude.shape
    lon, lat = longitude.ravel(), latitude.ravel()
    tree = shapely.STRtree([r.geometry.buffer(BOUNDARY_TOLERANCE_DEGREES) for r in regions])
    ids = np.asarray([r.id for r in regions], dtype=np.int16)
    assigned = np.full(lon.size, -1, dtype=np.int16)
    counts = np.zeros(lon.size, dtype=np.int16)
    for start in range(0, lon.size, chunk_size):
        stop = min(start + chunk_size, lon.size)
        x, y = lon[start:stop], lat[start:stop]
        pairs = tree.query(shapely.points(x, y), predicate="intersects")
        seam = np.flatnonzero(x == -180)
        if seam.size:
            extra = tree.query(shapely.points(np.full(seam.size, 180.0), y[seam]),
                               predicate="intersects")
            extra[0] = seam[extra[0]]
            pairs = np.unique(np.concatenate([pairs, extra], axis=1), axis=1)
        chosen = np.full(stop - start, np.iinfo(np.int16).max, dtype=np.int16)
        np.minimum.at(chosen, pairs[0], ids[pairs[1]])
        matched = np.bincount(pairs[0], minlength=stop - start)
        assigned[start:stop] = np.where(matched > 0, chosen, -1)
        counts[start:stop] = matched
    return assigned.reshape(original_shape), counts.reshape(original_shape)


def read_land_sea_mask(path: Path) -> xr.DataArray:
    """Load the small invariant mask in memory (safe for Windows Unicode paths).

    Never read a temperature file through this function. Extra dimensions must
    be singleton; the code must not silently pick one of several timestamps.
    """
    import netCDF4

    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("Expected the small ERA5 invariant LSM file, not an hourly archive.")
    with netCDF4.Dataset("lsm.nc", memory=path.read_bytes()) as dataset:
        lat_name = next((n for n in ("latitude", "lat") if n in dataset.variables), None)
        lon_name = next((n for n in ("longitude", "lon") if n in dataset.variables), None)
        if lat_name is None or lon_name is None or "lsm" not in dataset.variables:
            raise ValueError("LSM requires lsm, latitude/lat and longitude/lon variables.")
        lat = np.asarray(dataset.variables[lat_name][:], dtype=float)
        lon = np.asarray(dataset.variables[lon_name][:], dtype=float)
        var = dataset.variables["lsm"]
        values = np.asarray(np.ma.filled(var[:], np.nan), dtype=float)
        dims = list(var.dimensions)
        for axis in reversed(range(len(dims))):
            if dims[axis] not in (lat_name, lon_name):
                if values.shape[axis] != 1:
                    raise ValueError("LSM extra dimensions must have size 1.")
                values = np.take(values, 0, axis=axis)
                dims.pop(axis)
        if dims == [lon_name, lat_name]:
            values = values.T
        elif dims != [lat_name, lon_name]:
            raise ValueError("LSM must be on one-dimensional latitude/longitude axes.")
    if not np.isfinite(values).all() or np.any((values < -1e-6) | (values > 1 + 1e-6)):
        raise ValueError("LSM must be finite fractions in [0, 1].")
    # Only roundoff outside [0,1] is clipped; no missing values are invented.
    mask = xr.DataArray(np.clip(values, 0, 1), dims=("latitude", "longitude"),
                        coords={"latitude": lat, "longitude": lon % 360})
    if len(np.unique(lat)) != lat.size or len(np.unique(lon % 360)) != lon.size:
        raise ValueError("LSM has duplicate coordinate values.")
    return mask.sortby("latitude", ascending=False).sortby("longitude")


def build_grid(regions: list[Region], mask: xr.DataArray | None, threshold=0.5) -> xr.Dataset:
    if not 0 <= threshold < 1:
        raise ValueError("LSM threshold must lie in [0, 1).")
    lat = np.linspace(90, -90, 721)
    lon = np.arange(1440) * 0.25
    x, y = np.meshgrid(lon, lat)
    region_id, matches = assign_regions(x, y, regions)
    if mask is not None:
        # Label by coordinate value, never assume a downloaded file's row order.
        if (mask.shape != (721, 1440)
                or not np.allclose(mask.latitude, lat, atol=1e-8, rtol=0)
                or not np.allclose(mask.longitude, lon, atol=1e-8, rtol=0)):
            raise ValueError("LSM does not match the global 0.25-degree grid; no interpolation performed.")
        lsm = mask.values.astype(np.float32)
        land = (lsm > threshold).astype(np.int8)
    else:
        lsm = np.full(x.shape, np.nan, dtype=np.float32)
        land = np.full(x.shape, -1, dtype=np.int8)  # unknown, NOT ocean
    land_ids = [r.id for r in regions if "Land" in r.kind]
    land_capable = np.isin(region_id, land_ids)
    analysis = np.where(land < 0, -1, ((land == 1) & land_capable).astype(np.int8))
    unique = ~((np.abs(y) == 90) & (x != 0))
    eligible = np.where(analysis < 0, -1, ((analysis == 1) & unique).astype(np.int8))
    dims = ("latitude", "longitude")
    dataset = xr.Dataset({
        "ar6_region_id": (dims, region_id),
        "boundary_matches": (dims, matches),
        "lsm": (dims, lsm),
        "land_mask": (dims, land),
        "analysis_land_mask": (dims, analysis.astype(np.int8)),
        "spatial_cv_eligible": (dims, eligible.astype(np.int8)),
    }, coords={"latitude": lat, "longitude": lon})
    dataset.attrs.update({
        "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
        "paper_doi": "10.5194/essd-12-2959-2020", "schema_version": SCHEMA_VERSION,
        "lsm_threshold": threshold, "lsm_available": int(mask is not None),
        "membership": "WGS84 grid centres; shared boundaries: smallest official ID",
        "negative_mask": "-1 means unknown; it must not be cast to boolean",
        "pole_policy": "Keep ERA5 rectangular grid; spatial_cv_eligible retains only longitude=0 at each pole",
        "distance_policy": "Degrees are not km. Use geodesic distances or an audited regional projected CRS.",
    })
    return dataset


def catalogue(regions: list[Region]) -> pd.DataFrame:
    return pd.DataFrame([{"ar6_region_id": r.id, "ar6_region": r.acronym,
                          "ar6_region_name": r.name, "ar6_region_type": r.kind} for r in regions])


def attach_region_labels(table: pd.DataFrame, grid: xr.Dataset, region_table: pd.DataFrame,
                         *, lon_column="longitude", lat_column="latitude") -> pd.DataFrame:
    """Many annual rows per GRID are allowed; preserve values, index and row order.

    Requires original 0.25-degree grid centres. No nearest-cell guesses or
    interpolation. New grid_id_025 is a geographic key, not an existing station ID.
    """
    x, y = validate_points(table[lon_column].to_numpy(), table[lat_column].to_numpy())
    if (grid.sizes.get("latitude") != 721 or grid.sizes.get("longitude") != 1440
            or not np.allclose(grid.latitude, np.linspace(90, -90, 721))
            or not np.allclose(grid.longitude, np.arange(1440) * 0.25)):
        raise ValueError("Lookup must use the canonical global 0.25-degree axes.")
    col_float, row_float = (x % 360) * 4, (90 - y) * 4
    if (not np.allclose(col_float, np.rint(col_float), atol=1e-6, rtol=0)
            or not np.allclose(row_float, np.rint(row_float), atol=1e-6, rtol=0)):
        raise ValueError("Input coordinates are not exact 0.25-degree GRID centres.")
    col = np.rint(col_float).astype(int) % 1440
    row = np.rint(row_float).astype(int)
    new = {"grid_id_025": row * 1440 + col}
    for name in grid.data_vars:
        new[name] = grid[name].values[row, col]
    lookup = region_table.set_index("ar6_region_id")
    if not lookup.index.is_unique:
        raise ValueError("Region catalogue has duplicate IDs.")
    region_ids = pd.Series(new["ar6_region_id"])
    for name in ("ar6_region", "ar6_region_name", "ar6_region_type"):
        new[name] = region_ids.map(lookup[name]).to_numpy()
    collisions = set(table.columns) & set(new)
    if collisions:
        raise ValueError(f"Refusing to overwrite existing labels: {sorted(collisions)}")
    result = table.copy()
    for name, values in new.items():
        result[name] = values
    return result


def geodesic_distance_km(lon1, lat1, lon2, lat2):
    """WGS84 ellipsoid pair distances in km, including across the date line.

    Array arguments are elementwise/broadcast, not a global NxN allocation.
    This does not replace the existing Taiwan CV implementation automatically.
    """
    x1, y1 = validate_points(lon1, lat1)
    x2, y2 = validate_points(lon2, lat2)
    x1, y1, x2, y2 = np.broadcast_arrays(x1, y1, x2, y2)
    return np.asarray(Geod(ellps="WGS84").inv(x1, y1, x2, y2)[2]) / 1000


def plot_preview(grid: xr.Dataset, regions: list[Region], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(16, 8), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.955))
    order = np.argsort(normalize_longitudes(grid.longitude.values))
    lon = normalize_longitudes(grid.longitude.values)[order]
    data = grid.ar6_region_id.values[:, order].astype(float)
    known = bool(grid.attrs["lsm_available"])
    if known:
        data[grid.analysis_land_mask.values[:, order] != 1] = np.nan
    else:
        data[data >= 46] = np.nan
    colors = np.vstack([plt.get_cmap("tab20").colors,
                        plt.get_cmap("tab20b").colors, plt.get_cmap("tab20c").colors])
    ax.pcolormesh(lon, grid.latitude, data, cmap=ListedColormap(colors),
                  vmin=-0.5, vmax=59.5, shading="nearest", rasterized=True)
    for region in regions:
        if "Land" not in region.kind:
            continue
        parts = list(region.geometry.geoms) if region.geometry.geom_type == "MultiPolygon" else [region.geometry]
        for part in parts:
            bx, by = part.exterior.xy
            ax.plot(bx, by, color="0.3", linewidth=0.65)
        label_part = max(parts, key=lambda part: part.area)
        point = label_part.representative_point()
        ax.text(point.x, point.y, region.acronym, ha="center", va="center", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.5})
    if known:
        missed = (grid.land_mask.values[:, order] == 1) & (grid.analysis_land_mask.values[:, order] != 1)
        iy, ix = np.where(missed)
        if iy.size:
            ax.scatter(lon[ix], grid.latitude.values[iy], s=1, c="0.45",
                       label="LSM land outside AR6 land regions")
            ax.legend(loc="lower left", fontsize=8)
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude (degrees)", ylabel="Latitude (degrees)")
    mode = f"ERA5 LSM > {grid.attrs['lsm_threshold']:g}" if known else "POLYGONS ONLY: land/ocean not yet filtered"
    ax.set_title(f"IPCC AR6 reference regions (v4): 46 land-capable regions\n0.25-degree grid centres; {mode}")
    ax.grid(alpha=0.15)
    fig.text(0.5, 0.012, "Iturbide et al. (2020), Fig. 1(b). Colours identify regions, not temperature. No temperature data used.",
             ha="center", fontsize=9)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def prepare(output: Path, boundaries: Path, lsm_path: Path | None, threshold=0.5) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    regions = load_regions(ensure_boundaries(boundaries))
    if lsm_path is not None and not lsm_path.is_file():
        raise FileNotFoundError(f"LSM not found: {lsm_path}. Use --without-lsm for a labelled, unfiltered template.")
    configuration = {"schema_version": SCHEMA_VERSION, "boundaries_sha256": SOURCE_SHA256,
                     "boundary_tolerance_degrees": BOUNDARY_TOLERANCE_DEGREES,
                     "lsm_sha256": sha256(lsm_path) if lsm_path else None, "lsm_threshold": threshold}
    manifest_path = output / "region_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("configuration") != configuration:
            raise ValueError("Output belongs to different settings. Choose a new --output-directory; existing data preserved.")
        files = previous.get("output_sha256", {})
        if previous.get("status") == "complete" and files and all(
                (output / name).is_file() and sha256(output / name) == digest for name, digest in files.items()):
            print(f"Verified existing region lookup: {output}", flush=True)
            return previous
    manifest = {"status": "building", "configuration": configuration,
                "source_url": SOURCE_URL, "source_commit": SOURCE_COMMIT,
                "paper": "Iturbide et al. (2020), ESSD 12, 2959-2970, Fig. 1(b)",
                "doi": "10.5194/essd-12-2959-2020", "lsm_path": str(lsm_path) if lsm_path else None}
    write_json(manifest_path, manifest)
    grid = build_grid(regions, read_land_sea_mask(lsm_path) if lsm_path else None, threshold)
    region_table = catalogue(regions)
    ids, analysis = grid.ar6_region_id.values, grid.analysis_land_mask.values
    region_table["polygon_grid_count"] = [int((ids == r.id).sum()) for r in regions]
    region_table["analysis_land_grid_count"] = [int(((ids == r.id) & (analysis == 1)).sum()) for r in regions] if lsm_path else pd.NA
    region_table["spatial_cv_eligible_count"] = [int(((ids == r.id) & (grid.spatial_cv_eligible.values == 1)).sum()) for r in regions] if lsm_path else pd.NA
    region_table.to_csv(output / "region_catalogue.csv.part", index=False)
    (output / "region_catalogue.csv.part").replace(output / "region_catalogue.csv")
    grid.to_netcdf(output / "era5_ar6_grid.nc.part", engine="scipy")
    (output / "era5_ar6_grid.nc.part").replace(output / "era5_ar6_grid.nc")
    iy, ix = np.where(analysis == 1)
    land_points = pd.DataFrame({"latitude": grid.latitude.values[iy], "longitude": grid.longitude.values[ix]})
    land_lookup = attach_region_labels(land_points, grid, region_table)
    land_lookup.to_csv(output / "land_grid_lookup.csv.gz.part", index=False, compression="gzip")
    (output / "land_grid_lookup.csv.gz.part").replace(output / "land_grid_lookup.csv.gz")
    iy, ix = np.where((grid.land_mask.values == 1) & (analysis != 1))
    exceptions = attach_region_labels(pd.DataFrame({"latitude": grid.latitude.values[iy],
                                                    "longitude": grid.longitude.values[ix]}), grid, region_table)
    exceptions["exclusion_reason"] = np.where(exceptions.ar6_region_id < 0,
                                               "outside_all_ar6_polygons", "inside_ocean_only_region")
    exceptions.to_csv(output / "land_outside_ar6_land_regions.csv.part", index=False)
    (output / "land_outside_ar6_land_regions.csv.part").replace(output / "land_outside_ar6_land_regions.csv")
    plot_preview(grid, regions, output / "ar6_land_regions.png")
    missing = int(((grid.land_mask.values == 1) & (analysis != 1)).sum())
    manifest.update({"status": "complete", "grid_shape": [721, 1440], "grid_count": int(ids.size),
                     "land_region_count": 46, "ocean_region_count": 15, "unique_polygon_count": 58,
                     "land_status": "LSM filtered" if lsm_path else "unknown; do not use empty land lookup as ocean mask",
                     "lsm_land_grid_count": int((grid.land_mask.values == 1).sum()) if lsm_path else None,
                     "analysis_land_grid_count": int((analysis == 1).sum()) if lsm_path else None,
                     "spatial_cv_eligible_count": int((grid.spatial_cv_eligible.values == 1).sum()) if lsm_path else None,
                     "land_without_land_region_count": missing if lsm_path else None,
                     "unassigned_polygon_grid_count": int((ids < 0).sum()),
                     "boundary_tie_grid_count": int((grid.boundary_matches.values > 1).sum()),
                     "boundary_policy": "lowest official ID; both date-line edges queried",
                     "scope": "Grid centre membership only; no temperature aggregation, NN training or CV performed.",
                     "limitations": ["AR6 boundaries do not prove stationarity/isotropy/independence.",
                                     "LSM land outside land-capable polygons (including islands/source boundary gaps) is reported, not reassigned.",
                                     "Pole replicas remain in lookup; only longitude=0 is spatial-CV eligible.",
                                     "Use geodesic km or validate regional projection before spatial CV."]})
    output_files = ["era5_ar6_grid.nc", "region_catalogue.csv", "land_grid_lookup.csv.gz",
                    "land_outside_ar6_land_regions.csv", "ar6_land_regions.png"]
    manifest["output_sha256"] = {name: sha256(output / name) for name in output_files}
    write_json(manifest_path, manifest)
    if missing:
        warnings.warn(f"{missing} LSM-land grid centres lie outside AR6 land-capable polygons; reported, not silently reassigned.")
    return manifest


def apply_csv(input_path: Path, output_path: Path, directory: Path, *, lon_column="longitude",
              lat_column="latitude", land_only=False, region=None, chunksize=100_000) -> dict:
    if input_path.resolve() == output_path.resolve() or output_path.exists():
        raise ValueError("Use a new output filename; input and existing outputs are never overwritten.")
    manifest = json.loads((directory / "region_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Region preparation is incomplete; rerun prepare first.")
    for name in ("era5_ar6_grid.nc", "region_catalogue.csv"):
        if sha256(directory / name) != manifest["output_sha256"][name]:
            raise ValueError("Region lookup integrity failed; rerun prepare.")
    with xr.open_dataset(directory / "era5_ar6_grid.nc", engine="scipy") as handle:
        grid = handle.load()
    if land_only and not grid.attrs["lsm_available"]:
        raise ValueError("Cannot select land without an LSM. Prepare again in a new folder with --lsm.")
    region_table = pd.read_csv(directory / "region_catalogue.csv")
    if region is not None and region not in set(region_table.ar6_region):
        raise ValueError(f"Unknown AR6 region: {region}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    # Stream plain CSV to avoid holding a worldwide 50-year table in memory.
    if output_path.suffix.lower() != ".csv":
        raise ValueError("Output must end in .csv (input may be .csv.gz).")
    counts = {"input_rows": 0, "output_rows": 0}
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        first = True
        for chunk in pd.read_csv(input_path, chunksize=chunksize):
            counts["input_rows"] += len(chunk)
            result = attach_region_labels(chunk, grid, region_table, lon_column=lon_column, lat_column=lat_column)
            if land_only:
                result = result.loc[result.analysis_land_mask == 1]
            if region is not None:
                result = result.loc[result.ar6_region == region]
            counts["output_rows"] += len(result)
            result.to_csv(stream, index=False, header=first)
            first = False
    temporary.replace(output_path)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Prepare the 0.25-degree lookup without temperature data.")
    prep.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    prep.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES)
    mask = prep.add_mutually_exclusive_group()
    mask.add_argument("--lsm", type=Path, default=DEFAULT_LSM)
    mask.add_argument("--without-lsm", action="store_true")
    prep.add_argument("--land-threshold", type=float, default=0.5)
    apply = commands.add_parser("apply", help="Attach region labels to any long/wide annual GRID CSV.")
    apply.add_argument("--input", required=True, type=Path)
    apply.add_argument("--output", required=True, type=Path)
    apply.add_argument("--lookup-directory", type=Path, default=DEFAULT_OUTPUT)
    apply.add_argument("--lon-column", default="longitude")
    apply.add_argument("--lat-column", default="latitude")
    apply.add_argument("--land-only", action="store_true")
    apply.add_argument("--region", help="Optional region acronym, e.g. EAS (not a country).")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output_directory, args.boundaries,
                         None if args.without_lsm else args.lsm, args.land_threshold)
    else:
        result = apply_csv(args.input, args.output, args.lookup_directory,
                           lon_column=args.lon_column, lat_column=args.lat_column,
                           land_only=args.land_only, region=args.region)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
