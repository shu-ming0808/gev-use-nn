"""Export data-derived global map assets; never run NN or spatial CV.

Map 1 uses the pinned official AR6 land-capable polygons. Map 2 is actual
completed NN coverage, verified against the saved output registry and LSM.
Excluded/uncomputed cells are blue; support warnings do not imply uncomputed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.collections import QuadMesh
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr

from global_pipeline.global_climate_regions import (
    ROOT, DEFAULT_BOUNDARIES, DEFAULT_OUTPUT, SOURCE_URL, load_regions,
    normalize_longitudes, sha256,
)

DEFAULT_NN_ROOT = Path("D:/論文資料/ERA5/cheng_nn_21d_1976_2025")
OUTPUT = ROOT / "results/figures/global_maps_20261021_black_text"
COAST = ROOT / "data/shapefile/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp"
RED = "#C82032"
BLUE = "#629DCC"
LAND = "#E0C28E"


def verified_output(report_path: Path, filename: str) -> tuple[Path, str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") not in ("complete", "completed"):
        raise ValueError(f"Run not completed: {report_path}")
    path = report_path.parent / filename
    digest = sha256(path)
    if digest != report["output_sha256"][filename]:
        raise ValueError(f"Saved run checksum mismatch: {path}")
    return path, digest


def registry_mask(frame: pd.DataFrame, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Map recorded IDs to original ERA5 axes, checking positions and uniqueness."""
    if frame.grid_id.isna().any() or frame.grid_id.duplicated().any():
        raise ValueError("NN registry contains missing or duplicate GRID identifiers.")
    parts = frame.grid_id.str.extract(r"^G(\d{3})_(\d{4})$")
    if parts.isna().any().any():
        raise ValueError("Unexpected global GRID identifier format.")
    iy, ix = parts[0].to_numpy(int), parts[1].to_numpy(int)
    if np.any(iy >= len(lat)) or np.any(ix >= len(lon)):
        raise ValueError("GRID identifier is outside the coordinate array.")
    if not np.allclose(frame.latitude.to_numpy(), lat[iy], atol=1e-8, rtol=0):
        raise ValueError("Latitude does not match GRID identifier.")
    if not np.allclose(frame.longitude.to_numpy() % 360, lon[ix] % 360, atol=1e-8, rtol=0):
        raise ValueError("Longitude does not match GRID identifier.")
    if not (frame.lsm.gt(0.5) & frame.lsm.le(1)).all():
        raise ValueError("NN registry is not exclusively LSM > 0.5.")
    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    mask[iy, ix] = True
    return mask


def base_figure(title: str, subtitle: str, *, sea_color: str):
    fig = plt.figure(figsize=(14, 8.7), facecolor="white")
    ax = fig.add_axes([0.055, 0.17, 0.925, 0.74])
    ax.set_facecolor(sea_color)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_xticklabels(["180°", "120°W", "60°W", "0°", "60°E", "120°E", "180°"])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_yticklabels(["60°S", "30°S", "0°", "30°N", "60°N"])
    ax.tick_params(labelsize=11, colors="black", length=3, pad=5)
    ax.set_axisbelow(False)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_path_effects([path_effects.Normal()])
    for spine in ax.spines.values():
        spine.set_color("#64748B")
        spine.set_linewidth(0.7)
    fig.text(0.055, 0.965, title, fontsize=21, weight="bold", color="black")
    fig.text(0.055, 0.928, subtitle, fontsize=12, color="black")
    return fig, ax


def save(fig, output: Path, name: str) -> list[Path]:
    paths = []
    for extension in ("png", "svg"):
        # PNG is already raster. Only rasterise the million-cell layer for SVG,
        # avoiding mixed-mode clipping of tick labels in the PNG renderer.
        for mesh in fig.findobj(QuadMesh):
            mesh.set_rasterized(extension == "svg")
        fig.canvas.draw()
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def export(*, nn_root=DEFAULT_NN_ROOT, output=OUTPUT):
    output, nn_root = Path(output), Path(nn_root)
    regions = [r for r in load_regions(DEFAULT_BOUNDARIES) if "Land" in r.kind]
    grid_path, grid_digest = verified_output(DEFAULT_OUTPUT / "region_manifest.json", "era5_ar6_grid.nc")
    baseline, registry_digest = verified_output(nn_root / "run_20261011_01/run_report.json", "grid_index.csv")
    prediction, prediction_digest = verified_output(nn_root / "run_bound_20261012_01/run_report.json", "bound_coefficients_and_checks.csv")
    registry = pd.read_csv(baseline)
    coefficient_names = ["mu0", "beta_mu", "eta0", "beta_sigma", "xi0"]
    predicted = pd.read_csv(prediction, usecols=["grid_id", "latitude", "longitude", "lsm", *coefficient_names])
    if not registry.grid_id.equals(predicted.grid_id):
        raise ValueError("Original registry and bound NN predictions are not aligned.")
    if not np.isfinite(predicted[coefficient_names].to_numpy()).all():
        raise ValueError("Nonfinite coefficients: do not mark all registered grids as computed.")
    with xr.open_dataset(grid_path, engine="scipy") as opened:
        grid = opened.load()
    lat, lon = grid.latitude.values, grid.longitude.values
    computed = registry_mask(predicted, lat, lon)
    if not np.array_equal(computed, registry_mask(registry, lat, lon)):
        raise ValueError("Original and bound NN coverage differ.")
    eligible = grid.lsm.values > 0.5
    if not np.array_equal(computed, eligible) or not np.array_equal(eligible, grid.land_mask.values == 1):
        raise ValueError("Actual NN coverage differs from the displayed LSM land mask.")
    countries = gpd.read_file(COAST).to_crs("EPSG:4326")
    coastline = gpd.GeoSeries([countries.geometry.union_all()], crs="EPSG:4326")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

    fig, ax = base_figure("Global IPCC AR6 reference regions", "46 land-capable regions  |  Red lines: official v4 boundaries", sea_color="#EDF5FA")
    coastline.plot(ax=ax, facecolor="#E0E3DF", edgecolor="#8C979E", linewidth=0.4, zorder=2)
    for region in regions:
        parts = list(region.geometry.geoms) if region.geometry.geom_type == "MultiPolygon" else [region.geometry]
        for part in parts:
            x, y = part.exterior.xy
            ax.plot(x, y, color=RED, linewidth=1.1, zorder=4)
            for ring in part.interiors:
                x, y = ring.xy
                ax.plot(x, y, color=RED, linewidth=1.1, zorder=4)
        point = max(parts, key=lambda p: p.area).representative_point()
        ax.text(point.x, point.y, region.acronym, ha="center", va="center", fontsize=9.8,
                weight="medium", color="black", zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.55})
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#B3BEC7", linewidth=0.35, alpha=0.55, zorder=1)
    fig.legend(handles=[Line2D([0], [0], color=RED, linewidth=1.5, label="AR6 region boundary"),
                        Patch(facecolor="#E0E3DF", edgecolor="#8C979E", label="Land outline")],
               loc="lower center", bbox_to_anchor=(0.5, 0.078), ncol=2, frameon=False, fontsize=12, labelcolor="black")
    fig.text(0.055, 0.045, "Iturbide et al. (2020), Fig. 1(b) / AR6 v4. Regions are not spatial-CV folds.", fontsize=10.5, color="black")
    fig.text(0.055, 0.020, "Boundaries can extend over water; the land-sea mask is applied separately. Plate Carrée (longitude / latitude).", fontsize=10, color="black")
    files = save(fig, output, "01_global_ar6_red_boundaries")

    n_computed, n_other = int(computed.sum()), int((~computed).sum())
    fig, ax = base_figure("Global NN coverage after LSM > 0.5", "ERA5 0.25° grid  |  1976–2025 annual maxima  |  Verified completed NN outputs", sea_color=BLUE)
    order = np.argsort(normalize_longitudes(lon))
    x = normalize_longitudes(lon)[order]
    # Repeat the dateline centre so its two half-cells cover both map edges.
    x = np.append(x, x[0] + 360)
    values = computed[:, order]
    values = np.column_stack([values, values[:, 0]])
    ax.pcolormesh(x, lat, values.astype(np.uint8), shading="nearest", vmin=0, vmax=1,
                  cmap=ListedColormap([BLUE, LAND]), rasterized=True, antialiased=False, zorder=2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    # Draw ticks and graticules after the rasterised mesh (also for SVG).
    ax.set_axisbelow(False)
    ax.grid(color="white", linewidth=0.4, alpha=0.22, zorder=3)
    fig.legend(handles=[Patch(facecolor=LAND, edgecolor="none", label=f"NN-computed land: {n_computed:,} grids"),
                        Patch(facecolor=BLUE, edgecolor="none", label=f"Excluded / not computed: {n_other:,} grids")],
               loc="lower center", bbox_to_anchor=(0.5, 0.078), ncol=2, frameon=False, fontsize=12, labelcolor="black")
    fig.text(0.055, 0.045, "Colour shows computation coverage, not temperature or accuracy. Global spatial CV has not been run.", fontsize=10.5, color="black")
    fig.text(0.055, 0.020, "Counts include stored pole replicas and 844 computed land cells outside AR6 land polygons. Plate Carrée.", fontsize=10, color="black")
    files += save(fig, output, "02_global_lsm_nn_coverage")

    report = {
        "status": "completed", "figure_count": 2, "buffer_map_created": False,
        "scope": "Read-only export from existing data. No NN inference, GP, CV or model replacement.",
        "region_count": len(regions), "region_source_url": SOURCE_URL,
        "nn_computed_grids": n_computed, "excluded_or_uncomputed_grids": n_other,
        "total_grid_entries": int(computed.size), "coverage_exactly_matches_lsm_above_half": True,
        "outside_ar6_land_but_computed": int((computed & (grid.analysis_land_mask.values != 1)).sum()),
        "computed_but_not_spatial_cv_eligible": int((computed & (grid.spatial_cv_eligible.values != 1)).sum()),
        "count_note": "Grid-array entries, not unique physical points at the poles or area fractions.",
        "grid_path": str(grid_path), "grid_sha256": grid_digest,
        "registry_path": str(baseline), "registry_sha256": registry_digest,
        "prediction_path": str(prediction), "prediction_sha256": prediction_digest,
        "coastline_path": str(COAST), "coastline_shp_sha256": sha256(COAST),
        "boundaries_sha256": sha256(DEFAULT_BOUNDARIES),
        "source_sha256": sha256(Path(__file__)),
        "output_sha256": {p.name: sha256(p) for p in files},
    }
    (output / "figure_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "figures": 2, "nn_computed_grids": n_computed,
                      "excluded_or_uncomputed_grids": n_other}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nn-root", type=Path, default=DEFAULT_NN_ROOT)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    export(nn_root=arguments.nn_root, output=arguments.output_directory)
