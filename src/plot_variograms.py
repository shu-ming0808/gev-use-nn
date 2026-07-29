import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spatial_coordinates import project_lonlat_to_twd97_km


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPATIAL_DIR = os.path.join(PROJECT_ROOT, "data", "simulated", "spatial_gev")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
TABLE_DIR = os.path.join(PROJECT_ROOT, "results", "tables")


PARAMS = [
    ("mu", "mu"),
    ("sigma", "sigma"),
    ("xi", "xi"),
]


def lonlat_to_km(lon, lat):
    """Project WGS84 longitude/latitude to exact TWD97 kilometre coordinates."""
    return project_lonlat_to_twd97_km(lon, lat)


def empirical_variogram(coords, values, n_bins=12, max_pairs=300000, seed=111):
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(coords).all(axis=1) & np.isfinite(values)
    coords = coords[finite]
    values = values[finite]
    n = len(values)

    if n < 3:
        raise ValueError("At least three finite observations are required.")

    rng = np.random.default_rng(seed)
    total_pairs = n * (n - 1) // 2

    if total_pairs <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n - 1, size=max_pairs)
        j = np.where(j >= i, j + 1, j)

    distances = np.linalg.norm(coords[i] - coords[j], axis=1)
    semivar = 0.5 * (values[i] - values[j]) ** 2

    positive = distances > 0
    distances = distances[positive]
    semivar = semivar[positive]

    edges = np.linspace(0, np.nanpercentile(distances, 98), n_bins + 1)
    rows = []
    for bin_id in range(n_bins):
        left = edges[bin_id]
        right = edges[bin_id + 1]
        if bin_id == n_bins - 1:
            mask = (distances >= left) & (distances <= right)
        else:
            mask = (distances >= left) & (distances < right)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": bin_id + 1,
                "distance_km": float(np.mean(distances[mask])),
                "semivariance": float(np.mean(semivar[mask])),
                "n_pairs": int(mask.sum()),
            }
        )

    return pd.DataFrame(rows)


def load_original_station(scenario):
    path = os.path.join(SPATIAL_DIR, f"spatial_{scenario}_station_true_vs_nn.csv")
    df = pd.read_csv(path)
    return df, {"mu": "mu_hat", "sigma": "sigma_hat", "xi": "xi_hat"}


def load_gridded_true(scenario):
    path = os.path.join(SPATIAL_DIR, f"spatial_{scenario}_grid_true_params.csv")
    df = pd.read_csv(path)
    return df, {"mu": "true_mu", "sigma": "true_sigma", "xi": "true_xi"}


def build_variograms():
    datasets = [
        ("Original station", "annual", load_original_station),
        ("Original station", "monthly", load_original_station),
        ("Gridded true", "annual", load_gridded_true),
        ("Gridded true", "monthly", load_gridded_true),
    ]

    all_rows = []
    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=len(PARAMS),
        figsize=(12, 10),
        dpi=180,
        sharex=False,
        sharey=False,
    )

    for row_id, (dataset_name, scenario, loader) in enumerate(datasets):
        df, col_map = loader(scenario)
        coords = lonlat_to_km(df["lon"], df["lat"])

        for col_id, (param, title_param) in enumerate(PARAMS):
            variogram = empirical_variogram(
                coords,
                df[col_map[param]].to_numpy(dtype=np.float64),
                n_bins=10 if "station" in dataset_name.lower() else 14,
                max_pairs=300000,
                seed=111 + row_id * 10 + col_id,
            )
            variogram.insert(0, "param", param)
            variogram.insert(0, "scenario", scenario)
            variogram.insert(0, "dataset", dataset_name)
            all_rows.append(variogram)

            ax = axes[row_id, col_id]
            ax.plot(
                variogram["distance_km"],
                variogram["semivariance"],
                marker="o",
                linewidth=1.8,
                markersize=4,
            )
            if row_id == 0:
                ax.set_title(title_param)
            if col_id == 0:
                ax.set_ylabel(f"{dataset_name}\n{scenario}\nsemivariance")
            ax.set_xlabel("distance (km)")
            ax.grid(True, alpha=0.25)

    fig.suptitle("Empirical variograms for original station and gridded data", y=0.995)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, "variogram_original_vs_gridded_annual_monthly.png")
    table_path = os.path.join(TABLE_DIR, "variogram_original_vs_gridded_annual_monthly.csv")
    fig.savefig(fig_path, bbox_inches="tight")
    pd.concat(all_rows, ignore_index=True).to_csv(table_path, index=False, encoding="utf-8-sig")
    return fig_path, table_path


if __name__ == "__main__":
    fig_path, table_path = build_variograms()
    print("Saved figure:", fig_path)
    print("Saved table:", table_path)
