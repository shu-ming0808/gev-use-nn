import os
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRID_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "grid_gev_params.csv")
STATION_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "station_gev_params_with_loc.csv")

grid = pd.read_csv(GRID_PATH)
stations = pd.read_csv(STATION_PATH)

# =========================
# Taiwan shapefile
# =========================
world = gpd.read_file(r"C:\Users\User.DESKTOP-4RV84M1\Downloads\ne_50m_admin_0_countries\ne_50m_admin_0_countries.shp")
taiwan = world[world["NAME"].str.contains("Taiwan")]

# =========================
# grid → GeoDataFrame
# =========================
geometry = [Point(xy) for xy in zip(grid["lon"], grid["lat"])]
gdf = gpd.GeoDataFrame(grid, geometry=geometry, crs="EPSG:4326")

# clip
gdf_clipped = gpd.clip(gdf, taiwan)

# stations
station_geom = [Point(xy) for xy in zip(stations["lon"], stations["lat"])]
stations_gdf = gpd.GeoDataFrame(stations, geometry=station_geom, crs="EPSG:4326")

# =========================
# 畫三張圖
# =========================
def plot_param(column, title, filename):
    fig, ax = plt.subplots(figsize=(7, 7))

    taiwan.boundary.plot(ax=ax, color="black", linewidth=1)

    gdf_clipped.plot(
        column=column,
        ax=ax,
        cmap="viridis",
        markersize=10,
        legend=True
    )

    stations_gdf.plot(
        ax=ax,
        color="red",
        markersize=30,
        edgecolor="black"
    )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

    plt.tight_layout()

    out_path = os.path.join(PROJECT_ROOT, "results", "figures", filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)

plot_param("mu", "Spatial μ(s)", "map_mu.png")
plot_param("sigma", "Spatial σ(s)", "map_sigma.png")
plot_param("xi", "Spatial ξ(s)", "map_xi.png")