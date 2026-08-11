# Batch implementation of the TCCIP GRID preprocessing notebook.



from pathlib import Path
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import geopandas as gpd
from scipy.stats import genextreme as gev
from shapely import affinity
from shapely.geometry import Point
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# 路徑設定
# =========================
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPOSITORY_ROOT
PROJECT_SRC = REPOSITORY_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from project_paths import (  # noqa: E402
    FIGURE_DIR,
    MODEL_DIR,
    ORIGINAL_DATA_DIR,
    PROCESSED_DATA_DIR,
    SHAPEFILE_DIR,
)
from gev_nn import GEVNet, estimate_one as estimate_one_station  # noqa: E402
from prepare_daily_tmax_block_maxima import (  # noqa: E402
    prepare_daily_tmax_block_maxima,
)
from spatial_coordinates import (  # noqa: E402
    add_twd97_km_columns,
    center_train_test_coordinates,
)

RAW_DIR = ORIGINAL_DATA_DIR / "觀測_日資料_臺灣_最高溫"
PROCESSED_DIR = PROCESSED_DATA_DIR
FIG_DIR = FIGURE_DIR
MODEL_PATH = MODEL_DIR / "best_baseline_model.pth"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("PROJECT_ROOT:", PROJECT_ROOT)
print("RAW_DIR exists:", RAW_DIR.exists())
print("MODEL_PATH exists:", MODEL_PATH.exists())


def find_taiwan_boundary():
    candidates = [
        SHAPEFILE_DIR
        / "ne_50m_admin_0_countries"
        / "ne_50m_admin_0_countries.shp",
    ]
    for shp_path in candidates:
        if shp_path.exists():
            world = gpd.read_file(shp_path)
            taiwan = world[world["NAME"].str.contains("Taiwan", case=False, na=False)].to_crs("EPSG:4326")
            if not taiwan.empty:
                return taiwan
    raise FileNotFoundError("找不到 Taiwan shapefile，請確認 data/shapefile 是否存在。")


TAIWAN_BOUNDARY = find_taiwan_boundary()
TAIWAN_BOUNDARY_KM = TAIWAN_BOUNDARY.to_crs("EPSG:3826").copy()
TAIWAN_BOUNDARY_KM.geometry = TAIWAN_BOUNDARY_KM.geometry.apply(
    lambda geometry: affinity.scale(
        geometry,
        xfact=0.001,
        yfact=0.001,
        origin=(0.0, 0.0),
    )
)


def clip_points_to_taiwan(df, lon_col="lon", lat_col="lat"):
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")
    clipped = gpd.clip(gdf, TAIWAN_BOUNDARY)
    return pd.DataFrame(clipped.drop(columns="geometry")).reset_index(drop=True)



# =========================
# 由逐日最高溫建立每月極大值與發生日
# =========================
files = sorted(RAW_DIR.glob("觀測_日資料_臺灣_最高溫_*.csv"))
if not files:
    raise FileNotFoundError(
        "找不到 TCCIP 逐日最高溫檔。請登入 TCCIP 下載「網格化觀測日資料－"
        "最高溫－全臺－0.05度」，解壓縮至 data/original_data/"
        "觀測_日資料_臺灣_最高溫。降雨量日資料不可替代最高溫。"
    )
print("逐日最高溫年度檔數:", len(files))
print("年份範圍:", files[0].stem[-4:], "到", files[-1].stem[-4:])

daily_outputs = prepare_daily_tmax_block_maxima(
    raw_dir=RAW_DIR,
    output_dir=PROCESSED_DIR,
    pattern="觀測_日資料_臺灣_最高溫_*.csv",
    start_year=1960,
    min_daily_coverage=0.90,
)
monthly_events = pd.read_csv(
    daily_outputs["monthly_max_occurrences"],
    encoding="utf-8-sig",
)
daily_locations = pd.read_csv(
    daily_outputs["locations"],
    encoding="utf-8-sig",
)
monthly_long = monthly_events.merge(
    daily_locations[["station", "lon", "lat"]],
    on="station",
    how="left",
    validate="many_to_one",
)
monthly_long["date"] = pd.to_datetime(
    dict(year=monthly_long["year"], month=monthly_long["month"], day=1)
)
monthly_long["max_date"] = pd.to_datetime(monthly_long["max_date"])
monthly_long = monthly_long.rename(columns={"monthly_max_tmax_c": "max_temp"})
monthly_long = monthly_long[
    [
        "date", "max_date", "station", "lon", "lat", "max_temp",
        "all_tied_max_dates", "n_tied_max_dates",
    ]
]
monthly_long.to_csv(PROCESSED_DIR / "monthly_long_grid_temperature.csv", index=False, encoding="utf-8-sig")

station_location = monthly_long[["station", "lat", "lon"]].drop_duplicates("station").sort_values("station")
station_location = add_twd97_km_columns(station_location)
station_location.to_csv(PROCESSED_DIR / "grid_station_location.csv", index=False, encoding="utf-8-sig")

pivot_all = monthly_long.pivot_table(index="date", columns="station", values="max_temp", aggfunc="first").sort_index()
pivot_all.to_csv(PROCESSED_DIR / "pivot_grid_monthly_max_temperature_all.csv", encoding="utf-8-sig")

print("long shape:", monthly_long.shape)
print("pivot shape:", pivot_all.shape)
print("網格點數:", station_location.shape[0])
pivot_all.head()



# =========================
# EDA：缺值、coverage、時間範圍、溫度分布
# =========================
analysis_start = pd.Timestamp("1980-01-01")
pivot = pivot_all[pivot_all.index >= analysis_start].copy()

coverage = pivot.notna().mean().rename("valid_month_ratio").reset_index()
coverage = coverage.merge(station_location, on="station", how="left")
coverage["valid_months"] = pivot.notna().sum().values
coverage = coverage.sort_values("valid_month_ratio", ascending=False)
coverage.to_csv(PROCESSED_DIR / "grid_station_coverage_after_1980.csv", index=False, encoding="utf-8-sig")

MIN_MONTH_COVERAGE = 0.80
keep_stations = coverage.loc[coverage["valid_month_ratio"] >= MIN_MONTH_COVERAGE, "station"].tolist()
pivot_clean = pivot[keep_stations].copy()
pivot_clean.to_csv(PROCESSED_DIR / "pivot_grid_monthly_max_temperature_after_1980_clean.csv", encoding="utf-8-sig")

summary = pd.DataFrame({
    "item": [
        "raw_year_files", "all_grid_points", "months_after_1980", "kept_grid_points",
        "coverage_threshold", "temperature_min", "temperature_mean", "temperature_max"
    ],
    "value": [
        len(files), pivot_all.shape[1], pivot.shape[0], len(keep_stations),
        MIN_MONTH_COVERAGE, float(np.nanmin(pivot_clean.values)), float(np.nanmean(pivot_clean.values)), float(np.nanmax(pivot_clean.values))
    ]
})
summary.to_csv(PROCESSED_DIR / "eda_summary.csv", index=False, encoding="utf-8-sig")
print(summary)

# 圖 1：coverage 分布
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
ax.hist(coverage["valid_month_ratio"], bins=30, color="#2b8cbe", edgecolor="white")
ax.axvline(MIN_MONTH_COVERAGE, color="#d95f0e", linestyle="--", linewidth=2, label=f"keep >= {MIN_MONTH_COVERAGE:.0%}")
ax.set_title("Grid monthly data coverage after 1980")
ax.set_xlabel("valid month ratio")
ax.set_ylabel("grid count")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "eda_coverage_hist.png")
plt.close(fig)

# 圖 2：月最高溫整體分布
values = pivot_clean.to_numpy().ravel()
values = values[~np.isnan(values)]
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
ax.hist(values, bins=50, color="#41ab5d", edgecolor="white")
ax.set_title("Monthly maximum temperature distribution")
ax.set_xlabel("temperature")
ax.set_ylabel("count")
fig.tight_layout()
fig.savefig(FIG_DIR / "eda_temperature_hist.png")
plt.close(fig)

# 圖 3：每月全區平均最高溫時間序列
monthly_mean = pivot_clean.mean(axis=1)
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
monthly_mean.plot(ax=ax, color="#225ea8", linewidth=1.1)
ax.set_title("Regional mean of monthly maximum temperature")
ax.set_xlabel("date")
ax.set_ylabel("temperature")
fig.tight_layout()
fig.savefig(FIG_DIR / "eda_monthly_mean_timeseries.png")
plt.close(fig)

# 圖 4：有效網格點平均溫度空間分布
mean_by_station = pivot_clean.mean(axis=0).rename("mean_temp").reset_index()
mean_map = mean_by_station.merge(station_location, on="station", how="left")
fig, ax = plt.subplots(figsize=(6.2, 7.2), dpi=150)
sc = ax.scatter(mean_map["x_km"], mean_map["y_km"], c=mean_map["mean_temp"], s=9, cmap="turbo")
ax.set_title("Mean monthly maximum temperature by grid")
ax.set_xlabel("TWD97 Easting (km)")
ax.set_ylabel("TWD97 Northing (km)")
fig.colorbar(sc, ax=ax, label="temperature")
fig.tight_layout()
fig.savefig(FIG_DIR / "eda_mean_temperature_map.png")
plt.close(fig)

print("保留網格點:", len(keep_stations))
print("EDA figures saved to", FIG_DIR)



# =========================
# 保留年最大值作描述性比較；NN 正式輸入仍是 540 個月極大值
# =========================
annual_max = pivot_clean.resample("YE").max()
annual_max.index = annual_max.index.year
annual_max.index.name = "year"

# 年最大值輸出不是下方 NN 的輸入。
annual_max = annual_max[pivot_clean.columns]
annual_max.to_csv(PROCESSED_DIR / "annual_max_grid_temperature.csv", encoding="utf-8-sig")

annual_loc = station_location[
    station_location["station"].isin(pivot_clean.columns)
].copy()
annual_loc.to_csv(PROCESSED_DIR / "annual_grid_station_location.csv", index=False, encoding="utf-8-sig")

print("annual_max shape:", annual_max.shape)
annual_max.head()



# =========================
# 使用原本訓練好的 GEV NN 估計每個網格點參數
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
model = GEVNet().to(device)
try:
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
except TypeError:
    state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

results = []
for station in pivot_clean.columns:
    # 正式真實資料使用 1980--2024 的逐月極大值（最多 540 筆），
    # 而不是再次壓縮後的 45 筆年極大值。
    y = pivot_clean[station].dropna().to_numpy()
    if len(y) < 30:
        continue
    try:
        mu, sigma, xi = estimate_one_station(model, y, device)
        results.append({
            "station": station,
            "n_obs": int(len(y)),
            "mu_hat": mu,
            "sigma_hat": sigma,
            "log_sigma_hat": float(np.log(sigma)),
            "xi_hat": xi,
        })
    except Exception as exc:
        print("skip", station, exc)

station_gev = pd.DataFrame(results)
station_gev = station_gev.merge(annual_loc, on="station", how="left")
station_gev.to_csv(PROCESSED_DIR / "grid_station_gev_params_with_loc.csv", index=False, encoding="utf-8-sig")
print("estimated stations:", station_gev.shape[0])
station_gev.head()



# =========================
# 結果圖：NN 參數空間分布與簡單比較指標
# =========================
param_summary = station_gev[["mu_hat", "sigma_hat", "log_sigma_hat", "xi_hat"]].describe().T
param_summary.to_csv(PROCESSED_DIR / "nn_parameter_summary.csv", encoding="utf-8-sig")
print(param_summary)

plot_specs = [
    ("mu_hat", "NN estimated mu", "map_nn_mu.png"),
    ("sigma_hat", "NN estimated sigma", "map_nn_sigma.png"),
    ("xi_hat", "NN estimated xi", "map_nn_xi.png"),
]

for col, title, filename in plot_specs:
    fig, ax = plt.subplots(figsize=(6.2, 7.2), dpi=150)
    sc = ax.scatter(station_gev["x_km"], station_gev["y_km"], c=station_gev[col], s=9, cmap="turbo")
    ax.set_title(title)
    ax.set_xlabel("TWD97 Easting (km)")
    ax.set_ylabel("TWD97 Northing (km)")
    fig.colorbar(sc, ax=ax, label=col)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)

# 月極大值全期平均與 NN mu 的關係，檢查估計是否跟資料尺度一致
monthly_station_mean = (
    pivot_clean.mean(axis=0).rename("monthly_max_mean").reset_index()
)
check_df = station_gev.merge(monthly_station_mean, on="station", how="left")
check_df.to_csv(PROCESSED_DIR / "nn_estimate_scale_check.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(5.5, 5.2), dpi=150)
ax.scatter(check_df["monthly_max_mean"], check_df["mu_hat"], s=8, alpha=0.65, color="#756bb1")
ax.set_title("Monthly maximum mean vs NN mu")
ax.set_xlabel("mean monthly maximum temperature")
ax.set_ylabel("NN mu_hat")
fig.tight_layout()
fig.savefig(FIG_DIR / "check_monthly_mean_vs_mu.png")
plt.close(fig)

corr = check_df[["monthly_max_mean", "mu_hat", "sigma_hat", "xi_hat"]].corr(numeric_only=True)
corr.to_csv(PROCESSED_DIR / "result_correlation.csv", encoding="utf-8-sig")
print("figures saved to", FIG_DIR)



# =========================
# 空間平滑：沿用原專案 GP/Kriging 想法，但網格點很多時用抽樣避免過慢
# =========================
def fit_gp_and_predict(df, target_col, grid_df, max_train=800):
    train_df = df[["lon", "lat", "x_km", "y_km", target_col]].dropna().copy()
    if len(train_df) > max_train:
        train_df = train_df.sample(max_train, random_state=123)

    X, X_grid, _ = center_train_test_coordinates(
        train_df[["x_km", "y_km"]].to_numpy(),
        grid_df[["x_km", "y_km"]].to_numpy(),
    )
    y = train_df[target_col].to_numpy()

    kernel = C(1.0, (1e-2, 1e2)) * RBF(
        length_scale=50.0,
        length_scale_bounds=(1.0, 500.0),
    ) + WhiteKernel(
        noise_level=1e-4,
        noise_level_bounds=(1e-8, 1e-1),
    )
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True, random_state=123)
    gp.fit(X, y)
    pred, std = gp.predict(X_grid, return_std=True)
    return pred, std, str(gp.kernel_), len(train_df)

lon_grid = np.linspace(station_gev["lon"].min(), station_gev["lon"].max(), 90)
lat_grid = np.linspace(station_gev["lat"].min(), station_gev["lat"].max(), 90)
grid_raw = np.array([[lon, lat] for lat in lat_grid for lon in lon_grid])
grid_out = add_twd97_km_columns(
    pd.DataFrame(grid_raw, columns=["lon", "lat"])
)

kernel_rows = []
for source_col, out_col in [("mu_hat", "mu"), ("log_sigma_hat", "log_sigma"), ("xi_hat", "xi")]:
    pred, std, kernel_text, n_train = fit_gp_and_predict(station_gev, source_col, grid_out)
    grid_out[out_col] = pred
    grid_out[f"{out_col}_std"] = std
    kernel_rows.append({"target": out_col, "n_train_used": n_train, "kernel": kernel_text})

grid_out["sigma"] = np.exp(grid_out["log_sigma"])
grid_out.to_csv(PROCESSED_DIR / "kriging_grid_gev_params.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(kernel_rows).to_csv(PROCESSED_DIR / "kriging_kernel_summary.csv", index=False, encoding="utf-8-sig")

for col, title, filename in [("mu", "Kriging smoothed mu", "kriging_mu.png"), ("sigma", "Kriging smoothed sigma", "kriging_sigma.png"), ("xi", "Kriging smoothed xi", "kriging_xi.png")]:
    fig, ax = plt.subplots(figsize=(6.2, 7.2), dpi=150)
    sc = ax.scatter(grid_out["x_km"], grid_out["y_km"], c=grid_out[col], s=5, cmap="turbo")
    ax.set_title(title)
    ax.set_xlabel("TWD97 Easting (km)")
    ax.set_ylabel("TWD97 Northing (km)")
    fig.colorbar(sc, ax=ax, label=col)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)

print("kriging grid shape:", grid_out.shape)
grid_out.head()



# =========================
# 輸出清單
# =========================
outputs = sorted([p.relative_to(PROJECT_ROOT).as_posix() for p in PROCESSED_DIR.glob("*.csv")])
figures = sorted([p.relative_to(PROJECT_ROOT).as_posix() for p in FIG_DIR.glob("*.png")])
print("CSV outputs:")
for item in outputs:
    print(" -", item)
print("Figures:")
for item in figures:
    print(" -", item)


# =========================
# 模擬驗證：0.05 度完整矩形網格，並用台灣邊界遮罩區域外格點
# =========================
# 這段的目的不是替代真實資料分析，而是補一組「已知真值」的資料。
# 因為真實最高溫資料沒有真實 GEV 參數可比較，所以這裡用平滑函數先生成
# true_mu / true_sigma / true_xi，再從每個格點的 GEV 分布模擬 45 年年最大值，
# 接著用同一個 NN 估計參數，最後比較 true surface 與 predicted surface。
SIM_SEED = 20260525
SIM_YEARS = 45
SIM_GRID_STEP = 0.05
rng = np.random.default_rng(SIM_SEED)

lon_min = np.floor(annual_loc["lon"].min() / SIM_GRID_STEP) * SIM_GRID_STEP
lon_max = np.ceil(annual_loc["lon"].max() / SIM_GRID_STEP) * SIM_GRID_STEP
lat_min = np.floor(annual_loc["lat"].min() / SIM_GRID_STEP) * SIM_GRID_STEP
lat_max = np.ceil(annual_loc["lat"].max() / SIM_GRID_STEP) * SIM_GRID_STEP

sim_lons = np.arange(lon_min, lon_max + SIM_GRID_STEP / 2, SIM_GRID_STEP)
sim_lats = np.arange(lat_min, lat_max + SIM_GRID_STEP / 2, SIM_GRID_STEP)
sim_grid = pd.DataFrame(
    [(lon, lat) for lat in sim_lats for lon in sim_lons],
    columns=["lon", "lat"],
)
sim_grid = clip_points_to_taiwan(sim_grid)
sim_grid = add_twd97_km_columns(sim_grid)
sim_grid["station"] = (
    "SIM"
    + sim_grid["lon"].map(lambda x: f"{x:.2f}")
    + "_"
    + sim_grid["lat"].map(lambda x: f"{x:.2f}")
)

lon_s = (sim_grid["x_km"] - sim_grid["x_km"].mean()) / sim_grid["x_km"].std()
lat_s = (sim_grid["y_km"] - sim_grid["y_km"].mean()) / sim_grid["y_km"].std()

sim_grid["true_mu"] = (
    30.0
    + 1.6 * lat_s
    - 0.7 * lon_s
    + 0.9 * np.sin(np.pi * lon_s)
    + 0.6 * np.cos(np.pi * lat_s / 1.5)
)
sim_grid["true_log_sigma"] = (
    np.log(1.25)
    + 0.10 * lat_s
    + 0.08 * np.cos(np.pi * lon_s / 1.8)
)
sim_grid["true_sigma"] = np.exp(sim_grid["true_log_sigma"])
sim_grid["true_xi"] = (
    0.08
    + 0.04 * np.sin(np.pi * lat_s / 1.4)
    - 0.025 * np.cos(np.pi * lon_s / 1.7)
)
sim_grid["true_xi"] = sim_grid["true_xi"].clip(-0.20, 0.30)
sim_grid.to_csv(PROCESSED_DIR / "simulation_true_grid_gev_params.csv", index=False, encoding="utf-8-sig")

simulated_columns = {
    row.station: gev.rvs(
        # scipy.stats.genextreme 的 shape 參數 c = -xi。
        c=-row.true_xi,
        loc=row.true_mu,
        scale=row.true_sigma,
        size=SIM_YEARS,
        random_state=rng,
    )
    for row in sim_grid.itertuples(index=False)
}
sim_annual = pd.DataFrame(simulated_columns)
sim_annual.insert(0, "year", np.arange(1980, 1980 + SIM_YEARS))
sim_annual.to_csv(PROCESSED_DIR / "simulation_annual_max_grid.csv", index=False, encoding="utf-8-sig")

sim_results = []
for station in sim_grid["station"]:
    y = sim_annual[station].dropna().to_numpy()
    mu, sigma, shape_c = estimate_one_station(model, y, device)
    sim_results.append(
        {
            "station": station,
            "n_obs": int(len(y)),
            "mu_hat": mu,
            "sigma_hat": sigma,
            "log_sigma_hat": float(np.log(sigma)),
            "shape_c_hat": shape_c,
            "xi_hat": -shape_c,
        }
    )

sim_pred = pd.DataFrame(sim_results).merge(
    sim_grid[
        [
            "station",
            "lon",
            "lat",
            "x_km",
            "y_km",
            "true_mu",
            "true_sigma",
            "true_log_sigma",
            "true_xi",
        ]
    ],
    on="station",
    how="left",
)
sim_pred.to_csv(PROCESSED_DIR / "simulation_station_true_vs_nn.csv", index=False, encoding="utf-8-sig")


def summarize_simulation_error(df):
    rows = []
    for name, true_col, pred_col in [
        ("mu", "true_mu", "mu_hat"),
        ("sigma", "true_sigma", "sigma_hat"),
        ("xi", "true_xi", "xi_hat"),
    ]:
        err = df[pred_col] - df[true_col]
        rows.append(
            {
                "param": name,
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mae": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
                "correlation": float(df[[true_col, pred_col]].corr().iloc[0, 1]),
            }
        )
    return pd.DataFrame(rows)


sim_error = summarize_simulation_error(sim_pred)
sim_error.to_csv(PROCESSED_DIR / "simulation_error_summary.csv", index=False, encoding="utf-8-sig")
print("simulation grid shape:", sim_grid.shape)
print(sim_error)


def plot_three_param_surface(df, cols, filename, suptitle):
    plot_source = df.copy()
    if not {"x_km", "y_km"}.issubset(plot_source.columns):
        if not {"lon", "lat"}.issubset(plot_source.columns):
            raise ValueError("Surface data must contain lon/lat or x_km/y_km.")
        plot_source = add_twd97_km_columns(plot_source)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), dpi=150, constrained_layout=True)
    for ax, col, title in zip(axes, cols, ["mu", "sigma", "xi"]):
        plot_df = clip_points_to_taiwan(
            plot_source[["lon", "lat", "x_km", "y_km", col]].dropna()
        )
        TAIWAN_BOUNDARY_KM.boundary.plot(
            ax=ax,
            color="black",
            linewidth=0.8,
        )
        sc = ax.scatter(
            plot_df["x_km"],
            plot_df["y_km"],
            c=plot_df[col],
            s=8,
            cmap="turbo",
        )
        ax.set_title(title)
        ax.set_xlabel("TWD97 Easting (km)")
        ax.set_ylabel("TWD97 Northing (km)")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.82, label=title)
    fig.suptitle(suptitle)
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


plot_three_param_surface(
    sim_pred,
    ["true_mu", "true_sigma", "true_xi"],
    "simulation_true_surface_1x3.png",
    "Simulation true GEV parameter surfaces",
)
plot_three_param_surface(
    sim_pred,
    ["mu_hat", "sigma_hat", "xi_hat"],
    "simulation_predicted_surface_1x3.png",
    "Simulation NN predicted GEV parameter surfaces",
)

print("simulation figures saved to", FIG_DIR)
