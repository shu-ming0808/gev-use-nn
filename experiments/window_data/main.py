# This file mirrors the executable code in main.ipynb for batch runs.



from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import genextreme as gev
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# 路徑設定
# =========================
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

RAW_DIR = Path(r"C:\Users\User.DESKTOP-4RV84M1\Desktop\論文\fast parameter estimate\fast_parameter_using_NN\data\original_data\觀測_月資料_臺灣_最高溫")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
MODEL_PATH = PROJECT_ROOT / "models" / "best_baseline_model.pth"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("PROJECT_ROOT:", PROJECT_ROOT)
print("RAW_DIR exists:", RAW_DIR.exists())
print("MODEL_PATH exists:", MODEL_PATH.exists())



# =========================
# 讀取與合併每年最高溫月資料
# =========================
def read_one_year_temperature_csv(path: Path) -> pd.DataFrame:
    """讀單一年份最高溫檔，轉成 long format: date, station, lon, lat, max_temp。"""
    year_match = re.search(r"_(\d{4})\.csv$", path.name)
    year = int(year_match.group(1)) if year_match else None

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]

    df["LON"] = pd.to_numeric(df["LON"], errors="coerce").round(2)
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce").round(2)

    month_cols = [c for c in df.columns if re.fullmatch(r"\d{6}", str(c))]
    if year is not None:
        month_cols = [c for c in month_cols if str(c).startswith(str(year))]

    long_df = df.melt(
        id_vars=["LON", "LAT"],
        value_vars=month_cols,
        var_name="yyyymm",
        value_name="max_temp",
    )
    long_df["max_temp"] = pd.to_numeric(long_df["max_temp"], errors="coerce")
    long_df.loc[long_df["max_temp"] <= -90, "max_temp"] = np.nan
    long_df["date"] = pd.to_datetime(long_df["yyyymm"] + "01", format="%Y%m%d")
    long_df["station"] = "G" + long_df["LON"].map(lambda x: f"{x:.2f}") + "_" + long_df["LAT"].map(lambda x: f"{x:.2f}")
    return long_df[["date", "station", "LON", "LAT", "max_temp"]]

files = sorted(RAW_DIR.glob("觀測_月資料_臺灣_最高溫_*.csv"))
print("最高溫年度檔數:", len(files))
print("年份範圍:", files[0].stem[-4:], "到", files[-1].stem[-4:])

monthly_long = pd.concat([read_one_year_temperature_csv(p) for p in files], ignore_index=True)
monthly_long = monthly_long.rename(columns={"LON": "lon", "LAT": "lat"})
monthly_long.to_csv(PROCESSED_DIR / "monthly_long_grid_temperature.csv", index=False, encoding="utf-8-sig")

station_location = monthly_long[["station", "lat", "lon"]].drop_duplicates("station").sort_values("station")
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
sc = ax.scatter(mean_map["lon"], mean_map["lat"], c=mean_map["mean_temp"], s=9, cmap="turbo")
ax.set_title("Mean monthly maximum temperature by grid")
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
fig.colorbar(sc, ax=ax, label="temperature")
fig.tight_layout()
fig.savefig(FIG_DIR / "eda_mean_temperature_map.png")
plt.close(fig)

print("保留網格點:", len(keep_stations))
print("EDA figures saved to", FIG_DIR)



# =========================
# 轉成跟原本測站分析一樣的年最大值資料
# =========================
annual_max = pivot_clean.resample("YE").max()
annual_max.index = annual_max.index.year
annual_max.index.name = "year"

# 年最大值至少要有 30 年，對齊原 NN 訓練 sample size 下限
valid_annual_counts = annual_max.notna().sum()
annual_keep = valid_annual_counts[valid_annual_counts >= 30].index.tolist()
annual_max = annual_max[annual_keep]
annual_max.to_csv(PROCESSED_DIR / "annual_max_grid_temperature.csv", encoding="utf-8-sig")

annual_loc = station_location[station_location["station"].isin(annual_keep)].copy()
annual_loc.to_csv(PROCESSED_DIR / "annual_grid_station_location.csv", index=False, encoding="utf-8-sig")

print("annual_max shape:", annual_max.shape)
annual_max.head()



# =========================
# 使用原本訓練好的 GEV NN 估計每個網格點參數
# =========================
P_SET = np.array([0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999])

class GEVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)

def make_nn_input(y):
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    med = np.median(y)
    q1, q3 = np.quantile(y, [0.25, 0.75])
    iqr = q3 - q1
    if iqr <= 1e-12:
        raise ValueError("IQR too small")
    z = (y - med) / iqr
    q = np.quantile(z, P_SET)
    return q, med, iqr

def estimate_one_station(model, y, device):
    q, med, iqr = make_nn_input(y)
    x = torch.tensor(q, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy().ravel()
    mu_star, delta_star, xi_hat = pred
    sigma_hat = float(np.exp(delta_star) * iqr)
    mu_hat = float(mu_star * iqr + med)
    return mu_hat, sigma_hat, float(xi_hat)

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
for station in annual_max.columns:
    y = annual_max[station].dropna().to_numpy()
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
    sc = ax.scatter(station_gev["lon"], station_gev["lat"], c=station_gev[col], s=9, cmap="turbo")
    ax.set_title(title)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.colorbar(sc, ax=ax, label=col)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)

# 年最大值全區平均與 NN mu 的關係，檢查估計是否跟資料尺度一致
annual_station_mean = annual_max.mean(axis=0).rename("annual_max_mean").reset_index()
check_df = station_gev.merge(annual_station_mean, on="station", how="left")
check_df.to_csv(PROCESSED_DIR / "nn_estimate_scale_check.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(5.5, 5.2), dpi=150)
ax.scatter(check_df["annual_max_mean"], check_df["mu_hat"], s=8, alpha=0.65, color="#756bb1")
ax.set_title("Annual maximum mean vs NN mu")
ax.set_xlabel("mean annual maximum temperature")
ax.set_ylabel("NN mu_hat")
fig.tight_layout()
fig.savefig(FIG_DIR / "check_annual_mean_vs_mu.png")
plt.close(fig)

corr = check_df[["annual_max_mean", "mu_hat", "sigma_hat", "xi_hat"]].corr(numeric_only=True)
corr.to_csv(PROCESSED_DIR / "result_correlation.csv", encoding="utf-8-sig")
print("figures saved to", FIG_DIR)



# =========================
# 空間平滑：沿用原專案 GP/Kriging 想法，但網格點很多時用抽樣避免過慢
# =========================
def fit_gp_and_predict(df, target_col, grid_raw, max_train=800):
    train_df = df[["lon", "lat", target_col]].dropna().copy()
    if len(train_df) > max_train:
        train_df = train_df.sample(max_train, random_state=123)

    X_raw = train_df[["lon", "lat"]].to_numpy()
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X = (X_raw - X_mean) / X_std
    X_grid = (grid_raw - X_mean) / X_std
    y = train_df[target_col].to_numpy()

    kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10)) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True, random_state=123)
    gp.fit(X, y)
    pred, std = gp.predict(X_grid, return_std=True)
    return pred, std, str(gp.kernel_), len(train_df)

lon_grid = np.linspace(station_gev["lon"].min(), station_gev["lon"].max(), 90)
lat_grid = np.linspace(station_gev["lat"].min(), station_gev["lat"].max(), 90)
grid_raw = np.array([[lon, lat] for lat in lat_grid for lon in lon_grid])
grid_out = pd.DataFrame(grid_raw, columns=["lon", "lat"])

kernel_rows = []
for source_col, out_col in [("mu_hat", "mu"), ("log_sigma_hat", "log_sigma"), ("xi_hat", "xi")]:
    pred, std, kernel_text, n_train = fit_gp_and_predict(station_gev, source_col, grid_raw)
    grid_out[out_col] = pred
    grid_out[f"{out_col}_std"] = std
    kernel_rows.append({"target": out_col, "n_train_used": n_train, "kernel": kernel_text})

grid_out["sigma"] = np.exp(grid_out["log_sigma"])
grid_out.to_csv(PROCESSED_DIR / "kriging_grid_gev_params.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(kernel_rows).to_csv(PROCESSED_DIR / "kriging_kernel_summary.csv", index=False, encoding="utf-8-sig")

for col, title, filename in [("mu", "Kriging smoothed mu", "kriging_mu.png"), ("sigma", "Kriging smoothed sigma", "kriging_sigma.png"), ("xi", "Kriging smoothed xi", "kriging_xi.png")]:
    fig, ax = plt.subplots(figsize=(6.2, 7.2), dpi=150)
    sc = ax.scatter(grid_out["lon"], grid_out["lat"], c=grid_out[col], s=5, cmap="turbo")
    ax.set_title(title)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
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
# 模擬驗證：0.5 度網格真實 surface vs NN 預測 surface
# =========================
# 這段的目的不是替代真實資料分析，而是補一組「已知真值」的資料。
# 因為真實最高溫資料沒有真實 GEV 參數可比較，所以這裡用平滑函數先生成
# true_mu / true_sigma / true_xi，再從每個格點的 GEV 分布模擬 45 年年最大值，
# 接著用同一個 NN 估計參數，最後比較 true surface 與 predicted surface。
SIM_SEED = 20260525
SIM_YEARS = 45
SIM_GRID_STEP = 0.5
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
sim_grid["station"] = (
    "SIM"
    + sim_grid["lon"].map(lambda x: f"{x:.1f}")
    + "_"
    + sim_grid["lat"].map(lambda x: f"{x:.1f}")
)

lon_s = (sim_grid["lon"] - sim_grid["lon"].mean()) / sim_grid["lon"].std()
lat_s = (sim_grid["lat"] - sim_grid["lat"].mean()) / sim_grid["lat"].std()

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

sim_annual = pd.DataFrame({"year": np.arange(1980, 1980 + SIM_YEARS)})
for row in sim_grid.itertuples(index=False):
    # scipy.stats.genextreme 的 shape 參數 c = -xi。
    sim_annual[row.station] = gev.rvs(
        c=-row.true_xi,
        loc=row.true_mu,
        scale=row.true_sigma,
        size=SIM_YEARS,
        random_state=rng,
    )
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
    sim_grid[["station", "lon", "lat", "true_mu", "true_sigma", "true_log_sigma", "true_xi"]],
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
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), dpi=150, constrained_layout=True)
    for ax, col, title in zip(axes, cols, ["mu", "sigma", "xi"]):
        sc = ax.scatter(df["lon"], df["lat"], c=df[col], s=65, cmap="turbo")
        ax.set_title(title)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
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
