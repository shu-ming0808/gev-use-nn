import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARAM_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "station_gev_params.csv")
LOC_PATH = os.path.join(PROJECT_ROOT, "data", "original_data", "station_location.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "station_gev_params_with_loc.csv")

params = pd.read_csv(PARAM_PATH)
locs = pd.read_csv(LOC_PATH)

params["station"] = params["station"].astype(str).str.strip().str.upper()
locs["station"] = locs["station"].astype(str).str.strip().str.upper()

# 修正常見命名差異
params["station"] = params["station"].replace({
    "SUAO": "SU-AO",
    "SUN MOON LAKE": "SUNMOONLAKE"
})

locs["station"] = locs["station"].replace({
    "SUAO": "SU-AO",
    "SUN MOON LAKE": "SUNMOONLAKE"
})

df = pd.merge(params, locs, on="station", how="left")

missing = df[df["lat"].isna() | df["lon"].isna()]

if len(missing) > 0:
    print("Missing coordinates:")
    print(missing["station"].tolist())
else:
    print("All stations matched successfully.")

df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("Saved:", OUT_PATH)
print(df.head())