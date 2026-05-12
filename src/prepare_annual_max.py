import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(PROJECT_ROOT, "data", "original_data", "pivot_25stations.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "annual_max_25stations.csv")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")

# first column is Date
date_col = df.columns[0]
df[date_col] = pd.to_datetime(df[date_col])

df = df.set_index(date_col)

# keep data after 1980
df = df[df.index >= "1980-01-01"]

# annual block maxima
annual_max = df.resample("YE").max()

# save
annual_max.to_csv(OUT_PATH, encoding="utf-8-sig")

print("Saved:", OUT_PATH)
print("Shape:", annual_max.shape)
print(annual_max)  