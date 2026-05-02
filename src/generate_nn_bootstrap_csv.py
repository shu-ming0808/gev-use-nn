from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from bootstrap_nn import load_baseline_model, bootstrap_gev_nn

model, device = load_baseline_model("best_baseline_model.pth")

meta = pd.read_csv("sample_index.csv")

def process_row(i, row):
    x = np.load(row["file_path"])

    res = bootstrap_gev_nn(
        y=x,
        model=model,
        B=1000,
        alpha=0.05,
        seed=123 + i,
        device=device,
    )

    ci_df = res["ci"].set_index("param")

    return {
        "panel": row["panel"],
        "true_value": row["true_value"],
        "rep": row["rep"],
        "nn_width_mu": ci_df.loc["mu", "width"],
        "nn_width_sigma": ci_df.loc["sigma", "width"],
        "nn_width_xi": ci_df.loc["c", "width"],
    }

# 🔥 用所有核心
results = Parallel(n_jobs=-1)(
    delayed(process_row)(i, row)
    for i, row in meta.iterrows()
)

nn_df = pd.DataFrame(results)
nn_df.to_csv("nn_bootstrap_results.csv", index=False)

print("Done")