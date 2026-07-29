"""AIC comparison for GP kernels fitted to simulated GRID-level NN estimates.

This is a separate grid-input experiment.  The GP responses are the neural
network estimates at every simulated grid cell, not the sparse original
stations and not the known parameter truth.  Known truth remains available
for separate validation and is not used to calculate AIC.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor

from directional_kernel_tests import (
    RANDOM_STATE,
    make_gridded_truth,
    optimized_kernel,
    simulate_gridded_nn_estimates,
    metric_coordinates,
)

TARGETS = (
    ("mu", "mu_hat"),
    ("log_sigma", "log_sigma_hat"),
    ("xi", "xi_hat"),
)


SCRIPT_PATH = Path(__file__).resolve()
WINDOW_ROOT = SCRIPT_PATH.parents[1]
OUTPUT_PATH = (
    WINDOW_ROOT
    / "results"
    / "tables"
    / "simulated_grid_gp_aic_kernel_selection.csv"
)


def fit_grid_aic_models() -> pd.DataFrame:
    """Fit RBF and Matérn 0.5 to annual/monthly GRID NN estimates."""
    truth = make_gridded_truth()
    estimates = simulate_gridded_nn_estimates(truth)
    rows: list[dict] = []

    for scenario, data in estimates.items():
        data = data.copy()
        data["log_sigma_hat"] = np.log(
            data["sigma_hat"].to_numpy(dtype=np.float64)
        )
        coordinates = metric_coordinates(data)
        for parameter, response_column in TARGETS:
            response = data[response_column].to_numpy(dtype=np.float64)
            for kernel, nu in (("RBF", None), ("Matern", 0.5)):
                gp = GaussianProcessRegressor(
                    kernel=optimized_kernel(kernel, nu),
                    n_restarts_optimizer=10,
                    normalize_y=True,
                    random_state=RANDOM_STATE,
                )
                gp.fit(coordinates, response)
                log_likelihood = float(gp.log_marginal_likelihood_value_)
                n_hyperparameters = int(gp.kernel_.theta.size)
                aic = 2 * n_hyperparameters - 2 * log_likelihood
                rows.append(
                    {
                        "data_source": "simulated_grid_nn_estimates",
                        "scenario": scenario,
                        "n_grid_cells": len(data),
                        "parameter": parameter,
                        "kernel": kernel,
                        "nu": np.nan if nu is None else nu,
                        "log_marginal_likelihood": log_likelihood,
                        "n_hyperparameters": n_hyperparameters,
                        "AIC": aic,
                        "fitted_kernel": str(gp.kernel_),
                    }
                )

    result = pd.DataFrame(rows)
    result["delta_AIC"] = result.groupby(
        ["scenario", "parameter"]
    )["AIC"].transform(lambda values: values - values.min())
    relative_likelihood = np.exp(-0.5 * result["delta_AIC"])
    result["Akaike_weight"] = relative_likelihood / relative_likelihood.groupby(
        [result["scenario"], result["parameter"]]
    ).transform("sum")
    result["AIC_selected"] = result["delta_AIC"].eq(0.0)
    return result.sort_values(
        ["scenario", "parameter", "AIC"],
        ignore_index=True,
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = fit_grid_aic_models()
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    display_columns = [
        "scenario",
        "parameter",
        "kernel",
        "nu",
        "log_marginal_likelihood",
        "AIC",
        "delta_AIC",
        "Akaike_weight",
        "AIC_selected",
    ]
    print(result[display_columns].to_string(index=False))
    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
