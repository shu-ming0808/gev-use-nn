"""Shared neural-network inference utilities for GEV parameter estimation.

This module contains only reusable model and inverse-transformation logic.
Dataset-specific pipelines (for example, the TCCIP grid workflow) should
prepare their own time series and call :func:`estimate_one`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from project_paths import MODEL_DIR


P_SET = np.array(
    [
        0.0001,
        0.001,
        0.01,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        0.99,
        0.999,
        0.9999,
    ],
    dtype=np.float64,
)


class GEVNet(nn.Module):
    """Network architecture used by the baseline pretrained weights."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def robust_standardize(
    values: Iterable[float],
) -> tuple[np.ndarray, float, float]:
    """Standardize one sample with its median and interquartile range."""

    y = np.asarray(values, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        raise ValueError("The input series contains no finite observations.")

    median = float(np.median(y))
    q1, q3 = np.quantile(y, [0.25, 0.75])
    iqr = float(q3 - q1)
    if iqr <= 1e-12:
        raise ValueError("The input series has an IQR that is too small.")

    return (y - median) / iqr, median, iqr


def make_input(
    values: Iterable[float],
) -> tuple[np.ndarray, float, float]:
    """Convert one sample to the 11 empirical quantiles used by the NN."""

    z, median, iqr = robust_standardize(values)
    quantiles = np.quantile(z, P_SET).astype(np.float32)
    return quantiles, median, iqr


def invert_prediction(
    prediction: Iterable[float],
    standardized_values: np.ndarray,
    median: float,
    iqr: float,
) -> tuple[float, float, float]:
    """Convert standardized NN outputs to ``mu``, ``sigma`` and SciPy ``c``."""

    mu_star, delta_star, shape_c = [
        float(value) for value in np.asarray(prediction).reshape(-1)
    ]

    positive_part = float(np.exp(np.clip(delta_star, -30.0, 30.0)))
    if shape_c > 0.0:
        sigma_star = positive_part + shape_c * (
            float(np.max(standardized_values)) - mu_star
        )
    else:
        sigma_star = positive_part + shape_c * (
            float(np.min(standardized_values)) - mu_star
        )

    sigma_star = max(float(sigma_star), 1e-12)
    mu = float(mu_star * iqr + median)
    sigma = float(sigma_star * iqr)
    return mu, sigma, shape_c


def estimate_one(
    model: nn.Module,
    values: Iterable[float],
    device: str | torch.device,
) -> tuple[float, float, float]:
    """Estimate one time series with the pretrained NN.

    The third returned value is SciPy's shape convention ``c``.  The
    conventional GEV shape parameter used elsewhere in the project is
    ``xi = -c`` when that sign convention is required.
    """

    y = np.asarray(values, dtype=np.float64)
    y = y[np.isfinite(y)]
    quantiles, median, iqr = make_input(y)
    standardized = (y - median) / iqr

    x = torch.as_tensor(
        quantiles[None, :],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        prediction = model(x).detach().cpu().numpy()[0]

    return invert_prediction(prediction, standardized, median, iqr)


def load_baseline_model(
    model_path: str | Path | None = None,
    device: str | None = None,
) -> tuple[GEVNet, str]:
    """Load the baseline model using the canonical repository path."""

    selected_device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    weights_path = Path(model_path) if model_path else (
        MODEL_DIR / "best_baseline_model.pth"
    )

    model = GEVNet().to(selected_device)
    try:
        state = torch.load(
            weights_path,
            map_location=selected_device,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(weights_path, map_location=selected_device)
    model.load_state_dict(state)
    model.eval()
    return model, selected_device
