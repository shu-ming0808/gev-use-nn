import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import genextreme as gev
from torch.utils.data import DataLoader, Dataset

from simulate_data import P_SET, robust_standardize, split_train_valid


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMULATED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "simulated")
CONSTRAINT_DATA_PATH = os.path.join(
    SIMULATED_DATA_DIR,
    "gev_train_valid_constraint_seed111.npz",
)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_constraint_penalty_model.pth")
HISTORY_PATH = os.path.join(PROJECT_ROOT, "constraint_penalty_history.csv")
GRAD_CLIP_NORM = 5.0
DELTA_CLAMP = 30.0
DEFAULT_SAFETY_MARGIN = 0.01
DEFAULT_DELTA_MARGIN = 0.0


def set_seed(seed: int = 111) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_one_observation_with_extrema(mu: float, sigma: float, c: float, n: int, rng):
    sample = gev.rvs(c=c, loc=mu, scale=sigma, size=int(n), random_state=rng)
    z, sample_median, sample_iqr = robust_standardize(sample)

    x_feat = np.percentile(z, P_SET * 100).astype(np.float32)

    sc_loc = (mu - sample_median) / sample_iqr
    sc_scale = sigma / sample_iqr
    z_min = float(np.min(z))
    z_max = float(np.max(z))

    if c > np.finfo(float).eps:
        inside = sc_scale - c * (z_max - sc_loc)
    else:
        inside = sc_scale - c * (z_min - sc_loc)

    inside = max(float(inside), 1e-12)
    delta = np.log(inside)

    y_target = np.array([sc_loc, delta, c], dtype=np.float32)
    extrema = np.array([z_min, z_max], dtype=np.float32)
    return x_feat, y_target, extrema


def generate_constraint_dataset(seed: int = 111):
    """
    Generate the same author-style dataset as simulate_data.py, but also store
    standardized sample min/max for DeepExtrema-style xi bound penalties.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    n_total = 340000
    n_train = 300000
    n_valid = 40000

    n_set = np.rint(
        10 ** np.linspace(
            start=math.log(30, 10),
            stop=math.log(1000, 10),
            num=5,
        )
    ).astype(int)

    over_all_factor = int(np.rint(n_total / len(n_set)))

    shape_vals = rng.uniform(-1.0, 0.4, over_all_factor)
    scale_vals = 10 ** rng.uniform(np.log10(0.1), np.log10(40), over_all_factor)
    loc_vals = rng.uniform(1.0, 50.0, over_all_factor)

    loc_values = np.hstack([loc_vals] * len(n_set))
    scale_values = np.hstack([scale_vals] * len(n_set))
    shape_values = np.hstack([shape_vals] * len(n_set))
    n_values = np.repeat(n_set, over_all_factor)

    x_list = []
    y_list = []
    extrema_list = []

    for mu, sigma, c, sample_size in zip(loc_values, scale_values, shape_values, n_values):
        x_feat, y_target, extrema = generate_one_observation_with_extrema(
            mu=mu,
            sigma=sigma,
            c=c,
            n=int(sample_size),
            rng=rng,
        )
        x_list.append(x_feat)
        y_list.append(y_target)
        extrema_list.append(extrema)

    x = np.stack(x_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    extrema = np.stack(extrema_list).astype(np.float32)

    return x, y, extrema, n_train, n_valid, n_set


def save_constraint_dataset(
    x: np.ndarray,
    y: np.ndarray,
    extrema: np.ndarray,
    n_train: int,
    n_valid: int,
    n_set: np.ndarray,
    path: str = CONSTRAINT_DATA_PATH,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        X=x,
        Y=y,
        extrema=extrema,
        n_train=np.array(n_train, dtype=np.int64),
        n_valid=np.array(n_valid, dtype=np.int64),
        N_set=n_set,
    )


def load_constraint_dataset(path: str = CONSTRAINT_DATA_PATH):
    data = np.load(path)
    return (
        data["X"],
        data["Y"],
        data["extrema"],
        int(data["n_train"]),
        int(data["n_valid"]),
        data["N_set"],
    )


def load_or_generate_constraint_dataset(seed: int = 111, path: str = CONSTRAINT_DATA_PATH):
    if os.path.exists(path):
        print(f"Loading constraint dataset: {path}")
        return load_constraint_dataset(path)

    print(f"Generating constraint dataset and saving to: {path}")
    x, y, extrema, n_train, n_valid, n_set = generate_constraint_dataset(seed=seed)
    save_constraint_dataset(x, y, extrema, n_train, n_valid, n_set, path=path)
    return x, y, extrema, n_train, n_valid, n_set


class GEVConstraintDataset(Dataset):
    def __init__(self, x, y, extrema):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.extrema = torch.from_numpy(extrema).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.extrema[idx]


class GEVNet(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        return self.net(x)


def reconstruct_standardized_sigma(mu_star, delta, c, z_min, z_max, eps=1e-8):
    positive_part = torch.exp(torch.clamp(delta, min=-DELTA_CLAMP, max=DELTA_CLAMP))
    sigma_if_c_positive = positive_part + c * (z_max - mu_star)
    sigma_if_c_nonpositive = positive_part + c * (z_min - mu_star)
    sigma_star = torch.where(c > 0.0, sigma_if_c_positive, sigma_if_c_nonpositive)
    return torch.clamp(sigma_star, min=eps)


def xi_bounds_from_support(mu_star, sigma_star, z_min, z_max, tau=0.0, eps=1e-6):
    upper_denominator = torch.clamp(mu_star - z_min, min=eps)
    lower_denominator = torch.clamp(z_max - mu_star, min=eps)

    xi_lower = -(sigma_star / lower_denominator) * (1.0 + tau)
    xi_upper = (sigma_star / upper_denominator) * (1.0 + tau)
    return xi_lower, xi_upper


def compute_constraint_penalty(
    pred,
    extrema,
    tau=0.0,
    safety_margin=DEFAULT_SAFETY_MARGIN,
    delta_margin=DEFAULT_DELTA_MARGIN,
):
    mu_star = pred[:, 0]
    delta = pred[:, 1]
    c = pred[:, 2]
    z_min = extrema[:, 0]
    z_max = extrema[:, 1]

    xi = -c
    sigma_star = reconstruct_standardized_sigma(mu_star, delta, c, z_min, z_max)
    xi_lower, xi_upper = xi_bounds_from_support(
        mu_star=mu_star,
        sigma_star=sigma_star,
        z_min=z_min,
        z_max=z_max,
        tau=tau,
    )
    effective_margin = torch.clamp(
        torch.as_tensor(safety_margin, device=xi.device, dtype=xi.dtype),
        min=0.0,
    )
    xi_lower_safe = xi_lower + effective_margin
    xi_upper_safe = xi_upper - effective_margin

    lower_violation = F.relu(xi_lower_safe - xi)
    upper_violation = F.relu(xi - xi_upper_safe)

    if delta_margin > 0:
        # In the Fast reparameterization, exp(delta) is the positive slack
        # from the support boundary. Requiring exp(delta) >= delta_margin
        # is equivalent to delta >= log(delta_margin).
        delta_threshold = torch.log(
            torch.as_tensor(delta_margin, device=delta.device, dtype=delta.dtype)
        )
        delta_violation = F.relu(delta_threshold - delta)
    else:
        delta_violation = torch.zeros_like(delta)

    penalty = torch.mean(
        lower_violation ** 2
        + upper_violation ** 2
        + delta_violation ** 2
    )

    with torch.no_grad():
        violation_rate = torch.mean(
            (
                (xi < xi_lower_safe)
                | (xi > xi_upper_safe)
                | (delta_violation > 0)
            ).float()
        )
        mean_width = torch.mean(xi_upper_safe - xi_lower_safe)

    return penalty, violation_rate, mean_width


def compute_loss(
    pred,
    target,
    extrema,
    penalty_weight=0.1,
    tau=0.0,
    safety_margin=DEFAULT_SAFETY_MARGIN,
    delta_margin=DEFAULT_DELTA_MARGIN,
):
    loss_mu = torch.mean((pred[:, 0] - target[:, 0]) ** 2)
    loss_delta = torch.mean((pred[:, 1] - target[:, 1]) ** 2)
    loss_c = torch.mean((pred[:, 2] - target[:, 2]) ** 2)
    fast_loss = loss_mu + loss_delta + loss_c

    penalty, violation_rate, mean_width = compute_constraint_penalty(
        pred=pred,
        extrema=extrema,
        tau=tau,
        safety_margin=safety_margin,
        delta_margin=delta_margin,
    )
    total_loss = fast_loss + penalty_weight * penalty

    return {
        "total": total_loss,
        "fast": fast_loss,
        "mu": loss_mu,
        "delta": loss_delta,
        "c": loss_c,
        "penalty": penalty,
        "violation_rate": violation_rate,
        "mean_width": mean_width,
    }


def split_train_valid_with_extrema(x, y, extrema, n_train=300000, n_valid=40000):
    x_train, y_train, x_valid, y_valid = split_train_valid(x, y, n_train, n_valid)
    extrema_train = extrema[:n_train]
    extrema_valid = extrema[n_train:n_train + n_valid]
    return x_train, y_train, extrema_train, x_valid, y_valid, extrema_valid


def evaluate(
    model,
    loader,
    device,
    penalty_weight=0.1,
    tau=0.0,
    safety_margin=DEFAULT_SAFETY_MARGIN,
    delta_margin=DEFAULT_DELTA_MARGIN,
):
    model.eval()

    sums = {
        "val_total": 0.0,
        "val_fast": 0.0,
        "val_mu": 0.0,
        "val_delta": 0.0,
        "val_c": 0.0,
        "val_penalty": 0.0,
        "val_violation_rate": 0.0,
        "val_mean_width": 0.0,
    }

    with torch.no_grad():
        for xb, yb, eb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            eb = eb.to(device)
            pred = model(xb)
            losses = compute_loss(
                pred,
                yb,
                eb,
                penalty_weight=penalty_weight,
                tau=tau,
                safety_margin=safety_margin,
                delta_margin=delta_margin,
            )

            for key in sums:
                loss_key = key.replace("val_", "")
                sums[key] += float(losses[loss_key].item())

    n_batches = len(loader)
    return {key: value / n_batches for key, value in sums.items()}


def train_constraint_penalty(
    seed=111,
    penalty_weight=0.1,
    tau=0.0,
    safety_margin=DEFAULT_SAFETY_MARGIN,
    delta_margin=DEFAULT_DELTA_MARGIN,
    batch_size=128,
    epochs=150,
    patience=8,
    model_path=MODEL_PATH,
    history_path=HISTORY_PATH,
):
    set_seed(seed)
    x, y, extrema, n_train, n_valid, _ = load_or_generate_constraint_dataset(seed=seed)
    x_train, y_train, extrema_train, x_valid, y_valid, extrema_valid = (
        split_train_valid_with_extrema(x, y, extrema, n_train, n_valid)
    )

    train_loader = DataLoader(
        GEVConstraintDataset(x_train, y_train, extrema_train),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        GEVConstraintDataset(x_valid, y_valid, extrema_valid),
        batch_size=batch_size,
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GEVNet().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3,
    )

    history = {
        "epoch": [],
        "train_total": [],
        "train_fast": [],
        "train_mu": [],
        "train_delta": [],
        "train_c": [],
        "train_penalty": [],
        "train_violation_rate": [],
        "train_mean_width": [],
        "val_total": [],
        "val_fast": [],
        "val_mu": [],
        "val_delta": [],
        "val_c": [],
        "val_penalty": [],
        "val_violation_rate": [],
        "val_mean_width": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running = {
            "train_total": 0.0,
            "train_fast": 0.0,
            "train_mu": 0.0,
            "train_delta": 0.0,
            "train_c": 0.0,
            "train_penalty": 0.0,
            "train_violation_rate": 0.0,
            "train_mean_width": 0.0,
        }

        for xb, yb, eb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            eb = eb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            losses = compute_loss(
                pred,
                yb,
                eb,
                penalty_weight=penalty_weight,
                tau=tau,
                safety_margin=safety_margin,
                delta_margin=delta_margin,
            )
            if not torch.isfinite(losses["total"]):
                continue

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            for key in running:
                loss_key = key.replace("train_", "")
                running[key] += float(losses[loss_key].item())

        n_batches = len(train_loader)
        train_stats = {key: value / n_batches for key, value in running.items()}
        val_stats = evaluate(
            model,
            valid_loader,
            device,
            penalty_weight=penalty_weight,
            tau=tau,
            safety_margin=safety_margin,
            delta_margin=delta_margin,
        )
        scheduler.step(val_stats["val_total"])

        history["epoch"].append(epoch)
        for key, value in train_stats.items():
            history[key].append(value)
        for key, value in val_stats.items():
            history[key].append(value)

        print(
            f"Epoch {epoch:03d} | "
            f"Train={train_stats['train_total']:.6f} | "
            f"Val={val_stats['val_total']:.6f} | "
            f"Penalty={val_stats['val_penalty']:.6f} | "
            f"Viol={val_stats['val_violation_rate']:.4f}"
        )

        if val_stats["val_total"] < best_val_loss:
            best_val_loss = val_stats["val_total"]
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    summary_df = pd.DataFrame(history)
    summary_df.to_csv(history_path, index=False)
    return summary_df


if __name__ == "__main__":
    summary = train_constraint_penalty()
    print(summary.tail())
