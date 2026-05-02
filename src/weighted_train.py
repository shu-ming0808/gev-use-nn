import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from simulate_data import generate_dataset_author_style, split_train_valid


# ----------------------------
# Gradient utilities
# ----------------------------
def grad_norm_of_task(loss, params):
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True
    )

    total = 0.0
    for g in grads:
        if g is not None:
            total += torch.sum(g.detach() ** 2).item()

    return math.sqrt(total + 1e-12)


def compute_task_grad_norms(model, losses):
    shared_params = [p for p in model.trunk.parameters() if p.requires_grad]

    g_mu = grad_norm_of_task(losses["mu"], shared_params)
    g_delta = grad_norm_of_task(losses["delta"], shared_params)
    g_c = grad_norm_of_task(losses["c"], shared_params)

    return {
        "mu": g_mu,
        "delta": g_delta,
        "c": g_c,
    }


# ----------------------------
# Dynamic task weighter
# ----------------------------
class DynamicTaskWeighter:
    def __init__(self, alpha=1.0, ema_beta=0.9, min_weight=0.2, max_weight=5.0):
        self.alpha = alpha
        self.ema_beta = ema_beta
        self.min_weight = min_weight
        self.max_weight = max_weight

        self.ema_grad = {
            "mu": None,
            "delta": None,
            "c": None,
        }

        self.weights = {
            "mu": 1.0,
            "delta": 1.0,
            "c": 1.0,
        }

    def update_ema(self, grad_norms):
        smoothed = {}

        for k, g in grad_norms.items():
            if self.ema_grad[k] is None:
                self.ema_grad[k] = g
            else:
                self.ema_grad[k] = (
                    self.ema_beta * self.ema_grad[k]
                    + (1.0 - self.ema_beta) * g
                )
            smoothed[k] = float(self.ema_grad[k])

        return smoothed

    def compute_new_weights(self, grad_norms):
        smoothed = self.update_ema(grad_norms)

        raw = {}
        for k, g in smoothed.items():
            raw[k] = 1.0 / ((g + 1e-8) ** self.alpha)

        total_raw = sum(raw.values())
        n = len(raw)

        new_weights = {}
        for k, v in raw.items():
            w = (v / total_raw) * n
            w = max(self.min_weight, min(self.max_weight, w))
            new_weights[k] = float(w)

        self.weights = new_weights
        return new_weights

    def get_weighted_total_loss(self, losses):
        return (
            self.weights["mu"] * losses["mu"]
            + self.weights["delta"] * losses["delta"]
            + self.weights["c"] * losses["c"]
        )


# ----------------------------
# Dataset
# ----------------------------
class GEVQuantileDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ----------------------------
# Model（照論文 architecture）
# ----------------------------
class GEVNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
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
        )

        self.out = nn.Linear(128, 3)

    def forward(self, x):
        h = self.trunk(x)
        out = self.out(h)
        return {
            "mu": out[:, 0],
            "delta": out[:, 1],
            "c": out[:, 2],
        }


# ----------------------------
# Loss
# ----------------------------
def task_losses(pred, target):
    loss_mu = torch.mean((pred["mu"] - target[:, 0]) ** 2)
    loss_delta = torch.mean((pred["delta"] - target[:, 1]) ** 2)
    loss_c = torch.mean((pred["c"] - target[:, 2]) ** 2)
    return {
        "mu": loss_mu,
        "delta": loss_delta,
        "c": loss_c,
    }


# ----------------------------
# Validation
# ----------------------------
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_mu = 0.0
    total_delta = 0.0
    total_c = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            losses = task_losses(pred, yb)

            loss = losses["mu"] + losses["delta"] + losses["c"]

            total_loss += loss.item()
            total_mu += losses["mu"].item()
            total_delta += losses["delta"].item()
            total_c += losses["c"].item()

    n_batches = len(loader)

    return {
        "val_total": total_loss / n_batches,
        "val_mu": total_mu / n_batches,
        "val_delta": total_delta / n_batches,
        "val_c": total_c / n_batches,
    }


# ----------------------------
# Train weighted model
# ----------------------------
def train_weighted():
    X, Y, n_train, n_valid, _ = generate_dataset_author_style(seed=111)
    X_train, Y_train, X_valid, Y_valid = split_train_valid(X, Y, n_train, n_valid)

    train_loader = DataLoader(
        GEVQuantileDataset(X_train, Y_train),
        batch_size=128,
        shuffle=True
    )

    valid_loader = DataLoader(
        GEVQuantileDataset(X_valid, Y_valid),
        batch_size=128,
        shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GEVNet().to(device)

    # 與 baseline 盡量一致
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3
    )

    weighter = DynamicTaskWeighter(
        alpha=1.0,
        ema_beta=0.9,
        min_weight=0.2,
        max_weight=5.0
    )

    best_val_loss = float("inf")
    patience = 8
    patience_counter = 0
    epochs = 150

    history = {
        "epoch": [],
        "train_total": [],
        "train_mu": [],
        "train_delta": [],
        "train_c": [],
        "val_total": [],
        "val_mu": [],
        "val_delta": [],
        "val_c": [],
        "g_mu": [],
        "g_delta": [],
        "g_c": [],
        "w_mu": [],
        "w_delta": [],
        "w_c": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()

        sum_total = 0.0
        sum_mu = 0.0
        sum_delta = 0.0
        sum_c = 0.0

        sum_g_mu = 0.0
        sum_g_delta = 0.0
        sum_g_c = 0.0

        sum_w_mu = 0.0
        sum_w_delta = 0.0
        sum_w_c = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            losses = task_losses(pred, yb)

            grad_norms = compute_task_grad_norms(model, losses)

            weighter.compute_new_weights({
                "mu": grad_norms["mu"],
                "delta": grad_norms["delta"],
                "c": grad_norms["c"],
            })

            weighted_total = weighter.get_weighted_total_loss(losses)

            weighted_total.backward()
            optimizer.step()

            sum_total += weighted_total.item()
            sum_mu += losses["mu"].item()
            sum_delta += losses["delta"].item()
            sum_c += losses["c"].item()

            sum_g_mu += grad_norms["mu"]
            sum_g_delta += grad_norms["delta"]
            sum_g_c += grad_norms["c"]

            sum_w_mu += weighter.weights["mu"]
            sum_w_delta += weighter.weights["delta"]
            sum_w_c += weighter.weights["c"]

        n_batches = len(train_loader)

        train_stats = {
            "train_total": sum_total / n_batches,
            "train_mu": sum_mu / n_batches,
            "train_delta": sum_delta / n_batches,
            "train_c": sum_c / n_batches,
            "g_mu": sum_g_mu / n_batches,
            "g_delta": sum_g_delta / n_batches,
            "g_c": sum_g_c / n_batches,
            "w_mu": sum_w_mu / n_batches,
            "w_delta": sum_w_delta / n_batches,
            "w_c": sum_w_c / n_batches,
        }

        val_stats = evaluate(model, valid_loader, device)
        scheduler.step(val_stats["val_total"])

        history["epoch"].append(epoch)
        for k, v in train_stats.items():
            history[k].append(v)
        for k, v in val_stats.items():
            history[k].append(v)

        print(
            f"Epoch {epoch:03d} | "
            f"Train={train_stats['train_total']:.6f} | "
            f"Val={val_stats['val_total']:.6f} | "
            f"w=({train_stats['w_mu']:.3f}, {train_stats['w_delta']:.3f}, {train_stats['w_c']:.3f})"
        )

        if val_stats["val_total"] < best_val_loss:
            best_val_loss = val_stats["val_total"]
            patience_counter = 0
            torch.save(model.state_dict(), "best_weighted_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    summary_df = pd.DataFrame(history)
    summary_df.to_csv("weighted_history.csv", index=False)
    return summary_df


if __name__ == "__main__":
    summary_df = train_weighted()
    print(summary_df.tail())