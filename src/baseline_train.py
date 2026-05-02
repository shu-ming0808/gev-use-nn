import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from simulate_data import generate_dataset_author_style, split_train_valid


class GEVQuantileDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


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
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)


def compute_loss(pred, target):
    total = torch.mean((pred - target) ** 2)
    mu = torch.mean((pred[:, 0] - target[:, 0]) ** 2)
    delta = torch.mean((pred[:, 1] - target[:, 1]) ** 2)
    c = torch.mean((pred[:, 2] - target[:, 2]) ** 2)
    return total, mu, delta, c


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
            loss, mu, delta, c = compute_loss(pred, yb)

            total_loss += loss.item()
            total_mu += mu.item()
            total_delta += delta.item()
            total_c += c.item()

    n_batches = len(loader)
    return {
        "val_total": total_loss / n_batches,
        "val_mu": total_mu / n_batches,
        "val_delta": total_delta / n_batches,
        "val_c": total_c / n_batches,
    }


def train():
    X, Y, n_train, n_valid, _ = generate_dataset_author_style(seed=111)
    X_train, Y_train, X_valid, Y_valid = split_train_valid(X, Y, n_train, n_valid)

    train_loader = DataLoader(GEVQuantileDataset(X_train, Y_train), batch_size=128, shuffle=True)
    valid_loader = DataLoader(GEVQuantileDataset(X_valid, Y_valid), batch_size=128, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GEVNet().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
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
    }

    for epoch in range(1, epochs + 1):
        model.train()

        train_total = 0.0
        train_mu = 0.0
        train_delta = 0.0
        train_c = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            pred = model(xb)
            loss, mu, delta, c = compute_loss(pred, yb)

            loss.backward()
            optimizer.step()

            train_total += loss.item()
            train_mu += mu.item()
            train_delta += delta.item()
            train_c += c.item()

        n_batches = len(train_loader)
        train_stats = {
            "train_total": train_total / n_batches,
            "train_mu": train_mu / n_batches,
            "train_delta": train_delta / n_batches,
            "train_c": train_c / n_batches,
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
            f"Val={val_stats['val_total']:.6f}"
        )

        if val_stats["val_total"] < best_val_loss:
            best_val_loss = val_stats["val_total"]
            patience_counter = 0
            torch.save(model.state_dict(), "best_baseline_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    summary_df = pd.DataFrame(history)
    summary_df.to_csv("baseline_history.csv", index=False)
    return summary_df


if __name__ == "__main__":
    summary_df = train()
    print(summary_df.tail())