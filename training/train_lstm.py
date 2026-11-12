"""Train loop for the LSTM model with early stopping and checkpointing."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data.dataset import build_loaders
from models.lstm_model import build_lstm
from training.losses import QLIKELoss

CHECKPOINT_DIR = Path("outputs/checkpoints")
LOG_PATH = Path("outputs/training_log_lstm.csv")


def train_lstm(
    epochs: int = 150,
    patience: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = None,
    seed: int | None = 0,
    ckpt_name: str = "lstm_best.pt",
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, scaler, df = build_loaders(batch_size=batch_size)
    input_size = next(iter(train_loader))[0].shape[-1]

    init_log_var = float(np.log(train_loader.dataset.targets.mean()))
    model = build_lstm(input_size=input_size, device=device, init_log_var=init_log_var)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    criterion = QLIKELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    log_rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_losses.append(criterion(pred, y).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:4d} | train QLIKE {train_loss:.6f} | val QLIKE {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / ckpt_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # write training log
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Best val QLIKE: {best_val_loss:.6f}  |  checkpoint: {CHECKPOINT_DIR}/{ckpt_name}")
    return model, train_loader, val_loader, test_loader, scaler, df


def load_best_lstm(input_size: int = 8, device: str = "cpu", ckpt_name: str = "lstm_best.pt") -> object:
    model = build_lstm(input_size=input_size, device=device)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / ckpt_name, map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    train_lstm()
