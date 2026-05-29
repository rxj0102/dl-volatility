"""Train the Temporal Fusion Transformer using pytorch-forecasting + Lightning."""

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from data.dataset import build_tft_dataframes
from models.tft_model import build_tft_datasets, build_tft_model, get_tft_trainer

LOG_PATH = Path("outputs/training_log_tft.csv")


def train_tft(max_epochs: int = 100, patience: int = 15, batch_size: int = 64):
    scaled_df, train_mask, val_mask, test_mask, scaler = build_tft_dataframes()

    training_ds, validation_ds = build_tft_datasets(scaled_df, train_mask, val_mask)

    train_loader = training_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    tft = build_tft_model(training_ds)
    trainer = get_tft_trainer(max_epochs=max_epochs, patience=patience)

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = trainer.checkpoint_callback.best_model_path
    print(f"Best TFT checkpoint: {best_path}")

    # persist log
    if trainer.logger and hasattr(trainer.logger, "experiment"):
        try:
            metrics = pd.DataFrame(trainer.logger.experiment.metrics)
            metrics.to_csv(LOG_PATH, index=False)
        except Exception:
            pass

    return tft, training_ds, validation_ds, scaled_df, train_mask, val_mask, test_mask, best_path


def load_best_tft(checkpoint_path: str, training_dataset):
    from pytorch_forecasting import TemporalFusionTransformer

    tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    return tft


if __name__ == "__main__":
    train_tft()
