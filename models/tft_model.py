"""Temporal Fusion Transformer wrapper using pytorch-forecasting."""

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss, RMSE
from pytorch_forecasting.data import GroupNormalizer

from data.features import FEATURE_COLS

SEQ_LEN = 60
PRED_LEN = 1


def build_tft_datasets(scaled_df: pd.DataFrame, train_mask, val_mask):
    """Create pytorch-forecasting TimeSeriesDataSet objects."""
    max_encoder_length = SEQ_LEN
    max_prediction_length = PRED_LEN

    training = TimeSeriesDataSet(
        scaled_df[train_mask | val_mask],
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=FEATURE_COLS,
        target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, scaled_df[val_mask], predict=True, stop_randomization=True
    )

    return training, validation


def build_tft_model(training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=1e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        loss=RMSE(),
        log_interval=10,
        reduce_on_plateau_patience=5,
    )
    return tft


def get_tft_trainer(max_epochs: int = 100, patience: int = 15) -> pl.Trainer:
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    early_stop = EarlyStopping(
        monitor="val_loss", patience=patience, mode="min", verbose=False
    )
    checkpoint = ModelCheckpoint(
        dirpath="outputs/checkpoints",
        filename="tft_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint],
        enable_progress_bar=True,
        log_every_n_steps=5,
    )
    return trainer
