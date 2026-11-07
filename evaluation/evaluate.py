"""
Full evaluation pipeline:
  1. Load checkpoints & produce test-set forecasts for LSTM, TFT, GARCH, XGB.
  2. Compute QLIKE, RMSE-log-vol, DM tests, regime performance.
  3. Save outputs/model_comparison.csv, outputs/regime_performance.csv.
  4. Plot forecast_comparison.png and tft_attention_heatmap.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from data.dataset import build_loaders, build_tft_dataframes, SEQ_LEN
from data.features import download_and_engineer, FEATURE_COLS
from evaluation.metrics import qlike, rmse_log_vol, diebold_mariano, regime_qlike

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_END = "2020-12-31"


# ── helpers ───────────────────────────────────────────────────────────────────

def get_lstm_preds(device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    from training.train_lstm import load_best_lstm

    _, _, test_loader, scaler, df = build_loaders(batch_size=256)
    input_size = next(iter(test_loader))[0].shape[-1]
    model = load_best_lstm(input_size=input_size, device=device)

    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds.append(model(x).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def get_tft_preds(best_ckpt: str) -> tuple[np.ndarray, np.ndarray]:
    from pytorch_forecasting import TemporalFusionTransformer
    from models.tft_model import build_tft_datasets

    scaled_df, train_mask, val_mask, test_mask, scaler = build_tft_dataframes()
    training_ds, _ = build_tft_datasets(scaled_df, train_mask, val_mask)

    from pytorch_forecasting import TimeSeriesDataSet

    test_ds = TimeSeriesDataSet.from_dataset(
        training_ds, scaled_df[val_mask | test_mask], predict=True, stop_randomization=True
    )
    test_loader = test_ds.to_dataloader(train=False, batch_size=256, num_workers=0)

    tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    tft.eval()

    raw_preds, x = tft.predict(test_loader, mode="raw", return_x=True)
    preds = raw_preds["prediction"].squeeze(-1).cpu().numpy()

    targets = scaled_df.loc[test_mask, "target"].values[SEQ_LEN:]
    min_len = min(len(preds), len(targets))
    return preds[:min_len], targets[:min_len]


def get_tft_attention(best_ckpt: str, n_samples: int = 100):
    from pytorch_forecasting import TemporalFusionTransformer
    from models.tft_model import build_tft_datasets

    scaled_df, train_mask, val_mask, test_mask, scaler = build_tft_dataframes()
    training_ds, _ = build_tft_datasets(scaled_df, train_mask, val_mask)

    from pytorch_forecasting import TimeSeriesDataSet

    test_ds = TimeSeriesDataSet.from_dataset(
        training_ds, scaled_df[val_mask | test_mask], predict=True, stop_randomization=True
    )
    loader = test_ds.to_dataloader(train=False, batch_size=n_samples, num_workers=0)

    tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    tft.eval()

    raw_preds, x = tft.predict(loader, mode="raw", return_x=True)
    attention = raw_preds.get("attention", None)
    return attention


def get_baseline_preds(df: pd.DataFrame) -> dict[str, np.ndarray]:
    from models.baseline import BaselineForecaster

    bf = BaselineForecaster()
    bf.fit(df)
    test_df = df[df.index > VAL_END].dropna()

    garch_preds = bf.predict_garch(test_df)
    xgb_preds = bf.predict_xgb(test_df)
    targets = bf.get_test_targets(df).values

    min_len = min(len(garch_preds), len(xgb_preds), len(targets))
    return {
        "garch": garch_preds[:min_len],
        "xgb": xgb_preds[:min_len],
        "targets": targets[:min_len],
    }


# ── main ──────────────────────────────────────────────────────────────────────

def run_evaluation(tft_ckpt: str = None):
    print("Loading data...")
    df = download_and_engineer()
    test_df = df[df.index > VAL_END].dropna()
    vix = test_df["vix"].values

    print("Getting LSTM predictions...")
    lstm_preds, lstm_targets = get_lstm_preds()

    print("Getting baseline predictions...")
    baseline = get_baseline_preds(df)
    sigma2 = baseline["targets"]

    # align lengths across all models
    min_len = min(len(lstm_preds), len(baseline["garch"]), len(baseline["xgb"]))
    lstm_p = lstm_preds[:min_len]
    garch_p = baseline["garch"][:min_len]
    xgb_p = baseline["xgb"][:min_len]
    sigma2 = sigma2[:min_len]
    vix_aligned = vix[:min_len]

    all_preds = {"LSTM": lstm_p, "GARCH": garch_p, "XGB": xgb_p}

    # TFT (optional — skipped if no checkpoint)
    tft_p = None
    if tft_ckpt and Path(tft_ckpt).exists():
        print("Getting TFT predictions...")
        tft_p, _ = get_tft_preds(tft_ckpt)
        tft_p = tft_p[: min_len]
        all_preds["TFT"] = tft_p

    # ── metrics ───────────────────────────────────────────────────────────────
    rows = []
    for name, preds in all_preds.items():
        ql = qlike(sigma2, preds)
        rm = rmse_log_vol(sigma2, preds)
        dm_vs_garch = diebold_mariano(sigma2, preds, garch_p)
        rows.append(
            {
                "model": name,
                "QLIKE": round(ql, 6),
                "RMSE": round(rm, 6),
                "DM_vs_GARCH_stat": round(dm_vs_garch["dm_stat"], 4),
                "DM_p_value": round(dm_vs_garch["p_value"], 4),
            }
        )

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison:")
    print(comp_df.to_string(index=False))

    # ── regime performance ────────────────────────────────────────────────────
    regime_rows = []
    for name, preds in all_preds.items():
        reg = regime_qlike(sigma2, preds, vix_aligned)
        regime_rows.append(
            {
                "model": name,
                "low_vol_QLIKE": round(reg["low_vol_qlike"], 6),
                "high_vol_QLIKE": round(reg["high_vol_qlike"], 6),
                "n_low": reg["n_low"],
                "n_high": reg["n_high"],
            }
        )
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(OUTPUT_DIR / "regime_performance.csv", index=False)
    print("\nRegime performance:")
    print(regime_df.to_string(index=False))

    # ── forecast comparison plot ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    x_axis = np.arange(min_len)
    realized_vol = np.sqrt(sigma2) * np.sqrt(252)
    ax.plot(x_axis, realized_vol, label="Realized Vol", color="black", linewidth=1.2, alpha=0.8)
    colors = {"LSTM": "royalblue", "TFT": "darkorange", "GARCH": "forestgreen", "XGB": "crimson"}
    for name, preds in all_preds.items():
        pred_vol = np.sqrt(preds) * np.sqrt(252)
        ax.plot(x_axis, pred_vol, label=name, color=colors.get(name, None), linewidth=0.9, alpha=0.7)
    ax.set_title("Realized vs. Forecast Annualized Volatility (Test Set 2021–present)")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Annualized Volatility")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecast_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved forecast_comparison.png")

    # ── TFT attention heatmap ─────────────────────────────────────────────────
    if tft_ckpt and Path(tft_ckpt).exists():
        attention = get_tft_attention(tft_ckpt)
        if attention is not None:
            att_np = attention.squeeze().cpu().numpy()
            if att_np.ndim == 3:
                att_np = att_np.mean(axis=0)  # average over heads
            elif att_np.ndim == 1:
                att_np = att_np[np.newaxis, :]
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.heatmap(att_np[:20], ax=ax, cmap="YlOrRd", xticklabels=5, yticklabels=False)
            ax.set_title("TFT Encoder Attention Weights (first 20 samples)")
            ax.set_xlabel("Encoder time step (lag)")
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "tft_attention_heatmap.png", dpi=150)
            plt.close(fig)
            print("Saved tft_attention_heatmap.png")

    return comp_df, regime_df


if __name__ == "__main__":
    import sys

    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    run_evaluation(tft_ckpt=ckpt)
