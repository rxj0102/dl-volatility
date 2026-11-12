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
from evaluation.align import align_on_dates
from data.features import download_and_engineer, FEATURE_COLS
from evaluation.metrics import qlike, rmse_log_vol, diebold_mariano, regime_qlike

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_END = "2020-12-31"


# ── helpers ───────────────────────────────────────────────────────────────────

def get_lstm_preds(device: str = "cpu", ckpt_name: str = "lstm_best.pt") -> tuple[pd.Series, pd.Series]:
    from training.train_lstm import load_best_lstm

    _, _, test_loader, scaler, df = build_loaders(batch_size=256)
    input_size = next(iter(test_loader))[0].shape[-1]
    model = load_best_lstm(input_size=input_size, device=device, ckpt_name=ckpt_name)

    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds.append(model(x).cpu().numpy())
            targets.append(y.numpy())
    # the loader's windows end at test rows SEQ_LEN-1 .. n-1: attach those dates
    test_df = df[df.index > VAL_END]
    dates = test_df.index[SEQ_LEN - 1 :]
    return (pd.Series(np.concatenate(preds), index=dates),
            pd.Series(np.concatenate(targets), index=dates))


def get_tft_preds(best_ckpt: str) -> pd.Series:
    """TFT test forecasts indexed by the date whose target they forecast."""
    from pytorch_forecasting import TemporalFusionTransformer
    from models.tft_model import build_tft_datasets, build_tft_test_dataset

    scaled_df, train_mask, val_mask, test_mask, scaler = build_tft_dataframes()
    training_ds, _ = build_tft_datasets(scaled_df, train_mask, val_mask)
    test_ds = build_tft_test_dataset(training_ds, scaled_df, test_mask)
    test_loader = test_ds.to_dataloader(train=False, batch_size=256, num_workers=0)

    tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    tft.eval()
    out = tft.predict(test_loader, mode="prediction", return_index=True)
    preds = out.output.squeeze(-1).cpu().numpy()
    dates = scaled_df.set_index("time_idx")["forecast_date"].reindex(out.index["time_idx"].values)
    return pd.Series(preds, index=pd.DatetimeIndex(dates.values))


def get_tft_attention(best_ckpt: str, n_samples: int = 100):
    from pytorch_forecasting import TemporalFusionTransformer
    from models.tft_model import build_tft_datasets

    scaled_df, train_mask, val_mask, test_mask, scaler = build_tft_dataframes()
    training_ds, _ = build_tft_datasets(scaled_df, train_mask, val_mask)

    from pytorch_forecasting import TimeSeriesDataSet

    from models.tft_model import build_tft_test_dataset
    test_ds = build_tft_test_dataset(training_ds, scaled_df, test_mask)
    loader = test_ds.to_dataloader(train=False, batch_size=n_samples, num_workers=0)

    tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    tft.eval()

    out = tft.predict(loader, mode="raw", return_x=True)
    att = out.output.get("encoder_attention")  # (batch, decoder_len, heads, encoder_len)
    if att is None:
        return None
    return att[:n_samples].mean(dim=2)[:, 0, :]  # average over heads, one decoder step


def get_baseline_preds(df: pd.DataFrame) -> dict[str, pd.Series]:
    from models.baseline import BaselineForecaster
    bf = BaselineForecaster()
    bf.fit(df)
    test_df = df[df.index > VAL_END].dropna()
    return {
        "garch": bf.predict_garch(test_df),
        "xgb": bf.predict_xgb(test_df),
        "targets": bf.get_test_targets(df),
    }


def get_naive_preds(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Naive forecasts of the same target, from information through t-1 only.

    - "Random walk": yesterday's Garman-Klass variance.
    - "22d mean":    mean Garman-Klass variance over the previous 22 days.
    Any model worth reporting should beat these, not only GARCH, whose
    close-to-close variance is on a different scale from a range-based target.
    """
    t = df["target"]
    return {
        "Random walk": t.shift(1),
        "22d mean": t.rolling(22).mean().shift(1),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def run_evaluation(tft_ckpt: str = None):
    print("Loading data...")
    df = download_and_engineer()
    test_df = df[df.index > VAL_END].dropna()
    vix = test_df["vix"].values

    print("Getting LSTM predictions...")
    lstm_preds, _ = get_lstm_preds()

    print("Getting baseline predictions...")
    baseline = get_baseline_preds(df)
    all_preds_s = {"LSTM": lstm_preds, "GARCH": baseline["garch"], "XGB": baseline["xgb"]}
    all_preds_s.update(get_naive_preds(df))

    # TFT (optional, skipped if no checkpoint); forecasts carry their own dates
    if tft_ckpt and Path(tft_ckpt).exists():
        print("Getting TFT predictions...")
        all_preds_s["TFT"] = get_tft_preds(tft_ckpt)

    # align on COMMON DATES (models start forecasting on different days)
    aligned = align_on_dates(baseline["targets"], all_preds_s)
    sigma2 = aligned["target"].values
    garch_p = aligned["GARCH"].values
    rw_p = aligned["Random walk"].values
    vix_aligned = test_df["vix"].reindex(aligned.index).values
    all_preds = {k: aligned[k].values for k in all_preds_s}
    min_len = len(aligned)

    # ── metrics ───────────────────────────────────────────────────────────────
    rows = []
    for name, preds in all_preds.items():
        ql = qlike(sigma2, preds)
        rm = rmse_log_vol(sigma2, preds)
        dm_vs_garch = diebold_mariano(sigma2, preds, garch_p)
        dm_vs_rw = diebold_mariano(sigma2, preds, rw_p)
        rows.append(
            {
                "model": name,
                "QLIKE": round(ql, 6),
                "RMSE": round(rm, 6),
                "DM_vs_GARCH_stat": round(dm_vs_garch["dm_stat"], 4),
                "DM_p_value": round(dm_vs_garch["p_value"], 4),
                "DM_vs_RW_stat": round(dm_vs_rw["dm_stat"], 4),
                "DM_vs_RW_p_value": round(dm_vs_rw["p_value"], 4),
            }
        )

    comp_df = pd.DataFrame(rows)
    comp_df.attrs["n_days"] = min_len
    print(f"Common test days: {min_len} ({aligned.index[0].date()} to {aligned.index[-1].date()})")
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
            att_np = attention.cpu().numpy()  # (samples, encoder steps)
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
