"""
End-to-end runner: trains LSTM, trains TFT, then runs full evaluation.
Usage:
    python -m training.run_all
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-tft", action="store_true", help="Skip TFT training")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1: Training LSTM")
    print("=" * 60)
    from training.train_lstm import train_lstm

    train_lstm(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size)

    tft_ckpt = None
    if not args.skip_tft:
        print("\n" + "=" * 60)
        print("STEP 2: Training TFT")
        print("=" * 60)
        from training.train_tft import train_tft

        _, _, _, _, _, _, _, tft_ckpt = train_tft(
            max_epochs=args.epochs, patience=args.patience, batch_size=args.batch_size
        )

    print("\n" + "=" * 60)
    print("STEP 3: Evaluation")
    print("=" * 60)
    from evaluation.evaluate import run_evaluation

    run_evaluation(tft_ckpt=tft_ckpt)
    print("\nDone. Results saved to outputs/")


if __name__ == "__main__":
    main()
