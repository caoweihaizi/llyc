import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "runs" / "uggru_topo144_seq12"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check UGGRU training and evaluation outputs.")
    parser.add_argument("--run_dir", type=Path, default=RUN_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")


def main() -> None:
    args = parse_args()
    best_model = args.run_dir / "best_model.pt"
    train_log = args.run_dir / "train_log.csv"
    metrics_path = args.results_dir / "uggru_test_metrics.json"
    pred_path = args.results_dir / "uggru_predictions_test.npz"
    figures = [
        args.results_dir / "uggru_train_val_loss.png",
        args.results_dir / "uggru_test_util_scatter.png",
        args.results_dir / "uggru_test_congestion_confusion.png",
    ]

    for path in [best_model, train_log, metrics_path, pred_path, *figures]:
        require_file(path)
        print(f"Found: {path}")

    log_df = pd.read_csv(train_log)
    if len(log_df) < 1:
        raise RuntimeError("train_log.csv has no epoch rows.")
    print(f"train_log rows: {len(log_df)}")

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    print("Test metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value}")

    with np.load(pred_path) as data:
        print("Prediction arrays:")
        for key in data.files:
            array = data[key]
            print(f"- {key}: shape={array.shape}, dtype={array.dtype}")
            if np.isnan(array).any() or np.isinf(array).any():
                raise RuntimeError(f"{key} contains nan or inf.")
        cong_prob = data["cong_prob"]
        if cong_prob.min() < 0 or cong_prob.max() > 1:
            raise RuntimeError("cong_prob is outside [0, 1].")

    print("Check result: OK")


if __name__ == "__main__":
    main()
