import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

PREDICTIONS_PATH = RESULTS_DIR / "mc_dropout_predictions_test.npz"
METRICS_PATH = RESULTS_DIR / "mc_dropout_metrics.json"
RISK_TOPK_PATH = RESULTS_DIR / "mc_dropout_risk_topk.csv"

REQUIRED_ARRAYS = [
    "util_pred_mean",
    "util_pred_std",
    "cong_prob_mean",
    "cong_prob_std",
    "util_true",
    "cong_true",
    "risk_score",
    "congestion_risk_score",
]

REQUIRED_FIGURES = [
    "mc_dropout_uncertainty_error_scatter.png",
    "mc_dropout_uncertainty_bins.png",
    "mc_dropout_prediction_interval_example.png",
    "mc_dropout_risk_topk.png",
    "mc_dropout_congestion_confusion_threshold095.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MC Dropout uncertainty outputs.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required output file does not exist: {path}")


def check_finite(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or inf.")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    predictions_path = results_dir / PREDICTIONS_PATH.name
    metrics_path = results_dir / METRICS_PATH.name
    risk_topk_path = results_dir / RISK_TOPK_PATH.name

    for path in [predictions_path, metrics_path, risk_topk_path]:
        require_file(path)

    print(f"Metrics file: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    with np.load(predictions_path) as data:
        for name in REQUIRED_ARRAYS:
            if name not in data:
                raise KeyError(f"{predictions_path} is missing array: {name}")

        arrays = {name: data[name] for name in REQUIRED_ARRAYS}

    print("\nArray shapes:")
    for name, array in arrays.items():
        print(f"- {name}: shape={array.shape}, dtype={array.dtype}")
        check_finite(name, array)

    util_true_shape = arrays["util_true"].shape
    if arrays["risk_score"].shape != util_true_shape:
        raise ValueError(
            f"risk_score shape must match util_true: "
            f"{arrays['risk_score'].shape} vs {util_true_shape}"
        )
    if arrays["congestion_risk_score"].shape != util_true_shape:
        raise ValueError(
            f"congestion_risk_score shape must match util_true: "
            f"{arrays['congestion_risk_score'].shape} vs {util_true_shape}"
        )
    if np.any(arrays["util_pred_std"] < 0):
        raise ValueError("util_pred_std contains negative values.")
    if np.any((arrays["cong_prob_mean"] < 0.0) | (arrays["cong_prob_mean"] > 1.0)):
        raise ValueError("cong_prob_mean must be in [0, 1].")
    if np.any(arrays["cong_prob_std"] < 0):
        raise ValueError("cong_prob_std contains negative values.")

    print("\nFigure checks:")
    for figure_name in REQUIRED_FIGURES:
        figure_path = results_dir / figure_name
        require_file(figure_path)
        print(f"- {figure_path}: exists, size={figure_path.stat().st_size} bytes")

    print("\nUncertainty output check passed.")


if __name__ == "__main__":
    main()
