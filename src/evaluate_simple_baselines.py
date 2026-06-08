import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import LinkStateDataStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Last and HA prediction baselines.")
    parser.add_argument("--seq_len", type=int, default=12)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def feature_index(feature_names: np.ndarray, name: str) -> int:
    names = [str(item) for item in feature_names.tolist()]
    if name not in names:
        raise ValueError(f"Feature {name!r} not found in feature_names={names}")
    return names.index(name)


def compute_prf(pred: np.ndarray, true: np.ndarray) -> dict[str, float | int]:
    pred_bool = pred.astype(bool)
    true_bool = true.astype(bool)
    tp = int(np.logical_and(pred_bool, true_bool).sum())
    fp = int(np.logical_and(pred_bool, ~true_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~true_bool).sum())
    fn = int(np.logical_and(~pred_bool, true_bool).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "true_positive_ratio": float(true_bool.mean()),
        "predicted_positive_ratio": float(pred_bool.mean()),
    }


def compute_metrics(
    util_pred: np.ndarray,
    util_true: np.ndarray,
    load_pred: np.ndarray,
    load_true: np.ndarray,
    cong_pred: np.ndarray,
    cong_true: np.ndarray,
) -> dict[str, float | int]:
    util_diff = util_pred - util_true
    load_diff = load_pred - load_true
    metrics: dict[str, float | int] = {
        "MAE_util": float(np.abs(util_diff).mean()),
        "RMSE_util": float(np.sqrt((util_diff * util_diff).mean())),
        "MAE_load_norm": float(np.abs(load_diff).mean()),
        "RMSE_load_norm": float(np.sqrt((load_diff * load_diff).mean())),
    }
    metrics.update(compute_prf(cong_pred, cong_true >= 0.5))
    return metrics


def save_error_boxplot(
    last_abs_error: np.ndarray,
    ha_abs_error: np.ndarray,
    output_path: Path,
    sample_size: int = 200_000,
) -> None:
    rng = np.random.default_rng(SEED)
    errors = []
    labels = []
    for name, values in [("Last", last_abs_error.reshape(-1)), ("HA", ha_abs_error.reshape(-1))]:
        count = min(sample_size, values.size)
        indices = rng.choice(values.size, size=count, replace=False)
        errors.append(values[indices])
        labels.append(name)

    plt.figure(figsize=(6, 4.5))
    plt.boxplot(errors, tick_labels=labels, showfliers=False)
    plt.ylabel("absolute utilization error")
    plt.title("Simple baseline utilization error")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    samples_path = PROCESSED_DIR / f"samples_topo144_seq{args.seq_len}.npz"
    splits_path = PROCESSED_DIR / f"splits_topo144_seq{args.seq_len}.json"
    results_dir = ensure_dir(RESULTS_DIR)

    print(f"Loading samples: {samples_path}")
    store = LinkStateDataStore(samples_path=samples_path, splits_path=splits_path)
    test_start, test_end = store.split_bounds("test")

    util_idx = feature_index(store.feature_names, "utilization")
    load_idx = feature_index(store.feature_names, "load_mbps_norm")
    cong_idx = feature_index(store.feature_names, "congestion_label")

    x_test = store.X[test_start:test_end]
    y_util = store.y_utilization[test_start:test_end]
    y_load = store.y_load_mbps_norm[test_start:test_end]
    y_cong = store.y_congestion[test_start:test_end]

    last_util = x_test[:, -1, :, util_idx]
    last_load = x_test[:, -1, :, load_idx]
    last_cong = x_test[:, -1, :, cong_idx] >= 0.5
    ha_util = x_test[:, :, :, util_idx].mean(axis=1)
    ha_load = x_test[:, :, :, load_idx].mean(axis=1)
    ha_prob = x_test[:, :, :, cong_idx].mean(axis=1)
    ha_cong = ha_prob >= 0.5

    metrics = {
        "Last": compute_metrics(last_util, y_util, last_load, y_load, last_cong, y_cong),
        "HA": compute_metrics(ha_util, y_util, ha_load, y_load, ha_cong, y_cong),
    }

    csv_rows = []
    for model_name, model_metrics in metrics.items():
        row = {"model": model_name}
        row.update(model_metrics)
        csv_rows.append(row)

    json_path = results_dir / "simple_baselines_metrics.json"
    csv_path = results_dir / "simple_baselines_metrics.csv"
    plot_path = results_dir / "simple_baselines_util_error_boxplot.png"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(metrics), file, indent=2, ensure_ascii=False)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.9g")
    save_error_boxplot(np.abs(last_util - y_util), np.abs(ha_util - y_util), plot_path)

    print("Simple baseline metrics:")
    for name, model_metrics in metrics.items():
        print(
            f"- {name}: MAE_util={model_metrics['MAE_util']:.9g}, "
            f"RMSE_util={model_metrics['RMSE_util']:.9g}, F1={model_metrics['F1']:.9g}"
        )
    print("Output files:")
    print(f"- {json_path}")
    print(f"- {csv_path}")
    print(f"- {plot_path}")


if __name__ == "__main__":
    main()
