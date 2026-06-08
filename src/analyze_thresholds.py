import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_INPUT = RESULTS_DIR / "uggru_predictions_test.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze congestion classification thresholds for UGGRU test predictions."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--threshold_start", type=float, default=0.05)
    parser.add_argument("--threshold_end", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input prediction file does not exist: {path}")


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


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


def row_to_dict(row: pd.Series | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return to_builtin(row.to_dict())


def compute_threshold_metrics(
    cong_prob: np.ndarray,
    cong_true: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    true_bool = cong_true.astype(bool)
    total = int(true_bool.size)
    true_positive_ratio = float(true_bool.mean())
    records: list[dict[str, float | int]] = []

    for threshold in thresholds:
        pred_bool = cong_prob >= threshold
        tp = int(np.logical_and(pred_bool, true_bool).sum())
        fp = int(np.logical_and(pred_bool, ~true_bool).sum())
        tn = int(np.logical_and(~pred_bool, ~true_bool).sum())
        fn = int(np.logical_and(~pred_bool, true_bool).sum())

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)

        records.append(
            {
                "threshold": float(threshold),
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "predicted_positive_ratio": float(pred_bool.mean()),
                "true_positive_ratio": true_positive_ratio,
                "total_samples": total,
            }
        )

    return pd.DataFrame.from_records(records)


def select_best_rows(metrics: pd.DataFrame) -> dict[str, pd.Series | None]:
    best_f1 = metrics.loc[metrics["F1"].idxmax()]

    recall_90 = metrics[metrics["Recall"] >= 0.90]
    recall_90_best_precision = None
    if not recall_90.empty:
        recall_90_best_precision = recall_90.sort_values(
            ["Precision", "F1", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]

    recall_80 = metrics[metrics["Recall"] >= 0.80]
    recall_80_best_f1 = None
    if not recall_80.empty:
        recall_80_best_f1 = recall_80.sort_values(
            ["F1", "Precision", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]

    precision_30 = metrics[metrics["Precision"] >= 0.30]
    precision_30_best_recall = None
    if not precision_30.empty:
        precision_30_best_recall = precision_30.sort_values(
            ["Recall", "F1", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]

    ratio_gap = (
        metrics["predicted_positive_ratio"] - metrics["true_positive_ratio"]
    ).abs()
    closest_positive_ratio = metrics.loc[ratio_gap.idxmin()]

    return {
        "best_f1": best_f1,
        "recall_ge_0_90_best_precision": recall_90_best_precision,
        "recall_ge_0_80_best_f1": recall_80_best_f1,
        "precision_ge_0_30_best_recall": precision_30_best_recall,
        "closest_predicted_positive_ratio": closest_positive_ratio,
    }


def compute_pr_auc(cong_prob: np.ndarray, cong_true: np.ndarray) -> tuple[dict[str, float | None], str | None]:
    try:
        from sklearn.metrics import auc, average_precision_score, precision_recall_curve
    except ImportError:
        return (
            {
                "average_precision": None,
                "pr_auc": None,
            },
            "sklearn is not available; Average Precision / PR-AUC were skipped.",
        )

    true_int = cong_true.astype(np.int32)
    precision, recall, _ = precision_recall_curve(true_int, cong_prob)
    return (
        {
            "average_precision": float(average_precision_score(true_int, cong_prob)),
            "pr_auc": float(auc(recall, precision)),
        },
        None,
    )


def save_plot(metrics: pd.DataFrame, best_f1: pd.Series, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(metrics["threshold"], metrics["Precision"], label="Precision", linewidth=1.8)
    plt.plot(metrics["threshold"], metrics["Recall"], label="Recall", linewidth=1.8)
    plt.plot(metrics["threshold"], metrics["F1"], label="F1", linewidth=1.8)
    plt.axvline(
        float(best_f1["threshold"]),
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"Best F1 threshold={best_f1['threshold']:.2f}",
    )
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.title("UGGRU congestion threshold analysis")
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def format_row(row: pd.Series | None) -> str:
    if row is None:
        return "not found"
    return (
        f"threshold={row['threshold']:.2f}, "
        f"Precision={row['Precision']:.6f}, "
        f"Recall={row['Recall']:.6f}, "
        f"F1={row['F1']:.6f}, "
        f"pred_pos_ratio={row['predicted_positive_ratio']:.6f}, "
        f"true_pos_ratio={row['true_positive_ratio']:.6f}"
    )


def main() -> None:
    args = parse_args()
    require_file(args.input)
    results_dir = ensure_dir(args.results_dir)

    if args.threshold_step <= 0:
        raise ValueError(f"--threshold_step must be positive, got {args.threshold_step}")
    if args.threshold_start > args.threshold_end:
        raise ValueError(
            f"--threshold_start must be <= --threshold_end, got "
            f"{args.threshold_start} > {args.threshold_end}"
        )

    thresholds = np.round(
        np.arange(
            args.threshold_start,
            args.threshold_end + args.threshold_step / 2.0,
            args.threshold_step,
        ),
        6,
    )

    print(f"Loading predictions: {args.input}")
    with np.load(args.input) as data:
        if "cong_prob" not in data or "cong_true" not in data:
            raise KeyError("Prediction npz must contain cong_prob and cong_true arrays.")
        cong_prob = data["cong_prob"].reshape(-1).astype(np.float64)
        cong_true_raw = data["cong_true"].reshape(-1)

    if not np.all(np.isfinite(cong_prob)):
        raise ValueError("cong_prob contains NaN or inf.")
    if np.any((cong_prob < 0.0) | (cong_prob > 1.0)):
        min_prob = float(np.min(cong_prob))
        max_prob = float(np.max(cong_prob))
        raise ValueError(f"cong_prob must be in [0, 1], got min={min_prob}, max={max_prob}")

    if not np.all(np.isfinite(cong_true_raw.astype(np.float64))):
        raise ValueError("cong_true contains NaN or inf.")
    if not np.all((cong_true_raw == 0) | (cong_true_raw == 1)):
        unique_values = np.unique(cong_true_raw)
        raise ValueError(f"cong_true must contain only 0/1 values, got {unique_values[:20]}")

    cong_true = cong_true_raw.astype(np.int8)
    print(f"Flattened samples: {cong_prob.size}")
    print(f"True positive ratio: {float(cong_true.mean()):.6f}")
    print(f"Threshold count: {len(thresholds)}")

    metrics = compute_threshold_metrics(cong_prob, cong_true, thresholds)
    selected = select_best_rows(metrics)
    pr_auc, sklearn_warning = compute_pr_auc(cong_prob, cong_true)
    if sklearn_warning:
        print(f"Warning: {sklearn_warning}")

    csv_path = results_dir / "uggru_threshold_analysis.csv"
    json_path = results_dir / "uggru_threshold_analysis.json"
    plot_path = results_dir / "uggru_threshold_prf_curve.png"

    metrics.to_csv(csv_path, index=False, float_format="%.9g")
    save_plot(metrics, selected["best_f1"], plot_path)

    summary = {
        "input_path": str(args.input),
        "num_flattened_samples": int(cong_prob.size),
        "threshold_start": float(args.threshold_start),
        "threshold_end": float(args.threshold_end),
        "threshold_step": float(args.threshold_step),
        "threshold_count": int(len(thresholds)),
        "average_precision": pr_auc["average_precision"],
        "pr_auc": pr_auc["pr_auc"],
        "sklearn_warning": sklearn_warning,
        "best_f1": row_to_dict(selected["best_f1"]),
        "recall_ge_0_90_best_precision": row_to_dict(
            selected["recall_ge_0_90_best_precision"]
        ),
        "recall_ge_0_80_best_f1": row_to_dict(selected["recall_ge_0_80_best_f1"]),
        "precision_ge_0_30_best_recall": row_to_dict(
            selected["precision_ge_0_30_best_recall"]
        ),
        "closest_predicted_positive_ratio": row_to_dict(
            selected["closest_predicted_positive_ratio"]
        ),
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "plot": str(plot_path),
        },
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(summary), file, indent=2, ensure_ascii=False)

    print("\nThreshold analysis complete:")
    print(f"- best F1: {format_row(selected['best_f1'])}")
    print(
        "- recall >= 0.90, best precision: "
        f"{format_row(selected['recall_ge_0_90_best_precision'])}"
    )
    print(f"- recall >= 0.80, best F1: {format_row(selected['recall_ge_0_80_best_f1'])}")
    print(
        "- precision >= 0.30, best recall: "
        f"{format_row(selected['precision_ge_0_30_best_recall'])}"
    )
    print(
        "- closest predicted positive ratio: "
        f"{format_row(selected['closest_predicted_positive_ratio'])}"
    )
    print(f"- Average Precision: {pr_auc['average_precision']}")
    print(f"- PR-AUC: {pr_auc['pr_auc']}")
    print(f"- CSV: {csv_path}")
    print(f"- JSON: {json_path}")
    print(f"- plot: {plot_path}")


if __name__ == "__main__":
    main()
