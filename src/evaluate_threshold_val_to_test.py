import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LinkStateDataStore, LinkStateDataset
from models import UGGRU


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "runs" / "uggru_topo144_seq12" / "best_model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select congestion threshold on validation split and evaluate it on test split."
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold_start", type=float, default=0.05)
    parser.add_argument("--threshold_end", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")


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


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_metrics_at_threshold(
    cong_prob: np.ndarray,
    cong_true: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    pred_bool = cong_prob >= threshold
    true_bool = cong_true.astype(bool)

    tp = int(np.logical_and(pred_bool, true_bool).sum())
    fp = int(np.logical_and(pred_bool, ~true_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~true_bool).sum())
    fn = int(np.logical_and(~pred_bool, true_bool).sum())

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)

    return {
        "threshold": float(threshold),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "predicted_positive_ratio": float(pred_bool.mean()),
        "true_positive_ratio": float(true_bool.mean()),
    }


def scan_thresholds(
    cong_prob: np.ndarray,
    cong_true: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    records = [
        compute_metrics_at_threshold(cong_prob, cong_true, float(threshold))
        for threshold in thresholds
    ]
    return pd.DataFrame.from_records(records)


def select_threshold_rows(metrics: pd.DataFrame) -> dict[str, pd.Series | None]:
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

    return {
        "best_f1_threshold": best_f1,
        "recall_ge_0_90_best_precision": recall_90_best_precision,
        "recall_ge_0_80_best_f1": recall_80_best_f1,
        "precision_ge_0_30_best_recall": precision_30_best_recall,
    }


def validate_arrays(cong_prob: np.ndarray, cong_true: np.ndarray, split: str) -> None:
    if cong_prob.ndim != 1 or cong_true.ndim != 1:
        raise ValueError(f"{split}: cong_prob and cong_true must be flattened 1D arrays.")
    if cong_prob.shape != cong_true.shape:
        raise ValueError(
            f"{split}: cong_prob and cong_true shape mismatch: "
            f"{cong_prob.shape} vs {cong_true.shape}"
        )
    if not np.all(np.isfinite(cong_prob)):
        raise ValueError(f"{split}: cong_prob contains NaN or inf.")
    if np.any((cong_prob < 0.0) | (cong_prob > 1.0)):
        raise ValueError(
            f"{split}: cong_prob must be in [0, 1], got "
            f"min={float(cong_prob.min())}, max={float(cong_prob.max())}"
        )
    if not np.all(np.isfinite(cong_true.astype(np.float64))):
        raise ValueError(f"{split}: cong_true contains NaN or inf.")
    if not np.all((cong_true == 0) | (cong_true == 1)):
        unique_values = np.unique(cong_true)
        raise ValueError(f"{split}: cong_true must contain only 0/1, got {unique_values[:20]}")


def build_loader(
    split: str,
    store: LinkStateDataStore,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = LinkStateDataset(split=split, store=store)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def collect_congestion_predictions(
    model: UGGRU,
    loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    cong_probs: list[np.ndarray] = []
    cong_trues: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for x, _, _, y_cong in tqdm(loader, desc=f"Predicting {split} split"):
            x = x.to(device, non_blocking=True).float()
            y_cong = y_cong.to(device, non_blocking=True).float()
            _, _, cong_logit = model(x, adj)
            cong_prob = torch.sigmoid(cong_logit)
            cong_probs.append(cong_prob.cpu().numpy().astype(np.float32))
            cong_trues.append(y_cong.cpu().numpy().astype(np.float32))

    prob = np.concatenate(cong_probs, axis=0).reshape(-1).astype(np.float64)
    true = np.concatenate(cong_trues, axis=0).reshape(-1)
    validate_arrays(prob, true, split)
    return prob, true.astype(np.int8)


def save_val_prf_plot(metrics: pd.DataFrame, best_row: pd.Series, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(metrics["threshold"], metrics["Precision"], label="Val Precision", linewidth=1.8)
    plt.plot(metrics["threshold"], metrics["Recall"], label="Val Recall", linewidth=1.8)
    plt.plot(metrics["threshold"], metrics["F1"], label="Val F1", linewidth=1.8)
    plt.axvline(
        float(best_row["threshold"]),
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"Val best F1 threshold={best_row['threshold']:.2f}",
    )
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.title("Validation-selected congestion thresholds")
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def format_metrics(row: pd.Series | dict[str, Any] | None) -> str:
    if row is None:
        return "not found"
    getter = row.get if isinstance(row, dict) else row.__getitem__
    return (
        f"threshold={float(getter('threshold')):.2f}, "
        f"Precision={float(getter('Precision')):.6f}, "
        f"Recall={float(getter('Recall')):.6f}, "
        f"F1={float(getter('F1')):.6f}, "
        f"pred_pos_ratio={float(getter('predicted_positive_ratio')):.6f}, "
        f"true_pos_ratio={float(getter('true_positive_ratio')):.6f}"
    )


def main() -> None:
    args = parse_args()
    require_file(args.model_path)

    samples_path = PROCESSED_DIR / f"samples_topo144_seq{args.seq_len}.npz"
    splits_path = PROCESSED_DIR / f"splits_topo144_seq{args.seq_len}.json"
    adj_path = PROCESSED_DIR / "edge_adj_topo144.npy"
    for path in [samples_path, splits_path, adj_path]:
        require_file(path)

    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be positive, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"--num_workers must be >= 0, got {args.num_workers}")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: 当前未使用 GPU，UGGRU 阈值评估会明显变慢。")

    print("Val-to-test threshold evaluation settings:")
    print(f"- model_path: {args.model_path}")
    print(f"- device: {device}")
    if device.type == "cuda":
        print(f"- GPU: {torch.cuda.get_device_name(0)}")
    print(f"- batch_size: {args.batch_size}")
    print(f"- thresholds: {args.threshold_start} to {args.threshold_end}, step {args.threshold_step}")

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model = UGGRU(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    adj = torch.from_numpy(np.load(adj_path).astype(np.float32)).to(device)
    store = LinkStateDataStore(samples_path=samples_path, splits_path=splits_path)
    val_loader = build_loader(
        "val",
        store=store,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = build_loader(
        "test",
        store=store,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"- val samples: {len(val_loader.dataset)}")
    print(f"- test samples: {len(test_loader.dataset)}")

    val_prob, val_true = collect_congestion_predictions(model, val_loader, adj, device, "val")
    test_prob, test_true = collect_congestion_predictions(model, test_loader, adj, device, "test")

    val_metrics = scan_thresholds(val_prob, val_true, thresholds)
    selected_val = select_threshold_rows(val_metrics)

    selected_test: dict[str, dict[str, Any] | None] = {}
    for strategy_name, val_row in selected_val.items():
        if val_row is None:
            selected_test[strategy_name] = None
            continue
        threshold = float(val_row["threshold"])
        selected_test[strategy_name] = compute_metrics_at_threshold(test_prob, test_true, threshold)

    results_dir = ensure_dir(RESULTS_DIR)
    val_csv_path = results_dir / "uggru_val_threshold_analysis.csv"
    test_json_path = results_dir / "uggru_val_selected_threshold_test_metrics.json"
    plot_path = results_dir / "uggru_val_to_test_threshold_prf_curve.png"

    val_metrics.to_csv(val_csv_path, index=False, float_format="%.9g")
    save_val_prf_plot(val_metrics, selected_val["best_f1_threshold"], plot_path)

    summary = {
        "model_path": str(args.model_path),
        "samples_path": str(samples_path),
        "splits_path": str(splits_path),
        "adj_path": str(adj_path),
        "seq_len": int(args.seq_len),
        "batch_size": int(args.batch_size),
        "threshold_start": float(args.threshold_start),
        "threshold_end": float(args.threshold_end),
        "threshold_step": float(args.threshold_step),
        "val_flattened_samples": int(val_prob.size),
        "test_flattened_samples": int(test_prob.size),
        "selected_on_validation": {
            key: row_to_dict(value) for key, value in selected_val.items()
        },
        "applied_to_test": to_builtin(selected_test),
        "outputs": {
            "val_threshold_csv": str(val_csv_path),
            "test_metrics_json": str(test_json_path),
            "plot": str(plot_path),
        },
    }

    with test_json_path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(summary), file, indent=2, ensure_ascii=False)

    print("\nValidation-selected thresholds:")
    for key, row in selected_val.items():
        print(f"- {key}: {format_metrics(row)}")

    print("\nFixed-threshold test metrics:")
    for key, metrics in selected_test.items():
        print(f"- {key}: {format_metrics(metrics)}")

    print("\nOutput files:")
    print(f"- {val_csv_path}")
    print(f"- {test_json_path}")
    print(f"- {plot_path}")


if __name__ == "__main__":
    main()
