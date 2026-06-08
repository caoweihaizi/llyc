import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from baseline_models import GRUOnlyBaseline, LSTMOnlyBaseline
from dataset import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GRU-only or LSTM-only baseline.")
    parser.add_argument("--baseline", choices=["gru", "lstm"], required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
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


def build_model(baseline: str, model_config: dict[str, Any]) -> torch.nn.Module:
    if baseline == "gru":
        return GRUOnlyBaseline(**model_config)
    if baseline == "lstm":
        return LSTMOnlyBaseline(**model_config)
    raise ValueError(f"Unsupported baseline: {baseline}")


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


def save_scatter(util_pred: np.ndarray, util_true: np.ndarray, output_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    flat_pred = util_pred.reshape(-1)
    flat_true = util_true.reshape(-1)
    count = min(100_000, flat_true.size)
    indices = rng.choice(flat_true.size, size=count, replace=False)
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(flat_true[indices], flat_pred[indices], s=2, alpha=0.25)
    lim = max(float(flat_true[indices].max()), float(flat_pred[indices].max()), 1.0)
    plt.plot([0, lim], [0, lim], color="red", linewidth=1)
    plt.xlabel("true utilization")
    plt.ylabel("predicted utilization")
    plt.title("RNN baseline test utilization scatter")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_confusion(pred: np.ndarray, true: np.ndarray, output_path: Path) -> None:
    metrics = compute_prf(pred, true)
    matrix = np.array(
        [[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]],
        dtype=np.int64,
    )
    plt.figure(figsize=(5, 4.5))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["pred 0", "pred 1"])
    plt.yticks([0, 1], ["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
    plt.title("RNN baseline congestion confusion")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    require_file(args.model_path)
    results_dir = ensure_dir(RESULTS_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: 当前未使用 GPU，RNN baseline 评估会变慢。")
    print(f"device: {device}")
    print(f"baseline: {args.baseline}")
    print(f"threshold: {args.threshold}")

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model = build_model(args.baseline, checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples_path=PROCESSED_DIR / f"samples_topo144_seq{args.seq_len}.npz",
        splits_path=PROCESSED_DIR / f"splits_topo144_seq{args.seq_len}.json",
        pin_memory=device.type == "cuda",
    )

    util_preds = []
    util_trues = []
    load_preds = []
    load_trues = []
    cong_probs = []
    cong_trues = []
    with torch.no_grad():
        for x, y_util, y_load, y_cong in tqdm(
            test_loader,
            desc=f"Evaluating {args.baseline} baseline",
            leave=False,
            disable=True,
        ):
            x = x.to(device, non_blocking=True).float()
            util_pred, load_pred, cong_logit = model(x)
            cong_prob = torch.sigmoid(cong_logit)
            util_preds.append(util_pred.cpu().numpy().astype(np.float32))
            util_trues.append(y_util.numpy().astype(np.float32))
            load_preds.append(load_pred.cpu().numpy().astype(np.float32))
            load_trues.append(y_load.numpy().astype(np.float32))
            cong_probs.append(cong_prob.cpu().numpy().astype(np.float32))
            cong_trues.append(y_cong.numpy().astype(np.float32))

    util_pred = np.concatenate(util_preds, axis=0)
    util_true = np.concatenate(util_trues, axis=0)
    load_pred = np.concatenate(load_preds, axis=0)
    load_true = np.concatenate(load_trues, axis=0)
    cong_prob = np.concatenate(cong_probs, axis=0)
    cong_true = np.concatenate(cong_trues, axis=0)
    pred_cong = cong_prob >= args.threshold

    util_diff = util_pred - util_true
    load_diff = load_pred - load_true
    metrics: dict[str, Any] = {
        "baseline": args.baseline,
        "threshold": float(args.threshold),
        "MAE_util": float(np.abs(util_diff).mean()),
        "RMSE_util": float(np.sqrt((util_diff * util_diff).mean())),
        "MAE_load_norm": float(np.abs(load_diff).mean()),
        "RMSE_load_norm": float(np.sqrt((load_diff * load_diff).mean())),
        "used_cuda": bool(device.type == "cuda"),
    }
    metrics.update(compute_prf(pred_cong, cong_true >= 0.5))

    prefix = f"{args.baseline}_baseline"
    metrics_path = results_dir / f"{prefix}_test_metrics.json"
    pred_path = results_dir / f"{prefix}_predictions_test.npz"
    scatter_path = results_dir / f"{prefix}_test_util_scatter.png"
    confusion_path = results_dir / f"{prefix}_congestion_confusion.png"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(metrics), file, indent=2, ensure_ascii=False)
    np.savez_compressed(
        pred_path,
        util_pred=util_pred.astype(np.float32),
        util_true=util_true.astype(np.float32),
        load_pred=load_pred.astype(np.float32),
        load_true=load_true.astype(np.float32),
        cong_prob=cong_prob.astype(np.float32),
        cong_true=cong_true.astype(np.float32),
    )
    save_scatter(util_pred, util_true, scatter_path)
    save_confusion(pred_cong, cong_true >= 0.5, confusion_path)

    print("RNN baseline test metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value}")
    print("Output files:")
    print(f"- {metrics_path}")
    print(f"- {pred_path}")
    print(f"- {scatter_path}")
    print(f"- {confusion_path}")


if __name__ == "__main__":
    main()
