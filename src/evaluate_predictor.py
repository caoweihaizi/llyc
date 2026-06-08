import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset import create_dataloaders
from models import UGGRU


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained UGGRU predictor on test split.")
    parser.add_argument("--model_path", type=Path, default=PROJECT_ROOT / "runs" / "uggru_topo144_seq12" / "best_model.pt")
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def compute_prf(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    pred_bool = pred.astype(bool)
    true_bool = true.astype(bool)
    tp = np.logical_and(pred_bool, true_bool).sum()
    fp = np.logical_and(pred_bool, ~true_bool).sum()
    fn = np.logical_and(~pred_bool, true_bool).sum()
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(precision), float(recall), float(f1)


def save_scatter(util_pred: np.ndarray, util_true: np.ndarray, output_path: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    flat_pred = util_pred.reshape(-1)
    flat_true = util_true.reshape(-1)
    sample_size = min(100_000, flat_true.size)
    idx = rng.choice(flat_true.size, size=sample_size, replace=False)
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(flat_true[idx], flat_pred[idx], s=2, alpha=0.25)
    lim = max(float(flat_true[idx].max()), float(flat_pred[idx].max()), 1.0)
    plt.plot([0, lim], [0, lim], color="red", linewidth=1)
    plt.xlabel("true utilization")
    plt.ylabel("predicted utilization")
    plt.title("UGGRU test utilization scatter")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_confusion(pred: np.ndarray, true: np.ndarray, output_path: Path) -> None:
    pred_bool = pred.astype(bool)
    true_bool = true.astype(bool)
    tn = np.logical_and(~pred_bool, ~true_bool).sum()
    fp = np.logical_and(pred_bool, ~true_bool).sum()
    fn = np.logical_and(~pred_bool, true_bool).sum()
    tp = np.logical_and(pred_bool, true_bool).sum()
    matrix = np.array([[tn, fp], [fn, tp]], dtype=np.int64)

    plt.figure(figsize=(5, 4.5))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["pred 0", "pred 1"])
    plt.yticks([0, 1], ["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
    plt.title("Congestion confusion matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_train_val_loss(log_path: Path, output_path: Path) -> None:
    if not log_path.exists():
        return
    df = pd.read_csv(log_path)
    if df.empty:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("UGGRU train/val loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {args.model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: 当前未使用 GPU，UGGRU 评估会明显变慢。")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model = UGGRU(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples_path=PROCESSED_DIR / f"samples_topo144_seq{args.seq_len}.npz",
        splits_path=PROCESSED_DIR / f"splits_topo144_seq{args.seq_len}.json",
        pin_memory=device.type == "cuda",
    )
    adj = torch.from_numpy(np.load(PROCESSED_DIR / "edge_adj_topo144.npy").astype(np.float32)).to(device)

    util_preds = []
    util_trues = []
    load_preds = []
    load_trues = []
    cong_probs = []
    cong_trues = []

    with torch.no_grad():
        for x, y_util, y_load, y_cong in tqdm(test_loader, desc="Evaluating test split"):
            x = x.to(device, non_blocking=True).float()
            y_util = y_util.to(device, non_blocking=True).float()
            y_load = y_load.to(device, non_blocking=True).float()
            y_cong = y_cong.to(device, non_blocking=True).float()
            util_pred, load_pred, cong_logit = model(x, adj)
            cong_prob = torch.sigmoid(cong_logit)
            util_preds.append(util_pred.cpu().numpy().astype(np.float32))
            util_trues.append(y_util.cpu().numpy().astype(np.float32))
            load_preds.append(load_pred.cpu().numpy().astype(np.float32))
            load_trues.append(y_load.cpu().numpy().astype(np.float32))
            cong_probs.append(cong_prob.cpu().numpy().astype(np.float32))
            cong_trues.append(y_cong.cpu().numpy().astype(np.float32))

    util_pred = np.concatenate(util_preds, axis=0)
    util_true = np.concatenate(util_trues, axis=0)
    load_pred = np.concatenate(load_preds, axis=0)
    load_true = np.concatenate(load_trues, axis=0)
    cong_prob = np.concatenate(cong_probs, axis=0)
    cong_true = np.concatenate(cong_trues, axis=0)
    pred_congestion = cong_prob >= 0.5

    util_diff = util_pred - util_true
    load_diff = load_pred - load_true
    precision, recall, f1 = compute_prf(pred_congestion, cong_true >= 0.5)
    metrics = {
        "MAE_util": float(np.abs(util_diff).mean()),
        "RMSE_util": float(np.sqrt((util_diff * util_diff).mean())),
        "MAPE_util": float((np.abs(util_diff) / np.maximum(np.abs(util_true), 1e-6)).mean()),
        "MAE_load_norm": float(np.abs(load_diff).mean()),
        "RMSE_load_norm": float(np.sqrt((load_diff * load_diff).mean())),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "positive_ratio_true": float((cong_true >= 0.5).mean()),
        "positive_ratio_pred": float(pred_congestion.mean()),
    }

    metrics_path = RESULTS_DIR / "uggru_test_metrics.json"
    pred_path = RESULTS_DIR / "uggru_predictions_test.npz"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    np.savez_compressed(
        pred_path,
        util_pred=util_pred.astype(np.float32),
        util_true=util_true.astype(np.float32),
        load_pred=load_pred.astype(np.float32),
        load_true=load_true.astype(np.float32),
        cong_prob=cong_prob.astype(np.float32),
        cong_true=cong_true.astype(np.float32),
    )

    save_scatter(util_pred, util_true, RESULTS_DIR / "uggru_test_util_scatter.png")
    save_confusion(pred_congestion, cong_true >= 0.5, RESULTS_DIR / "uggru_test_congestion_confusion.png")
    plot_train_val_loss(args.model_path.parent / "train_log.csv", RESULTS_DIR / "uggru_train_val_loss.png")

    print("Test metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.9g}")
    print("Output files:")
    print(f"- {metrics_path}")
    print(f"- {pred_path}")
    print(f"- {RESULTS_DIR / 'uggru_train_val_loss.png'}")
    print(f"- {RESULTS_DIR / 'uggru_test_util_scatter.png'}")
    print(f"- {RESULTS_DIR / 'uggru_test_congestion_confusion.png'}")
    print(f"Prediction archive size MB: {pred_path.stat().st_size / 1024**2:.2f}")


if __name__ == "__main__":
    main()
