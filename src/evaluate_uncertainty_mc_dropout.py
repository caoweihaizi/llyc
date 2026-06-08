import argparse
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LinkStateDataStore, LinkStateDataset
from models import UGGRU


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "runs" / "uggru_topo144_seq12" / "best_model.pt"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate UGGRU uncertainty on test split with MC Dropout."
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mc_samples", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--risk_lambda", type=float, default=1.0)
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


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable randomness only for dropout layers during MC inference."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def build_test_loader(
    seq_len: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    samples_path = PROCESSED_DIR / f"samples_topo144_seq{seq_len}.npz"
    splits_path = PROCESSED_DIR / f"splits_topo144_seq{seq_len}.json"
    require_file(samples_path)
    require_file(splits_path)
    store = LinkStateDataStore(samples_path=samples_path, splits_path=splits_path)
    test_dataset = LinkStateDataset("test", store=store)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def collect_truth(loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    util_true_batches = []
    load_true_batches = []
    cong_true_batches = []
    for _, y_util, y_load, y_cong in tqdm(loader, desc="Collecting test labels"):
        util_true_batches.append(y_util.numpy().astype(np.float32))
        load_true_batches.append(y_load.numpy().astype(np.float32))
        cong_true_batches.append(y_cong.numpy().astype(np.float32))
    return (
        np.concatenate(util_true_batches, axis=0),
        np.concatenate(load_true_batches, axis=0),
        np.concatenate(cong_true_batches, axis=0),
    )


def run_one_mc_pass(
    model: UGGRU,
    loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
    pass_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    util_batches = []
    load_batches = []
    cong_prob_batches = []
    model.eval()
    enable_mc_dropout(model)

    with torch.no_grad():
        for x, _, _, _ in tqdm(loader, desc=f"MC pass {pass_index}", leave=False):
            x = x.to(device, non_blocking=True).float()
            util_pred, load_pred, cong_logit = model(x, adj)
            cong_prob = torch.sigmoid(cong_logit)
            util_batches.append(util_pred.cpu().numpy().astype(np.float32))
            load_batches.append(load_pred.cpu().numpy().astype(np.float32))
            cong_prob_batches.append(cong_prob.cpu().numpy().astype(np.float32))

    return (
        np.concatenate(util_batches, axis=0),
        np.concatenate(load_batches, axis=0),
        np.concatenate(cong_prob_batches, axis=0),
    )


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


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_flat = x.reshape(-1).astype(np.float64)
    y_flat = y.reshape(-1).astype(np.float64)
    x_std = float(x_flat.std())
    y_std = float(y_flat.std())
    if x_std == 0.0 or y_std == 0.0:
        return 0.0
    return float(np.corrcoef(x_flat, y_flat)[0, 1])


def topk_congestion_rate(
    score: np.ndarray,
    cong_true: np.ndarray,
    fractions: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> dict[str, float]:
    flat_score = score.reshape(-1)
    flat_true = cong_true.reshape(-1).astype(bool)
    total = flat_score.size
    order = np.argsort(flat_score)[::-1]
    rates = {}
    for fraction in fractions:
        top_count = max(1, int(np.ceil(total * fraction)))
        selected = order[:top_count]
        rates[f"top{int(fraction * 100)}_congestion_rate"] = float(flat_true[selected].mean())
    return rates


def save_uncertainty_error_scatter(
    pred_std: np.ndarray,
    abs_error: np.ndarray,
    output_path: Path,
    sample_size: int = 100_000,
) -> None:
    rng = np.random.default_rng(SEED)
    flat_std = pred_std.reshape(-1)
    flat_error = abs_error.reshape(-1)
    count = min(sample_size, flat_std.size)
    indices = rng.choice(flat_std.size, size=count, replace=False)
    plt.figure(figsize=(6, 5))
    plt.scatter(flat_std[indices], flat_error[indices], s=2, alpha=0.25)
    plt.xlabel("util_pred_std")
    plt.ylabel("absolute error")
    plt.title("MC Dropout uncertainty vs absolute error")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_uncertainty_bins(pred_std: np.ndarray, abs_error: np.ndarray, output_path: Path) -> pd.DataFrame:
    flat_std = pred_std.reshape(-1)
    flat_error = abs_error.reshape(-1)
    quantiles = np.quantile(flat_std, np.linspace(0.0, 1.0, 11))
    quantiles = np.unique(quantiles)
    records = []

    if len(quantiles) <= 2:
        records.append(
            {
                "bin": 0,
                "std_min": float(flat_std.min()),
                "std_max": float(flat_std.max()),
                "count": int(flat_std.size),
                "mean_abs_error": float(flat_error.mean()),
            }
        )
    else:
        for bin_index in range(len(quantiles) - 1):
            left = quantiles[bin_index]
            right = quantiles[bin_index + 1]
            if bin_index == len(quantiles) - 2:
                mask = (flat_std >= left) & (flat_std <= right)
            else:
                mask = (flat_std >= left) & (flat_std < right)
            if not np.any(mask):
                continue
            records.append(
                {
                    "bin": int(bin_index),
                    "std_min": float(left),
                    "std_max": float(right),
                    "count": int(mask.sum()),
                    "mean_abs_error": float(flat_error[mask].mean()),
                }
            )

    df = pd.DataFrame.from_records(records)
    plt.figure(figsize=(8, 4.5))
    plt.bar(df["bin"].astype(str), df["mean_abs_error"])
    plt.xlabel("util_pred_std quantile bin")
    plt.ylabel("mean absolute error")
    plt.title("Average absolute error by uncertainty bin")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return df


def save_prediction_interval_example(
    util_mean: np.ndarray,
    util_std: np.ndarray,
    util_true: np.ndarray,
    output_path: Path,
    max_points: int = 500,
) -> None:
    edge_index = int(np.argmax(util_true.mean(axis=0)))
    length = min(max_points, util_true.shape[0])
    x_axis = np.arange(length)
    mean = util_mean[:length, edge_index]
    std = util_std[:length, edge_index]
    true = util_true[:length, edge_index]

    plt.figure(figsize=(12, 4.8))
    plt.plot(x_axis, true, label="true utilization", linewidth=1.2)
    plt.plot(x_axis, mean, label="pred mean", linewidth=1.2)
    plt.fill_between(
        x_axis,
        mean - 2.0 * std,
        mean + 2.0 * std,
        alpha=0.25,
        label="pred mean +/- 2std",
    )
    plt.xlabel("test sample index")
    plt.ylabel("utilization")
    plt.title(f"Prediction interval example, edge index {edge_index}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_risk_topk_plot(risk_rows: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    for score_name, group in risk_rows.groupby("score_name"):
        plt.plot(
            group["top_percent"],
            group["true_congestion_rate"],
            marker="o",
            linewidth=1.8,
            label=score_name,
        )
    plt.xlabel("top-k risk percentage")
    plt.ylabel("true congestion rate")
    plt.title("True congestion rate in top-risk positions")
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.legend()
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
    plt.title("MC Dropout congestion confusion, threshold=0.95")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    require_file(args.model_path)
    adj_path = PROCESSED_DIR / "edge_adj_topo144.npy"
    require_file(adj_path)

    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be positive, got {args.batch_size}")
    if args.mc_samples <= 1:
        raise ValueError(f"--mc_samples must be > 1 to compute std, got {args.mc_samples}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("MC Dropout evaluation settings:")
    print(f"- Python version: {platform.python_version()}")
    print(f"- torch.__version__: {torch.__version__}")
    print(f"- torch.version.cuda: {torch.version.cuda}")
    print(f"- torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"- device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"- GPU: {torch.cuda.get_device_name(0)}")
        print(f"- GPU total memory GB: {props.total_memory / 1024**3:.2f}")
    else:
        print("WARNING: 当前未使用 GPU，MC Dropout 推理会明显变慢。")
    print(f"- MC samples: {args.mc_samples}")
    print(f"- threshold: {args.threshold}")
    print(f"- risk_lambda: {args.risk_lambda}")

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model = UGGRU(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    adj = torch.from_numpy(np.load(adj_path).astype(np.float32)).to(device)
    test_loader = build_test_loader(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    util_true, load_true, cong_true = collect_truth(test_loader)
    print(f"- test util_true shape: {util_true.shape}")

    util_samples = []
    load_samples = []
    cong_prob_samples = []
    try:
        for sample_index in tqdm(range(args.mc_samples), desc="MC Dropout samples"):
            util_pred, load_pred, cong_prob = run_one_mc_pass(
                model=model,
                loader=test_loader,
                adj=adj,
                device=device,
                pass_index=sample_index + 1,
            )
            util_samples.append(util_pred)
            load_samples.append(load_pred)
            cong_prob_samples.append(cong_prob)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            print("CUDA out of memory during MC Dropout inference.")
            print("请尝试 --batch_size 4 或 --batch_size 2。")
        raise

    util_stack = np.stack(util_samples, axis=0).astype(np.float32)
    load_stack = np.stack(load_samples, axis=0).astype(np.float32)
    cong_prob_stack = np.stack(cong_prob_samples, axis=0).astype(np.float32)

    util_pred_mean = util_stack.mean(axis=0).astype(np.float32)
    util_pred_std = util_stack.std(axis=0).astype(np.float32)
    load_pred_mean = load_stack.mean(axis=0).astype(np.float32)
    load_pred_std = load_stack.std(axis=0).astype(np.float32)
    cong_prob_mean = cong_prob_stack.mean(axis=0).astype(np.float32)
    cong_prob_std = cong_prob_stack.std(axis=0).astype(np.float32)

    del util_stack, load_stack, cong_prob_stack

    risk_score = (util_pred_mean + args.risk_lambda * util_pred_std).astype(np.float32)
    congestion_risk_score = (
        cong_prob_mean + args.risk_lambda * cong_prob_std
    ).astype(np.float32)
    pred_congestion = cong_prob_mean >= args.threshold

    util_diff = util_pred_mean - util_true
    load_diff = load_pred_mean - load_true
    abs_error = np.abs(util_diff)
    coverage_1std = np.abs(util_true - util_pred_mean) <= util_pred_std
    coverage_2std = np.abs(util_true - util_pred_mean) <= 2.0 * util_pred_std
    class_metrics = compute_prf(pred_congestion, cong_true >= 0.5)
    util_risk_rates = topk_congestion_rate(risk_score, cong_true)
    cong_risk_rates = topk_congestion_rate(congestion_risk_score, cong_true)

    risk_rows = []
    for score_name, rates in [
        ("risk_score", util_risk_rates),
        ("congestion_risk_score", cong_risk_rates),
    ]:
        for key, value in rates.items():
            top_percent = int(key.replace("top", "").replace("_congestion_rate", ""))
            risk_rows.append(
                {
                    "score_name": score_name,
                    "top_percent": top_percent,
                    "true_congestion_rate": value,
                }
            )
    risk_topk = pd.DataFrame.from_records(risk_rows)

    metrics = {
        "mc_samples": int(args.mc_samples),
        "threshold": float(args.threshold),
        "risk_lambda": float(args.risk_lambda),
        "MAE_util": float(abs_error.mean()),
        "RMSE_util": float(np.sqrt((util_diff * util_diff).mean())),
        "MAE_load_norm": float(np.abs(load_diff).mean()),
        "RMSE_load_norm": float(np.sqrt((load_diff * load_diff).mean())),
        "coverage_1std": float(coverage_1std.mean()),
        "coverage_2std": float(coverage_2std.mean()),
        "uncertainty_error_corr": pearson_corr(util_pred_std, abs_error),
        "mean_pred_std": float(util_pred_std.mean()),
        "p50_pred_std": float(np.percentile(util_pred_std, 50)),
        "p90_pred_std": float(np.percentile(util_pred_std, 90)),
        "p95_pred_std": float(np.percentile(util_pred_std, 95)),
        "p99_pred_std": float(np.percentile(util_pred_std, 99)),
        **class_metrics,
        "risk_top1_congestion_rate": util_risk_rates["top1_congestion_rate"],
        "risk_top5_congestion_rate": util_risk_rates["top5_congestion_rate"],
        "risk_top10_congestion_rate": util_risk_rates["top10_congestion_rate"],
        "congestion_risk_top1_congestion_rate": cong_risk_rates["top1_congestion_rate"],
        "congestion_risk_top5_congestion_rate": cong_risk_rates["top5_congestion_rate"],
        "congestion_risk_top10_congestion_rate": cong_risk_rates["top10_congestion_rate"],
        "used_cuda": bool(device.type == "cuda"),
    }

    results_dir = ensure_dir(RESULTS_DIR)
    predictions_path = results_dir / "mc_dropout_predictions_test.npz"
    metrics_path = results_dir / "mc_dropout_metrics.json"
    risk_topk_path = results_dir / "mc_dropout_risk_topk.csv"
    scatter_path = results_dir / "mc_dropout_uncertainty_error_scatter.png"
    bins_path = results_dir / "mc_dropout_uncertainty_bins.png"
    interval_path = results_dir / "mc_dropout_prediction_interval_example.png"
    risk_plot_path = results_dir / "mc_dropout_risk_topk.png"
    confusion_path = results_dir / "mc_dropout_congestion_confusion_threshold095.png"

    np.savez_compressed(
        predictions_path,
        util_pred_mean=util_pred_mean.astype(np.float32),
        util_pred_std=util_pred_std.astype(np.float32),
        load_pred_mean=load_pred_mean.astype(np.float32),
        load_pred_std=load_pred_std.astype(np.float32),
        cong_prob_mean=cong_prob_mean.astype(np.float32),
        cong_prob_std=cong_prob_std.astype(np.float32),
        util_true=util_true.astype(np.float32),
        load_true=load_true.astype(np.float32),
        cong_true=cong_true.astype(np.float32),
        risk_score=risk_score.astype(np.float32),
        congestion_risk_score=congestion_risk_score.astype(np.float32),
    )
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(metrics), file, indent=2, ensure_ascii=False)
    risk_topk.to_csv(risk_topk_path, index=False, float_format="%.9g")

    save_uncertainty_error_scatter(util_pred_std, abs_error, scatter_path)
    save_uncertainty_bins(util_pred_std, abs_error, bins_path)
    save_prediction_interval_example(util_pred_mean, util_pred_std, util_true, interval_path)
    save_risk_topk_plot(risk_topk, risk_plot_path)
    save_confusion(pred_congestion, cong_true >= 0.5, confusion_path)

    print("\nMC Dropout uncertainty evaluation complete:")
    for key in [
        "mc_samples",
        "threshold",
        "MAE_util",
        "RMSE_util",
        "coverage_1std",
        "coverage_2std",
        "uncertainty_error_corr",
        "Precision",
        "Recall",
        "F1",
        "risk_top1_congestion_rate",
        "risk_top5_congestion_rate",
        "risk_top10_congestion_rate",
    ]:
        value = metrics[key]
        print(f"- {key}: {value:.9g}" if isinstance(value, float) else f"- {key}: {value}")
    print("Output files:")
    for path in [
        predictions_path,
        metrics_path,
        risk_topk_path,
        scatter_path,
        bins_path,
        interval_path,
        risk_plot_path,
        confusion_path,
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
