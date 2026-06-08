import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from dataset import create_dataloaders
from models import UGGRU, count_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RUNS_DIR = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the UGGRU traffic predictor.")
    parser.add_argument("--model", choices=["uggru"], default="uggru")
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--gcn_hidden", type=int, default=32)
    parser.add_argument("--gru_hidden", type=int, default=64)
    parser.add_argument("--gru_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Python version: {sys.version}")
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU total memory GB: {props.total_memory / 1024**3:.2f}")
    else:
        print("WARNING: 当前未使用 GPU，UGGRU 训练会明显变慢。")
    return device


def move_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(tensor.to(device, non_blocking=True).float() for tensor in batch)  # type: ignore[return-value]


def compute_classification_metrics(logits: torch.Tensor, target: torch.Tensor) -> tuple[int, int, int]:
    pred = logits >= 0.0
    truth = target >= 0.5
    tp = int((pred & truth).sum().item())
    fp = int((pred & ~truth).sum().item())
    fn = int((~pred & truth).sum().item())
    return tp, fp, fn


def finalize_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def run_epoch(
    model: UGGRU,
    loader,
    adj: torch.Tensor,
    device: torch.device,
    mse_loss: nn.Module,
    bce_loss: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    desc: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_samples = 0
    loss_sum = util_loss_sum = load_loss_sum = cong_loss_sum = 0.0
    mae_sum = sse_sum = element_count = 0.0
    tp = fp = fn = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        x, y_util, y_load, y_cong = move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            util_pred, load_pred, cong_logit = model(x, adj)
            util_loss = mse_loss(util_pred, y_util)
            load_loss = mse_loss(load_pred, y_load)
            cong_loss = bce_loss(cong_logit, y_cong)
            loss = util_loss + 0.3 * load_loss + 0.5 * cong_loss

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = x.shape[0]
        total_samples += batch_size
        loss_sum += float(loss.item()) * batch_size
        util_loss_sum += float(util_loss.item()) * batch_size
        load_loss_sum += float(load_loss.item()) * batch_size
        cong_loss_sum += float(cong_loss.item()) * batch_size

        diff = util_pred.detach() - y_util
        mae_sum += float(diff.abs().sum().item())
        sse_sum += float((diff * diff).sum().item())
        element_count += float(diff.numel())
        batch_tp, batch_fp, batch_fn = compute_classification_metrics(cong_logit.detach(), y_cong)
        tp += batch_tp
        fp += batch_fp
        fn += batch_fn

    precision, recall, f1 = finalize_prf(tp, fp, fn)
    return {
        "loss": loss_sum / total_samples,
        "util_loss": util_loss_sum / total_samples,
        "load_loss": load_loss_sum / total_samples,
        "cong_loss": cong_loss_sum / total_samples,
        "MAE_util": mae_sum / element_count,
        "RMSE_util": float(np.sqrt(sse_sum / element_count)),
        "cong_precision": precision,
        "cong_recall": recall,
        "cong_f1": f1,
    }


def evaluate_test(model, loader, adj, device, mse_loss, bce_loss) -> dict[str, float]:
    with torch.no_grad():
        metrics = run_epoch(
            model=model,
            loader=loader,
            adj=adj,
            device=device,
            mse_loss=mse_loss,
            bce_loss=bce_loss,
            optimizer=None,
            grad_clip=0.0,
            desc="Test",
        )
    return {
        "test_loss": metrics["loss"],
        "test_util_loss": metrics["util_loss"],
        "test_load_loss": metrics["load_loss"],
        "test_cong_loss": metrics["cong_loss"],
        "test_MAE_util": metrics["MAE_util"],
        "test_RMSE_util": metrics["RMSE_util"],
        "test_cong_precision": metrics["cong_precision"],
        "test_cong_recall": metrics["cong_recall"],
        "test_cong_f1": metrics["cong_f1"],
    }


def save_checkpoint(path: Path, model: UGGRU, optimizer, epoch: int, config: dict, val_loss: float) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": config["model_config"],
            "train_config": config,
            "val_loss": val_loss,
        },
        path,
    )


def plot_loss(log_path: Path, output_path: Path) -> None:
    import pandas as pd

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
    set_seed(args.seed)
    device = get_device()

    run_dir = RUNS_DIR / "uggru_topo144_seq12"
    result_dir = RESULTS_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    samples_path = PROCESSED_DIR / f"samples_topo144_seq{args.seq_len}.npz"
    splits_path = PROCESSED_DIR / f"splits_topo144_seq{args.seq_len}.json"
    adj_path = PROCESSED_DIR / "edge_adj_topo144.npy"

    pin_memory = device.type == "cuda"
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples_path=samples_path,
        splits_path=splits_path,
        pin_memory=pin_memory,
    )
    store = train_loader.dataset.store
    train_start, train_end = store.split_bounds("train")
    train_cong = store.y_congestion[train_start:train_end]
    positive_count = float(train_cong.sum())
    total_count = float(train_cong.size)
    negative_count = total_count - positive_count
    if positive_count <= 0:
        raise RuntimeError("Train split has zero positive congestion labels; cannot compute pos_weight.")
    pos_weight_value = negative_count / positive_count

    adj = torch.from_numpy(np.load(adj_path).astype(np.float32)).to(device)
    model_config = {
        "num_features": int(store.X.shape[-1]),
        "gcn_hidden": args.gcn_hidden,
        "gru_hidden": args.gru_hidden,
        "gru_layers": args.gru_layers,
        "dropout": args.dropout,
    }
    model = UGGRU(**model_config).to(device)
    print(f"Model parameters: {count_parameters(model)}")
    print(f"batch_size: {args.batch_size}")
    print(f"epochs: {args.epochs}")
    print(f"train/val/test samples: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"y_congestion train positive ratio: {positive_count / total_count:.9g}")
    print(f"BCE pos_weight: {pos_weight_value:.9g}")
    print(f"X shape: {store.X.shape}")
    print(f"adj shape: {tuple(adj.shape)}")

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_config = vars(args).copy()
    train_config.update(
        {
            "samples_path": str(samples_path),
            "splits_path": str(splits_path),
            "adj_path": str(adj_path),
            "device": str(device),
            "model_config": model_config,
            "pos_weight": pos_weight_value,
        }
    )
    with (run_dir / "train_config.json").open("w", encoding="utf-8") as file:
        json.dump(train_config, file, indent=2, ensure_ascii=False)

    log_path = run_dir / "train_log.csv"
    best_model_path = run_dir / "best_model.pt"
    last_model_path = run_dir / "last_model.pt"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_util_loss",
        "train_load_loss",
        "train_cong_loss",
        "val_loss",
        "val_util_loss",
        "val_load_loss",
        "val_cong_loss",
        "val_MAE_util",
        "val_RMSE_util",
        "val_cong_precision",
        "val_cong_recall",
        "val_cong_f1",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, adj, device, mse_loss, bce_loss, optimizer, args.grad_clip, f"Epoch {epoch} train"
            )
            with torch.no_grad():
                val_metrics = run_epoch(
                    model, val_loader, adj, device, mse_loss, bce_loss, None, args.grad_clip, f"Epoch {epoch} val"
                )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_util_loss": train_metrics["util_loss"],
                "train_load_loss": train_metrics["load_loss"],
                "train_cong_loss": train_metrics["cong_loss"],
                "val_loss": val_metrics["loss"],
                "val_util_loss": val_metrics["util_loss"],
                "val_load_loss": val_metrics["load_loss"],
                "val_cong_loss": val_metrics["cong_loss"],
                "val_MAE_util": val_metrics["MAE_util"],
                "val_RMSE_util": val_metrics["RMSE_util"],
                "val_cong_precision": val_metrics["cong_precision"],
                "val_cong_recall": val_metrics["cong_recall"],
                "val_cong_f1": val_metrics["cong_f1"],
            }
            with log_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(row)

            print(
                f"Epoch {epoch}: train_loss={row['train_loss']:.6f}, "
                f"val_loss={row['val_loss']:.6f}, val_MAE={row['val_MAE_util']:.6f}, "
                f"val_F1={row['val_cong_f1']:.6f}"
            )

            save_checkpoint(last_model_path, model, optimizer, epoch, train_config, row["val_loss"])
            if row["val_loss"] < best_val_loss:
                best_val_loss = row["val_loss"]
                epochs_without_improvement = 0
                save_checkpoint(best_model_path, model, optimizer, epoch, train_config, best_val_loss)
            else:
                epochs_without_improvement += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("CUDA out of memory. 请尝试 --batch_size 4 或 --batch_size 2。")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        raise

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        test_metrics = evaluate_test(model, test_loader, adj, device, mse_loss, bce_loss)

    metrics_path = result_dir / "uggru_test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2, ensure_ascii=False)
    plot_loss(log_path, result_dir / "uggru_train_val_loss.png")

    print("Training complete.")
    print("Output files:")
    print(f"- {best_model_path}")
    print(f"- {last_model_path}")
    print(f"- {log_path}")
    print(f"- {run_dir / 'train_config.json'}")
    print(f"- {metrics_path}")
    print(f"- {result_dir / 'uggru_train_val_loss.png'}")


if __name__ == "__main__":
    main()
