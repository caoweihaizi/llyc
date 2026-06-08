import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from baseline_models import GRUOnlyBaseline, LSTMOnlyBaseline, count_parameters
from dataset import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "runs"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRU-only or LSTM-only baselines.")
    parser.add_argument("--baseline", choices=["gru", "lstm"], required=True)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--rnn_hidden", type=int, default=64)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=SEED)
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
        print("WARNING: 当前未使用 GPU，RNN baseline 训练会明显变慢。")
    return device


def move_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(tensor.to(device, non_blocking=True).float() for tensor in batch)  # type: ignore[return-value]


def finalize_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def build_model(baseline: str, model_config: dict[str, Any]) -> nn.Module:
    if baseline == "gru":
        return GRUOnlyBaseline(**model_config)
    if baseline == "lstm":
        return LSTMOnlyBaseline(**model_config)
    raise ValueError(f"Unsupported baseline: {baseline}")


def run_epoch(
    model: nn.Module,
    loader,
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
    loss_sum = 0.0
    mae_sum = 0.0
    sse_sum = 0.0
    element_count = 0.0
    tp = fp = fn = 0

    for batch in tqdm(loader, desc=desc, leave=False, disable=True):
        x, y_util, y_load, y_cong = move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            util_pred, load_pred, cong_logit = model(x)
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
        diff = util_pred.detach() - y_util
        mae_sum += float(diff.abs().sum().item())
        sse_sum += float((diff * diff).sum().item())
        element_count += float(diff.numel())
        pred = cong_logit.detach() >= 0.0
        truth = y_cong >= 0.5
        tp += int((pred & truth).sum().item())
        fp += int((pred & ~truth).sum().item())
        fn += int((~pred & truth).sum().item())

    precision, recall, f1 = finalize_prf(tp, fp, fn)
    return {
        "loss": loss_sum / total_samples,
        "MAE_util": mae_sum / element_count,
        "RMSE_util": float(np.sqrt(sse_sum / element_count)),
        "cong_precision": precision,
        "cong_recall": recall,
        "cong_f1": f1,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_config: dict[str, Any],
    val_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": train_config["model_config"],
            "train_config": train_config,
            "val_loss": val_loss,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    samples_path = PROCESSED_DIR / f"samples_topo144_seq{args.seq_len}.npz"
    splits_path = PROCESSED_DIR / f"splits_topo144_seq{args.seq_len}.json"
    run_dir = RUNS_DIR / f"{args.baseline}_baseline_topo144_seq{args.seq_len}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples_path=samples_path,
        splits_path=splits_path,
        pin_memory=device.type == "cuda",
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

    model_config = {
        "num_features": int(store.X.shape[-1]),
        "rnn_hidden": args.rnn_hidden,
        "rnn_layers": args.rnn_layers,
        "dropout": args.dropout,
    }
    model = build_model(args.baseline, model_config).to(device)
    print(f"baseline: {args.baseline}")
    print(f"Model parameters: {count_parameters(model)}")
    print(f"train/val/test samples: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"train positive ratio: {positive_count / total_count:.9g}")
    print(f"pos_weight: {pos_weight_value:.9g}")
    print(f"X shape: {store.X.shape}")

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_config = vars(args).copy()
    train_config.update(
        {
            "samples_path": str(samples_path),
            "splits_path": str(splits_path),
            "device": str(device),
            "model_config": model_config,
            "pos_weight": pos_weight_value,
            "used_cuda": bool(device.type == "cuda"),
        }
    )
    with (run_dir / "train_config.json").open("w", encoding="utf-8") as file:
        json.dump(to_builtin(train_config), file, indent=2, ensure_ascii=False)

    log_path = run_dir / "train_log.csv"
    best_model_path = run_dir / "best_model.pt"
    last_model_path = run_dir / "last_model.pt"
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
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
                model=model,
                loader=train_loader,
                device=device,
                mse_loss=mse_loss,
                bce_loss=bce_loss,
                optimizer=optimizer,
                grad_clip=args.grad_clip,
                desc=f"{args.baseline.upper()} epoch {epoch} train",
            )
            with torch.no_grad():
                val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    mse_loss=mse_loss,
                    bce_loss=bce_loss,
                    optimizer=None,
                    grad_clip=args.grad_clip,
                    desc=f"{args.baseline.upper()} epoch {epoch} val",
                )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
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
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            print("CUDA out of memory. 请尝试 --batch_size 8 或 --batch_size 4。")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        raise

    print("RNN baseline training complete.")
    print("Output files:")
    print(f"- {best_model_path}")
    print(f"- {last_model_path}")
    print(f"- {log_path}")
    print(f"- {run_dir / 'train_config.json'}")


if __name__ == "__main__":
    main()
