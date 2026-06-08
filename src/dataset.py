import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class LinkStateDataStore:
    """CPU-side holder for the compressed sample archive.

    The archive is loaded once and shared by train/val/test datasets. Batches are
    moved to GPU by the training loop, not by the dataset.
    """

    def __init__(
        self,
        samples_path: str | Path = PROCESSED_DIR / "samples_topo144_seq12.npz",
        splits_path: str | Path = PROCESSED_DIR / "splits_topo144_seq12.json",
    ) -> None:
        self.samples_path = Path(samples_path)
        self.splits_path = Path(splits_path)
        if not self.samples_path.exists():
            raise FileNotFoundError(f"Sample archive does not exist: {self.samples_path}")
        if not self.splits_path.exists():
            raise FileNotFoundError(f"Split file does not exist: {self.splits_path}")

        with np.load(self.samples_path) as data:
            self.X = data["X"].astype(np.float32, copy=False)
            self.y_utilization = data["y_utilization"].astype(np.float32, copy=False)
            self.y_load_mbps_norm = data["y_load_mbps_norm"].astype(np.float32, copy=False)
            self.y_congestion = data["y_congestion"].astype(np.float32, copy=False)
            self.feature_names = data["feature_names"]
            self.edge_ids = data["edge_ids"]
            self.times = data["times"]

        with self.splits_path.open("r", encoding="utf-8") as file:
            self.splits: dict[str, Any] = json.load(file)

    def split_bounds(self, split: str) -> tuple[int, int]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val, or test, got {split}")
        return int(self.splits[f"{split}_start"]), int(self.splits[f"{split}_end"])


class LinkStateDataset(Dataset):
    def __init__(
        self,
        split: str,
        samples_path: str | Path = PROCESSED_DIR / "samples_topo144_seq12.npz",
        splits_path: str | Path = PROCESSED_DIR / "splits_topo144_seq12.json",
        store: LinkStateDataStore | None = None,
    ) -> None:
        self.store = store or LinkStateDataStore(samples_path, splits_path)
        self.split = split
        self.start, self.end = self.store.split_bounds(split)

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        real_index = self.start + index
        x = torch.from_numpy(self.store.X[real_index])
        y_util = torch.from_numpy(self.store.y_utilization[real_index])
        y_load = torch.from_numpy(self.store.y_load_mbps_norm[real_index])
        y_cong = torch.from_numpy(self.store.y_congestion[real_index])
        return x, y_util, y_load, y_cong


def create_dataloaders(
    batch_size: int,
    num_workers: int,
    samples_path: str | Path = PROCESSED_DIR / "samples_topo144_seq12.npz",
    splits_path: str | Path = PROCESSED_DIR / "splits_topo144_seq12.json",
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")

    store = LinkStateDataStore(samples_path=samples_path, splits_path=splits_path)
    train_dataset = LinkStateDataset("train", store=store)
    val_dataset = LinkStateDataset("val", store=store)
    test_dataset = LinkStateDataset("test", store=store)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
