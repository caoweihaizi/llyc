import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "runs"

FEATURE_NAMES = np.array(
    [
        "utilization",
        "load_mbps_norm",
        "delay_ms_norm",
        "queue_len_norm",
        "remain_visible_time_norm",
        "congestion_label",
    ]
)
REQUIRED_COLUMNS = [
    "time",
    "edge_id",
    "load_mbps",
    "utilization",
    "delay_ms",
    "queue_len",
    "remain_visible_time",
    "congestion_label",
    "next_load_mbps",
    "next_utilization",
    "next_congestion_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build supervised prediction samples from link states.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--runs_dir", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DIR / "link_state_topo144_shortest_delay.csv",
    )
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--chunk_times", type=int, default=512)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing sample outputs.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def safe_std(std_value: float) -> float:
    if not np.isfinite(std_value) or abs(std_value) < 1e-12:
        return 1.0
    return float(std_value)


def compute_splits(num_samples: int) -> dict:
    train_count = int(num_samples * 0.70)
    val_count = int(num_samples * 0.15)
    test_count = num_samples - train_count - val_count
    train_start = 0
    train_end = train_count
    val_start = train_end
    val_end = val_start + val_count
    test_start = val_end
    test_end = test_start + test_count
    return {
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
        "num_samples": num_samples,
    }


def read_link_state_matrices(input_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    print(f"Reading necessary columns from {input_path}")
    df = pd.read_csv(input_path, usecols=REQUIRED_COLUMNS)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Input link-state CSV missing columns: {missing}")

    print("Sorting by time and edge_id")
    df = df.sort_values(["time", "edge_id"], kind="mergesort").reset_index(drop=True)
    times = np.sort(df["time"].unique().astype(np.int32))
    edge_ids = np.sort(df["edge_id"].unique().astype(np.int32))
    num_times = len(times)
    num_edges = len(edge_ids)

    if len(df) != num_times * num_edges:
        raise ValueError(
            f"Expected dense [time, edge] table, got rows={len(df)}, "
            f"num_times*num_edges={num_times * num_edges}"
        )
    if not np.array_equal(df["time"].to_numpy(dtype=np.int32).reshape(num_times, num_edges)[:, 0], times):
        raise ValueError("CSV rows are not dense by time after sorting.")
    if not np.array_equal(df["edge_id"].to_numpy(dtype=np.int32).reshape(num_times, num_edges)[0], edge_ids):
        raise ValueError("CSV rows are not dense by edge_id after sorting.")

    matrices = {}
    for column in tqdm(REQUIRED_COLUMNS[2:], desc="Reshaping feature/label matrices"):
        dtype = np.int8 if "label" in column else np.float32
        matrices[column] = df[column].to_numpy(dtype=dtype).reshape(num_times, num_edges)

    del df
    return matrices, times, edge_ids


def build_scaler(matrices: dict[str, np.ndarray], train_end_sample: int, seq_len: int) -> dict:
    train_time_end = train_end_sample + seq_len - 1
    train_slice = slice(0, train_time_end + 1)
    scaler = {}
    for name in ["load_mbps", "delay_ms", "queue_len", "remain_visible_time"]:
        values = matrices[name][train_slice].astype(np.float64, copy=False)
        mean = float(values.mean())
        raw_std = float(values.std())
        std = safe_std(raw_std)
        scaler[name] = {
            "mean": mean,
            "std": std,
            "raw_std": raw_std,
            "std_was_zero": bool(abs(raw_std) < 1e-12),
        }
    return scaler


def normalize_feature_matrix(
    matrices: dict[str, np.ndarray],
    scaler: dict,
) -> np.ndarray:
    num_times, num_edges = matrices["utilization"].shape
    features = np.empty((num_times, num_edges, len(FEATURE_NAMES)), dtype=np.float32)
    features[:, :, 0] = matrices["utilization"]
    features[:, :, 1] = (
        (matrices["load_mbps"] - scaler["load_mbps"]["mean"]) / scaler["load_mbps"]["std"]
    ).astype(np.float32)
    features[:, :, 2] = (
        (matrices["delay_ms"] - scaler["delay_ms"]["mean"]) / scaler["delay_ms"]["std"]
    ).astype(np.float32)
    features[:, :, 3] = (
        (matrices["queue_len"] - scaler["queue_len"]["mean"]) / scaler["queue_len"]["std"]
    ).astype(np.float32)
    features[:, :, 4] = (
        (matrices["remain_visible_time"] - scaler["remain_visible_time"]["mean"])
        / scaler["remain_visible_time"]["std"]
    ).astype(np.float32)
    features[:, :, 5] = matrices["congestion_label"].astype(np.float32)
    return features


def write_samples_memmap(
    features: np.ndarray,
    output_path: Path,
    num_samples: int,
    seq_len: int,
    chunk_times: int,
) -> np.memmap:
    if chunk_times <= 0:
        raise ValueError(f"--chunk_times must be positive, got {chunk_times}")

    shape = (num_samples, seq_len, features.shape[1], features.shape[2])
    x_memmap = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )

    for start in tqdm(range(0, num_samples, chunk_times), desc="Building X sample windows"):
        end = min(start + chunk_times, num_samples)
        source = features[start : end + seq_len - 1]
        windows = sliding_window_view(source, window_shape=seq_len, axis=0)
        # sliding_window_view places the window axis last: [N, E, F, L].
        x_memmap[start:end] = np.moveaxis(windows, -1, 1).astype(np.float32, copy=False)

    x_memmap.flush()
    return x_memmap


def save_npz_compressed(
    output_path: Path,
    x_memmap: np.memmap,
    y_utilization: np.ndarray,
    y_load_mbps_norm: np.ndarray,
    y_congestion: np.ndarray,
    edge_ids: np.ndarray,
    sample_times: np.ndarray,
) -> None:
    print(f"Writing compressed sample archive: {output_path}")
    np.savez_compressed(
        output_path,
        X=x_memmap,
        y_utilization=y_utilization.astype(np.float32, copy=False),
        y_load_mbps_norm=y_load_mbps_norm.astype(np.float32, copy=False),
        y_congestion=y_congestion.astype(np.int8, copy=False),
        feature_names=FEATURE_NAMES,
        edge_ids=edge_ids.astype(np.int32, copy=False),
        times=sample_times.astype(np.int32, copy=False),
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    processed_dir = ensure_dir(args.processed_dir)
    temp_dir = ensure_dir(args.runs_dir / "sample_build_tmp")
    topo_name = str(config.get("topo_name", "topo144"))
    seq_len = int(args.seq_len if args.seq_len is not None else config.get("seq_len", 12))

    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if not args.input.exists():
        raise FileNotFoundError(f"Link-state CSV does not exist: {args.input}")

    suffix = f"{topo_name}_seq{seq_len}"
    samples_path = processed_dir / f"samples_{suffix}.npz"
    splits_path = processed_dir / f"splits_{suffix}.json"
    scaler_path = processed_dir / f"sample_scaler_{suffix}.json"
    stats_path = processed_dir / f"sample_build_stats_{suffix}.json"
    x_memmap_path = temp_dir / f"X_{suffix}.npy"

    output_paths = [samples_path, splits_path, scaler_path, stats_path]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Sample outputs already exist. Pass --overwrite to replace them:\n"
            + "\n".join(str(path) for path in existing)
        )
    for path in existing:
        path.unlink()
    if x_memmap_path.exists():
        x_memmap_path.unlink()

    matrices, times, edge_ids = read_link_state_matrices(args.input)
    num_times, num_edges = matrices["utilization"].shape
    num_samples = num_times - seq_len + 1
    if num_samples <= 0:
        raise ValueError(f"Not enough time steps {num_times} for seq_len={seq_len}")

    splits = compute_splits(num_samples)
    splits.update(
        {
            "seq_len": seq_len,
            "num_edges": int(num_edges),
            "num_features": int(len(FEATURE_NAMES)),
            "time_order": "chronological",
            "shuffle": False,
        }
    )

    scaler = build_scaler(matrices, splits["train_end"], seq_len)
    features = normalize_feature_matrix(matrices, scaler)

    label_time_indices = np.arange(seq_len - 1, num_times, dtype=np.int32)
    sample_times = times[label_time_indices]
    y_utilization = matrices["next_utilization"][label_time_indices].astype(np.float32)
    y_load_mbps_norm = (
        (matrices["next_load_mbps"][label_time_indices] - scaler["load_mbps"]["mean"])
        / scaler["load_mbps"]["std"]
    ).astype(np.float32)
    y_congestion = matrices["next_congestion_label"][label_time_indices].astype(np.int8)

    x_memmap = write_samples_memmap(
        features=features,
        output_path=x_memmap_path,
        num_samples=num_samples,
        seq_len=seq_len,
        chunk_times=args.chunk_times,
    )
    save_npz_compressed(
        output_path=samples_path,
        x_memmap=x_memmap,
        y_utilization=y_utilization,
        y_load_mbps_norm=y_load_mbps_norm,
        y_congestion=y_congestion,
        edge_ids=edge_ids,
        sample_times=sample_times,
    )

    with splits_path.open("w", encoding="utf-8") as file:
        json.dump(splits, file, indent=2, ensure_ascii=False)

    scaler_payload = {
        "feature_names": FEATURE_NAMES.tolist(),
        "scaler": scaler,
        "normalization": {
            "utilization": "identity",
            "congestion_label": "identity",
            "load_mbps_norm": "z_score_using_train_time_range",
            "delay_ms_norm": "z_score_using_train_time_range",
            "queue_len_norm": "z_score_using_train_time_range",
            "remain_visible_time_norm": "z_score_using_train_time_range_safe_zero_std",
            "y_load_mbps_norm": "same_as_load_mbps_norm",
        },
    }
    with scaler_path.open("w", encoding="utf-8") as file:
        json.dump(scaler_payload, file, indent=2, ensure_ascii=False)

    stats = {
        "input": str(args.input),
        "num_times": int(num_times),
        "num_edges": int(num_edges),
        "seq_len": int(seq_len),
        "num_samples": int(num_samples),
        "expected_num_samples_formula": "num_times - seq_len + 1",
        "X_shape": list(x_memmap.shape),
        "y_utilization_shape": list(y_utilization.shape),
        "y_load_mbps_norm_shape": list(y_load_mbps_norm.shape),
        "y_congestion_shape": list(y_congestion.shape),
        "feature_names": FEATURE_NAMES.tolist(),
        "sample_time_min": int(sample_times.min()),
        "sample_time_max": int(sample_times.max()),
        "temp_x_memmap": str(x_memmap_path),
        "temp_x_memmap_removed_after_npz": True,
    }
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, ensure_ascii=False)

    del x_memmap
    if x_memmap_path.exists():
        x_memmap_path.unlink()
    try:
        if not any(temp_dir.iterdir()):
            shutil.rmtree(temp_dir)
    except OSError:
        pass

    print("Prediction samples built:")
    for key, value in stats.items():
        print(f"- {key}: {value}")
    print("\nOutput files:")
    print(f"- {samples_path}")
    print(f"- {splits_path}")
    print(f"- {scaler_path}")
    print(f"- {stats_path}")


if __name__ == "__main__":
    main()
