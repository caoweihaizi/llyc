import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check prediction sample archive and edge graph.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--link_state",
        type=Path,
        default=PROCESSED_DIR / "link_state_topo144_shortest_delay.csv",
    )
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--sample_index", type=int, default=123)
    parser.add_argument("--edge_index", type=int, default=17)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")


def print_distribution(name: str, values: np.ndarray) -> None:
    percentiles = np.percentile(values, [50, 90, 95, 99])
    print(f"- {name} min: {values.min():.9g}")
    print(f"- {name} mean: {values.mean():.9g}")
    print(f"- {name} p50: {percentiles[0]:.9g}")
    print(f"- {name} p90: {percentiles[1]:.9g}")
    print(f"- {name} p95: {percentiles[2]:.9g}")
    print(f"- {name} p99: {percentiles[3]:.9g}")
    print(f"- {name} max: {values.max():.9g}")


def split_slice(splits: dict, name: str) -> slice:
    return slice(int(splits[f"{name}_start"]), int(splits[f"{name}_end"]))


def save_figures(
    y_utilization: np.ndarray,
    y_congestion: np.ndarray,
    splits: dict,
    results_dir: Path,
) -> tuple[Path, Path]:
    ensure_dir(results_dir)
    util_path = results_dir / "sample_y_utilization_distribution.png"
    ratio_path = results_dir / "sample_y_congestion_ratio_split.png"

    plt.figure(figsize=(9, 5))
    plt.hist(y_utilization.reshape(-1), bins=100, log=True)
    plt.xlabel("y_utilization")
    plt.ylabel("count (log)")
    plt.title("Target utilization distribution")
    plt.tight_layout()
    plt.savefig(util_path, dpi=200)
    plt.close()

    names = ["train", "val", "test"]
    ratios = [
        float(y_congestion[split_slice(splits, name)].mean())
        for name in names
    ]
    plt.figure(figsize=(7, 4.5))
    plt.bar(names, ratios)
    plt.ylabel("positive ratio")
    plt.title("y_congestion positive ratio by split")
    plt.tight_layout()
    plt.savefig(ratio_path, dpi=200)
    plt.close()
    return util_path, ratio_path


def find_link_state_rows(link_state_path: Path, time_value: int, edge_id: int) -> pd.DataFrame:
    usecols = ["time", "edge_id", "utilization", "next_utilization"]
    for chunk in pd.read_csv(link_state_path, usecols=usecols, chunksize=200_000):
        subset = chunk[(chunk["time"] == time_value) & (chunk["edge_id"] == edge_id)]
        if not subset.empty:
            return subset
    return pd.DataFrame(columns=usecols)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    processed_dir = args.processed_dir
    results_dir = ensure_dir(args.results_dir)
    topo_name = str(config.get("topo_name", "topo144"))
    seq_len = int(args.seq_len if args.seq_len is not None else config.get("seq_len", 12))
    suffix = f"{topo_name}_seq{seq_len}"

    samples_path = processed_dir / f"samples_{suffix}.npz"
    splits_path = processed_dir / f"splits_{suffix}.json"
    adjacency_path = processed_dir / f"edge_adj_{topo_name}.npy"
    scaler_path = processed_dir / f"sample_scaler_{suffix}.json"

    for path in [samples_path, splits_path, adjacency_path, scaler_path, args.link_state]:
        require_file(path)

    with splits_path.open("r", encoding="utf-8") as file:
        splits = json.load(file)
    with scaler_path.open("r", encoding="utf-8") as file:
        scaler = json.load(file)

    data = np.load(samples_path)
    X = data["X"]
    y_utilization = data["y_utilization"]
    y_load_mbps_norm = data["y_load_mbps_norm"]
    y_congestion = data["y_congestion"]
    feature_names = data["feature_names"]
    edge_ids = data["edge_ids"]
    times = data["times"]
    adjacency = np.load(adjacency_path)

    print("Prediction sample summary:")
    print(f"- X.shape: {X.shape}")
    print(f"- y_utilization.shape: {y_utilization.shape}")
    print(f"- y_load_mbps_norm.shape: {y_load_mbps_norm.shape}")
    print(f"- y_congestion.shape: {y_congestion.shape}")
    print(f"- feature_names: {feature_names.tolist()}")
    print(f"- edge_ids count: {len(edge_ids)}")
    print(f"- times range: [{int(times.min())}, {int(times.max())}]")

    problems = []
    num_samples = X.shape[0]
    expected_y_shape = (num_samples, 288)
    if X.ndim != 4:
        problems.append(f"X.ndim must be 4, got {X.ndim}")
    if X.shape[1] != seq_len:
        problems.append(f"X.shape[1] must be {seq_len}, got {X.shape[1]}")
    if X.shape[2] != 288:
        problems.append(f"X.shape[2] must be 288, got {X.shape[2]}")
    if X.shape[3] != 6:
        problems.append(f"X.shape[3] must be 6, got {X.shape[3]}")
    for name, values in [
        ("y_utilization", y_utilization),
        ("y_load_mbps_norm", y_load_mbps_norm),
        ("y_congestion", y_congestion),
    ]:
        if values.shape != expected_y_shape:
            problems.append(f"{name}.shape must be {expected_y_shape}, got {values.shape}")

    if adjacency.shape != (288, 288):
        problems.append(f"edge adjacency shape must be (288, 288), got {adjacency.shape}")
    if not np.allclose(adjacency, adjacency.T):
        problems.append("edge adjacency is not symmetric")

    for name, values in [
        ("X", X),
        ("y_utilization", y_utilization),
        ("y_load_mbps_norm", y_load_mbps_norm),
        ("y_congestion", y_congestion),
        ("edge_adj", adjacency),
    ]:
        if np.isnan(values).any() or np.isinf(values).any():
            problems.append(f"{name} contains nan or inf")

    split_order_ok = (
        splits["train_start"] == 0
        and splits["train_start"] < splits["train_end"] <= splits["val_start"]
        and splits["val_start"] < splits["val_end"] <= splits["test_start"]
        and splits["test_start"] < splits["test_end"] == splits["num_samples"]
    )
    if not split_order_ok:
        problems.append("train/val/test splits are not chronological or overlap")
    if splits["num_samples"] != num_samples:
        problems.append(f"splits num_samples mismatch: {splits['num_samples']} vs {num_samples}")

    print("\ny_congestion positive ratio:")
    print(f"- overall: {float(y_congestion.mean()):.9g}")
    for name in ["train", "val", "test"]:
        print(f"- {name}: {float(y_congestion[split_slice(splits, name)].mean()):.9g}")

    print("\ny_utilization statistics:")
    print_distribution("y_utilization", y_utilization.reshape(-1))

    print("\nX feature statistics:")
    for feature_idx, feature_name in enumerate(feature_names):
        values = X[:, :, :, feature_idx]
        print(
            f"- {feature_name}: mean={values.mean():.9g}, std={values.std():.9g}, "
            f"min={values.min():.9g}, max={values.max():.9g}"
        )

    sample_index = min(max(args.sample_index, 0), num_samples - 1)
    edge_position = min(max(args.edge_index, 0), len(edge_ids) - 1)
    edge_id = int(edge_ids[edge_position])
    time_value = int(times[sample_index])
    row = find_link_state_rows(args.link_state, time_value=time_value, edge_id=edge_id)
    if row.empty:
        problems.append(
            f"Could not find link-state row for time={time_value}, edge_id={edge_id}"
        )
    else:
        record = row.iloc[0]
        y_error = abs(float(y_utilization[sample_index, edge_position]) - float(record["next_utilization"]))
        x_error = abs(float(X[sample_index, -1, edge_position, 0]) - float(record["utilization"]))
        print("\nSample cross-check:")
        print(f"- sample_index: {sample_index}")
        print(f"- time: {time_value}")
        print(f"- edge_id: {edge_id}")
        print(f"- y_utilization error: {y_error:.12g}")
        print(f"- X last-step utilization error: {x_error:.12g}")
        if y_error > 1e-7:
            problems.append("y_utilization does not match next_utilization in source CSV")
        if x_error > 1e-7:
            problems.append("X last-step utilization does not match utilization in source CSV")

    util_fig, ratio_fig = save_figures(y_utilization, y_congestion, splits, results_dir)

    print("\nScaler file:")
    print(f"- feature_names: {scaler.get('feature_names')}")
    print("\nOutput figures:")
    print(f"- {util_fig}")
    print(f"- {ratio_fig}")

    if problems:
        print("\nCheck result: FAILED")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)

    print("\nCheck result: OK")
    print("\nChecked files:")
    print(f"- {samples_path}")
    print(f"- {splits_path}")
    print(f"- {adjacency_path}")
    print(f"- {scaler_path}")


if __name__ == "__main__":
    main()
