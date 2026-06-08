import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

REQUIRED_COLUMNS = [
    "time",
    "edge_id",
    "src_sat",
    "dst_sat",
    "load_mbps",
    "capacity_mbps",
    "utilization",
    "delay_ms",
    "queue_len",
    "remain_visible_time",
    "congestion_label",
    "next_load_mbps",
    "next_utilization",
    "next_congestion_label",
]

NUMERIC_COLUMNS = [
    "time",
    "edge_id",
    "src_sat",
    "dst_sat",
    "load_mbps",
    "capacity_mbps",
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
    parser = argparse.ArgumentParser(description="Check generated LEO link-state CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DIR / "link_state_topo144_shortest_delay.csv",
    )
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--chunksize", type=int, default=200_000)
    return parser.parse_args()


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def infer_expected_time_count(path: Path) -> tuple[int | None, str]:
    match = re.search(r"_debug(\d+)", path.stem)
    if match:
        return int(match.group(1)) - 1, f"debug{match.group(1)}"
    return 48383, ""


def print_distribution(name: str, values: np.ndarray) -> None:
    percentiles = np.percentile(values, [50, 90, 95, 99])
    print(f"- {name} min: {values.min():.9g}")
    print(f"- {name} mean: {values.mean():.9g}")
    print(f"- {name} p50: {percentiles[0]:.9g}")
    print(f"- {name} p90: {percentiles[1]:.9g}")
    print(f"- {name} p95: {percentiles[2]:.9g}")
    print(f"- {name} p99: {percentiles[3]:.9g}")
    print(f"- {name} max: {values.max():.9g}")


def save_curves(
    per_time: pd.DataFrame,
    results_dir: Path,
    suffix_label: str,
) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{suffix_label}" if suffix_label else ""
    avg_path = results_dir / f"leo_average_utilization_curve{suffix}.png"
    mlu_path = results_dir / f"leo_mlu_curve{suffix}.png"

    plt.figure(figsize=(12, 4.5))
    plt.plot(per_time["time"], per_time["mean"], linewidth=0.8)
    plt.xlabel("time")
    plt.ylabel("average utilization")
    plt.title("Average utilization across all ISL edges")
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(avg_path, dpi=200)
    plt.close()

    plt.figure(figsize=(12, 4.5))
    plt.plot(per_time["time"], per_time["max"], linewidth=0.8)
    plt.xlabel("time")
    plt.ylabel("maximum link utilization")
    plt.title("Maximum link utilization")
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(mlu_path, dpi=200)
    plt.close()

    return avg_path, mlu_path


def sample_next_label_check(path: Path, edge_count: int, sample_times: int = 20) -> list[str]:
    rows_to_read = max(edge_count * sample_times, edge_count * 2)
    sample = pd.read_csv(
        path,
        usecols=[
            "time",
            "edge_id",
            "load_mbps",
            "utilization",
            "congestion_label",
            "next_load_mbps",
            "next_utilization",
            "next_congestion_label",
        ],
        nrows=rows_to_read,
    )
    problems = []
    if sample["time"].nunique() < 2:
        problems.append("Not enough sampled time steps to check next_* labels.")
        return problems

    current_rows = sample.iloc[:-edge_count].reset_index(drop=True)
    next_rows = sample.iloc[edge_count:].reset_index(drop=True)
    same_edge = current_rows["edge_id"].to_numpy() == next_rows["edge_id"].to_numpy()
    if not bool(np.all(same_edge)):
        problems.append("Sampled rows are not ordered by time -> edge_id as expected.")
        return problems

    if not np.allclose(
        current_rows["next_load_mbps"].to_numpy(),
        next_rows["load_mbps"].to_numpy(),
        rtol=1e-7,
        atol=1e-7,
    ):
        problems.append("Sample next_load_mbps does not match next time load_mbps.")
    if not np.allclose(
        current_rows["next_utilization"].to_numpy(),
        next_rows["utilization"].to_numpy(),
        rtol=1e-7,
        atol=1e-7,
    ):
        problems.append("Sample next_utilization does not match next time utilization.")
    if not np.array_equal(
        current_rows["next_congestion_label"].to_numpy(),
        next_rows["congestion_label"].to_numpy(),
    ):
        problems.append("Sample next_congestion_label does not match next time congestion_label.")
    return problems


def main() -> None:
    args = parse_args()
    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input link-state file does not exist: {input_path}")
    if args.chunksize <= 0:
        raise ValueError(f"--chunksize must be positive, got {args.chunksize}")

    file_size = input_path.stat().st_size
    head = pd.read_csv(input_path, nrows=5)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in head.columns]
    if missing_columns:
        raise ValueError(f"Input file is missing required columns: {missing_columns}")

    print("Link-state file:")
    print(f"- path: {input_path}")
    print(f"- size: {format_size(file_size)}")
    print("\nTraffic unit used for this dataset:")
    print("Abilene raw unit = 100 bytes / 5 minutes")
    print("load_mbps = raw * 100 * 8 / 300 / 1e6")
    print("\nFirst 5 rows:")
    print(head.to_string(index=False))

    total_rows = 0
    time_values: set[int] = set()
    edge_values: set[int] = set()
    src_min = np.inf
    src_max = -np.inf
    dst_min = np.inf
    dst_max = -np.inf
    congestion_count = 0
    next_congestion_count = 0
    nan_inf_found = False
    negative_found = False
    nonpositive_delay_found = False
    bad_capacity_found = False
    edge_time_counts: dict[int, int] = {}
    per_time_frames = []
    utilization_values = []
    next_utilization_values = []

    reader = pd.read_csv(input_path, chunksize=args.chunksize)
    for chunk in tqdm(reader, desc="Checking link-state CSV"):
        total_rows += len(chunk)
        time_values.update(chunk["time"].astype(int).unique().tolist())
        edge_values.update(chunk["edge_id"].astype(int).unique().tolist())

        src_min = min(src_min, float(chunk["src_sat"].min()))
        src_max = max(src_max, float(chunk["src_sat"].max()))
        dst_min = min(dst_min, float(chunk["dst_sat"].min()))
        dst_max = max(dst_max, float(chunk["dst_sat"].max()))

        numeric = chunk[NUMERIC_COLUMNS]
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
            nan_inf_found = True
        nonnegative_columns = [
            "time",
            "edge_id",
            "src_sat",
            "dst_sat",
            "load_mbps",
            "capacity_mbps",
            "utilization",
            "delay_ms",
            "queue_len",
            "remain_visible_time",
            "congestion_label",
            "next_load_mbps",
            "next_utilization",
            "next_congestion_label",
        ]
        if (chunk[nonnegative_columns] < 0).any().any():
            negative_found = True
        if (chunk["delay_ms"] <= 0).any():
            nonpositive_delay_found = True
        if not np.allclose(chunk["capacity_mbps"].to_numpy(), 1000.0):
            bad_capacity_found = True

        congestion_count += int((chunk["congestion_label"] == 1).sum())
        next_congestion_count += int((chunk["next_congestion_label"] == 1).sum())
        utilization_values.append(chunk["utilization"].to_numpy(dtype=np.float64))
        next_utilization_values.append(chunk["next_utilization"].to_numpy(dtype=np.float64))

        edge_counts = chunk.groupby("edge_id", sort=False).size()
        for edge_id, count in edge_counts.items():
            edge_time_counts[int(edge_id)] = edge_time_counts.get(int(edge_id), 0) + int(count)

        per_time = chunk.groupby("time", as_index=False)["utilization"].agg(["mean", "max"]).reset_index()
        per_time_frames.append(per_time)

    utilization = np.concatenate(utilization_values)
    next_utilization = np.concatenate(next_utilization_values)
    per_time_all = pd.concat(per_time_frames, ignore_index=True)
    per_time = per_time_all.groupby("time", as_index=False).agg({"mean": "mean", "max": "max"})
    mlu = per_time["max"].to_numpy(dtype=np.float64)

    time_count = len(time_values)
    edge_count = len(edge_values)
    expected_time_count, suffix_label = infer_expected_time_count(input_path)
    avg_path, mlu_path = save_curves(per_time, args.results_dir, suffix_label)

    print("\nSummary:")
    print(f"- total rows: {total_rows}")
    print(f"- time count: {time_count}")
    print(f"- edge_id count: {edge_count}")
    print(f"- src_sat range: [{int(src_min)}, {int(src_max)}]")
    print(f"- dst_sat range: [{int(dst_min)}, {int(dst_max)}]")

    print("\nUtilization:")
    print_distribution("utilization", utilization)
    print("\nNext utilization:")
    print_distribution("next_utilization", next_utilization)

    print("\nCongestion:")
    print(f"- congestion_label=1 ratio: {congestion_count / total_rows:.9g}")
    print(f"- next_congestion_label=1 ratio: {next_congestion_count / total_rows:.9g}")

    print("\nMLU:")
    print(f"- MLU min: {mlu.min():.9g}")
    print(f"- MLU mean: {mlu.mean():.9g}")
    print(f"- MLU p95: {np.percentile(mlu, 95):.9g}")
    print(f"- MLU p99: {np.percentile(mlu, 99):.9g}")
    print(f"- MLU max: {mlu.max():.9g}")

    edge_count_values = np.array(list(edge_time_counts.values()), dtype=np.int64)
    print("\nPer-edge time counts:")
    print(f"- min: {edge_count_values.min()}")
    print(f"- max: {edge_count_values.max()}")
    print(f"- all equal: {bool(np.all(edge_count_values == edge_count_values[0]))}")

    problems = []
    warnings = []

    if edge_count != 288:
        problems.append(f"Expected 288 edge_id values, got {edge_count}")
    if expected_time_count is not None and time_count != expected_time_count:
        problems.append(f"Expected {expected_time_count} time values, got {time_count}")
    if nan_inf_found:
        problems.append("File contains NaN or inf values.")
    if negative_found:
        problems.append("File contains negative values.")
    if nonpositive_delay_found:
        problems.append("delay_ms must be positive.")
    if bad_capacity_found:
        problems.append("capacity_mbps is not always 1000.")
    if int(src_min) < 0 or int(src_max) > 143 or int(dst_min) < 0 or int(dst_max) > 143:
        problems.append(
            f"src_sat/dst_sat must be in [0, 143], got src=[{src_min}, {src_max}], "
            f"dst=[{dst_min}, {dst_max}]"
        )
    if np.isclose(utilization.min(), utilization.max()):
        problems.append("utilization does not vary over time/edges.")
    if not np.all(edge_count_values == edge_count_values[0]):
        problems.append("Each edge does not have the same number of time rows.")

    problems.extend(sample_next_label_check(input_path, edge_count=edge_count))

    if np.allclose(utilization, 0.0):
        warnings.append("utilization is almost all zero.")
    if congestion_count == 0:
        warnings.append("congestion_label is all zero.")
    if utilization.max() > 10:
        warnings.append(
            "max utilization is greater than 10. This may indicate capacity_mbps=1000 "
            "is small for some traffic peaks; later experiments can adjust link capacity "
            "or keep the resulting congestion scenario."
        )

    print("\nOutput figures:")
    print(f"- {avg_path}")
    print(f"- {mlu_path}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    if problems:
        print("\nCheck result: FAILED")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)

    print("\nCheck result: OK")


if __name__ == "__main__":
    main()
