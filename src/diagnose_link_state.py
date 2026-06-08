import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether the LEO link-state dataset is ready for sample construction."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DIR / "link_state_topo144_shortest_delay.csv",
    )
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--sample_edges", type=int, default=8)
    parser.add_argument("--sample_times", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_time_edge_slice(input_path: Path, sample_pairs: pd.DataFrame) -> pd.DataFrame:
    times_needed = sorted(
        set(sample_pairs["time"].astype(int).tolist())
        | set((sample_pairs["time"] + 1).astype(int).tolist())
    )
    edges_needed = sorted(sample_pairs["edge_id"].astype(int).unique().tolist())
    frames = []

    for chunk in pd.read_csv(
        input_path,
        usecols=["time", "edge_id", "utilization", "next_utilization"],
        chunksize=200_000,
    ):
        subset = chunk[
            chunk["time"].isin(times_needed) & chunk["edge_id"].isin(edges_needed)
        ]
        if not subset.empty:
            frames.append(subset)

    if not frames:
        return pd.DataFrame(columns=["time", "edge_id", "utilization", "next_utilization"])
    return pd.concat(frames, ignore_index=True)


def check_next_labels(
    input_path: Path,
    edge_ids: np.ndarray,
    time_values: np.ndarray,
    sample_edges: int,
    sample_times: int,
    seed: int,
) -> tuple[bool, float, int]:
    rng = np.random.default_rng(seed)
    selected_edges = rng.choice(
        edge_ids,
        size=min(sample_edges, len(edge_ids)),
        replace=False,
    )
    valid_times = time_values[time_values < time_values.max()]
    selected_times = rng.choice(
        valid_times,
        size=min(sample_times, len(valid_times)),
        replace=False,
    )

    sample_pairs = pd.DataFrame(
        [(int(time), int(edge)) for time in selected_times for edge in selected_edges],
        columns=["time", "edge_id"],
    )
    slice_df = load_time_edge_slice(input_path, sample_pairs)
    lookup = {
        (int(row.time), int(row.edge_id)): (
            float(row.utilization),
            float(row.next_utilization),
        )
        for row in slice_df.itertuples(index=False)
    }

    max_error = 0.0
    checked = 0
    for row in sample_pairs.itertuples(index=False):
        key = (int(row.time), int(row.edge_id))
        next_key = (int(row.time) + 1, int(row.edge_id))
        if key not in lookup or next_key not in lookup:
            return False, max_error, checked
        current_next = lookup[key][1]
        actual_next = lookup[next_key][0]
        max_error = max(max_error, abs(current_next - actual_next))
        checked += 1

    return bool(max_error <= 1e-7), max_error, checked


def save_bar_plot(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 4.5))
    plt.bar(df["edge_id"].astype(str), df["congestion_count"])
    plt.xlabel("edge_id")
    plt.ylabel("congestion count")
    plt.title("Top 10 congested ISL edges")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_time_plot(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 4.5))
    plt.plot(df["time"], df["congestion_count"], linewidth=0.8)
    plt.xlabel("time")
    plt.ylabel("congested link count")
    plt.title("Congested ISL count by time")
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input link-state CSV does not exist: {args.input}")
    if args.chunksize <= 0:
        raise ValueError(f"--chunksize must be positive, got {args.chunksize}")

    results_dir = ensure_dir(args.results_dir)

    edge_congestion_counts: dict[int, int] = {}
    time_congestion_counts: dict[int, int] = {}
    edge_ids_seen: set[int] = set()
    time_values_seen: set[int] = set()
    total_rows = 0
    total_congestion = 0
    max_utilization_error = 0.0

    usecols = [
        "time",
        "edge_id",
        "load_mbps",
        "capacity_mbps",
        "utilization",
        "congestion_label",
    ]
    reader = pd.read_csv(args.input, usecols=usecols, chunksize=args.chunksize)
    for chunk in tqdm(reader, desc="Diagnosing link-state CSV"):
        total_rows += len(chunk)
        edge_ids_seen.update(chunk["edge_id"].astype(int).unique().tolist())
        time_values_seen.update(chunk["time"].astype(int).unique().tolist())

        expected_utilization = chunk["load_mbps"] / chunk["capacity_mbps"]
        error = (chunk["utilization"] - expected_utilization).abs().max()
        max_utilization_error = max(max_utilization_error, float(error))

        congested = chunk[chunk["congestion_label"] == 1]
        total_congestion += len(congested)

        edge_counts = congested.groupby("edge_id").size()
        for edge_id, count in edge_counts.items():
            edge_key = int(edge_id)
            edge_congestion_counts[edge_key] = edge_congestion_counts.get(edge_key, 0) + int(count)

        time_counts = congested.groupby("time").size()
        for time_id, count in time_counts.items():
            time_key = int(time_id)
            time_congestion_counts[time_key] = time_congestion_counts.get(time_key, 0) + int(count)

    all_edges = sorted(edge_ids_seen)
    edge_counts_df = pd.DataFrame(
        {
            "edge_id": all_edges,
            "congestion_count": [edge_congestion_counts.get(edge_id, 0) for edge_id in all_edges],
        }
    )
    edge_counts_df = edge_counts_df.sort_values(
        ["congestion_count", "edge_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    top10_edges = edge_counts_df.head(10).copy()

    all_times = sorted(time_values_seen)
    time_counts_df = pd.DataFrame(
        {
            "time": all_times,
            "congestion_count": [time_congestion_counts.get(time_id, 0) for time_id in all_times],
        }
    )

    top10_congestion = int(top10_edges["congestion_count"].sum())
    top10_ratio = top10_congestion / total_congestion if total_congestion > 0 else 0.0
    congested_time_count = int((time_counts_df["congestion_count"] > 0).sum())
    congested_time_ratio = congested_time_count / len(time_counts_df)
    max_simultaneous_congested_links = int(time_counts_df["congestion_count"].max())
    mean_congested_links_per_time = float(time_counts_df["congestion_count"].mean())

    edge_ids = np.array(all_edges, dtype=np.int32)
    time_values = np.array(all_times, dtype=np.int32)
    next_ok, next_max_error, next_checked = check_next_labels(
        input_path=args.input,
        edge_ids=edge_ids,
        time_values=time_values,
        sample_edges=args.sample_edges,
        sample_times=args.sample_times,
        seed=args.seed,
    )

    diagnosis_path = results_dir / "link_state_diagnosis.txt"
    top_edges_path = results_dir / "top_congested_edges.csv"
    time_counts_path = results_dir / "congestion_count_by_time.csv"
    top_edges_plot_path = results_dir / "top_congested_edges.png"
    time_plot_path = results_dir / "congestion_count_by_time.png"

    top10_edges.to_csv(top_edges_path, index=False)
    time_counts_df.to_csv(time_counts_path, index=False)
    save_bar_plot(top10_edges, top_edges_plot_path)
    save_time_plot(time_counts_df, time_plot_path)

    lines = [
        "Link-State Dataset Diagnosis",
        f"input: {args.input}",
        f"total_rows: {total_rows}",
        f"edge_count: {len(edge_ids_seen)}",
        f"time_count: {len(time_values_seen)}",
        f"total_congestion_samples: {total_congestion}",
        "",
        "Top 10 congested edges:",
        top10_edges.to_string(index=False),
        "",
        f"top10_congestion_ratio: {top10_ratio:.9g}",
        f"congested_time_count: {congested_time_count}",
        f"congested_time_ratio: {congested_time_ratio:.9g}",
        f"max_simultaneous_congested_links: {max_simultaneous_congested_links}",
        f"mean_congested_links_per_time: {mean_congested_links_per_time:.9g}",
        f"max_utilization_formula_abs_error: {max_utilization_error:.12g}",
        f"next_label_sample_passed: {next_ok}",
        f"next_label_sample_checked_pairs: {next_checked}",
        f"next_label_sample_max_error: {next_max_error:.12g}",
        "",
        "Output files:",
        f"- {top_edges_path}",
        f"- {time_counts_path}",
        f"- {top_edges_plot_path}",
        f"- {time_plot_path}",
    ]
    diagnosis_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"- {diagnosis_path}")


if __name__ == "__main__":
    main()
