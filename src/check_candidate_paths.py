import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check topo144 candidate path outputs.")
    parser.add_argument("--k", type=int, default=5, help="Candidate path K used in the filename.")
    parser.add_argument(
        "--candidate_path",
        type=Path,
        default=None,
        help="Optional explicit candidate path pickle.",
    )
    parser.add_argument(
        "--edges_path",
        type=Path,
        default=PROCESSED_DIR / "leo_topo144_edges.csv",
        help="Input topo144 ISL edge CSV.",
    )
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {path}")


def load_records(path: Path) -> list[dict]:
    with path.open("rb") as file:
        records = pickle.load(file)
    if not isinstance(records, list):
        raise TypeError(f"Candidate path pickle must contain a list of records, got {type(records)}")
    return records


def build_edge_lookup(edges: pd.DataFrame) -> tuple[set[int], dict[tuple[int, int], int]]:
    required_columns = {"edge_id", "src_sat", "dst_sat"}
    missing_columns = required_columns - set(edges.columns)
    if missing_columns:
        raise ValueError(f"Missing required edge columns: {sorted(missing_columns)}")

    valid_edge_ids = set()
    edge_lookup: dict[tuple[int, int], int] = {}
    for row in edges.itertuples(index=False):
        edge_id = int(row.edge_id)
        src_sat = int(row.src_sat)
        dst_sat = int(row.dst_sat)
        valid_edge_ids.add(edge_id)
        edge_lookup[tuple(sorted((src_sat, dst_sat)))] = edge_id
    return valid_edge_ids, edge_lookup


def validate_record(
    record: dict,
    valid_edge_ids: set[int],
    edge_lookup: dict[tuple[int, int], int],
    max_sat_id: int,
) -> list[str]:
    errors = []
    sat_path = record.get("sat_path", [])
    edge_path = record.get("edge_path", [])
    is_valid = bool(record.get("is_valid", False))

    if not is_valid:
        return errors

    if len(sat_path) < 2:
        errors.append("valid path has fewer than 2 satellites")
    if len(edge_path) != len(sat_path) - 1:
        errors.append("edge_path length does not match sat_path hop count")

    for sat in sat_path:
        if not isinstance(sat, (int, np.integer)) or int(sat) < 0 or int(sat) > max_sat_id:
            errors.append(f"satellite id out of range: {sat}")

    for edge_id in edge_path:
        if int(edge_id) not in valid_edge_ids:
            errors.append(f"edge_id does not exist: {edge_id}")

    for idx in range(max(0, len(sat_path) - 1)):
        key = tuple(sorted((int(sat_path[idx]), int(sat_path[idx + 1]))))
        expected_edge_id = edge_lookup.get(key)
        if expected_edge_id is None:
            errors.append(f"satellite hop has no edge: {key}")
            continue
        if idx < len(edge_path) and int(edge_path[idx]) != expected_edge_id:
            errors.append(
                f"edge_path mismatch at hop {idx}: expected {expected_edge_id}, got {edge_path[idx]}"
            )

    return errors


def plot_hop_distribution(hop_counts: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    bins = np.arange(hop_counts.min(), hop_counts.max() + 2) - 0.5
    plt.hist(hop_counts, bins=bins, edgecolor="black", linewidth=0.6)
    plt.xlabel("hop_count")
    plt.ylabel("candidate path count")
    plt.title("Candidate path hop-count distribution")
    plt.grid(axis="y", linewidth=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    processed_dir = ensure_dir(args.processed_dir)
    results_dir = ensure_dir(args.results_dir)

    candidate_path = (
        args.candidate_path
        if args.candidate_path is not None
        else processed_dir / f"candidate_paths_topo144_k{args.k}.pkl"
    )
    report_path = results_dir / "candidate_path_check_report.json"
    fig_path = results_dir / "candidate_path_hop_count_distribution.png"

    require_file(candidate_path)
    require_file(args.edges_path)

    records = load_records(candidate_path)
    edges = pd.read_csv(args.edges_path).sort_values("edge_id").reset_index(drop=True)
    valid_edge_ids, edge_lookup = build_edge_lookup(edges)
    max_sat_id = int(max(edges["src_sat"].max(), edges["dst_sat"].max()))
    satellites = set(range(max_sat_id + 1))

    pair_paths: dict[tuple[int, int], list[tuple[int, ...]]] = defaultdict(list)
    validation_errors = []
    valid_records = []
    invalid_records = []

    for idx, record in enumerate(records):
        errors = validate_record(record, valid_edge_ids, edge_lookup, max_sat_id)
        if errors:
            validation_errors.append({"record_index": idx, "errors": errors, "record": record})

        if bool(record.get("is_valid", False)):
            valid_records.append(record)
            pair = (int(record["src_sat"]), int(record["dst_sat"]))
            pair_paths[pair].append(tuple(int(sat) for sat in record["sat_path"]))
        else:
            invalid_records.append(record)

    all_pairs = [(src, dst) for src in satellites for dst in satellites if src != dst]
    counts = np.asarray([len(pair_paths.get(pair, [])) for pair in all_pairs], dtype=np.int32)
    hop_counts = np.asarray([record["hop_count"] for record in valid_records], dtype=np.float64)

    duplicate_pairs = []
    duplicate_path_count = 0
    for pair, paths in pair_paths.items():
        unique_count = len(set(paths))
        if unique_count != len(paths):
            duplicate_pairs.append({"src_sat": pair[0], "dst_sat": pair[1], "duplicate_count": len(paths) - unique_count})
            duplicate_path_count += len(paths) - unique_count

    no_path_pairs = [{"src_sat": pair[0], "dst_sat": pair[1]} for pair in all_pairs if len(pair_paths.get(pair, [])) == 0]

    if len(hop_counts):
        plot_hop_distribution(hop_counts.astype(np.int32), fig_path)

    report = {
        "candidate_path": str(candidate_path),
        "edges_path": str(args.edges_path),
        "total_records": int(len(records)),
        "valid_records": int(len(valid_records)),
        "invalid_records": int(len(invalid_records)),
        "satellite_pair_total": int(len(all_pairs)),
        "average_candidate_paths_per_pair": float(np.mean(counts)) if len(counts) else 0.0,
        "min_candidate_paths_per_pair": int(np.min(counts)) if len(counts) else 0,
        "max_candidate_paths_per_pair": int(np.max(counts)) if len(counts) else 0,
        "hop_count_mean": float(np.mean(hop_counts)) if len(hop_counts) else None,
        "hop_count_p50": float(np.percentile(hop_counts, 50)) if len(hop_counts) else None,
        "hop_count_p95": float(np.percentile(hop_counts, 95)) if len(hop_counts) else None,
        "hop_count_max": int(np.max(hop_counts)) if len(hop_counts) else None,
        "has_no_path_pair": bool(len(no_path_pairs) > 0),
        "no_path_pair_count": int(len(no_path_pairs)),
        "has_duplicate_path": bool(duplicate_path_count > 0),
        "duplicate_path_count": int(duplicate_path_count),
        "duplicate_pair_count": int(len(duplicate_pairs)),
        "validation_error_count": int(len(validation_errors)),
        "validation_passed": bool(len(validation_errors) == 0 and len(no_path_pairs) == 0 and duplicate_path_count == 0),
        "figure_path": str(fig_path),
        "note": "Candidate paths are checked only by topology continuity and hop-based edge mapping.",
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("Candidate path check complete:")
    print(f"- candidate path pkl: {candidate_path}")
    print(f"- satellite pair total: {report['satellite_pair_total']}")
    print(f"- average candidate paths per pair: {report['average_candidate_paths_per_pair']:.4f}")
    print(
        "- hop_count mean / p50 / p95 / max: "
        f"{report['hop_count_mean']:.4f} / {report['hop_count_p50']:.4f} / "
        f"{report['hop_count_p95']:.4f} / {report['hop_count_max']}"
    )
    print(f"- has no-path pair: {report['has_no_path_pair']} ({report['no_path_pair_count']})")
    print(f"- has duplicate path: {report['has_duplicate_path']} ({report['duplicate_path_count']})")
    print(f"- validation errors: {report['validation_error_count']}")
    print(f"- report: {report_path}")
    print(f"- hop distribution figure: {fig_path}")

    if report["validation_error_count"] > 0:
        raise SystemExit("Candidate path validation failed. See report JSON for details.")


if __name__ == "__main__":
    main()
