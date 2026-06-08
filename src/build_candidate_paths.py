import argparse
import json
import pickle
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import networkx as nx
except ImportError as exc:
    raise SystemExit(
        "networkx is required for candidate path generation. "
        "Please install it with: pip install networkx"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hop-based K-shortest candidate paths for topo144 satellite pairs."
    )
    parser.add_argument("--k", type=int, default=5, help="Maximum candidate paths per satellite pair.")
    parser.add_argument(
        "--edges_path",
        type=Path,
        default=PROCESSED_DIR / "leo_topo144_edges.csv",
        help="Input topo144 ISL edge CSV.",
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory for candidate path outputs.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {path}")


def build_graph(edges: pd.DataFrame) -> tuple[nx.Graph, dict[tuple[int, int], int]]:
    graph = nx.Graph()
    edge_lookup: dict[tuple[int, int], int] = {}

    required_columns = {"edge_id", "src_sat", "dst_sat"}
    missing_columns = required_columns - set(edges.columns)
    if missing_columns:
        raise ValueError(f"Missing required edge columns: {sorted(missing_columns)}")

    for row in edges.itertuples(index=False):
        edge_id = int(row.edge_id)
        src_sat = int(row.src_sat)
        dst_sat = int(row.dst_sat)
        graph.add_edge(src_sat, dst_sat, weight=1.0, edge_id=edge_id)
        edge_lookup[tuple(sorted((src_sat, dst_sat)))] = edge_id

    return graph, edge_lookup


def sat_path_to_edge_path(
    sat_path: list[int],
    edge_lookup: dict[tuple[int, int], int],
) -> list[int]:
    edge_path = []
    for idx in range(len(sat_path) - 1):
        key = tuple(sorted((int(sat_path[idx]), int(sat_path[idx + 1]))))
        if key not in edge_lookup:
            raise KeyError(f"No edge_id for satellite hop {key}")
        edge_path.append(edge_lookup[key])
    return edge_path


def generate_pair_paths(
    graph: nx.Graph,
    edge_lookup: dict[tuple[int, int], int],
    src_sat: int,
    dst_sat: int,
    k: int,
) -> list[dict]:
    try:
        path_iter = nx.shortest_simple_paths(graph, source=src_sat, target=dst_sat, weight="weight")
        sat_paths = list(islice(path_iter, k))
    except nx.NetworkXNoPath:
        sat_paths = []

    records = []
    for path_id, sat_path in enumerate(sat_paths):
        sat_path = [int(node) for node in sat_path]
        edge_path = sat_path_to_edge_path(sat_path, edge_lookup)
        records.append(
            {
                "src_sat": int(src_sat),
                "dst_sat": int(dst_sat),
                "path_id": int(path_id),
                "sat_path": sat_path,
                "edge_path": [int(edge_id) for edge_id in edge_path],
                "hop_count": int(len(edge_path)),
                "is_valid": True,
            }
        )

    if not records:
        records.append(
            {
                "src_sat": int(src_sat),
                "dst_sat": int(dst_sat),
                "path_id": -1,
                "sat_path": [],
                "edge_path": [],
                "hop_count": -1,
                "is_valid": False,
            }
        )

    return records


def make_summary_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "src_sat": record["src_sat"],
                "dst_sat": record["dst_sat"],
                "path_id": record["path_id"],
                "sat_path": json.dumps(record["sat_path"], separators=(",", ":")),
                "edge_path": json.dumps(record["edge_path"], separators=(",", ":")),
                "hop_count": record["hop_count"],
                "is_valid": record["is_valid"],
            }
        )
    return pd.DataFrame(rows)


def compute_stats(records: list[dict], graph: nx.Graph, k: int) -> dict:
    valid_records = [record for record in records if record["is_valid"]]
    pair_counts: dict[tuple[int, int], int] = {}
    for src_sat in graph.nodes:
        for dst_sat in graph.nodes:
            if src_sat != dst_sat:
                pair_counts[(int(src_sat), int(dst_sat))] = 0
    for record in valid_records:
        pair = (int(record["src_sat"]), int(record["dst_sat"]))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    counts = np.asarray(list(pair_counts.values()), dtype=np.int32)
    hop_counts = np.asarray([record["hop_count"] for record in valid_records], dtype=np.float64)

    return {
        "k": int(k),
        "num_satellites": int(graph.number_of_nodes()),
        "num_isl_edges": int(graph.number_of_edges()),
        "total_satellite_pairs": int(len(pair_counts)),
        "total_path_records": int(len(records)),
        "valid_path_records": int(len(valid_records)),
        "pairs_with_no_path": int(np.sum(counts == 0)),
        "average_paths_per_pair": float(np.mean(counts)) if len(counts) else 0.0,
        "min_paths_per_pair": int(np.min(counts)) if len(counts) else 0,
        "max_paths_per_pair": int(np.max(counts)) if len(counts) else 0,
        "hop_count_mean": float(np.mean(hop_counts)) if len(hop_counts) else None,
        "hop_count_p50": float(np.percentile(hop_counts, 50)) if len(hop_counts) else None,
        "hop_count_p95": float(np.percentile(hop_counts, 95)) if len(hop_counts) else None,
        "hop_count_max": int(np.max(hop_counts)) if len(hop_counts) else None,
        "generation_note": "Hop-based K-shortest paths only; risk_score is not used in this stage.",
    }


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError(f"--k must be positive, got {args.k}")

    require_file(args.edges_path)
    processed_dir = ensure_dir(args.processed_dir)

    edges = pd.read_csv(args.edges_path).sort_values("edge_id").reset_index(drop=True)
    graph, edge_lookup = build_graph(edges)
    satellites = sorted(int(node) for node in graph.nodes)

    output_pkl = processed_dir / f"candidate_paths_topo144_k{args.k}.pkl"
    output_csv = processed_dir / f"candidate_paths_topo144_k{args.k}_summary.csv"
    output_json = processed_dir / f"candidate_paths_topo144_k{args.k}_stats.json"

    print("Candidate path generation settings:")
    print(f"- input edges: {args.edges_path}")
    print(f"- satellites: {len(satellites)}")
    print(f"- ISL edges: {graph.number_of_edges()}")
    print(f"- K: {args.k}")
    print("- weight: hop-based weight=1")
    print("- risk_score: not used in this stage")

    records: list[dict] = []
    pair_total = len(satellites) * (len(satellites) - 1)
    with tqdm(total=pair_total, desc="Generating candidate paths") as progress:
        for src_sat in satellites:
            for dst_sat in satellites:
                if src_sat == dst_sat:
                    continue
                records.extend(generate_pair_paths(graph, edge_lookup, src_sat, dst_sat, args.k))
                progress.update(1)

    with output_pkl.open("wb") as file:
        pickle.dump(records, file, protocol=pickle.HIGHEST_PROTOCOL)

    summary = make_summary_frame(records)
    summary.to_csv(output_csv, index=False)

    stats = compute_stats(records, graph, args.k)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, ensure_ascii=False)

    print("\nCandidate path generation complete:")
    print(f"- candidate paths pkl: {output_pkl}")
    print(f"- summary csv: {output_csv}")
    print(f"- stats json: {output_json}")
    print(f"- satellite pair total: {stats['total_satellite_pairs']}")
    print(f"- average paths per pair: {stats['average_paths_per_pair']:.4f}")
    print(
        "- hop_count mean / p50 / p95 / max: "
        f"{stats['hop_count_mean']:.4f} / {stats['hop_count_p50']:.4f} / "
        f"{stats['hop_count_p95']:.4f} / {stats['hop_count_max']}"
    )
    print(f"- pairs with no path: {stats['pairs_with_no_path']}")


if __name__ == "__main__":
    main()
