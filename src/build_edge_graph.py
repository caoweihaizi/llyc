import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build edge-to-edge adjacency for topo144 ISLs.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--edges_path",
        type=Path,
        default=PROCESSED_DIR / "leo_topo144_edges.csv",
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


def build_binary_edge_adjacency(edges: pd.DataFrame) -> np.ndarray:
    edges = edges.sort_values("edge_id").reset_index(drop=True)
    edge_ids = edges["edge_id"].to_numpy(dtype=np.int32)
    expected_ids = np.arange(len(edges), dtype=np.int32)
    if not np.array_equal(edge_ids, expected_ids):
        raise ValueError("edge_id values must be contiguous and sorted from 0 to num_edges-1.")

    src = edges["src_sat"].to_numpy(dtype=np.int32)
    dst = edges["dst_sat"].to_numpy(dtype=np.int32)
    num_edges = len(edges)
    adjacency = np.zeros((num_edges, num_edges), dtype=np.float32)

    for i in range(num_edges):
        shares_src = (src == src[i]) | (dst == src[i])
        shares_dst = (src == dst[i]) | (dst == dst[i])
        adjacent = shares_src | shares_dst
        adjacent[i] = False
        adjacency[i, adjacent] = 1.0

    adjacency = np.maximum(adjacency, adjacency.T)
    return adjacency


def normalize_adjacency(binary_adjacency: np.ndarray) -> np.ndarray:
    adjacency_with_loop = binary_adjacency + np.eye(binary_adjacency.shape[0], dtype=np.float32)
    degree = adjacency_with_loop.sum(axis=1)
    inv_sqrt_degree = np.power(degree, -0.5, where=degree > 0)
    inv_sqrt_degree[degree <= 0] = 0.0
    return (
        inv_sqrt_degree[:, None]
        * adjacency_with_loop
        * inv_sqrt_degree[None, :]
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    processed_dir = ensure_dir(args.processed_dir)
    topo_name = str(config.get("topo_name", "topo144"))

    if not args.edges_path.exists():
        raise FileNotFoundError(f"Edge file does not exist: {args.edges_path}")

    edges = pd.read_csv(args.edges_path)
    binary_adjacency = build_binary_edge_adjacency(edges)
    norm_adjacency = normalize_adjacency(binary_adjacency)

    degree = binary_adjacency.sum(axis=1)
    stats = {
        "num_edges": int(binary_adjacency.shape[0]),
        "adjacency_shape": list(binary_adjacency.shape),
        "min_degree": float(degree.min()),
        "max_degree": float(degree.max()),
        "mean_degree": float(degree.mean()),
        "num_nonzero": int(np.count_nonzero(binary_adjacency)),
        "is_symmetric": bool(np.allclose(binary_adjacency, binary_adjacency.T)),
        "has_self_loop_after_norm": bool(np.all(np.diag(norm_adjacency) > 0)),
    }

    norm_path = processed_dir / f"edge_adj_{topo_name}.npy"
    binary_path = processed_dir / f"edge_adj_binary_{topo_name}.npy"
    stats_path = processed_dir / f"edge_graph_stats_{topo_name}.json"

    np.save(norm_path, norm_adjacency)
    np.save(binary_path, binary_adjacency)
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, ensure_ascii=False)

    print("Edge graph built:")
    for key, value in stats.items():
        print(f"- {key}: {value}")
    print("\nOutput files:")
    print(f"- {norm_path}")
    print(f"- {binary_path}")
    print(f"- {stats_path}")


if __name__ == "__main__":
    main()
