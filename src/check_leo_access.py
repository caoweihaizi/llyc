import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check stage-2 LEO access outputs.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--expected_time_steps", type=int, default=48384)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required stage-2 file does not exist: {path}. "
            "Run build_leo_constellation.py and ground_sat_access.py first."
        )


def check_duplicate_edges(edges: pd.DataFrame) -> bool:
    edge_pairs = edges[["src_sat", "dst_sat"]].copy()
    edge_pairs["a"] = edge_pairs[["src_sat", "dst_sat"]].min(axis=1)
    edge_pairs["b"] = edge_pairs[["src_sat", "dst_sat"]].max(axis=1)
    return bool(edge_pairs.duplicated(["a", "b"]).any())


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    processed_dir = args.processed_dir

    topo_name = str(config.get("topo_name", "topo144"))
    min_elevation_deg = float(config.get("min_elevation_deg", 15))
    num_planes = int(config["num_planes"])
    sats_per_plane = int(config["sats_per_plane"])
    total_sats = num_planes * sats_per_plane

    edges_path = processed_dir / f"leo_{topo_name}_edges.csv"
    positions_path = processed_dir / f"leo_{topo_name}_sat_positions.npy"
    access_path = processed_dir / f"gateway_access_{topo_name}.npy"
    stats_path = processed_dir / f"gateway_access_stats_{topo_name}.csv"

    for path in [edges_path, positions_path, access_path, stats_path]:
        require_file(path)
        print(f"Found: {path}")

    edges = pd.read_csv(edges_path)
    sat_positions = np.load(positions_path, mmap_mode="r")
    gateway_access = np.load(access_path, mmap_mode="r")
    stats = pd.read_csv(stats_path)

    print("\nStage-2 output summary:")
    print(f"- min_elevation_deg: {min_elevation_deg}")
    print(f"- edges: {len(edges)}")
    print(f"- sat_positions shape: {sat_positions.shape}")
    print(f"- gateway_access shape: {gateway_access.shape}")
    print(f"- gateway_access min sat id: {int(gateway_access.min())}")
    print(f"- gateway_access max sat id: {int(gateway_access.max())}")

    print("\nUnique satellites used per gateway:")
    for gateway_id in range(gateway_access.shape[1]):
        unique_count = len(np.unique(gateway_access[:, gateway_id]))
        gateway_name = stats.iloc[gateway_id]["gateway_name"] if gateway_id < len(stats) else f"gateway_{gateway_id}"
        print(f"- {gateway_id:02d} {gateway_name}: {unique_count}")

    print("\nFallback rates:")
    print(f"- mean fallback_rate: {stats['fallback_rate'].mean():.6f}")
    print(f"- max fallback_rate: {stats['fallback_rate'].max():.6f}")

    problems = []
    expected_positions_shape = (args.expected_time_steps, total_sats, 3)
    expected_access_shape = (args.expected_time_steps, 12)

    if tuple(sat_positions.shape) != expected_positions_shape:
        problems.append(
            f"Expected sat_positions shape {expected_positions_shape}, got {sat_positions.shape}"
        )
    if tuple(gateway_access.shape) != expected_access_shape:
        problems.append(
            f"Expected gateway_access shape {expected_access_shape}, got {gateway_access.shape}"
        )

    if len(edges) != 288:
        problems.append(f"Expected 288 topo144 ISL edges, got {len(edges)}")

    if int(gateway_access.min()) < 0 or int(gateway_access.max()) >= total_sats:
        problems.append(
            f"Gateway access satellite ids must be in [0, {total_sats - 1}], "
            f"got min={int(gateway_access.min())}, max={int(gateway_access.max())}"
        )

    if check_duplicate_edges(edges):
        problems.append("Edges contain duplicate undirected pairs.")

    degrees = np.zeros(total_sats, dtype=np.int32)
    for src, dst in edges[["src_sat", "dst_sat"]].to_numpy(dtype=np.int32):
        degrees[src] += 1
        degrees[dst] += 1
    average_degree = float(degrees.mean())
    print(f"\nISL degree summary:")
    print(f"- min degree: {int(degrees.min())}")
    print(f"- max degree: {int(degrees.max())}")
    print(f"- average degree: {average_degree:.6f}")
    if not np.all(degrees == 4):
        problems.append(
            "Each satellite should have degree 4, "
            f"got min={int(degrees.min())}, max={int(degrees.max())}"
        )

    if problems:
        print("\nCheck result: FAILED")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)

    print("\nCheck result: OK")
    print("\nChecked files:")
    print(f"- {edges_path}")
    print(f"- {positions_path}")
    print(f"- {access_path}")
    print(f"- {stats_path}")


if __name__ == "__main__":
    main()
