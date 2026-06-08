import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

EARTH_RADIUS_KM = 6371.0
SPEED_OF_LIGHT_KM_PER_MS = 299.792458
REMAIN_VISIBLE_TIME_PLACEHOLDER = 9999.0

OUTPUT_COLUMNS = [
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
    parser = argparse.ArgumentParser(
        description="Map Abilene OD traffic onto topo144 ISLs and generate link states."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--od_path",
        type=Path,
        default=PROCESSED_DIR / "od_matrices_full.npy",
    )
    parser.add_argument(
        "--edges_path",
        type=Path,
        default=PROCESSED_DIR / "leo_topo144_edges.csv",
    )
    parser.add_argument(
        "--sat_positions_path",
        type=Path,
        default=PROCESSED_DIR / "leo_topo144_sat_positions.npy",
    )
    parser.add_argument(
        "--gateway_access_path",
        type=Path,
        default=PROCESSED_DIR / "gateway_access_topo144.npy",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional debug limit. Debug output filenames include max_steps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output CSV.",
    )
    parser.add_argument(
        "--traffic_unit",
        choices=["abilene_100bytes_per_5min"],
        default="abilene_100bytes_per_5min",
        help="Unit of od_matrices_full.npy values. Only Abilene official raw unit is supported.",
    )
    parser.add_argument(
        "--flush_steps",
        type=int,
        default=100,
        help="Number of time steps buffered before appending to CSV.",
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


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {path}")


def lla_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_km: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    radius = EARTH_RADIUS_KM + alt_km
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def compute_edge_delays_ms(sat_positions_t: np.ndarray, edges: pd.DataFrame) -> np.ndarray:
    sat_ecef = lla_to_ecef(
        sat_positions_t[:, 0],
        sat_positions_t[:, 1],
        sat_positions_t[:, 2],
    )
    src = edges["src_sat"].to_numpy(dtype=np.int32)
    dst = edges["dst_sat"].to_numpy(dtype=np.int32)
    distance_km = np.linalg.norm(sat_ecef[src] - sat_ecef[dst], axis=1)
    return distance_km / SPEED_OF_LIGHT_KM_PER_MS


def abilene_raw_to_mbps(raw_value: np.ndarray | float) -> np.ndarray | float:
    """
    Convert Abilene traffic matrix raw value to Mbps.

    Abilene official unit:
    raw_value unit = 100 bytes / 5 minutes

    5 minutes = 300 seconds
    Mbps = raw_value * 100 * 8 / 300 / 1e6
    """
    return raw_value * 100.0 * 8.0 / 300.0 / 1e6


def build_delay_graph(edges: pd.DataFrame, delay_ms: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    for row, delay in zip(edges.itertuples(index=False), delay_ms):
        graph.add_edge(
            int(row.src_sat),
            int(row.dst_sat),
            weight=float(delay),
            edge_id=int(row.edge_id),
        )
    return graph


def build_edge_lookup(edges: pd.DataFrame) -> dict[tuple[int, int], int]:
    lookup = {}
    for row in edges.itertuples(index=False):
        src = int(row.src_sat)
        dst = int(row.dst_sat)
        lookup[tuple(sorted((src, dst)))] = int(row.edge_id)
    return lookup


def path_nodes_to_edge_ids(
    path: list[int],
    edge_lookup: dict[tuple[int, int], int],
) -> list[int]:
    return [
        edge_lookup[tuple(sorted((int(path[i]), int(path[i + 1]))))]
        for i in range(len(path) - 1)
    ]


def get_shortest_path_edge_ids(
    src_sat: int,
    dst_sat: int,
    graph: nx.Graph,
    edge_lookup: dict[tuple[int, int], int],
    path_cache: dict[tuple[int, int], list[int]],
) -> list[int] | None:
    key = (int(src_sat), int(dst_sat))
    if key in path_cache:
        return path_cache[key]

    try:
        path = nx.shortest_path(graph, source=key[0], target=key[1], weight="weight")
    except nx.NetworkXNoPath:
        return None

    edge_ids = path_nodes_to_edge_ids(path, edge_lookup)
    path_cache[key] = edge_ids
    path_cache[(key[1], key[0])] = edge_ids
    return edge_ids


def aggregate_od_by_access_pair(
    od_matrix: np.ndarray,
    gateway_access_t: np.ndarray,
    traffic_unit: str,
) -> dict[tuple[int, int], float]:
    if traffic_unit != "abilene_100bytes_per_5min":
        raise ValueError(f"Unsupported traffic_unit: {traffic_unit}")

    pair_loads: dict[tuple[int, int], float] = {}

    for src_gateway in range(od_matrix.shape[0]):
        for dst_gateway in range(od_matrix.shape[1]):
            if src_gateway == dst_gateway:
                continue
            raw_flow = float(od_matrix[src_gateway, dst_gateway])
            flow_mbps = float(abilene_raw_to_mbps(raw_flow))
            if flow_mbps <= 0:
                continue

            src_sat = int(gateway_access_t[src_gateway])
            dst_sat = int(gateway_access_t[dst_gateway])
            if src_sat == dst_sat:
                continue

            key = (src_sat, dst_sat)
            pair_loads[key] = pair_loads.get(key, 0.0) + flow_mbps

    return pair_loads


def compute_link_state(
    time_index: int,
    od_matrix: np.ndarray,
    gateway_access_t: np.ndarray,
    sat_positions_t: np.ndarray,
    edges: pd.DataFrame,
    graph: nx.Graph,
    edge_lookup: dict[tuple[int, int], int],
    path_cache: dict[tuple[int, int], list[int]],
    capacity_mbps: float,
    congestion_threshold: float,
    traffic_unit: str,
) -> dict[str, np.ndarray]:
    edge_count = len(edges)
    load_mbps = np.zeros(edge_count, dtype=np.float64)
    delay_ms = compute_edge_delays_ms(sat_positions_t, edges)
    pair_loads = aggregate_od_by_access_pair(od_matrix, gateway_access_t, traffic_unit)

    no_path_count = 0
    for (src_sat, dst_sat), flow_mbps in pair_loads.items():
        path_edge_ids = get_shortest_path_edge_ids(
            src_sat=src_sat,
            dst_sat=dst_sat,
            graph=graph,
            edge_lookup=edge_lookup,
            path_cache=path_cache,
        )
        if path_edge_ids is None:
            no_path_count += 1
            continue
        load_mbps[path_edge_ids] += flow_mbps

    if no_path_count:
        print(f"Warning: time {time_index} skipped {no_path_count} OD access pairs with no path.")

    utilization = load_mbps / capacity_mbps
    queue_len = np.maximum(0.0, utilization - 0.7) * 100.0
    congestion_label = (utilization > congestion_threshold).astype(np.int8)

    # The current Walker-like 4-ISL edge set is fixed, so remain_visible_time is
    # a placeholder feature. It can be replaced with a real remaining visibility
    # time when dynamic ISLs or TLE/SGP4 propagation are introduced.
    remain_visible_time = np.full(edge_count, REMAIN_VISIBLE_TIME_PLACEHOLDER, dtype=np.float64)

    return {
        "load_mbps": load_mbps,
        "delay_ms": delay_ms,
        "utilization": utilization,
        "queue_len": queue_len,
        "remain_visible_time": remain_visible_time,
        "congestion_label": congestion_label,
    }


def make_output_frame(
    time_index: int,
    edges: pd.DataFrame,
    state: dict[str, np.ndarray],
    next_state: dict[str, np.ndarray],
    capacity_mbps: float,
) -> pd.DataFrame:
    edge_count = len(edges)
    frame = pd.DataFrame(
        {
            "time": np.full(edge_count, time_index, dtype=np.int32),
            "edge_id": edges["edge_id"].to_numpy(dtype=np.int32),
            "src_sat": edges["src_sat"].to_numpy(dtype=np.int32),
            "dst_sat": edges["dst_sat"].to_numpy(dtype=np.int32),
            "load_mbps": state["load_mbps"],
            "capacity_mbps": np.full(edge_count, capacity_mbps, dtype=np.float64),
            "utilization": state["utilization"],
            "delay_ms": state["delay_ms"],
            "queue_len": state["queue_len"],
            "remain_visible_time": state["remain_visible_time"],
            "congestion_label": state["congestion_label"],
            "next_load_mbps": next_state["load_mbps"],
            "next_utilization": next_state["utilization"],
            "next_congestion_label": next_state["congestion_label"],
        },
        columns=OUTPUT_COLUMNS,
    )
    return frame


def append_frames_to_csv(frames: list[pd.DataFrame], output_path: Path, write_header: bool) -> bool:
    if not frames:
        return write_header

    block = pd.concat(frames, ignore_index=True)
    block.to_csv(
        output_path,
        mode="a",
        index=False,
        header=write_header,
        float_format="%.9g",
    )
    frames.clear()
    return False


def save_curves(
    time_values: list[int],
    average_utilization: list[float],
    mlu_values: list[float],
    results_dir: Path,
    debug_suffix: str,
) -> tuple[Path, Path]:
    ensure_dir(results_dir)

    avg_path = results_dir / f"leo_average_utilization_curve{debug_suffix}.png"
    mlu_path = results_dir / f"leo_mlu_curve{debug_suffix}.png"

    plt.figure(figsize=(12, 4.5))
    plt.plot(time_values, average_utilization, linewidth=0.8)
    plt.xlabel("time")
    plt.ylabel("average utilization")
    plt.title("Average utilization across all ISL edges")
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(avg_path, dpi=200)
    plt.close()

    plt.figure(figsize=(12, 4.5))
    plt.plot(time_values, mlu_values, linewidth=0.8)
    plt.xlabel("time")
    plt.ylabel("maximum link utilization")
    plt.title("Maximum link utilization")
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(mlu_path, dpi=200)
    plt.close()

    return avg_path, mlu_path


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    processed_dir = ensure_dir(args.processed_dir)
    results_dir = ensure_dir(args.results_dir)

    for path in [args.od_path, args.edges_path, args.sat_positions_path, args.gateway_access_path]:
        require_file(path)

    topo_name = str(config.get("topo_name", "topo144"))
    capacity_mbps = float(config.get("capacity_mbps", 1000))
    congestion_threshold = float(config.get("congestion_threshold", 0.8))
    debug_suffix = "" if args.max_steps is None else f"_debug{args.max_steps}"
    output_path = processed_dir / f"link_state_{topo_name}_shortest_delay{debug_suffix}.csv"

    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Output file already exists: {output_path}. "
            "Pass --overwrite to replace it."
        )
    if output_path.exists() and args.overwrite:
        output_path.unlink()

    if args.flush_steps <= 0:
        raise ValueError(f"--flush_steps must be positive, got {args.flush_steps}")

    od_matrices = np.load(args.od_path, mmap_mode="r")
    sat_positions = np.load(args.sat_positions_path, mmap_mode="r")
    gateway_access = np.load(args.gateway_access_path, mmap_mode="r")
    edges = pd.read_csv(args.edges_path).sort_values("edge_id").reset_index(drop=True)

    full_steps = int(od_matrices.shape[0])
    if args.max_steps is not None:
        if args.max_steps <= 1:
            raise ValueError(f"--max_steps must be > 1 to create next labels, got {args.max_steps}")
        time_steps = min(args.max_steps, full_steps)
    else:
        time_steps = full_steps

    if sat_positions.shape[0] < time_steps or gateway_access.shape[0] < time_steps:
        raise ValueError(
            "Input time lengths are inconsistent: "
            f"od={od_matrices.shape}, sat_positions={sat_positions.shape}, "
            f"gateway_access={gateway_access.shape}, requested={time_steps}"
        )

    initial_delay_ms = compute_edge_delays_ms(sat_positions[0], edges)
    graph = build_delay_graph(edges, initial_delay_ms)
    edge_lookup = build_edge_lookup(edges)
    path_cache: dict[tuple[int, int], list[int]] = {}

    print("Link-state simulation settings:")
    print(f"- topo_name: {topo_name}")
    print(f"- time_steps: {time_steps}")
    print(f"- output rows: {(time_steps - 1) * len(edges)}")
    print(f"- edges: {len(edges)}")
    print(f"- capacity_mbps: {capacity_mbps}")
    print(f"- congestion_threshold: {congestion_threshold}")
    print(f"- traffic_unit: {args.traffic_unit}")
    print("Traffic unit: Abilene raw unit = 100 bytes / 5 minutes")
    print("Conversion: Mbps = raw * 100 * 8 / 300 / 1e6")
    print(
        "- routing: NetworkX shortest-delay Dijkstra on t=0 delays. "
        "For the current fixed Walker-like geometry, ISL distances are time-invariant."
    )

    pending_frames: list[pd.DataFrame] = []
    write_header = True
    average_utilization: list[float] = []
    mlu_values: list[float] = []
    written_times: list[int] = []

    prev_state = compute_link_state(
        time_index=0,
        od_matrix=od_matrices[0],
        gateway_access_t=gateway_access[0],
        sat_positions_t=sat_positions[0],
        edges=edges,
        graph=graph,
        edge_lookup=edge_lookup,
        path_cache=path_cache,
        capacity_mbps=capacity_mbps,
        congestion_threshold=congestion_threshold,
        traffic_unit=args.traffic_unit,
    )

    for time_index in tqdm(range(1, time_steps), desc="Simulating LEO link states"):
        current_state = compute_link_state(
            time_index=time_index,
            od_matrix=od_matrices[time_index],
            gateway_access_t=gateway_access[time_index],
            sat_positions_t=sat_positions[time_index],
            edges=edges,
            graph=graph,
            edge_lookup=edge_lookup,
            path_cache=path_cache,
            capacity_mbps=capacity_mbps,
            congestion_threshold=congestion_threshold,
            traffic_unit=args.traffic_unit,
        )

        output_time = time_index - 1
        pending_frames.append(
            make_output_frame(
                time_index=output_time,
                edges=edges,
                state=prev_state,
                next_state=current_state,
                capacity_mbps=capacity_mbps,
            )
        )
        written_times.append(output_time)
        average_utilization.append(float(prev_state["utilization"].mean()))
        mlu_values.append(float(prev_state["utilization"].max()))

        if len(pending_frames) >= args.flush_steps:
            write_header = append_frames_to_csv(pending_frames, output_path, write_header)

        prev_state = current_state

    append_frames_to_csv(pending_frames, output_path, write_header)
    avg_path, mlu_path = save_curves(
        time_values=written_times,
        average_utilization=average_utilization,
        mlu_values=mlu_values,
        results_dir=results_dir,
        debug_suffix=debug_suffix,
    )

    print("\nSimulation complete:")
    print(f"- output CSV: {output_path}")
    print(f"- rows written: {(time_steps - 1) * len(edges)}")
    print(f"- time values written: {time_steps - 1}")
    print(f"- path cache entries: {len(path_cache)}")
    print(f"- average utilization curve: {avg_path}")
    print(f"- MLU curve: {mlu_path}")


if __name__ == "__main__":
    main()
