import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

EARTH_RADIUS_KM = 6371.0
EARTH_MU_KM3_S2 = 398600.4418
EARTH_ROTATION_RAD_PER_MIN = 2.0 * np.pi / 1436.0
TIME_STEP_MINUTES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a topo144 Walker-like LEO constellation."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--od_path",
        type=Path,
        default=PROCESSED_DIR / "od_matrices_full.npy",
        help="Parsed Abilene OD matrix file used only to read time length T.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional debug limit. Debug outputs include max_steps in filenames.",
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


def get_time_steps(od_path: Path, max_steps: int | None) -> tuple[int, int]:
    if not od_path.exists():
        raise FileNotFoundError(
            f"OD matrix file does not exist: {od_path}. "
            "Run python src/parse_abilene.py --traffic_value_mode first first."
        )

    od_matrices = np.load(od_path, mmap_mode="r")
    full_steps = int(od_matrices.shape[0])
    if max_steps is None:
        return full_steps, full_steps
    if max_steps <= 0:
        raise ValueError(f"--max_steps must be positive, got {max_steps}")
    return min(max_steps, full_steps), full_steps


def sat_id(plane: int, index: int, sats_per_plane: int) -> int:
    return plane * sats_per_plane + index


def build_isl_edges(
    num_planes: int,
    sats_per_plane: int,
    capacity_mbps: float,
) -> pd.DataFrame:
    edges = {}

    def add_edge(
        src_plane: int,
        src_index: int,
        dst_plane: int,
        dst_index: int,
        link_type: str,
    ) -> None:
        src = sat_id(src_plane, src_index, sats_per_plane)
        dst = sat_id(dst_plane, dst_index, sats_per_plane)
        key = tuple(sorted((src, dst)))
        if key not in edges:
            edges[key] = {
                "src_sat": key[0],
                "dst_sat": key[1],
                "src_plane": key[0] // sats_per_plane,
                "src_index": key[0] % sats_per_plane,
                "dst_plane": key[1] // sats_per_plane,
                "dst_index": key[1] % sats_per_plane,
                "capacity_mbps": capacity_mbps,
                "link_type": link_type,
            }

    for plane in range(num_planes):
        for index in range(sats_per_plane):
            add_edge(
                src_plane=plane,
                src_index=index,
                dst_plane=plane,
                dst_index=(index + 1) % sats_per_plane,
                link_type="intra_orbit",
            )

    for plane in range(num_planes):
        next_plane = (plane + 1) % num_planes
        for index in range(sats_per_plane):
            add_edge(
                src_plane=plane,
                src_index=index,
                dst_plane=next_plane,
                dst_index=index,
                link_type="inter_orbit",
            )

    rows = []
    for edge_id, edge in enumerate(sorted(edges.values(), key=lambda item: (item["src_sat"], item["dst_sat"]))):
        rows.append({"edge_id": edge_id, **edge})
    return pd.DataFrame(rows)


def generate_walker_like_positions(
    time_steps: int,
    num_planes: int,
    sats_per_plane: int,
    altitude_km: float,
    inclination_deg: float,
    time_step_minutes: int,
    batch_size: int = 1024,
) -> np.ndarray:
    """Generate approximate LEO sub-satellite positions.

    This simplified circular-orbit interface is intentionally isolated so it can
    be replaced by TLE/SGP4 propagation in a later stage.
    """
    total_sats = num_planes * sats_per_plane
    orbit_radius_km = EARTH_RADIUS_KM + altitude_km
    inclination = np.deg2rad(inclination_deg)
    mean_motion_rad_per_min = np.sqrt(EARTH_MU_KM3_S2 / orbit_radius_km**3) * 60.0

    sat_planes = np.repeat(np.arange(num_planes), sats_per_plane)
    sat_indices = np.tile(np.arange(sats_per_plane), num_planes)
    raan = 2.0 * np.pi * sat_planes / num_planes
    phase_offset = 2.0 * np.pi * sat_planes / (num_planes * sats_per_plane)
    initial_argument = 2.0 * np.pi * sat_indices / sats_per_plane + phase_offset

    positions = np.empty((time_steps, total_sats, 3), dtype=np.float32)
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_inc = np.cos(inclination)
    sin_inc = np.sin(inclination)

    for start in tqdm(range(0, time_steps, batch_size), desc="Generating satellite positions"):
        end = min(start + batch_size, time_steps)
        time_minutes = np.arange(start, end, dtype=np.float64)[:, None] * time_step_minutes
        argument = initial_argument[None, :] + mean_motion_rad_per_min * time_minutes

        cos_u = np.cos(argument)
        sin_u = np.sin(argument)

        x_eci = orbit_radius_km * (cos_raan[None, :] * cos_u - sin_raan[None, :] * sin_u * cos_inc)
        y_eci = orbit_radius_km * (sin_raan[None, :] * cos_u + cos_raan[None, :] * sin_u * cos_inc)
        z_eci = orbit_radius_km * (sin_u * sin_inc)

        earth_theta = EARTH_ROTATION_RAD_PER_MIN * time_minutes
        cos_theta = np.cos(earth_theta)
        sin_theta = np.sin(earth_theta)

        x_ecef = cos_theta * x_eci + sin_theta * y_eci
        y_ecef = -sin_theta * x_eci + cos_theta * y_eci
        z_ecef = z_eci

        longitude = np.rad2deg(np.arctan2(y_ecef, x_ecef))
        hyp = np.sqrt(x_ecef**2 + y_ecef**2)
        latitude = np.rad2deg(np.arctan2(z_ecef, hyp))

        positions[start:end, :, 0] = latitude.astype(np.float32)
        positions[start:end, :, 1] = longitude.astype(np.float32)
        positions[start:end, :, 2] = np.float32(altitude_km)

    return positions


def save_snapshot(positions: np.ndarray, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    snapshot = positions[0]

    plt.figure(figsize=(11, 5.5))
    plt.scatter(snapshot[:, 1], snapshot[:, 0], s=18, c=np.arange(snapshot.shape[0]), cmap="viridis")
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.xlabel("longitude_deg")
    plt.ylabel("latitude_deg")
    plt.title("topo144 satellite snapshot at t=0")
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    processed_dir = ensure_dir(args.processed_dir)
    results_dir = ensure_dir(args.results_dir)

    topo_name = str(config.get("topo_name", "topo144"))
    num_planes = int(config["num_planes"])
    sats_per_plane = int(config["sats_per_plane"])
    altitude_km = float(config["altitude_km"])
    inclination_deg = float(config["inclination_deg"])
    capacity_mbps = float(config["capacity_mbps"])
    total_sats = num_planes * sats_per_plane

    time_steps, full_steps = get_time_steps(args.od_path, args.max_steps)
    debug_suffix = "" if args.max_steps is None else f"_debug{time_steps}"

    edges = build_isl_edges(num_planes, sats_per_plane, capacity_mbps)
    edges_path = processed_dir / f"leo_{topo_name}_edges.csv"
    edges.to_csv(edges_path, index=False)

    positions = generate_walker_like_positions(
        time_steps=time_steps,
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        time_step_minutes=TIME_STEP_MINUTES,
    )
    positions_path = processed_dir / f"leo_{topo_name}_sat_positions{debug_suffix}.npy"
    np.save(positions_path, positions)

    meta = {
        "topo_name": topo_name,
        "num_planes": num_planes,
        "sats_per_plane": sats_per_plane,
        "total_sats": total_sats,
        "altitude_km": altitude_km,
        "inclination_deg": inclination_deg,
        "num_edges": int(len(edges)),
        "time_steps": int(time_steps),
        "full_time_steps": int(full_steps),
        "time_step_minutes": TIME_STEP_MINUTES,
        "debug": args.max_steps is not None,
    }
    meta_path = processed_dir / f"leo_{topo_name}_meta{debug_suffix}.json"
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)

    snapshot_path = results_dir / f"leo_{topo_name}_snapshot{debug_suffix}.png"
    save_snapshot(positions, snapshot_path)

    print("LEO constellation build complete:")
    print(f"- topo_name: {topo_name}")
    print(f"- total_sats: {total_sats}")
    print(f"- time_steps: {time_steps}")
    print(f"- edges: {len(edges)}")
    print("\nOutput files:")
    print(f"- {edges_path}")
    print(f"- {positions_path}")
    print(f"- {meta_path}")
    print(f"- {snapshot_path}")


if __name__ == "__main__":
    main()
