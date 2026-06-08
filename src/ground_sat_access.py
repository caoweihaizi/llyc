import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select gateway access satellites for each Abilene time step."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--gateways_path",
        type=Path,
        default=PROCESSED_DIR / "abilene_gateways.csv",
    )
    parser.add_argument(
        "--sat_positions_path",
        type=Path,
        default=None,
        help="Optional satellite positions file. Defaults to leo_{topo_name}_sat_positions.npy.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional debug limit. Debug outputs include max_steps in filenames.",
    )
    parser.add_argument(
        "--min_elevation_deg",
        type=float,
        default=None,
        help="Override configs/base.yaml min_elevation_deg. Default comes from config, now 15 deg.",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def geodetic_to_ecef(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    altitude_km: np.ndarray | float,
) -> np.ndarray:
    latitude = np.deg2rad(latitude_deg)
    longitude = np.deg2rad(longitude_deg)
    radius = EARTH_RADIUS_KM + altitude_km
    x = radius * np.cos(latitude) * np.cos(longitude)
    y = radius * np.cos(latitude) * np.sin(longitude)
    z = radius * np.sin(latitude)
    return np.stack([x, y, z], axis=-1)


def compute_best_access_batch(
    sat_positions_batch: np.ndarray,
    gateway_ecef: np.ndarray,
    gateway_up: np.ndarray,
    min_elevation_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sat_ecef = geodetic_to_ecef(
        sat_positions_batch[:, :, 0],
        sat_positions_batch[:, :, 1],
        sat_positions_batch[:, :, 2],
    )

    los = sat_ecef[:, None, :, :] - gateway_ecef[None, :, None, :]
    los_norm = np.linalg.norm(los, axis=-1)
    los_norm = np.maximum(los_norm, 1e-12)
    sin_elevation = np.sum(los * gateway_up[None, :, None, :], axis=-1) / los_norm
    sin_elevation = np.clip(sin_elevation, -1.0, 1.0)
    elevation_deg = np.rad2deg(np.arcsin(sin_elevation))

    best_sat = np.argmax(elevation_deg, axis=2).astype(np.int32)
    best_elevation = np.take_along_axis(
        elevation_deg,
        best_sat[:, :, None],
        axis=2,
    )[:, :, 0]
    visible_success = best_elevation > min_elevation_deg
    return best_sat, visible_success, best_elevation


def save_access_example(
    gateway_access: np.ndarray,
    gateways: pd.DataFrame,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    total_steps = min(500, gateway_access.shape[0])
    gateway_count = min(3, gateway_access.shape[1])
    time_index = np.arange(total_steps)

    plt.figure(figsize=(12, 5))
    for gateway_id in range(gateway_count):
        label = str(gateways.iloc[gateway_id]["name"])
        plt.plot(
            time_index,
            gateway_access[:total_steps, gateway_id],
            linewidth=1.0,
            label=label,
        )

    plt.xlabel("time index")
    plt.ylabel("access satellite id")
    plt.title("Gateway access satellites over first 500 time steps")
    plt.legend()
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
    min_elevation_deg = (
        float(args.min_elevation_deg)
        if args.min_elevation_deg is not None
        else float(config.get("min_elevation_deg", 15))
    )
    sat_positions_path = args.sat_positions_path or (
        processed_dir / f"leo_{topo_name}_sat_positions.npy"
    )

    if not args.gateways_path.exists():
        raise FileNotFoundError(
            f"Gateway file does not exist: {args.gateways_path}. "
            "Run python src/parse_abilene_topology.py first."
        )
    if not sat_positions_path.exists():
        raise FileNotFoundError(
            f"Satellite positions file does not exist: {sat_positions_path}. "
            "Run python src/build_leo_constellation.py first."
        )
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be positive, got {args.batch_size}")

    gateways = pd.read_csv(args.gateways_path)
    required_columns = {"name", "latitude", "longitude"}
    missing_columns = sorted(required_columns - set(gateways.columns))
    if missing_columns:
        raise ValueError(f"Gateway CSV missing required columns: {missing_columns}")

    sat_positions = np.load(sat_positions_path, mmap_mode="r")
    full_steps = int(sat_positions.shape[0])
    time_steps = full_steps if args.max_steps is None else min(args.max_steps, full_steps)
    if time_steps <= 0:
        raise ValueError(f"--max_steps must be positive, got {args.max_steps}")

    debug_suffix = "" if args.max_steps is None else f"_debug{time_steps}"
    num_gateways = len(gateways)
    gateway_access = np.empty((time_steps, num_gateways), dtype=np.int32)
    visible_success_counts = np.zeros(num_gateways, dtype=np.int64)
    fallback_counts = np.zeros(num_gateways, dtype=np.int64)

    gateway_ecef = geodetic_to_ecef(
        gateways["latitude"].to_numpy(dtype=np.float64),
        gateways["longitude"].to_numpy(dtype=np.float64),
        0.0,
    )
    gateway_up = gateway_ecef / np.linalg.norm(gateway_ecef, axis=1, keepdims=True)

    for start in tqdm(range(0, time_steps, args.batch_size), desc="Selecting gateway access satellites"):
        end = min(start + args.batch_size, time_steps)
        best_sat, visible_success, _ = compute_best_access_batch(
            sat_positions[start:end],
            gateway_ecef=gateway_ecef,
            gateway_up=gateway_up,
            min_elevation_deg=min_elevation_deg,
        )
        gateway_access[start:end] = best_sat
        visible_success_counts += visible_success.sum(axis=0)
        fallback_counts += (~visible_success).sum(axis=0)

    stats_rows = []
    for gateway_id, gateway in gateways.reset_index(drop=True).iterrows():
        used_satellites = np.unique(gateway_access[:, gateway_id])
        stats_rows.append(
            {
                "gateway_id": gateway_id,
                "gateway_name": gateway["name"],
                "total_steps": time_steps,
                "visible_success_steps": int(visible_success_counts[gateway_id]),
                "fallback_steps": int(fallback_counts[gateway_id]),
                "fallback_rate": float(fallback_counts[gateway_id] / time_steps),
                "unique_satellites_used": int(len(used_satellites)),
            }
        )

    stats = pd.DataFrame(stats_rows)
    access_path = processed_dir / f"gateway_access_{topo_name}{debug_suffix}.npy"
    stats_path = processed_dir / f"gateway_access_stats_{topo_name}{debug_suffix}.csv"
    example_path = results_dir / f"gateway_access_example{debug_suffix}.png"

    np.save(access_path, gateway_access)
    stats.to_csv(stats_path, index=False)
    save_access_example(gateway_access, gateways, example_path)

    print("Gateway access selection complete:")
    print(f"- gateways: {num_gateways}")
    print(f"- time_steps: {time_steps}")
    print(f"- min_elevation_deg: {min_elevation_deg}")
    print(f"- gateway_access shape: {gateway_access.shape}")
    print(f"- mean fallback_rate: {stats['fallback_rate'].mean():.6f}")
    print(f"- max fallback_rate: {stats['fallback_rate'].max():.6f}")
    print("- unique satellites used per gateway:")
    for row in stats.itertuples(index=False):
        print(
            f"  gateway {int(row.gateway_id):02d} {row.gateway_name}: "
            f"{int(row.unique_satellites_used)}"
        )
    print("\nOutput files:")
    print(f"- {access_path}")
    print(f"- {stats_path}")
    print(f"- {example_path}")


if __name__ == "__main__":
    main()
