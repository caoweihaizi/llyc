import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXPECTED_T = 2016 * 24
EXPECTED_NODES = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check parsed Abilene data files.")
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--t_tolerance",
        type=int,
        default=2016,
        help="Allowed absolute difference from 2016*24 time slots.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required processed file does not exist: {path}. "
            "Run the Abilene parsing scripts first."
        )


def check_od_matrices(path: Path, t_tolerance: int) -> list[str]:
    problems = []
    od_matrices = np.load(path)

    print(f"Loaded OD matrices: {path}")
    print(f"- Shape: {od_matrices.shape}")

    if od_matrices.ndim != 3 or od_matrices.shape[1:] != (
        EXPECTED_NODES,
        EXPECTED_NODES,
    ):
        problems.append(f"Expected OD shape [T, 12, 12], got {od_matrices.shape}")
    else:
        print("- Shape check: OK")

    t_value = od_matrices.shape[0]
    if abs(t_value - EXPECTED_T) > t_tolerance:
        problems.append(
            f"T is not close to {EXPECTED_T}: got {t_value}, "
            f"tolerance is {t_tolerance}"
        )
    else:
        print(f"- T close to {EXPECTED_T}: OK ({t_value})")

    if np.isnan(od_matrices).any():
        problems.append("OD matrices contain nan values.")
    else:
        print("- NaN check: OK")

    if np.isinf(od_matrices).any():
        problems.append("OD matrices contain inf values.")
    else:
        print("- Inf check: OK")

    if (od_matrices < 0).any():
        problems.append("OD matrices contain negative values.")
    else:
        print("- Negative value check: OK")

    if od_matrices.ndim == 3 and od_matrices.shape[1] == od_matrices.shape[2]:
        diagonals = np.diagonal(od_matrices, axis1=1, axis2=2)
        if not np.allclose(diagonals, 0.0):
            problems.append("OD matrix diagonals are not all zero.")
        else:
            print("- Diagonal zero check: OK")

    return problems


def check_gateways(path: Path) -> list[str]:
    problems = []
    gateways = pd.read_csv(path)

    print(f"\nLoaded gateways: {path}")
    print(f"- Gateway rows: {len(gateways)}")

    required_columns = {"name", "city", "latitude", "longitude"}
    missing_columns = sorted(required_columns - set(gateways.columns))
    if missing_columns:
        problems.append(f"Gateway CSV missing columns: {missing_columns}")

    if len(gateways) != EXPECTED_NODES:
        problems.append(f"Expected 12 gateways, got {len(gateways)}")
    else:
        print("- Gateway count check: OK")

    if not missing_columns:
        print("\nFirst 5 gateways:")
        print(gateways[["name", "city", "latitude", "longitude"]].head().to_string(index=False))

    return problems


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir

    od_path = processed_dir / "od_matrices_full.npy"
    gateways_path = processed_dir / "abilene_gateways.csv"
    links_path = processed_dir / "abilene_links.csv"

    for path in [od_path, gateways_path, links_path]:
        require_file(path)
        print(f"Found: {path}")

    problems = []
    problems.extend(check_od_matrices(od_path, args.t_tolerance))
    problems.extend(check_gateways(gateways_path))

    links = pd.read_csv(links_path)
    print(f"\nLoaded links: {links_path}")
    print(f"- Link rows: {len(links)}")
    print(f"- Link columns: {list(links.columns)}")

    if problems:
        print("\nCheck result: FAILED")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)

    print("\nCheck result: OK")
    print("\nChecked files:")
    print(f"- {od_path}")
    print(f"- {gateways_path}")
    print(f"- {links_path}")


if __name__ == "__main__":
    main()
