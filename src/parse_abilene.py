import argparse
import csv
import gzip
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import ensure_dir, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "abilene"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
LOG_DIR = PROJECT_ROOT / "logs"

NUM_NODES = 12
NUM_OD_FLOWS = NUM_NODES * NUM_NODES
VALUES_PER_FLOW = 5
EXPECTED_FIELDS = NUM_OD_FLOWS * VALUES_PER_FLOW
EXPECTED_ROWS_PER_WEEK = 2016


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse Abilene X*.gz traffic matrices into OD matrix arrays."
    )
    parser.add_argument(
        "--traffic_value_mode",
        choices=["first", "mean5", "sum5"],
        default="first",
        help="How to reduce the 5 values attached to each OD flow.",
    )
    parser.add_argument(
        "--start_week",
        type=int,
        default=1,
        help="First Abilene week to parse, inclusive. Valid range: 1-24.",
    )
    parser.add_argument(
        "--end_week",
        type=int,
        default=24,
        help="Last Abilene week to parse, inclusive. Valid range: 1-24.",
    )
    parser.add_argument("--raw_dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--log_dir", type=Path, default=LOG_DIR)
    return parser.parse_args()


def validate_week_range(start_week: int, end_week: int) -> None:
    if not 1 <= start_week <= 24:
        raise ValueError(f"--start_week must be in [1, 24], got {start_week}")
    if not 1 <= end_week <= 24:
        raise ValueError(f"--end_week must be in [1, 24], got {end_week}")
    if start_week > end_week:
        raise ValueError(
            f"--start_week must be <= --end_week, got {start_week} > {end_week}"
        )


def reduce_flow_values(values: np.ndarray, mode: str) -> np.ndarray:
    grouped = values.reshape(NUM_OD_FLOWS, VALUES_PER_FLOW)
    if mode == "first":
        return grouped[:, 0]
    if mode == "mean5":
        return grouped.mean(axis=1)
    if mode == "sum5":
        return grouped.sum(axis=1)
    raise ValueError(f"Unsupported traffic_value_mode: {mode}")


def parse_tm_line(line: str, mode: str) -> np.ndarray:
    parts = line.strip().split()
    if len(parts) != EXPECTED_FIELDS:
        raise ValueError(f"expected {EXPECTED_FIELDS} fields, got {len(parts)}")

    try:
        values = np.array([float(item) for item in parts], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"non-numeric field found: {exc}") from exc

    flows = reduce_flow_values(values, mode)
    matrix = flows.reshape(NUM_NODES, NUM_NODES)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix[matrix < 0] = 0.0
    np.fill_diagonal(matrix, 0.0)
    return matrix.astype(np.float32, copy=False)


def parse_week_file(
    gz_path: Path,
    week: int,
    mode: str,
    error_writer: csv.writer,
) -> tuple[list[np.ndarray], dict[str, int | str]]:
    if not gz_path.exists():
        raise FileNotFoundError(
            f"Required Abilene file does not exist: {gz_path}. "
            "Run python src/download_abilene.py first."
        )

    matrices = []
    valid_rows = 0
    invalid_rows = 0

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as file:
        iterator = tqdm(
            enumerate(file, start=1),
            total=EXPECTED_ROWS_PER_WEEK,
            desc=f"Parsing {gz_path.name}",
            unit="row",
        )
        for line_number, line in iterator:
            if not line.strip():
                invalid_rows += 1
                error_writer.writerow(
                    [week, gz_path.name, line_number, "empty line"]
                )
                continue

            try:
                matrices.append(parse_tm_line(line, mode))
                valid_rows += 1
            except ValueError as exc:
                invalid_rows += 1
                error_writer.writerow([week, gz_path.name, line_number, str(exc)])

    return matrices, {
        "week": week,
        "filename": gz_path.name,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
    }


def save_week_lengths(rows: list[dict[str, int | str]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["week", "filename", "valid_rows", "invalid_rows"]
        )
        writer.writeheader()
        writer.writerows(rows)


def print_statistics(od_matrices: np.ndarray, week_rows: list[dict[str, int | str]]) -> None:
    values = od_matrices.reshape(-1)
    percentiles = np.percentile(values, [50, 90, 95, 99])

    print("\nAbilene OD statistics:")
    print(f"- Total time slots T: {od_matrices.shape[0]}")
    print(f"- Shape: {od_matrices.shape}")
    print(f"- Min: {values.min():.6f}")
    print(f"- Max: {values.max():.6f}")
    print(f"- Mean: {values.mean():.6f}")
    print(f"- P50: {percentiles[0]:.6f}")
    print(f"- P90: {percentiles[1]:.6f}")
    print(f"- P95: {percentiles[2]:.6f}")
    print(f"- P99: {percentiles[3]:.6f}")
    print("- Valid rows per week:")
    for row in week_rows:
        print(
            f"  week {int(row['week']):02d} ({row['filename']}): "
            f"{row['valid_rows']} valid, {row['invalid_rows']} invalid"
        )


def save_total_traffic_curve(od_matrices: np.ndarray, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    total_traffic = od_matrices.sum(axis=(1, 2))

    plt.figure(figsize=(14, 5))
    plt.plot(np.arange(total_traffic.shape[0]), total_traffic, linewidth=0.8)
    plt.xlabel("time index")
    plt.ylabel("total OD traffic")
    plt.title("Abilene total OD traffic curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    validate_week_range(args.start_week, args.end_week)
    set_seed(42)

    processed_dir = ensure_dir(args.processed_dir)
    results_dir = ensure_dir(args.results_dir)
    log_dir = ensure_dir(args.log_dir)

    error_log_path = log_dir / "abilene_parse_errors.txt"
    all_matrices = []
    week_rows = []

    with error_log_path.open("w", newline="", encoding="utf-8") as error_file:
        error_writer = csv.writer(error_file)
        error_writer.writerow(["week", "filename", "line_number", "error"])

        for week in range(args.start_week, args.end_week + 1):
            gz_path = args.raw_dir / f"X{week:02d}.gz"
            matrices, row = parse_week_file(
                gz_path=gz_path,
                week=week,
                mode=args.traffic_value_mode,
                error_writer=error_writer,
            )
            all_matrices.extend(matrices)
            week_rows.append(row)

    if not all_matrices:
        raise RuntimeError(
            "No valid Abilene traffic matrix rows were parsed. "
            f"See error log: {error_log_path}"
        )

    od_matrices = np.stack(all_matrices, axis=0).astype(np.float32, copy=False)

    od_output_path = processed_dir / "od_matrices_full.npy"
    week_lengths_path = processed_dir / "abilene_week_lengths.csv"
    curve_path = results_dir / "abilene_total_traffic_curve.png"

    np.save(od_output_path, od_matrices)
    save_week_lengths(week_rows, week_lengths_path)
    save_total_traffic_curve(od_matrices, curve_path)

    print_statistics(od_matrices, week_rows)
    print("\nOutput files:")
    print(f"- {od_output_path}")
    print(f"- {week_lengths_path}")
    print(f"- {curve_path}")
    print(f"- {error_log_path}")


if __name__ == "__main__":
    main()
