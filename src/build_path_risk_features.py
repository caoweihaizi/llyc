import argparse
import csv
import json
from ast import literal_eval
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

ABILENE_RAW_TO_MBPS = 100.0 * 8.0 / 300.0 / 1e6

FEATURE_DESCRIPTIONS = {
    "od_demand_mbps": "当前 gateway OD 需求，Abilene raw 按 raw * 100 * 8 / 300 / 1e6 转为 Mbps",
    "hop_count": "候选路径跳数",
    "path_delay_ms_sum": "路径当前链路传播时延之和",
    "path_delay_ms_mean": "路径当前链路传播时延均值",
    "path_current_util_max": "路径当前链路利用率最大值",
    "path_current_util_mean": "路径当前链路利用率均值",
    "path_current_util_sum": "路径当前链路利用率之和",
    "path_current_congestion_count": "路径当前拥塞链路数量",
    "path_current_congestion_ratio": "路径当前拥塞链路比例",
    "path_pred_util_max": "路径预测下一时刻链路利用率最大值",
    "path_pred_util_mean": "路径预测下一时刻链路利用率均值",
    "path_pred_util_sum": "路径预测下一时刻链路利用率之和",
    "path_util_uncertainty_max": "路径利用率预测标准差最大值",
    "path_util_uncertainty_mean": "路径利用率预测标准差均值",
    "path_cong_prob_max": "路径预测拥塞概率最大值",
    "path_cong_prob_mean": "路径预测拥塞概率均值",
    "path_cong_prob_sum": "路径预测拥塞概率之和",
    "path_cong_uncertainty_max": "路径拥塞概率标准差最大值",
    "path_cong_uncertainty_mean": "路径拥塞概率标准差均值",
    "path_risk_score_max": "路径 risk_score 最大值，risk_score = util_pred_mean + lambda * util_pred_std",
    "path_risk_score_mean": "路径 risk_score 均值",
    "path_risk_score_sum": "路径 risk_score 之和",
    "path_congestion_risk_score_max": "路径 congestion_risk_score 最大值",
    "path_congestion_risk_score_mean": "路径 congestion_risk_score 均值",
    "path_congestion_risk_score_sum": "路径 congestion_risk_score 之和",
    "is_shortest_path": "是否为第 0 条 hop-based shortest candidate path",
    "is_min_risk_path": "同一 sample 和 gateway OD pair 内 path_risk_score_max 最小的路径",
    "is_min_cong_prob_path": "同一 sample 和 gateway OD pair 内 path_cong_prob_max 最小的路径",
    "is_min_delay_path": "同一 sample 和 gateway OD pair 内 path_delay_ms_sum 最小的路径",
    "rank_by_risk": "同组内按 path_risk_score_max 从小到大的排名，1 表示最低风险",
    "rank_by_cong_prob": "同组内按 path_cong_prob_max 从小到大的排名，1 表示最低拥塞概率",
    "rank_by_delay": "同组内按 path_delay_ms_sum 从小到大的排名，1 表示最低时延",
}

FEATURE_NAMES = list(FEATURE_DESCRIPTIONS.keys())

CSV_COLUMNS = [
    "sample_idx",
    "time_t",
    "target_time",
    "src_gateway",
    "dst_gateway",
    "src_sat",
    "dst_sat",
    "path_id",
    "hop_count",
    "od_demand_mbps",
    "is_valid",
    *[name for name in FEATURE_NAMES if name not in {"od_demand_mbps", "hop_count"}],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate link-level MC Dropout predictions into path-level risk features."
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional debug sample limit.")
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--candidate_summary",
        type=Path,
        default=None,
        help="Candidate path summary CSV. Defaults to candidate_paths_topo144_k{k}_summary.csv.",
    )
    parser.add_argument(
        "--mc_predictions",
        type=Path,
        default=RESULTS_DIR / "mc_dropout_predictions_test.npz",
    )
    parser.add_argument(
        "--samples_path",
        type=Path,
        default=PROCESSED_DIR / "samples_topo144_seq12.npz",
    )
    parser.add_argument(
        "--splits_path",
        type=Path,
        default=PROCESSED_DIR / "splits_topo144_seq12.json",
    )
    parser.add_argument(
        "--gateway_access_path",
        type=Path,
        default=PROCESSED_DIR / "gateway_access_topo144.npy",
    )
    parser.add_argument(
        "--od_path",
        type=Path,
        default=PROCESSED_DIR / "od_matrices_full.npy",
    )
    parser.add_argument(
        "--link_state_path",
        type=Path,
        default=PROCESSED_DIR / "link_state_topo144_shortest_delay.csv",
    )
    parser.add_argument("--chunk_size", type=int, default=500_000)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {path}")


def suffix(max_samples: int | None) -> str:
    return "" if max_samples is None else f"_debug{max_samples}"


def parse_list_cell(value: object, row_number: int, column: str) -> list[int]:
    try:
        parsed = literal_eval(str(value))
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Failed to parse {column} at CSV row {row_number}: {value}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{column} at CSV row {row_number} is not a list: {value}")
    return [int(item) for item in parsed]


def load_candidate_paths(summary_path: Path, k: int) -> dict[tuple[int, int], list[dict]]:
    require_file(summary_path)
    paths: dict[tuple[int, int], list[dict]] = {}

    for row_idx, row in enumerate(pd.read_csv(summary_path).itertuples(index=False), start=2):
        sat_path = parse_list_cell(row.sat_path, row_idx, "sat_path")
        edge_path = parse_list_cell(row.edge_path, row_idx, "edge_path")
        record = {
            "src_sat": int(row.src_sat),
            "dst_sat": int(row.dst_sat),
            "path_id": int(row.path_id),
            "sat_path": sat_path,
            "edge_path": edge_path,
            "hop_count": int(row.hop_count),
            "is_valid": bool(row.is_valid),
        }
        paths.setdefault((record["src_sat"], record["dst_sat"]), []).append(record)

    for pair, records in paths.items():
        records.sort(key=lambda item: item["path_id"])
        if len(records) < k:
            raise ValueError(f"Satellite pair {pair} has only {len(records)} candidate paths; expected {k}.")
    return paths


def load_test_times(samples_path: Path, splits_path: Path, max_samples: int | None) -> tuple[np.ndarray, np.ndarray, dict]:
    require_file(samples_path)
    require_file(splits_path)
    samples = np.load(samples_path, allow_pickle=False)
    if "times" not in samples.files:
        raise KeyError("samples_topo144_seq12.npz does not contain 'times'; cannot align test predictions.")

    with splits_path.open("r", encoding="utf-8") as file:
        splits = json.load(file)

    test_start = int(splits["test_start"])
    test_end = int(splits["test_end"])
    test_indices = np.arange(test_start, test_end, dtype=np.int32)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"--max_samples must be positive, got {max_samples}")
        test_indices = test_indices[:max_samples]

    times = samples["times"]
    test_times = times[test_indices].astype(np.int32)
    return test_indices, test_times, splits


def load_mc_predictions(mc_path: Path, n_samples: int) -> dict[str, np.ndarray]:
    require_file(mc_path)
    mc = np.load(mc_path, allow_pickle=False)
    required = [
        "util_pred_mean",
        "util_pred_std",
        "cong_prob_mean",
        "cong_prob_std",
        "risk_score",
        "congestion_risk_score",
    ]
    missing = [key for key in required if key not in mc.files]
    if missing:
        raise KeyError(f"Missing required MC prediction arrays: {missing}")

    predictions = {key: mc[key][:n_samples].astype(np.float32, copy=False) for key in required}
    for key, array in predictions.items():
        if array.shape[0] != n_samples:
            raise ValueError(f"{key} has too few samples: {array.shape[0]} < {n_samples}")
    return predictions


def load_current_link_state(
    link_state_path: Path,
    test_times: np.ndarray,
    edge_count: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_file(link_state_path)
    time_to_index = {int(time): idx for idx, time in enumerate(test_times.tolist())}
    needed_times = set(time_to_index)

    n_samples = len(test_times)
    delay = np.full((n_samples, edge_count), np.nan, dtype=np.float32)
    util = np.full((n_samples, edge_count), np.nan, dtype=np.float32)
    cong = np.full((n_samples, edge_count), np.nan, dtype=np.float32)

    usecols = ["time", "edge_id", "delay_ms", "utilization", "congestion_label"]
    print("Loading current link_state rows for test_times by chunks...")
    for chunk in tqdm(
        pd.read_csv(link_state_path, usecols=usecols, chunksize=chunk_size),
        desc="Filtering link_state",
    ):
        filtered = chunk[chunk["time"].isin(needed_times)]
        if filtered.empty:
            continue
        for row in filtered.itertuples(index=False):
            sample_pos = time_to_index[int(row.time)]
            edge_id = int(row.edge_id)
            delay[sample_pos, edge_id] = float(row.delay_ms)
            util[sample_pos, edge_id] = float(row.utilization)
            cong[sample_pos, edge_id] = float(row.congestion_label)

    if np.isnan(delay).any() or np.isnan(util).any() or np.isnan(cong).any():
        missing_count = int(np.isnan(delay).sum() + np.isnan(util).sum() + np.isnan(cong).sum())
        raise ValueError(f"Missing current link_state values after chunk loading: {missing_count}")

    return delay, util, cong


def gateway_pairs() -> np.ndarray:
    return np.asarray([(src, dst) for src in range(12) for dst in range(12) if src != dst], dtype=np.int16)


def aggregate_path(
    edge_path: list[int],
    demand_mbps: float,
    delay_row: np.ndarray,
    util_row: np.ndarray,
    cong_row: np.ndarray,
    pred: dict[str, np.ndarray],
    sample_pos: int,
    hop_count: int,
) -> dict[str, float]:
    if len(edge_path) == 0:
        values = {name: 0.0 for name in FEATURE_NAMES}
        values["od_demand_mbps"] = float(demand_mbps)
        values["hop_count"] = float(hop_count)
        return values

    edges = np.asarray(edge_path, dtype=np.int32)
    current_delay = delay_row[edges]
    current_util = util_row[edges]
    current_cong = cong_row[edges]
    pred_util = pred["util_pred_mean"][sample_pos, edges]
    util_std = pred["util_pred_std"][sample_pos, edges]
    cong_prob = pred["cong_prob_mean"][sample_pos, edges]
    cong_std = pred["cong_prob_std"][sample_pos, edges]
    risk = pred["risk_score"][sample_pos, edges]
    cong_risk = pred["congestion_risk_score"][sample_pos, edges]

    return {
        "od_demand_mbps": float(demand_mbps),
        "hop_count": float(hop_count),
        "path_delay_ms_sum": float(current_delay.sum()),
        "path_delay_ms_mean": float(current_delay.mean()),
        "path_current_util_max": float(current_util.max()),
        "path_current_util_mean": float(current_util.mean()),
        "path_current_util_sum": float(current_util.sum()),
        "path_current_congestion_count": float(current_cong.sum()),
        "path_current_congestion_ratio": float(current_cong.mean()),
        "path_pred_util_max": float(pred_util.max()),
        "path_pred_util_mean": float(pred_util.mean()),
        "path_pred_util_sum": float(pred_util.sum()),
        "path_util_uncertainty_max": float(util_std.max()),
        "path_util_uncertainty_mean": float(util_std.mean()),
        "path_cong_prob_max": float(cong_prob.max()),
        "path_cong_prob_mean": float(cong_prob.mean()),
        "path_cong_prob_sum": float(cong_prob.sum()),
        "path_cong_uncertainty_max": float(cong_std.max()),
        "path_cong_uncertainty_mean": float(cong_std.mean()),
        "path_risk_score_max": float(risk.max()),
        "path_risk_score_mean": float(risk.mean()),
        "path_risk_score_sum": float(risk.sum()),
        "path_congestion_risk_score_max": float(cong_risk.max()),
        "path_congestion_risk_score_mean": float(cong_risk.mean()),
        "path_congestion_risk_score_sum": float(cong_risk.sum()),
        "is_shortest_path": 0.0,
        "is_min_risk_path": 0.0,
        "is_min_cong_prob_path": 0.0,
        "is_min_delay_path": 0.0,
        "rank_by_risk": 0.0,
        "rank_by_cong_prob": 0.0,
        "rank_by_delay": 0.0,
    }


def local_same_sat_paths(sat_id: int, k: int) -> list[dict]:
    return [
        {
            "src_sat": int(sat_id),
            "dst_sat": int(sat_id),
            "path_id": int(path_id),
            "sat_path": [int(sat_id)],
            "edge_path": [],
            "hop_count": 0,
            "is_valid": True,
        }
        for path_id in range(k)
    ]


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float32)
    return ranks


def update_strategy_accumulators(accumulators: dict[str, dict[str, float]], strategy: str, row: dict) -> None:
    fields = [
        "hop_count",
        "path_delay_ms_sum",
        "path_current_util_max",
        "path_pred_util_max",
        "path_cong_prob_max",
        "path_risk_score_max",
        "od_demand_mbps",
    ]
    acc = accumulators[strategy]
    acc["count"] += 1.0
    for field in fields:
        acc[field] += float(row[field])


def write_feature_name_json(path: Path) -> None:
    payload = {
        "feature_names": FEATURE_NAMES,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "note": "features array excludes identifier columns; IDs and masks are stored as separate NPZ arrays.",
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_plots(
    csv_path: Path,
    results_dir: Path,
    strategy_csv: Path,
    debug_suffix: str,
    sample_rows: int = 250_000,
) -> list[Path]:
    print("Generating path-risk figures from CSV samples/statistics...")
    usecols = [
        "path_risk_score_max",
        "path_cong_prob_max",
        "path_delay_ms_sum",
        "path_id",
        "is_min_risk_path",
    ]
    chunks = pd.read_csv(csv_path, usecols=usecols, chunksize=200_000)
    sampled_parts = []
    path_id_counts = np.zeros(32, dtype=np.int64)
    for chunk in chunks:
        min_risk = chunk[chunk["is_min_risk_path"].astype(bool)]
        for path_id, count in min_risk["path_id"].value_counts().items():
            if int(path_id) >= len(path_id_counts):
                path_id_counts = np.pad(path_id_counts, (0, int(path_id) - len(path_id_counts) + 1))
            path_id_counts[int(path_id)] += int(count)
        if sum(len(part) for part in sampled_parts) < sample_rows:
            sampled_parts.append(chunk.head(max(0, sample_rows - sum(len(part) for part in sampled_parts))))

    sample = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame(columns=usecols)
    paths = {
        "risk_dist": results_dir / f"path_risk_score_distribution{debug_suffix}.png",
        "cong_dist": results_dir / f"path_cong_prob_distribution{debug_suffix}.png",
        "delay_risk": results_dir / f"path_delay_vs_risk_scatter{debug_suffix}.png",
        "shortest_vs_min_risk": results_dir / f"shortest_vs_min_risk_comparison{debug_suffix}.png",
        "min_risk_path_id": results_dir / f"min_risk_path_id_distribution{debug_suffix}.png",
    }

    plt.figure(figsize=(8, 4.5))
    plt.hist(sample["path_risk_score_max"], bins=80, edgecolor="black", linewidth=0.3)
    plt.xlabel("path_risk_score_max")
    plt.ylabel("path count")
    plt.title("Path risk score distribution")
    plt.tight_layout()
    plt.savefig(paths["risk_dist"], dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.hist(sample["path_cong_prob_max"], bins=80, edgecolor="black", linewidth=0.3)
    plt.xlabel("path_cong_prob_max")
    plt.ylabel("path count")
    plt.title("Path congestion probability distribution")
    plt.tight_layout()
    plt.savefig(paths["cong_dist"], dpi=200)
    plt.close()

    scatter = sample.sample(n=min(len(sample), 100_000), random_state=42) if len(sample) else sample
    plt.figure(figsize=(7, 5))
    plt.scatter(scatter["path_delay_ms_sum"], scatter["path_risk_score_max"], s=2, alpha=0.25)
    plt.xlabel("path_delay_ms_sum")
    plt.ylabel("path_risk_score_max")
    plt.title("Path delay vs risk")
    plt.tight_layout()
    plt.savefig(paths["delay_risk"], dpi=200)
    plt.close()

    strategy_df = pd.read_csv(strategy_csv)
    metric_cols = ["path_risk_score_max", "path_delay_ms_sum", "hop_count"]
    plot_df = strategy_df[strategy_df["strategy"].isin(["shortest_path", "min_risk_path"])]
    x = np.arange(len(metric_cols))
    width = 0.35
    plt.figure(figsize=(8, 4.5))
    for idx, strategy in enumerate(["shortest_path", "min_risk_path"]):
        vals = [float(plot_df.loc[plot_df["strategy"] == strategy, col].iloc[0]) for col in metric_cols]
        plt.bar(x + (idx - 0.5) * width, vals, width=width, label=strategy)
    plt.xticks(x, metric_cols, rotation=15, ha="right")
    plt.title("Shortest path vs min-risk path")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["shortest_vs_min_risk"], dpi=200)
    plt.close()

    valid_counts = path_id_counts[: int(np.nonzero(path_id_counts)[0].max() + 1) if np.any(path_id_counts) else 0]
    plt.figure(figsize=(7, 4.5))
    plt.bar(np.arange(len(valid_counts)), valid_counts)
    plt.xlabel("path_id selected as min-risk")
    plt.ylabel("selection count")
    plt.title("Min-risk path_id distribution")
    plt.tight_layout()
    plt.savefig(paths["min_risk_path_id"], dpi=200)
    plt.close()

    return list(paths.values())


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError(f"--k must be positive, got {args.k}")
    if args.chunk_size <= 0:
        raise ValueError(f"--chunk_size must be positive, got {args.chunk_size}")

    processed_dir = ensure_dir(args.processed_dir)
    results_dir = ensure_dir(args.results_dir)
    debug_suffix = suffix(args.max_samples)

    candidate_summary = args.candidate_summary or processed_dir / f"candidate_paths_topo144_k{args.k}_summary.csv"
    npz_path = processed_dir / f"path_risk_features_test_topo144_k{args.k}{debug_suffix}.npz"
    csv_path = processed_dir / f"path_risk_features_test_topo144_k{args.k}{debug_suffix}.csv"
    names_path = processed_dir / f"path_risk_feature_names_topo144_k{args.k}{debug_suffix}.json"
    stats_path = results_dir / f"path_risk_feature_stats_topo144_k{args.k}{debug_suffix}.json"
    strategy_path = results_dir / f"path_risk_strategy_comparison_topo144_k{args.k}{debug_suffix}.csv"

    print("Loading candidate paths...")
    path_dict = load_candidate_paths(candidate_summary, args.k)
    test_indices, test_times, splits = load_test_times(args.samples_path, args.splits_path, args.max_samples)
    n_samples = len(test_times)
    target_times = test_times + 1
    print(f"Aligned test samples: {n_samples}")
    print(f"test time range: {int(test_times.min())} - {int(test_times.max())}")

    predictions = load_mc_predictions(args.mc_predictions, n_samples)
    edge_count = predictions["util_pred_mean"].shape[1]
    current_delay, current_util, current_cong = load_current_link_state(
        args.link_state_path,
        test_times,
        edge_count=edge_count,
        chunk_size=args.chunk_size,
    )

    require_file(args.gateway_access_path)
    require_file(args.od_path)
    gateway_access = np.load(args.gateway_access_path, mmap_mode="r")
    od_matrices = np.load(args.od_path, mmap_mode="r")
    gw_pairs = gateway_pairs()
    pair_count = len(gw_pairs)

    features = np.zeros((n_samples, pair_count, args.k, len(FEATURE_NAMES)), dtype=np.float32)
    valid_mask = np.zeros((n_samples, pair_count, args.k), dtype=bool)
    src_sat_array = np.zeros((n_samples, pair_count), dtype=np.int16)
    dst_sat_array = np.zeros((n_samples, pair_count), dtype=np.int16)

    strategy_acc = {
        name: {
            "count": 0.0,
            "hop_count": 0.0,
            "path_delay_ms_sum": 0.0,
            "path_current_util_max": 0.0,
            "path_pred_util_max": 0.0,
            "path_cong_prob_max": 0.0,
            "path_risk_score_max": 0.0,
            "od_demand_mbps": 0.0,
        }
        for name in ["shortest_path", "min_risk_path", "min_cong_prob_path", "min_delay_path"]
    }

    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    shortest_min_risk_diff = 0
    total_groups = 0
    min_risk_risk_delta_sum = 0.0
    min_risk_hop_delta_sum = 0.0

    try:
        for sample_pos, time_t in enumerate(tqdm(test_times, desc="Building path risk features")):
            access = gateway_access[int(time_t)]
            od_mbps = od_matrices[int(time_t)] * ABILENE_RAW_TO_MBPS
            rows_to_write = []

            for pair_idx, (src_gw, dst_gw) in enumerate(gw_pairs):
                src_sat = int(access[int(src_gw)])
                dst_sat = int(access[int(dst_gw)])
                src_sat_array[sample_pos, pair_idx] = src_sat
                dst_sat_array[sample_pos, pair_idx] = dst_sat
                demand_mbps = float(od_mbps[int(src_gw), int(dst_gw)])

                candidate_records = path_dict.get((src_sat, dst_sat))
                if candidate_records is None and src_sat == dst_sat:
                    candidate_records = local_same_sat_paths(src_sat, args.k)
                if candidate_records is None:
                    raise KeyError(f"No candidate paths for satellite pair {(src_sat, dst_sat)}")

                group_rows = []
                for path_pos, path_record in enumerate(candidate_records[: args.k]):
                    is_valid = bool(path_record["is_valid"])
                    if not is_valid:
                        values = {name: 0.0 for name in FEATURE_NAMES}
                    else:
                        values = aggregate_path(
                            edge_path=path_record["edge_path"],
                            demand_mbps=demand_mbps,
                            delay_row=current_delay[sample_pos],
                            util_row=current_util[sample_pos],
                            cong_row=current_cong[sample_pos],
                            pred=predictions,
                            sample_pos=sample_pos,
                            hop_count=int(path_record["hop_count"]),
                        )
                    values["is_shortest_path"] = 1.0 if int(path_record["path_id"]) == 0 else 0.0
                    group_rows.append((path_pos, path_record, values, is_valid))

                risk_vals = np.asarray([item[2]["path_risk_score_max"] for item in group_rows], dtype=np.float32)
                cong_vals = np.asarray([item[2]["path_cong_prob_max"] for item in group_rows], dtype=np.float32)
                delay_vals = np.asarray([item[2]["path_delay_ms_sum"] for item in group_rows], dtype=np.float32)

                min_risk_idx = int(np.argmin(risk_vals))
                min_cong_idx = int(np.argmin(cong_vals))
                min_delay_value = float(np.min(delay_vals))
                rank_risk = rank_values(risk_vals)
                rank_cong = rank_values(cong_vals)
                rank_delay = rank_values(delay_vals)

                total_groups += 1
                if min_risk_idx != 0:
                    shortest_min_risk_diff += 1
                min_risk_risk_delta_sum += float(risk_vals[0] - risk_vals[min_risk_idx])
                min_risk_hop_delta_sum += float(group_rows[min_risk_idx][2]["hop_count"] - group_rows[0][2]["hop_count"])

                for path_pos, path_record, values, is_valid in group_rows:
                    values["is_min_risk_path"] = 1.0 if path_pos == min_risk_idx else 0.0
                    values["is_min_cong_prob_path"] = 1.0 if path_pos == min_cong_idx else 0.0
                    values["is_min_delay_path"] = 1.0 if abs(values["path_delay_ms_sum"] - min_delay_value) <= 1e-9 else 0.0
                    values["rank_by_risk"] = float(rank_risk[path_pos])
                    values["rank_by_cong_prob"] = float(rank_cong[path_pos])
                    values["rank_by_delay"] = float(rank_delay[path_pos])

                    for feature_idx, feature_name in enumerate(FEATURE_NAMES):
                        features[sample_pos, pair_idx, path_pos, feature_idx] = float(values[feature_name])
                    valid_mask[sample_pos, pair_idx, path_pos] = is_valid

                    row = {
                        "sample_idx": int(test_indices[sample_pos]),
                        "time_t": int(time_t),
                        "target_time": int(target_times[sample_pos]),
                        "src_gateway": int(src_gw),
                        "dst_gateway": int(dst_gw),
                        "src_sat": int(src_sat),
                        "dst_sat": int(dst_sat),
                        "path_id": int(path_record["path_id"]),
                        "hop_count": int(path_record["hop_count"]),
                        "od_demand_mbps": f"{demand_mbps:.9g}",
                        "is_valid": bool(is_valid),
                    }
                    for feature_name in FEATURE_NAMES:
                        if feature_name in {"od_demand_mbps", "hop_count"}:
                            continue
                        row[feature_name] = f"{values[feature_name]:.9g}"
                    rows_to_write.append(row)

                strategy_indices = {
                    "shortest_path": 0,
                    "min_risk_path": min_risk_idx,
                    "min_cong_prob_path": min_cong_idx,
                    "min_delay_path": int(np.argmin(delay_vals)),
                }
                for strategy, idx in strategy_indices.items():
                    update_strategy_accumulators(strategy_acc, strategy, group_rows[idx][2])

            writer.writerows(rows_to_write)
    finally:
        csv_file.close()

    np.savez_compressed(
        npz_path,
        features=features,
        feature_names=np.asarray(FEATURE_NAMES),
        gw_pairs=gw_pairs,
        time_t=test_times.astype(np.int32),
        target_time=target_times.astype(np.int32),
        src_sat=src_sat_array,
        dst_sat=dst_sat_array,
        valid_mask=valid_mask,
    )
    write_feature_name_json(names_path)

    strategy_rows = []
    for strategy, acc in strategy_acc.items():
        row = {"strategy": strategy}
        count = max(acc["count"], 1.0)
        for key, value in acc.items():
            if key != "count":
                row[key] = value / count
        strategy_rows.append(row)
    strategy_df = pd.DataFrame(strategy_rows)
    strategy_df.to_csv(strategy_path, index=False)

    figure_paths = save_plots(csv_path, results_dir, strategy_path, debug_suffix)

    stats = {
        "alignment_assumption": (
            "mc_dropout_predictions_test[i] corresponds to samples test_indices[i] and "
            "time_t = samples['times'][test_indices[i]]. Gateway access and current link_state use time_t; "
            "prediction arrays represent risk around target_time = time_t + 1."
        ),
        "same_satellite_gateway_pair_policy": (
            "If src_gateway and dst_gateway access the same satellite at time_t, the OD flow does not traverse ISLs. "
            "This script emits K local zero-hop paths with empty edge_path and zero path risk/current-link features."
        ),
        "debug_max_samples": args.max_samples,
        "k": int(args.k),
        "n_test": int(n_samples),
        "gw_pair_count_per_sample": int(pair_count),
        "feature_count": int(len(FEATURE_NAMES)),
        "features_shape": list(features.shape),
        "gw_pairs_shape": list(gw_pairs.shape),
        "csv_rows": int(n_samples * pair_count * args.k),
        "valid_mask_all_true": bool(valid_mask.all()),
        "has_nan": bool(np.isnan(features).any()),
        "has_inf": bool(np.isinf(features).any()),
        "shortest_min_risk_different_ratio": float(shortest_min_risk_diff / total_groups),
        "min_risk_mean_risk_score_reduction_vs_shortest": float(min_risk_risk_delta_sum / total_groups),
        "min_risk_mean_hop_count_increase_vs_shortest": float(min_risk_hop_delta_sum / total_groups),
        "output_npz": str(npz_path),
        "output_csv": str(csv_path),
        "feature_names_json": str(names_path),
        "strategy_comparison_csv": str(strategy_path),
        "figure_paths": [str(path) for path in figure_paths],
    }
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, ensure_ascii=False)

    print("\nPath risk feature build complete:")
    print(f"- npz: {npz_path}")
    print(f"- csv: {csv_path}")
    print(f"- feature names: {names_path}")
    print(f"- stats: {stats_path}")
    print(f"- strategy comparison: {strategy_path}")
    print(f"- features shape: {features.shape}")
    print(f"- gw_pairs shape: {gw_pairs.shape}")
    print(f"- csv rows: {stats['csv_rows']}")
    print(f"- shortest vs min-risk different ratio: {stats['shortest_min_risk_different_ratio']:.6f}")
    print(
        "- min-risk mean risk reduction / hop increase vs shortest: "
        f"{stats['min_risk_mean_risk_score_reduction_vs_shortest']:.6f} / "
        f"{stats['min_risk_mean_hop_count_increase_vs_shortest']:.6f}"
    )
    print("- figures:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
