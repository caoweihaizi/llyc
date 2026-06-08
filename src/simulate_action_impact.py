import argparse
import json
import pickle
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

IMPACT_FEATURE_DESCRIPTIONS = {
    "hop_count": "候选路径跳数；同卫星接入 OD 为 0",
    "od_demand_mbps": "当前 OD demand，按 Abilene raw * 100 * 8 / 300 / 1e6 转为 Mbps",
    "is_zero_hop": "src_sat == dst_sat 时为 1，此时不经过 ISL",
    "is_valid": "动作是否有效",
    "pre_mlu": "动作前全网最大链路利用率",
    "pre_congestion_count": "动作前全网拥塞链路数量",
    "pre_congestion_ratio": "动作前全网拥塞链路比例",
    "pre_avg_utilization": "动作前全网平均链路利用率",
    "pre_total_load_mbps": "动作前全网链路负载总和",
    "path_pre_load_sum": "动作前候选路径链路负载总和",
    "path_pre_util_max": "动作前候选路径最大链路利用率",
    "path_pre_util_mean": "动作前候选路径平均链路利用率",
    "path_pre_congestion_count": "动作前候选路径拥塞链路数量",
    "path_delay_ms_sum": "候选路径当前 delay_ms 之和",
    "added_load_mbps": "本动作向路径每条 ISL 增加的 Mbps；zero-hop 为 0",
    "affected_edge_count": "受该动作影响的 ISL 数量",
    "added_load_edge_sum": "全网链路负载增量总和，等于 od_demand_mbps * affected_edge_count",
    "path_post_load_sum": "动作后候选路径链路负载总和",
    "path_post_util_max": "动作后候选路径最大链路利用率",
    "path_post_util_mean": "动作后候选路径平均链路利用率",
    "path_post_congestion_count": "动作后候选路径拥塞链路数量",
    "path_new_congestion_count": "路径上由非拥塞跨越为拥塞的链路数量",
    "path_congestion_crossing_ratio": "路径新增拥塞链路比例",
    "post_mlu": "动作后全网最大链路利用率",
    "post_congestion_count": "动作后全网拥塞链路数量",
    "post_congestion_ratio": "动作后全网拥塞链路比例",
    "post_avg_utilization": "动作后全网平均链路利用率",
    "post_total_load_mbps": "动作后全网链路负载总和",
    "delta_mlu": "post_mlu - pre_mlu",
    "delta_congestion_count": "post_congestion_count - pre_congestion_count",
    "delta_congestion_ratio": "post_congestion_ratio - pre_congestion_ratio",
    "delta_avg_utilization": "post_avg_utilization - pre_avg_utilization",
    "delta_total_load_mbps": "post_total_load_mbps - pre_total_load_mbps",
    "path_pred_util_max": "第9阶段路径预测利用率最大值",
    "path_pred_util_mean": "第9阶段路径预测利用率均值",
    "path_cong_prob_max": "第9阶段路径拥塞概率最大值",
    "path_cong_prob_mean": "第9阶段路径拥塞概率均值",
    "path_risk_score_max": "第9阶段路径 risk_score 最大值",
    "path_risk_score_mean": "第9阶段路径 risk_score 均值",
    "path_risk_score_sum": "第9阶段路径 risk_score 总和",
    "path_congestion_risk_score_max": "第9阶段路径 congestion_risk_score 最大值",
    "path_congestion_risk_score_mean": "第9阶段路径 congestion_risk_score 均值",
    "path_util_uncertainty_mean": "第9阶段路径 utilization 不确定性均值",
    "path_util_uncertainty_max": "第9阶段路径 utilization 不确定性最大值",
    "action_cost_delay_term": "action_cost 中的 delay 分项",
    "action_cost_post_mlu_term": "action_cost 中的 post_mlu 分项，100 * post_mlu",
    "action_cost_delta_congestion_term": "action_cost 中的拥塞增量分项，10 * delta_congestion_count",
    "action_cost_risk_term": "action_cost 中的风险分项，10 * path_risk_score_max",
    "action_cost": "启发式动作代价，不是强化学习奖励函数",
}

IMPACT_FEATURE_NAMES = list(IMPACT_FEATURE_DESCRIPTIONS.keys())

STRATEGIES = [
    "shortest_path",
    "min_delay_path",
    "min_risk_path",
    "min_cong_prob_path",
    "min_action_cost_path",
]

STRATEGY_METRICS = [
    "hop_count",
    "path_delay_ms_sum",
    "od_demand_mbps",
    "pre_mlu",
    "post_mlu",
    "delta_mlu",
    "pre_congestion_count",
    "post_congestion_count",
    "delta_congestion_count",
    "path_post_util_max",
    "path_new_congestion_count",
    "path_risk_score_max",
    "path_cong_prob_max",
    "action_cost",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate rule-based single-OD routing action impacts.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--congestion_threshold", type=float, default=0.8)
    parser.add_argument("--chunk_size", type=int, default=500_000)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--path_risk_npz",
        type=Path,
        default=None,
        help="Defaults to path_risk_features_test_topo144_k{k}[debug].npz.",
    )
    parser.add_argument(
        "--path_risk_names",
        type=Path,
        default=None,
        help="Defaults to path_risk_feature_names_topo144_k{k}[debug].json.",
    )
    parser.add_argument(
        "--candidate_paths",
        type=Path,
        default=None,
        help="Defaults to candidate_paths_topo144_k{k}.pkl.",
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
    return parser.parse_args()


def suffix(max_samples: int | None) -> str:
    return "" if max_samples is None else f"_debug{max_samples}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {path}")


def load_candidate_paths(path: Path) -> dict[tuple[int, int, int], list[int]]:
    require_file(path)
    with path.open("rb") as file:
        records = pickle.load(file)
    out: dict[tuple[int, int, int], list[int]] = {}
    for record in records:
        if bool(record.get("is_valid", False)):
            out[(int(record["src_sat"]), int(record["dst_sat"]), int(record["path_id"]))] = [
                int(edge_id) for edge_id in record["edge_path"]
            ]
    return out


def load_current_link_state(
    link_state_path: Path,
    test_times: np.ndarray,
    edge_count: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    require_file(link_state_path)
    time_to_index = {int(time): idx for idx, time in enumerate(test_times.tolist())}
    needed_times = set(time_to_index)
    n_samples = len(test_times)
    load = np.full((n_samples, edge_count), np.nan, dtype=np.float32)
    capacity = np.full((n_samples, edge_count), np.nan, dtype=np.float32)
    util = np.full((n_samples, edge_count), np.nan, dtype=np.float32)
    cong = np.full((n_samples, edge_count), np.nan, dtype=np.float32)
    delay = np.full((n_samples, edge_count), np.nan, dtype=np.float32)

    usecols = ["time", "edge_id", "load_mbps", "capacity_mbps", "utilization", "congestion_label", "delay_ms"]
    print("Loading current link_state rows for action impact...")
    for chunk in tqdm(
        pd.read_csv(link_state_path, usecols=usecols, chunksize=chunk_size),
        desc="Filtering link_state",
    ):
        filtered = chunk[chunk["time"].isin(needed_times)]
        if filtered.empty:
            continue
        for row in filtered.itertuples(index=False):
            idx = time_to_index[int(row.time)]
            edge_id = int(row.edge_id)
            load[idx, edge_id] = float(row.load_mbps)
            capacity[idx, edge_id] = float(row.capacity_mbps)
            util[idx, edge_id] = float(row.utilization)
            cong[idx, edge_id] = float(row.congestion_label)
            delay[idx, edge_id] = float(row.delay_ms)

    if any(np.isnan(arr).any() for arr in [load, capacity, util, cong, delay]):
        raise ValueError("Missing link_state values after chunk filtering.")
    return load, capacity, util, cong, delay


def path_edges(candidate_paths: dict[tuple[int, int, int], list[int]], src_sat: int, dst_sat: int, path_id: int) -> list[int]:
    if src_sat == dst_sat:
        return []
    key = (int(src_sat), int(dst_sat), int(path_id))
    if key not in candidate_paths:
        raise KeyError(f"No candidate path for {key}")
    return candidate_paths[key]


def choose_index(rows: np.ndarray, metric_idx: int, hop_idx: int, path_ids: np.ndarray) -> int:
    order = np.lexsort((path_ids, rows[:, hop_idx], rows[:, metric_idx]))
    return int(order[0])


def init_accumulators() -> dict[str, dict[str, float]]:
    return {strategy: {"count": 0.0, **{metric: 0.0 for metric in STRATEGY_METRICS}} for strategy in STRATEGIES}


def update_acc(accumulators: dict[str, dict[str, float]], strategy: str, values: np.ndarray, feature_index: dict[str, int]) -> None:
    acc = accumulators[strategy]
    acc["count"] += 1.0
    for metric in STRATEGY_METRICS:
        acc[metric] += float(values[feature_index[metric]])


def make_strategy_frame(accumulators: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        acc = accumulators[strategy]
        count = max(acc["count"], 1.0)
        row = {"strategy": strategy}
        for metric in STRATEGY_METRICS:
            row[metric] = acc[metric] / count
        rows.append(row)
    return pd.DataFrame(rows)


def make_delta_frame(strategy_df: pd.DataFrame) -> pd.DataFrame:
    base = strategy_df[strategy_df["strategy"] == "shortest_path"].iloc[0]
    rows = []
    mapping = [
        ("min_risk_path", "min_risk_path"),
        ("min_cong_prob_path", "min_cong_prob_path"),
        ("min_action_cost_path", "min_action_cost_path"),
    ]
    for strategy, label in mapping:
        row = strategy_df[strategy_df["strategy"] == strategy].iloc[0]
        rows.append(
            {
                "strategy_vs_shortest": label,
                "delta_hop_count": row["hop_count"] - base["hop_count"],
                "delta_delay_ms": row["path_delay_ms_sum"] - base["path_delay_ms_sum"],
                "delta_post_mlu": row["post_mlu"] - base["post_mlu"],
                "delta_delta_mlu": row["delta_mlu"] - base["delta_mlu"],
                "delta_post_congestion_count": row["post_congestion_count"] - base["post_congestion_count"],
                "delta_delta_congestion_count": row["delta_congestion_count"] - base["delta_congestion_count"],
                "delta_action_cost": row["action_cost"] - base["action_cost"],
                "delta_risk_score": row["path_risk_score_max"] - base["path_risk_score_max"],
                "delta_cong_prob": row["path_cong_prob_max"] - base["path_cong_prob_max"],
            }
        )
    return pd.DataFrame(rows)


def save_plots(
    strategy_df: pd.DataFrame,
    shortest_costs: np.ndarray,
    min_risk_costs: np.ndarray,
    min_action_path_ids: np.ndarray,
    results_dir: Path,
    debug_suffix: str,
) -> list[Path]:
    paths = [
        results_dir / f"action_impact_strategy_post_mlu{debug_suffix}.png",
        results_dir / f"action_impact_strategy_delta_mlu{debug_suffix}.png",
        results_dir / f"action_impact_strategy_congestion_delta{debug_suffix}.png",
        results_dir / f"action_impact_strategy_cost{debug_suffix}.png",
        results_dir / f"action_impact_shortest_vs_min_risk_scatter{debug_suffix}.png",
        results_dir / f"action_impact_selected_path_id_distribution{debug_suffix}.png",
    ]

    for metric, path, ylabel in [
        ("post_mlu", paths[0], "mean post_mlu"),
        ("delta_mlu", paths[1], "mean delta_mlu"),
        ("delta_congestion_count", paths[2], "mean delta_congestion_count"),
        ("action_cost", paths[3], "mean action_cost"),
    ]:
        plt.figure(figsize=(8, 4.5))
        plt.bar(strategy_df["strategy"], strategy_df[metric])
        plt.ylabel(ylabel)
        plt.xticks(rotation=18, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    rng = np.random.default_rng(42)
    sample_size = min(len(shortest_costs), 100_000)
    sample_idx = rng.choice(len(shortest_costs), size=sample_size, replace=False)
    plt.figure(figsize=(6, 6))
    plt.scatter(shortest_costs[sample_idx], min_risk_costs[sample_idx], s=2, alpha=0.25)
    lim_min = min(float(shortest_costs[sample_idx].min()), float(min_risk_costs[sample_idx].min()))
    lim_max = max(float(shortest_costs[sample_idx].max()), float(min_risk_costs[sample_idx].max()))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], color="red", linewidth=1.0)
    plt.xlabel("shortest_path action_cost")
    plt.ylabel("min_risk_path action_cost")
    plt.title("Shortest vs min-risk action cost")
    plt.tight_layout()
    plt.savefig(paths[4], dpi=200)
    plt.close()

    counts = np.bincount(min_action_path_ids.astype(np.int32), minlength=int(min_action_path_ids.max()) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.bar(np.arange(len(counts)), counts)
    plt.xlabel("path_id selected by min_action_cost")
    plt.ylabel("selection count")
    plt.tight_layout()
    plt.savefig(paths[5], dpi=200)
    plt.close()

    return paths


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError(f"--k must be positive, got {args.k}")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError(f"--max_samples must be positive, got {args.max_samples}")

    processed_dir = ensure_dir(args.processed_dir)
    results_dir = ensure_dir(args.results_dir)
    debug_suffix = suffix(args.max_samples)

    path_risk_npz = args.path_risk_npz or processed_dir / f"path_risk_features_test_topo144_k{args.k}{debug_suffix}.npz"
    path_risk_names = args.path_risk_names or processed_dir / f"path_risk_feature_names_topo144_k{args.k}{debug_suffix}.json"
    candidate_path = args.candidate_paths or processed_dir / f"candidate_paths_topo144_k{args.k}.pkl"
    output_npz = processed_dir / f"action_impact_features_test_topo144_k{args.k}{debug_suffix}.npz"
    names_json = processed_dir / f"action_impact_feature_names_topo144_k{args.k}{debug_suffix}.json"
    stats_json = results_dir / f"action_impact_stats_topo144_k{args.k}{debug_suffix}.json"
    strategy_csv = results_dir / f"action_impact_strategy_comparison_topo144_k{args.k}{debug_suffix}.csv"
    delta_csv = results_dir / f"action_impact_strategy_delta_topo144_k{args.k}{debug_suffix}.csv"

    for path in [path_risk_npz, path_risk_names, candidate_path, args.od_path]:
        require_file(path)

    risk_npz = np.load(path_risk_npz, allow_pickle=False)
    risk_features = risk_npz["features"]
    risk_feature_names = [str(name) for name in risk_npz["feature_names"]]
    risk_idx = {name: idx for idx, name in enumerate(risk_feature_names)}
    gw_pairs = risk_npz["gw_pairs"]
    time_t = risk_npz["time_t"]
    target_time = risk_npz["target_time"]
    src_sat_arr = risk_npz["src_sat"]
    dst_sat_arr = risk_npz["dst_sat"]
    valid_mask = risk_npz["valid_mask"]

    if args.max_samples is not None:
        risk_features = risk_features[: args.max_samples]
        gw_pairs = gw_pairs
        time_t = time_t[: args.max_samples]
        target_time = target_time[: args.max_samples]
        src_sat_arr = src_sat_arr[: args.max_samples]
        dst_sat_arr = dst_sat_arr[: args.max_samples]
        valid_mask = valid_mask[: args.max_samples]

    n_samples, pair_count, k, _ = risk_features.shape
    if k != args.k:
        raise ValueError(f"Risk feature K mismatch: {k} != {args.k}")

    candidate_paths = load_candidate_paths(candidate_path)
    edge_count = 288
    pre_load, pre_capacity, pre_util, pre_cong_label, pre_delay = load_current_link_state(
        args.link_state_path,
        time_t,
        edge_count=edge_count,
        chunk_size=args.chunk_size,
    )
    od_matrices = np.load(args.od_path, mmap_mode="r")

    impact_idx = {name: idx for idx, name in enumerate(IMPACT_FEATURE_NAMES)}
    impact = np.zeros((n_samples, pair_count, args.k, len(IMPACT_FEATURE_NAMES)), dtype=np.float32)

    accumulators = init_accumulators()
    group_count = n_samples * pair_count
    shortest_costs = np.zeros(group_count, dtype=np.float32)
    min_risk_costs = np.zeros(group_count, dtype=np.float32)
    min_action_path_ids = np.zeros(group_count, dtype=np.int16)

    min_risk_diff = 0
    min_action_diff = 0
    min_risk_action_cost_improvement_sum = 0.0
    min_risk_delta_mlu_improvement_sum = 0.0
    min_risk_delta_congestion_improvement_sum = 0.0
    min_action_cost_improvement_sum = 0.0
    zero_hop_action_count = 0

    print("Simulating single-OD action impacts...")
    for sample_pos in tqdm(range(n_samples), desc="Action impact"):
        load_row = pre_load[sample_pos]
        cap_row = pre_capacity[sample_pos]
        util_row = pre_util[sample_pos]
        cong_bool = pre_cong_label[sample_pos] > 0.5
        delay_row = pre_delay[sample_pos]

        pre_mlu = float(util_row.max())
        pre_cong_count = float(cong_bool.sum())
        pre_cong_ratio = float(pre_cong_count / edge_count)
        pre_avg_util = float(util_row.mean())
        pre_total_load = float(load_row.sum())

        od_mbps = od_matrices[int(time_t[sample_pos])] * ABILENE_RAW_TO_MBPS

        for pair_idx, (src_gw, dst_gw) in enumerate(gw_pairs):
            src_sat = int(src_sat_arr[sample_pos, pair_idx])
            dst_sat = int(dst_sat_arr[sample_pos, pair_idx])
            demand_mbps = float(od_mbps[int(src_gw), int(dst_gw)])
            group_values = np.zeros((args.k, len(IMPACT_FEATURE_NAMES)), dtype=np.float32)
            path_ids = np.arange(args.k, dtype=np.int16)

            for path_id in range(args.k):
                edge_path = path_edges(candidate_paths, src_sat, dst_sat, path_id)
                affected = len(edge_path)
                is_zero = affected == 0
                if is_zero:
                    zero_hop_action_count += 1

                values = group_values[path_id]
                values[impact_idx["hop_count"]] = float(affected)
                values[impact_idx["od_demand_mbps"]] = demand_mbps
                values[impact_idx["is_zero_hop"]] = 1.0 if is_zero else 0.0
                values[impact_idx["is_valid"]] = 1.0 if bool(valid_mask[sample_pos, pair_idx, path_id]) else 0.0
                values[impact_idx["pre_mlu"]] = pre_mlu
                values[impact_idx["pre_congestion_count"]] = pre_cong_count
                values[impact_idx["pre_congestion_ratio"]] = pre_cong_ratio
                values[impact_idx["pre_avg_utilization"]] = pre_avg_util
                values[impact_idx["pre_total_load_mbps"]] = pre_total_load

                if is_zero:
                    path_pre_load_sum = 0.0
                    path_pre_util_max = 0.0
                    path_pre_util_mean = 0.0
                    path_pre_cong_count = 0.0
                    path_delay_sum = 0.0
                    added_load = 0.0
                    added_load_edge_sum = 0.0
                    path_post_load_sum = 0.0
                    path_post_util_max = 0.0
                    path_post_util_mean = 0.0
                    path_post_cong_count = 0.0
                    path_new_cong_count = 0.0
                    crossing_ratio = 0.0
                    post_mlu = pre_mlu
                    post_cong_count = pre_cong_count
                    post_avg_util = pre_avg_util
                    post_total_load = pre_total_load
                else:
                    edges = np.asarray(edge_path, dtype=np.int32)
                    pre_path_load = load_row[edges]
                    pre_path_util = util_row[edges]
                    pre_path_cong = cong_bool[edges]
                    path_pre_load_sum = float(pre_path_load.sum())
                    path_pre_util_max = float(pre_path_util.max())
                    path_pre_util_mean = float(pre_path_util.mean())
                    path_pre_cong_count = float(pre_path_cong.sum())
                    path_delay_sum = float(delay_row[edges].sum())

                    added_load = demand_mbps
                    added_load_edge_sum = demand_mbps * affected
                    post_path_load = pre_path_load + demand_mbps
                    post_path_util = post_path_load / cap_row[edges]
                    post_path_cong = post_path_util > args.congestion_threshold
                    new_cong = (~pre_path_cong) & post_path_cong

                    path_post_load_sum = float(post_path_load.sum())
                    path_post_util_max = float(post_path_util.max())
                    path_post_util_mean = float(post_path_util.mean())
                    path_post_cong_count = float(post_path_cong.sum())
                    path_new_cong_count = float(new_cong.sum())
                    crossing_ratio = float(path_new_cong_count / affected)

                    post_mlu = float(max(pre_mlu, path_post_util_max))
                    post_cong_count = pre_cong_count + path_new_cong_count
                    post_total_load = pre_total_load + added_load_edge_sum
                    post_avg_util = float((util_row.sum() + np.sum(demand_mbps / cap_row[edges])) / edge_count)

                post_cong_ratio = float(post_cong_count / edge_count)
                delta_mlu = post_mlu - pre_mlu
                delta_cong_count = post_cong_count - pre_cong_count
                delta_cong_ratio = post_cong_ratio - pre_cong_ratio
                delta_avg_util = post_avg_util - pre_avg_util
                delta_total_load = post_total_load - pre_total_load

                values[impact_idx["path_pre_load_sum"]] = path_pre_load_sum
                values[impact_idx["path_pre_util_max"]] = path_pre_util_max
                values[impact_idx["path_pre_util_mean"]] = path_pre_util_mean
                values[impact_idx["path_pre_congestion_count"]] = path_pre_cong_count
                values[impact_idx["path_delay_ms_sum"]] = path_delay_sum
                values[impact_idx["added_load_mbps"]] = added_load
                values[impact_idx["affected_edge_count"]] = float(affected)
                values[impact_idx["added_load_edge_sum"]] = added_load_edge_sum
                values[impact_idx["path_post_load_sum"]] = path_post_load_sum
                values[impact_idx["path_post_util_max"]] = path_post_util_max
                values[impact_idx["path_post_util_mean"]] = path_post_util_mean
                values[impact_idx["path_post_congestion_count"]] = path_post_cong_count
                values[impact_idx["path_new_congestion_count"]] = path_new_cong_count
                values[impact_idx["path_congestion_crossing_ratio"]] = crossing_ratio
                values[impact_idx["post_mlu"]] = post_mlu
                values[impact_idx["post_congestion_count"]] = post_cong_count
                values[impact_idx["post_congestion_ratio"]] = post_cong_ratio
                values[impact_idx["post_avg_utilization"]] = post_avg_util
                values[impact_idx["post_total_load_mbps"]] = post_total_load
                values[impact_idx["delta_mlu"]] = delta_mlu
                values[impact_idx["delta_congestion_count"]] = delta_cong_count
                values[impact_idx["delta_congestion_ratio"]] = delta_cong_ratio
                values[impact_idx["delta_avg_utilization"]] = delta_avg_util
                values[impact_idx["delta_total_load_mbps"]] = delta_total_load

                for dst_name, src_name in [
                    ("path_pred_util_max", "path_pred_util_max"),
                    ("path_pred_util_mean", "path_pred_util_mean"),
                    ("path_cong_prob_max", "path_cong_prob_max"),
                    ("path_cong_prob_mean", "path_cong_prob_mean"),
                    ("path_risk_score_max", "path_risk_score_max"),
                    ("path_risk_score_mean", "path_risk_score_mean"),
                    ("path_risk_score_sum", "path_risk_score_sum"),
                    ("path_congestion_risk_score_max", "path_congestion_risk_score_max"),
                    ("path_congestion_risk_score_mean", "path_congestion_risk_score_mean"),
                    ("path_util_uncertainty_mean", "path_util_uncertainty_mean"),
                    ("path_util_uncertainty_max", "path_util_uncertainty_max"),
                ]:
                    values[impact_idx[dst_name]] = risk_features[sample_pos, pair_idx, path_id, risk_idx[src_name]]

                delay_term = path_delay_sum
                post_mlu_term = 100.0 * post_mlu
                delta_cong_term = 10.0 * delta_cong_count
                risk_term = 10.0 * values[impact_idx["path_risk_score_max"]]
                values[impact_idx["action_cost_delay_term"]] = delay_term
                values[impact_idx["action_cost_post_mlu_term"]] = post_mlu_term
                values[impact_idx["action_cost_delta_congestion_term"]] = delta_cong_term
                values[impact_idx["action_cost_risk_term"]] = risk_term
                values[impact_idx["action_cost"]] = delay_term + post_mlu_term + delta_cong_term + risk_term

            impact[sample_pos, pair_idx] = group_values

            idx_shortest = 0
            idx_min_delay = choose_index(group_values, impact_idx["path_delay_ms_sum"], impact_idx["hop_count"], path_ids)
            idx_min_risk = choose_index(group_values, impact_idx["path_risk_score_max"], impact_idx["hop_count"], path_ids)
            idx_min_cong = choose_index(group_values, impact_idx["path_cong_prob_max"], impact_idx["hop_count"], path_ids)
            idx_min_cost = choose_index(group_values, impact_idx["action_cost"], impact_idx["hop_count"], path_ids)

            group_flat_idx = sample_pos * pair_count + pair_idx
            shortest_costs[group_flat_idx] = group_values[idx_shortest, impact_idx["action_cost"]]
            min_risk_costs[group_flat_idx] = group_values[idx_min_risk, impact_idx["action_cost"]]
            min_action_path_ids[group_flat_idx] = idx_min_cost

            if idx_min_risk != idx_shortest:
                min_risk_diff += 1
            if idx_min_cost != idx_shortest:
                min_action_diff += 1

            min_risk_action_cost_improvement_sum += float(
                group_values[idx_shortest, impact_idx["action_cost"]] - group_values[idx_min_risk, impact_idx["action_cost"]]
            )
            min_risk_delta_mlu_improvement_sum += float(
                group_values[idx_shortest, impact_idx["delta_mlu"]] - group_values[idx_min_risk, impact_idx["delta_mlu"]]
            )
            min_risk_delta_congestion_improvement_sum += float(
                group_values[idx_shortest, impact_idx["delta_congestion_count"]]
                - group_values[idx_min_risk, impact_idx["delta_congestion_count"]]
            )
            min_action_cost_improvement_sum += float(
                group_values[idx_shortest, impact_idx["action_cost"]] - group_values[idx_min_cost, impact_idx["action_cost"]]
            )

            for strategy, idx in [
                ("shortest_path", idx_shortest),
                ("min_delay_path", idx_min_delay),
                ("min_risk_path", idx_min_risk),
                ("min_cong_prob_path", idx_min_cong),
                ("min_action_cost_path", idx_min_cost),
            ]:
                update_acc(accumulators, strategy, group_values[idx], impact_idx)

    np.savez_compressed(
        output_npz,
        impact_features=impact,
        impact_feature_names=np.asarray(IMPACT_FEATURE_NAMES),
        gw_pairs=gw_pairs.astype(np.int16),
        time_t=time_t.astype(np.int32),
        target_time=target_time.astype(np.int32),
        src_sat=src_sat_arr.astype(np.int16),
        dst_sat=dst_sat_arr.astype(np.int16),
        valid_mask=valid_mask.astype(bool),
    )

    with names_json.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "impact_feature_names": IMPACT_FEATURE_NAMES,
                "impact_feature_descriptions": IMPACT_FEATURE_DESCRIPTIONS,
                "action_cost_formula": (
                    "action_cost = 1.0 * path_delay_ms_sum + 100.0 * post_mlu + "
                    "10.0 * delta_congestion_count + 10.0 * path_risk_score_max"
                ),
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    strategy_df = make_strategy_frame(accumulators)
    strategy_df.to_csv(strategy_csv, index=False)
    delta_df = make_delta_frame(strategy_df)
    delta_df.to_csv(delta_csv, index=False)
    figure_paths = save_plots(strategy_df, shortest_costs, min_risk_costs, min_action_path_ids, results_dir, debug_suffix)

    stats = {
        "assumptions": [
            "This is a single-OD action impact proxy.",
            "The base link_state is generated by shortest-delay Dijkstra and is treated as the current network state.",
            "The simulation adds one OD demand to the selected candidate path and does not remove existing traffic or reroute the full network.",
            "Therefore metrics are local/proxy action impacts, not a complete global post-routing state.",
            "If src_sat == dst_sat, edge_path is empty, added_load is zero, and post state equals pre state. Future RL action masks should keep only path_id=0 for such equivalent zero-hop actions.",
        ],
        "congestion_threshold": float(args.congestion_threshold),
        "debug_max_samples": args.max_samples,
        "impact_features_shape": list(impact.shape),
        "n_test": int(n_samples),
        "gw_pair_count": int(pair_count),
        "k": int(args.k),
        "f_impact": int(len(IMPACT_FEATURE_NAMES)),
        "has_nan": bool(np.isnan(impact).any()),
        "has_inf": bool(np.isinf(impact).any()),
        "zero_hop_action_count": int(zero_hop_action_count),
        "zero_hop_action_ratio": float(zero_hop_action_count / impact[..., 0].size),
        "min_risk_vs_shortest_different_ratio": float(min_risk_diff / group_count),
        "min_action_cost_vs_shortest_different_ratio": float(min_action_diff / group_count),
        "min_risk_mean_delta_mlu_improvement_vs_shortest": float(min_risk_delta_mlu_improvement_sum / group_count),
        "min_risk_mean_delta_congestion_count_improvement_vs_shortest": float(
            min_risk_delta_congestion_improvement_sum / group_count
        ),
        "min_risk_mean_action_cost_improvement_vs_shortest": float(min_risk_action_cost_improvement_sum / group_count),
        "min_action_cost_mean_action_cost_improvement_vs_shortest": float(min_action_cost_improvement_sum / group_count),
        "output_npz": str(output_npz),
        "feature_names_json": str(names_json),
        "strategy_comparison_csv": str(strategy_csv),
        "strategy_delta_csv": str(delta_csv),
        "selected_records_csv": None,
        "selected_records_note": "Not written for the full run to avoid another multi-million-row CSV; selected strategy summaries are stored in strategy comparison and delta CSVs.",
        "figure_paths": [str(path) for path in figure_paths],
    }
    with stats_json.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, ensure_ascii=False)

    print("\nAction impact simulation complete:")
    print(f"- npz: {output_npz}")
    print(f"- feature names: {names_json}")
    print(f"- stats: {stats_json}")
    print(f"- strategy comparison: {strategy_csv}")
    print(f"- strategy delta: {delta_csv}")
    print(f"- impact_features shape: {impact.shape}")
    print(f"- zero-hop actions: {zero_hop_action_count} ({stats['zero_hop_action_ratio']:.6f})")
    print(f"- min-risk vs shortest different ratio: {stats['min_risk_vs_shortest_different_ratio']:.6f}")
    print(f"- min-action-cost vs shortest different ratio: {stats['min_action_cost_vs_shortest_different_ratio']:.6f}")
    print(f"- min-risk action_cost improvement vs shortest: {stats['min_risk_mean_action_cost_improvement_vs_shortest']:.6f}")
    print(f"- min-action-cost improvement vs shortest: {stats['min_action_cost_mean_action_cost_improvement_vs_shortest']:.6f}")
    print("- figures:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
