import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

ABILENE_RAW_TO_MBPS = 100.0 * 8.0 / 300.0 / 1e6
REQUIRED_STRATEGIES = {
    "shortest_path",
    "min_delay_path",
    "min_risk_path",
    "min_cong_prob_path",
    "min_action_cost_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check rule-based routing action impact outputs.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--congestion_threshold", type=float, default=0.8)
    parser.add_argument("--sample_checks", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=500_000)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def suffix(max_samples: int | None) -> str:
    return "" if max_samples is None else f"_debug{max_samples}"


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


def path_edges(candidate_paths: dict[tuple[int, int, int], list[int]], src_sat: int, dst_sat: int, path_id: int) -> list[int]:
    if src_sat == dst_sat:
        return []
    key = (int(src_sat), int(dst_sat), int(path_id))
    if key not in candidate_paths:
        raise KeyError(f"No candidate path for {key}")
    return candidate_paths[key]


def load_link_state_for_times(link_state_path: Path, times: np.ndarray, edge_count: int, chunk_size: int):
    time_to_index = {int(time): idx for idx, time in enumerate(times.tolist())}
    needed = set(time_to_index)
    load = np.full((len(times), edge_count), np.nan, dtype=np.float32)
    capacity = np.full((len(times), edge_count), np.nan, dtype=np.float32)
    util = np.full((len(times), edge_count), np.nan, dtype=np.float32)
    cong = np.full((len(times), edge_count), np.nan, dtype=np.float32)

    usecols = ["time", "edge_id", "load_mbps", "capacity_mbps", "utilization", "congestion_label"]
    for chunk in tqdm(pd.read_csv(link_state_path, usecols=usecols, chunksize=chunk_size), desc="Loading sampled link_state"):
        filtered = chunk[chunk["time"].isin(needed)]
        if filtered.empty:
            continue
        for row in filtered.itertuples(index=False):
            idx = time_to_index[int(row.time)]
            edge_id = int(row.edge_id)
            load[idx, edge_id] = float(row.load_mbps)
            capacity[idx, edge_id] = float(row.capacity_mbps)
            util[idx, edge_id] = float(row.utilization)
            cong[idx, edge_id] = float(row.congestion_label)

    if any(np.isnan(arr).any() for arr in [load, capacity, util, cong]):
        raise ValueError("Missing sampled link_state values.")
    return load, capacity, util, cong, time_to_index


def recompute_action(
    edge_path: list[int],
    demand_mbps: float,
    load_row: np.ndarray,
    cap_row: np.ndarray,
    util_row: np.ndarray,
    cong_row: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    edge_count = len(load_row)
    pre_mlu = float(util_row.max())
    pre_cong_bool = cong_row > 0.5
    pre_cong_count = float(pre_cong_bool.sum())
    pre_total_load = float(load_row.sum())

    if len(edge_path) == 0:
        return {
            "path_post_util_max": 0.0,
            "post_mlu": pre_mlu,
            "delta_congestion_count": 0.0,
            "delta_total_load_mbps": 0.0,
            "post_total_load_mbps": pre_total_load,
        }

    edges = np.asarray(edge_path, dtype=np.int32)
    post_path_load = load_row[edges] + demand_mbps
    post_path_util = post_path_load / cap_row[edges]
    new_cong = (~pre_cong_bool[edges]) & (post_path_util > threshold)
    return {
        "path_post_util_max": float(post_path_util.max()),
        "post_mlu": float(max(pre_mlu, float(post_path_util.max()))),
        "delta_congestion_count": float(new_cong.sum()),
        "delta_total_load_mbps": float(demand_mbps * len(edge_path)),
        "post_total_load_mbps": float(pre_total_load + demand_mbps * len(edge_path)),
    }


def main() -> None:
    args = parse_args()
    debug_suffix = suffix(args.max_samples)
    processed_dir = args.processed_dir
    results_dir = args.results_dir

    impact_npz_path = processed_dir / f"action_impact_features_test_topo144_k{args.k}{debug_suffix}.npz"
    names_path = processed_dir / f"action_impact_feature_names_topo144_k{args.k}{debug_suffix}.json"
    stats_path = results_dir / f"action_impact_stats_topo144_k{args.k}{debug_suffix}.json"
    strategy_path = results_dir / f"action_impact_strategy_comparison_topo144_k{args.k}{debug_suffix}.csv"
    report_path = results_dir / f"action_impact_check_report_topo144_k{args.k}{debug_suffix}.json"
    candidate_path = processed_dir / f"candidate_paths_topo144_k{args.k}.pkl"
    link_state_path = processed_dir / "link_state_topo144_shortest_delay.csv"
    od_path = processed_dir / "od_matrices_full.npy"

    for path in [impact_npz_path, names_path, stats_path, strategy_path, candidate_path, link_state_path, od_path]:
        require_file(path)

    data = np.load(impact_npz_path, allow_pickle=False)
    impact = data["impact_features"]
    feature_names = [str(name) for name in data["impact_feature_names"]]
    gw_pairs = data["gw_pairs"]
    time_t = data["time_t"]
    valid_mask = data["valid_mask"]
    src_sat = data["src_sat"]
    dst_sat = data["dst_sat"]
    idx = {name: pos for pos, name in enumerate(feature_names)}

    with names_path.open("r", encoding="utf-8") as file:
        names_payload = json.load(file)
    expected_names = names_payload["impact_feature_names"]
    with stats_path.open("r", encoding="utf-8") as file:
        stats = json.load(file)

    strategy_df = pd.read_csv(strategy_path)
    strategy_set = set(strategy_df["strategy"].astype(str))

    shape_ok = (
        impact.ndim == 4
        and impact.shape[1] == 132
        and impact.shape[2] == args.k
        and impact.shape[3] == len(feature_names)
        and gw_pairs.shape == (132, 2)
        and valid_mask.shape == impact.shape[:3]
    )
    feature_names_complete = expected_names == feature_names
    valid_mask_all_true = bool(valid_mask.all())
    has_nan = bool(np.isnan(impact).any())
    has_inf = bool(np.isinf(impact).any())
    post_mlu_ok = bool(np.all(impact[..., idx["post_mlu"]] + 1e-7 >= impact[..., idx["pre_mlu"]]))
    post_load_ok = bool(
        np.all(impact[..., idx["post_total_load_mbps"]] + 1e-4 >= impact[..., idx["pre_total_load_mbps"]])
    )
    delta_load_expected = impact[..., idx["od_demand_mbps"]] * impact[..., idx["affected_edge_count"]]
    delta_load_ok = bool(np.allclose(impact[..., idx["delta_total_load_mbps"]], delta_load_expected, atol=2e-3))

    zero = impact[..., idx["is_zero_hop"]] > 0.5
    zero_checks_ok = bool(
        np.all(impact[..., idx["affected_edge_count"]][zero] == 0)
        and np.allclose(impact[..., idx["delta_mlu"]][zero], 0.0, atol=1e-8)
        and np.allclose(impact[..., idx["delta_congestion_count"]][zero], 0.0, atol=1e-8)
        and np.allclose(impact[..., idx["delta_total_load_mbps"]][zero], 0.0, atol=1e-8)
    )

    nonzero_positions = np.argwhere(~zero)
    sample_n = min(args.sample_checks, len(nonzero_positions))
    rng = np.random.default_rng(42)
    selected = nonzero_positions[rng.choice(len(nonzero_positions), size=sample_n, replace=False)]
    sampled_times = np.unique(time_t[selected[:, 0]]).astype(np.int32)
    load, capacity, util, cong, time_to_local = load_link_state_for_times(
        link_state_path,
        sampled_times,
        edge_count=impact.shape[3] and 288,
        chunk_size=args.chunk_size,
    )
    od_matrices = np.load(od_path, mmap_mode="r")
    candidate_paths = load_candidate_paths(candidate_path)

    max_abs_error = 0.0
    failures = []
    for sample_pos, pair_idx, path_id in selected:
        sample_pos = int(sample_pos)
        pair_idx = int(pair_idx)
        path_id = int(path_id)
        local_time_idx = time_to_local[int(time_t[sample_pos])]
        demand_mbps = float(od_matrices[int(time_t[sample_pos]), int(gw_pairs[pair_idx, 0]), int(gw_pairs[pair_idx, 1])] * ABILENE_RAW_TO_MBPS)
        edge_path = path_edges(candidate_paths, int(src_sat[sample_pos, pair_idx]), int(dst_sat[sample_pos, pair_idx]), path_id)
        expected = recompute_action(
            edge_path=edge_path,
            demand_mbps=demand_mbps,
            load_row=load[local_time_idx],
            cap_row=capacity[local_time_idx],
            util_row=util[local_time_idx],
            cong_row=cong[local_time_idx],
            threshold=args.congestion_threshold,
        )
        for name, expected_value in expected.items():
            actual = float(impact[sample_pos, pair_idx, path_id, idx[name]])
            err = abs(actual - expected_value)
            max_abs_error = max(max_abs_error, err)
            if err > 2e-3:
                failures.append(
                    {
                        "sample_pos": sample_pos,
                        "pair_idx": pair_idx,
                        "path_id": path_id,
                        "feature": name,
                        "actual": actual,
                        "expected": expected_value,
                        "abs_error": err,
                    }
                )
                break

    sample_recompute_passed = len(failures) == 0
    strategies_ok = REQUIRED_STRATEGIES.issubset(strategy_set)

    report = {
        "impact_npz": str(impact_npz_path),
        "stats_json": str(stats_path),
        "strategy_comparison_csv": str(strategy_path),
        "impact_features_shape": list(impact.shape),
        "shape_ok": shape_ok,
        "feature_names_complete": feature_names_complete,
        "valid_mask_all_true": valid_mask_all_true,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "post_mlu_ge_pre_mlu": post_mlu_ok,
        "post_total_load_ge_pre_total_load": post_load_ok,
        "delta_total_load_matches_demand_times_edges": delta_load_ok,
        "zero_hop_checks_ok": zero_checks_ok,
        "zero_hop_action_count": int(zero.sum()),
        "sample_recompute_checks": int(sample_n),
        "sample_recompute_passed": sample_recompute_passed,
        "sample_recompute_max_abs_error": float(max_abs_error),
        "sample_recompute_failures": failures[:5],
        "strategies_ok": strategies_ok,
        "strategies_found": sorted(strategy_set),
        "stats_has_nan": stats.get("has_nan"),
        "stats_has_inf": stats.get("has_inf"),
        "validation_passed": bool(
            shape_ok
            and feature_names_complete
            and valid_mask_all_true
            and not has_nan
            and not has_inf
            and post_mlu_ok
            and post_load_ok
            and delta_load_ok
            and zero_checks_ok
            and sample_recompute_passed
            and strategies_ok
        ),
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("Action impact check complete:")
    print(f"- report: {report_path}")
    print(f"- impact_features shape: {impact.shape}")
    print(f"- has NaN / Inf: {has_nan} / {has_inf}")
    print(f"- zero-hop checks ok: {zero_checks_ok}")
    print(f"- sample recompute passed: {sample_recompute_passed}")
    print(f"- max recompute abs error: {max_abs_error:.9g}")
    print(f"- strategies ok: {strategies_ok}")
    print(f"- validation passed: {report['validation_passed']}")

    if not report["validation_passed"]:
        raise SystemExit("Action impact validation failed. See report JSON.")


if __name__ == "__main__":
    main()
