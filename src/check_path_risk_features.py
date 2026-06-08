import argparse
import json
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

FEATURES_TO_RECHECK = [
    "path_delay_ms_sum",
    "path_delay_ms_mean",
    "path_current_util_max",
    "path_current_util_mean",
    "path_current_util_sum",
    "path_current_congestion_count",
    "path_current_congestion_ratio",
    "path_pred_util_max",
    "path_pred_util_mean",
    "path_pred_util_sum",
    "path_util_uncertainty_max",
    "path_util_uncertainty_mean",
    "path_cong_prob_max",
    "path_cong_prob_mean",
    "path_cong_prob_sum",
    "path_cong_uncertainty_max",
    "path_cong_uncertainty_mean",
    "path_risk_score_max",
    "path_risk_score_mean",
    "path_risk_score_sum",
    "path_congestion_risk_score_max",
    "path_congestion_risk_score_mean",
    "path_congestion_risk_score_sum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check path-level risk feature outputs.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--sample_checks", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=500_000)
    return parser.parse_args()


def suffix(max_samples: int | None) -> str:
    return "" if max_samples is None else f"_debug{max_samples}"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {path}")


def parse_list(value: object) -> list[int]:
    return [int(item) for item in literal_eval(str(value))]


def load_candidate_paths(summary_path: Path) -> dict[tuple[int, int, int], list[int]]:
    require_file(summary_path)
    out: dict[tuple[int, int, int], list[int]] = {}
    for row in pd.read_csv(summary_path).itertuples(index=False):
        out[(int(row.src_sat), int(row.dst_sat), int(row.path_id))] = parse_list(row.edge_path)
    return out


def load_sample_csv_rows(csv_path: Path, n: int) -> pd.DataFrame:
    require_file(csv_path)
    total_rows = sum(1 for _ in csv_path.open("r", encoding="utf-8")) - 1
    if total_rows <= 0:
        raise ValueError(f"CSV has no data rows: {csv_path}")
    n = min(n, total_rows)
    rng = np.random.default_rng(42)
    selected_positions = set(int(pos) for pos in rng.choice(total_rows, size=n, replace=False))

    rows = []
    for chunk_start, chunk in enumerate(pd.read_csv(csv_path, chunksize=200_000)):
        offset = chunk_start * 200_000
        local_positions = [pos - offset for pos in selected_positions if offset <= pos < offset + len(chunk)]
        if local_positions:
            rows.append(chunk.iloc[local_positions])
    return pd.concat(rows, ignore_index=True)


def load_current_link_state_for_times(link_state_path: Path, times: np.ndarray, edge_count: int, chunk_size: int):
    time_to_index = {int(time): idx for idx, time in enumerate(times.tolist())}
    needed = set(time_to_index)
    delay = np.full((len(times), edge_count), np.nan, dtype=np.float32)
    util = np.full((len(times), edge_count), np.nan, dtype=np.float32)
    cong = np.full((len(times), edge_count), np.nan, dtype=np.float32)

    usecols = ["time", "edge_id", "delay_ms", "utilization", "congestion_label"]
    for chunk in tqdm(pd.read_csv(link_state_path, usecols=usecols, chunksize=chunk_size), desc="Loading sampled link_state"):
        filtered = chunk[chunk["time"].isin(needed)]
        if filtered.empty:
            continue
        for row in filtered.itertuples(index=False):
            idx = time_to_index[int(row.time)]
            edge_id = int(row.edge_id)
            delay[idx, edge_id] = float(row.delay_ms)
            util[idx, edge_id] = float(row.utilization)
            cong[idx, edge_id] = float(row.congestion_label)

    if np.isnan(delay).any() or np.isnan(util).any() or np.isnan(cong).any():
        raise ValueError("Missing sampled link_state values.")
    return delay, util, cong, time_to_index


def recompute(edge_path: list[int], delay_row, util_row, cong_row, mc, sample_pos: int) -> dict[str, float]:
    if len(edge_path) == 0:
        return {feature: 0.0 for feature in FEATURES_TO_RECHECK}

    edges = np.asarray(edge_path, dtype=np.int32)
    current_delay = delay_row[edges]
    current_util = util_row[edges]
    current_cong = cong_row[edges]
    pred_util = mc["util_pred_mean"][sample_pos, edges]
    util_std = mc["util_pred_std"][sample_pos, edges]
    cong_prob = mc["cong_prob_mean"][sample_pos, edges]
    cong_std = mc["cong_prob_std"][sample_pos, edges]
    risk = mc["risk_score"][sample_pos, edges]
    cong_risk = mc["congestion_risk_score"][sample_pos, edges]
    return {
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
    }


def main() -> None:
    args = parse_args()
    debug_suffix = suffix(args.max_samples)
    processed_dir = args.processed_dir
    results_dir = args.results_dir

    npz_path = processed_dir / f"path_risk_features_test_topo144_k{args.k}{debug_suffix}.npz"
    csv_path = processed_dir / f"path_risk_features_test_topo144_k{args.k}{debug_suffix}.csv"
    names_path = processed_dir / f"path_risk_feature_names_topo144_k{args.k}{debug_suffix}.json"
    candidate_summary = processed_dir / f"candidate_paths_topo144_k{args.k}_summary.csv"
    mc_path = results_dir / "mc_dropout_predictions_test.npz"
    link_state_path = processed_dir / "link_state_topo144_shortest_delay.csv"
    report_path = results_dir / f"path_risk_feature_check_report{debug_suffix}.json"

    for path in [npz_path, csv_path, names_path, candidate_summary, mc_path, link_state_path]:
        require_file(path)

    data = np.load(npz_path, allow_pickle=False)
    features = data["features"]
    feature_names = [str(name) for name in data["feature_names"]]
    gw_pairs = data["gw_pairs"]
    time_t = data["time_t"]
    src_sat = data["src_sat"]
    dst_sat = data["dst_sat"]
    valid_mask = data["valid_mask"]

    with names_path.open("r", encoding="utf-8") as file:
        names_payload = json.load(file)
    expected_feature_names = names_payload["feature_names"]
    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    csv_rows = sum(1 for _ in csv_path.open("r", encoding="utf-8")) - 1
    expected_rows = int(features.shape[0] * features.shape[1] * features.shape[2])

    shape_ok = (
        features.ndim == 4
        and features.shape[1] == 132
        and features.shape[2] == args.k
        and gw_pairs.shape == (132, 2)
        and valid_mask.shape == features.shape[:3]
        and src_sat.shape == features.shape[:2]
        and dst_sat.shape == features.shape[:2]
        and csv_rows == expected_rows
    )
    feature_names_complete = expected_feature_names == feature_names
    valid_mask_all_true = bool(valid_mask.all())
    has_nan = bool(np.isnan(features).any())
    has_inf = bool(np.isinf(features).any())

    hop = features[..., feature_index["hop_count"]]
    path0_hop_is_min = bool(np.all(hop[:, :, 0] <= hop.min(axis=2) + 1e-6))

    rank_risk = features[..., feature_index["rank_by_risk"]]
    sorted_ranks_ok = bool(np.all(np.sort(rank_risk, axis=2) == np.arange(1, args.k + 1)))

    min_risk_count = features[..., feature_index["is_min_risk_path"]].sum(axis=2)
    min_cong_count = features[..., feature_index["is_min_cong_prob_path"]].sum(axis=2)
    min_delay_count = features[..., feature_index["is_min_delay_path"]].sum(axis=2)
    min_risk_one_per_group = bool(np.all(min_risk_count == 1))
    min_cong_one_per_group = bool(np.all(min_cong_count == 1))
    min_delay_at_least_one_per_group = bool(np.all(min_delay_count >= 1))
    min_delay_groups_with_ties = int(np.sum(min_delay_count > 1))

    sampled_rows = load_sample_csv_rows(csv_path, args.sample_checks)
    sampled_times = np.sort(sampled_rows["time_t"].unique().astype(np.int32))
    edge_count = np.load(mc_path, allow_pickle=False)["util_pred_mean"].shape[1]
    delay, util, cong, time_to_local = load_current_link_state_for_times(
        link_state_path,
        sampled_times,
        edge_count=edge_count,
        chunk_size=args.chunk_size,
    )
    mc = np.load(mc_path, allow_pickle=False)
    candidate_paths = load_candidate_paths(candidate_summary)

    test_time_to_pos = {int(time): idx for idx, time in enumerate(time_t.tolist())}
    max_abs_error = 0.0
    failed_samples = []
    for row in sampled_rows.itertuples(index=False):
        sample_pos = test_time_to_pos[int(row.time_t)]
        local_time_idx = time_to_local[int(row.time_t)]
        key = (int(row.src_sat), int(row.dst_sat), int(row.path_id))
        if key in candidate_paths:
            edge_path = candidate_paths[key]
        elif int(row.src_sat) == int(row.dst_sat):
            edge_path = []
        else:
            raise KeyError(f"No candidate edge_path for {key}")
        expected = recompute(
            edge_path=edge_path,
            delay_row=delay[local_time_idx],
            util_row=util[local_time_idx],
            cong_row=cong[local_time_idx],
            mc=mc,
            sample_pos=sample_pos,
        )
        for feature in FEATURES_TO_RECHECK:
            actual = float(getattr(row, feature))
            error = abs(actual - expected[feature])
            max_abs_error = max(max_abs_error, error)
            if error > 5e-5:
                failed_samples.append(
                    {
                        "time_t": int(row.time_t),
                        "src_gateway": int(row.src_gateway),
                        "dst_gateway": int(row.dst_gateway),
                        "path_id": int(row.path_id),
                        "feature": feature,
                        "actual": actual,
                        "expected": expected[feature],
                        "abs_error": error,
                    }
                )
                break

    sample_aggregation_passed = len(failed_samples) == 0

    report = {
        "npz_path": str(npz_path),
        "csv_path": str(csv_path),
        "features_shape": list(features.shape),
        "gw_pairs_shape": list(gw_pairs.shape),
        "csv_rows": int(csv_rows),
        "shape_ok": shape_ok,
        "feature_names_complete": feature_names_complete,
        "valid_mask_all_true": valid_mask_all_true,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "sample_aggregation_checks": int(len(sampled_rows)),
        "sample_aggregation_passed": sample_aggregation_passed,
        "sample_aggregation_max_abs_error": float(max_abs_error),
        "sample_aggregation_failed_count": int(len(failed_samples)),
        "sample_aggregation_first_failures": failed_samples[:5],
        "path0_hop_count_is_min": path0_hop_is_min,
        "rank_by_risk_is_1_to_k": sorted_ranks_ok,
        "is_min_risk_one_per_group": min_risk_one_per_group,
        "is_min_cong_prob_one_per_group": min_cong_one_per_group,
        "is_min_delay_at_least_one_per_group": min_delay_at_least_one_per_group,
        "min_delay_groups_with_ties": min_delay_groups_with_ties,
        "validation_passed": bool(
            shape_ok
            and feature_names_complete
            and valid_mask_all_true
            and not has_nan
            and not has_inf
            and sample_aggregation_passed
            and path0_hop_is_min
            and sorted_ranks_ok
            and min_risk_one_per_group
            and min_cong_one_per_group
            and min_delay_at_least_one_per_group
        ),
        "note": "min_delay_path may have multiple True values per group when there are exact delay ties.",
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("Path risk feature check complete:")
    print(f"- report: {report_path}")
    print(f"- features shape: {features.shape}")
    print(f"- gw_pairs shape: {gw_pairs.shape}")
    print(f"- csv rows: {csv_rows}")
    print(f"- has NaN / Inf: {has_nan} / {has_inf}")
    print(f"- sample aggregation passed: {sample_aggregation_passed}")
    print(f"- max aggregation abs error: {max_abs_error:.9g}")
    print(f"- path_id=0 hop_count is min: {path0_hop_is_min}")
    print(f"- rank_by_risk is 1..K: {sorted_ranks_ok}")
    print(f"- validation passed: {report['validation_passed']}")

    if not report["validation_passed"]:
        raise SystemExit("Path risk feature validation failed. See report JSON.")


if __name__ == "__main__":
    main()
