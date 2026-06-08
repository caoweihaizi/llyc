import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from leo_routing_env import LeoRoutingEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

POLICIES = ["random_valid", "shortest", "min_raw_cost", "min_risk", "min_cong_prob"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate heuristic policies in LeoRoutingEnv.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def suffix(max_samples: int | None) -> str:
    return "" if max_samples is None else f"_debug{max_samples}"


def choose_action(env: LeoRoutingEnv, policy: str, rng: np.random.Generator) -> int:
    mask = env.action_masks()
    valid = np.flatnonzero(mask)
    if policy == "random_valid":
        return int(rng.choice(valid))
    if policy == "shortest":
        return 0 if mask[0] else int(valid[0])

    values = env._current_values()
    scores = np.full(env.k, np.inf, dtype=np.float32)
    if policy == "min_raw_cost":
        for action in valid:
            scores[action] = env.compute_raw_cost(int(action))
    elif policy == "min_risk":
        scores[valid] = values[valid, env.feature_index["path_risk_score_max"]]
    elif policy == "min_cong_prob":
        scores[valid] = values[valid, env.feature_index["path_cong_prob_max"]]
    else:
        raise ValueError(f"Unknown policy: {policy}")
    return int(np.argmin(scores))


def evaluate_policy(policy: str, k: int, max_samples: int | None, seed: int) -> dict:
    env = LeoRoutingEnv(k=k)
    options = {}
    if max_samples is not None:
        options["max_samples"] = max_samples
    env.reset(seed=seed, options=options)
    rng = np.random.default_rng(seed)

    action_counts = np.zeros(k, dtype=np.int64)
    sums = {
        "reward": 0.0,
        "raw_cost": 0.0,
        "delay": 0.0,
        "post_mlu": 0.0,
        "delta_mlu": 0.0,
        "delta_congestion_count": 0.0,
        "risk_score": 0.0,
        "cong_prob": 0.0,
    }
    invalid_count = 0
    zero_hop_count = 0
    steps = 0

    total_steps = (max_samples if max_samples is not None else env.n_test) * env.gw_pair_count
    with tqdm(total=total_steps, desc=f"Policy {policy}") as progress:
        while True:
            action = choose_action(env, policy, rng)
            _, reward, terminated, truncated, info = env.step(action)
            steps += 1
            action_counts[action] += 1
            invalid_count += int(info["invalid_action"])
            zero_hop_count += int(info["is_zero_hop"])
            sums["reward"] += float(reward)
            sums["raw_cost"] += float(info["raw_cost"])
            sums["delay"] += float(info["path_delay_ms_sum"])
            sums["post_mlu"] += float(info["post_mlu"])
            sums["delta_mlu"] += float(info["delta_mlu"])
            sums["delta_congestion_count"] += float(info["delta_congestion_count"])
            sums["risk_score"] += float(info["path_risk_score_max"])
            sums["cong_prob"] += float(info["path_cong_prob_max"])
            progress.update(1)
            if terminated or truncated:
                break

    return {
        "policy": policy,
        "steps": int(steps),
        "mean_reward": sums["reward"] / steps,
        "total_reward": sums["reward"],
        "mean_raw_cost": sums["raw_cost"] / steps,
        "mean_delay": sums["delay"] / steps,
        "mean_post_mlu": sums["post_mlu"] / steps,
        "mean_delta_mlu": sums["delta_mlu"] / steps,
        "mean_delta_congestion_count": sums["delta_congestion_count"] / steps,
        "mean_risk_score": sums["risk_score"] / steps,
        "mean_cong_prob": sums["cong_prob"] / steps,
        "invalid_action_count": int(invalid_count),
        "zero_hop_count": int(zero_hop_count),
        "action_id_distribution": {str(i): int(v) for i, v in enumerate(action_counts.tolist())},
    }


def save_figures(df: pd.DataFrame, results_dir: Path, debug_suffix: str) -> list[Path]:
    paths = [
        results_dir / f"routing_env_policy_reward_comparison{debug_suffix}.png",
        results_dir / f"routing_env_policy_cost_comparison{debug_suffix}.png",
        results_dir / f"routing_env_policy_action_distribution{debug_suffix}.png",
    ]
    plt.figure(figsize=(8, 4.8))
    plt.bar(df["policy"], df["mean_reward"])
    plt.ylabel("mean_reward")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(paths[0], dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.bar(df["policy"], df["mean_raw_cost"])
    plt.ylabel("mean_raw_cost")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(paths[1], dpi=200)
    plt.close()

    action_df = pd.DataFrame(
        [
            {"policy": row["policy"], "action": int(action), "count": count}
            for row in df.to_dict("records")
            for action, count in row["action_id_distribution"].items()
        ]
    )
    pivot = action_df.pivot(index="policy", columns="action", values="count").fillna(0)
    pivot.plot(kind="bar", stacked=True, figsize=(9, 5))
    plt.ylabel("action count")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(paths[2], dpi=200)
    plt.close()
    return paths


def main() -> None:
    args = parse_args()
    debug_suffix = suffix(args.max_samples)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    rows = [evaluate_policy(policy, args.k, args.max_samples, seed=42) for policy in POLICIES]
    df = pd.DataFrame(rows)
    csv_path = args.results_dir / f"routing_env_heuristic_policy_eval_topo144_k{args.k}{debug_suffix}.csv"
    json_path = args.results_dir / f"routing_env_heuristic_policy_eval_topo144_k{args.k}{debug_suffix}.json"
    df.to_csv(csv_path, index=False)
    best_policy = str(df.loc[df["mean_reward"].idxmax(), "policy"])
    payload = {
        "k": args.k,
        "max_samples": args.max_samples,
        "best_policy_by_mean_reward": best_policy,
        "policies": rows,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    figure_paths = save_figures(df, args.results_dir, debug_suffix)

    print("Heuristic policy evaluation complete:")
    print(f"- csv: {csv_path}")
    print(f"- json: {json_path}")
    print(f"- best policy by mean_reward: {best_policy}")
    print(df[["policy", "mean_reward", "mean_raw_cost", "mean_post_mlu", "mean_delta_congestion_count"]].to_string(index=False))
    print("- figures:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
