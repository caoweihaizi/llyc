import argparse
import json
from pathlib import Path
from typing import Callable

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from leo_routing_env import LeoRoutingEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "runs" / "masked_ppo_topo144_k5_smoke" / "maskable_ppo_smoke_model.zip"


class FixedEpisodeOptionsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_samples: int | None, random_start: bool = False):
        super().__init__(env)
        self.max_samples = max_samples
        self.random_start = random_start

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        merged_options = dict(options or {})
        if self.max_samples is not None:
            merged_options.setdefault("max_samples", self.max_samples)
        merged_options.setdefault("random_start", self.random_start)
        return self.env.reset(seed=seed, options=merged_options)

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MaskablePPO smoke model and heuristic policies.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_env(k: int, max_samples: int | None, seed: int) -> gym.Env:
    env = LeoRoutingEnv(k=k, seed=seed)
    return FixedEpisodeOptionsWrapper(env, max_samples=max_samples, random_start=False)


def choose_heuristic_action(env: LeoRoutingEnv, policy: str, rng: np.random.Generator) -> int:
    mask = env.action_masks()
    valid_actions = np.flatnonzero(mask)
    if policy == "random_valid":
        return int(rng.choice(valid_actions))
    if policy == "shortest":
        return 0 if mask[0] else int(valid_actions[0])

    values = env.unwrapped._current_values()
    if policy == "min_raw_cost":
        scores = np.array([env.unwrapped.compute_raw_cost(a) for a in range(env.action_space.n)], dtype=np.float64)
    elif policy == "min_risk":
        scores = values[:, env.unwrapped.feature_index["path_risk_score_max"]].astype(np.float64)
    elif policy == "min_cong_prob":
        scores = values[:, env.unwrapped.feature_index["path_cong_prob_max"]].astype(np.float64)
    else:
        raise ValueError(f"Unsupported heuristic policy: {policy}")
    scores[~mask] = np.inf
    return int(np.argmin(scores))


def rollout_policy(
    env: gym.Env,
    policy_name: str,
    action_fn: Callable[[np.ndarray], int],
    seed: int,
) -> dict:
    obs, _ = env.reset(seed=seed)
    done = False
    metrics = {
        "policy": policy_name,
        "steps": 0,
        "total_reward": 0.0,
        "raw_cost": [],
        "delay": [],
        "post_mlu": [],
        "delta_mlu": [],
        "delta_congestion_count": [],
        "risk_score": [],
        "cong_prob": [],
        "invalid_action_count": 0,
        "zero_hop_count": 0,
        "action_counts": np.zeros(env.action_space.n, dtype=np.int64),
    }

    while not done:
        action = int(action_fn(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        metrics["steps"] += 1
        metrics["total_reward"] += float(reward)
        metrics["raw_cost"].append(float(info["raw_cost"]))
        metrics["delay"].append(float(info["path_delay_ms_sum"]))
        metrics["post_mlu"].append(float(info["post_mlu"]))
        metrics["delta_mlu"].append(float(info["delta_mlu"]))
        metrics["delta_congestion_count"].append(float(info["delta_congestion_count"]))
        metrics["risk_score"].append(float(info["path_risk_score_max"]))
        metrics["cong_prob"].append(float(info["path_cong_prob_max"]))
        metrics["invalid_action_count"] += int(info["invalid_action"])
        metrics["zero_hop_count"] += int(info["is_zero_hop"])
        metrics["action_counts"][action] += 1

    steps = max(metrics["steps"], 1)
    return {
        "policy": policy_name,
        "steps": metrics["steps"],
        "mean_reward": metrics["total_reward"] / steps,
        "total_reward": metrics["total_reward"],
        "mean_raw_cost": float(np.mean(metrics["raw_cost"])),
        "mean_delay": float(np.mean(metrics["delay"])),
        "mean_post_mlu": float(np.mean(metrics["post_mlu"])),
        "mean_delta_mlu": float(np.mean(metrics["delta_mlu"])),
        "mean_delta_congestion_count": float(np.mean(metrics["delta_congestion_count"])),
        "mean_risk_score": float(np.mean(metrics["risk_score"])),
        "mean_cong_prob": float(np.mean(metrics["cong_prob"])),
        "invalid_action_count": int(metrics["invalid_action_count"]),
        "zero_hop_count": int(metrics["zero_hop_count"]),
        "action_id_distribution": {str(i): int(v) for i, v in enumerate(metrics["action_counts"].tolist())},
    }


def save_figures(df: pd.DataFrame) -> list[Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    reward_path = RESULTS_DIR / "masked_ppo_smoke_policy_reward_comparison.png"
    plt.figure(figsize=(9, 4.5))
    plt.bar(df["policy"], df["mean_reward"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("mean reward")
    plt.tight_layout()
    plt.savefig(reward_path, dpi=200)
    plt.close()
    paths.append(reward_path)

    cost_path = RESULTS_DIR / "masked_ppo_smoke_policy_cost_comparison.png"
    plt.figure(figsize=(9, 4.5))
    plt.bar(df["policy"], df["mean_raw_cost"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("mean raw cost")
    plt.tight_layout()
    plt.savefig(cost_path, dpi=200)
    plt.close()
    paths.append(cost_path)

    action_path = RESULTS_DIR / "masked_ppo_smoke_action_distribution.png"
    action_rows = []
    for row in df.itertuples(index=False):
        dist = row.action_id_distribution
        if isinstance(dist, str):
            dist = json.loads(dist.replace("'", '"'))
        total = sum(dist.values()) or 1
        for action_id, count in dist.items():
            action_rows.append({"policy": row.policy, "action": int(action_id), "ratio": count / total})
    action_df = pd.DataFrame(action_rows)
    pivot = action_df.pivot(index="policy", columns="action", values="ratio").fillna(0.0)
    pivot.plot(kind="bar", stacked=True, figsize=(10, 4.8))
    plt.ylabel("action ratio")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(action_path, dpi=200)
    plt.close()
    paths.append(action_path)
    return paths


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Missing MaskablePPO model: {args.model_path}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = MaskablePPO.load(args.model_path)
    rng = np.random.default_rng(args.seed)
    policy_names = ["maskable_ppo", "random_valid", "shortest", "min_raw_cost", "min_risk", "min_cong_prob"]
    rows = []

    for policy_name in policy_names:
        env = make_env(args.k, args.max_samples, args.seed)
        if policy_name == "maskable_ppo":
            def action_fn(obs, env=env, model=model):
                masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=masks, deterministic=True)
                return int(action)
        else:
            def action_fn(obs, env=env, policy_name=policy_name, rng=rng):
                return choose_heuristic_action(env, policy_name, rng)
        row = rollout_policy(env, policy_name, action_fn, args.seed)
        rows.append(row)
        print(
            f"{policy_name}: mean_reward={row['mean_reward']:.6f}, "
            f"mean_raw_cost={row['mean_raw_cost']:.6f}, invalid={row['invalid_action_count']}"
        )

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / f"masked_ppo_smoke_eval_topo144_k{args.k}.csv"
    json_path = RESULTS_DIR / f"masked_ppo_smoke_eval_topo144_k{args.k}.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)
    fig_paths = save_figures(df)

    print("MaskedPPO smoke evaluation complete:")
    print(f"- csv: {csv_path}")
    print(f"- json: {json_path}")
    for path in fig_paths:
        print(f"- figure: {path}")


if __name__ == "__main__":
    main()
