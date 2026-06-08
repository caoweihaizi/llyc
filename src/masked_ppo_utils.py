from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO

from leo_routing_env import LeoRoutingEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


class FixedEpisodeOptionsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_samples: int | None, random_start: bool = False):
        super().__init__(env)
        self.max_samples = max_samples
        self.random_start = random_start

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        merged_options = dict(options or {})
        if self.max_samples is not None:
            merged_options.setdefault("max_samples", self.max_samples)
        merged_options.setdefault("random_start", self.random_start)
        return self.env.reset(seed=seed, options=merged_options)

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    if str(value).lower() in {"none", "null", "-1"}:
        return None
    return int(value)


def make_env(k: int, max_samples: int | None, seed: int, random_start: bool, reward_mode: str = "relative_to_shortest"):
    env = LeoRoutingEnv(k=k, seed=seed, reward_mode=reward_mode)
    return FixedEpisodeOptionsWrapper(env, max_samples=max_samples, random_start=random_start)


def choose_heuristic_action(env: gym.Env, policy: str, rng: np.random.Generator) -> int:
    mask = env.action_masks()
    valid_actions = np.flatnonzero(mask)
    if policy == "random_valid":
        return int(rng.choice(valid_actions))
    if policy == "shortest":
        return 0 if mask[0] else int(valid_actions[0])

    unwrapped = env.unwrapped
    values = unwrapped._current_values()
    if policy == "min_raw_cost":
        scores = np.array([unwrapped.compute_raw_cost(a) for a in range(env.action_space.n)], dtype=np.float64)
    elif policy == "min_risk":
        scores = values[:, unwrapped.feature_index["path_risk_score_max"]].astype(np.float64)
    elif policy == "min_cong_prob":
        scores = values[:, unwrapped.feature_index["path_cong_prob_max"]].astype(np.float64)
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
    obs, reset_info = env.reset(seed=seed)
    done = False
    raw_cost = []
    delay = []
    post_mlu = []
    delta_mlu = []
    delta_congestion_count = []
    risk_score = []
    cong_prob = []
    action_counts = np.zeros(env.action_space.n, dtype=np.int64)
    total_reward = 0.0
    invalid_action_count = 0
    zero_hop_count = 0
    steps = 0

    while not done:
        action = int(action_fn(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
        total_reward += float(reward)
        raw_cost.append(float(info["raw_cost"]))
        delay.append(float(info["path_delay_ms_sum"]))
        post_mlu.append(float(info["post_mlu"]))
        delta_mlu.append(float(info["delta_mlu"]))
        delta_congestion_count.append(float(info["delta_congestion_count"]))
        risk_score.append(float(info["path_risk_score_max"]))
        cong_prob.append(float(info["path_cong_prob_max"]))
        invalid_action_count += int(info["invalid_action"])
        zero_hop_count += int(info["is_zero_hop"])
        action_counts[action] += 1

    steps = max(steps, 1)
    evaluated_samples = int(reset_info["episode_end_sample"] - reset_info["start_sample"])
    return {
        "policy": policy_name,
        "total_steps": steps,
        "steps": steps,
        "evaluated_samples": evaluated_samples,
        "evaluated_od_pairs": steps,
        "mean_reward": total_reward / steps,
        "total_reward": total_reward,
        "mean_raw_cost": float(np.mean(raw_cost)),
        "mean_delay": float(np.mean(delay)),
        "mean_post_mlu": float(np.mean(post_mlu)),
        "mean_delta_mlu": float(np.mean(delta_mlu)),
        "mean_delta_congestion_count": float(np.mean(delta_congestion_count)),
        "mean_risk_score": float(np.mean(risk_score)),
        "mean_cong_prob": float(np.mean(cong_prob)),
        "invalid_action_count": int(invalid_action_count),
        "zero_hop_count": int(zero_hop_count),
        "action_id_distribution": {str(i): int(v) for i, v in enumerate(action_counts.tolist())},
    }


def rollout_maskable_ppo(model: MaskablePPO, env: gym.Env, seed: int, policy_name: str = "maskable_ppo") -> dict:
    def action_fn(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        return int(action)

    return rollout_policy(env, policy_name, action_fn, seed)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, indent=2, ensure_ascii=False)
