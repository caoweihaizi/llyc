from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    print("gymnasium is not installed. Please install it before Masked PPO training: pip install gymnasium")

    class _Discrete:
        def __init__(self, n: int):
            self.n = int(n)

        def sample(self) -> int:
            return int(np.random.randint(self.n))

    class _Box:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = dtype

    class _Env:
        metadata: dict[str, Any] = {}

        def reset(self, seed: int | None = None, options: dict | None = None):
            if seed is not None:
                np.random.seed(seed)

    class _Spaces:
        Box = _Box
        Discrete = _Discrete

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OBS_FEATURE_NAMES = [
    "hop_count",
    "od_demand_mbps",
    "is_zero_hop",
    "affected_edge_count",
    "added_load_mbps",
    "added_load_edge_sum",
    "pre_mlu",
    "pre_congestion_count",
    "pre_congestion_ratio",
    "pre_avg_utilization",
    "pre_total_load_mbps",
    "path_delay_ms_sum",
    "path_pre_load_sum",
    "path_pre_util_max",
    "path_pre_util_mean",
    "path_pre_congestion_count",
    "path_pred_util_max",
    "path_pred_util_mean",
    "path_cong_prob_max",
    "path_cong_prob_mean",
    "path_risk_score_max",
    "path_risk_score_mean",
    "path_risk_score_sum",
    "path_congestion_risk_score_max",
    "path_congestion_risk_score_mean",
    "path_util_uncertainty_mean",
    "path_util_uncertainty_max",
]

UNNORMALIZED_FEATURES = {"is_zero_hop"}

DEFAULT_REWARD_WEIGHTS = {
    "w_delay": 1.0,
    "w_mlu": 100.0,
    "w_delta_mlu": 100.0,
    "w_congestion": 10.0,
    "w_risk": 10.0,
    "w_cong_prob": 5.0,
    "w_switch": 0.0,
}


class LeoRoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        k: int = 5,
        action_impact_path: Path | None = None,
        feature_names_path: Path | None = None,
        scaler_path: Path | None = None,
        reward_mode: str = "relative_to_shortest",
        reward_weights: dict[str, float] | None = None,
        invalid_action_penalty: float = -1000.0,
        mask_high_risk: bool = False,
        risk_threshold: float = 0.95,
        mask_high_cong_prob: bool = False,
        cong_prob_threshold: float = 0.95,
        seed: int = 42,
    ):
        super().__init__()
        self.k = int(k)
        self.action_impact_path = action_impact_path or PROCESSED_DIR / f"action_impact_features_test_topo144_k{self.k}.npz"
        self.feature_names_path = feature_names_path or PROCESSED_DIR / f"action_impact_feature_names_topo144_k{self.k}.json"
        self.scaler_path = scaler_path or PROCESSED_DIR / f"rl_env_feature_scaler_topo144_k{self.k}.json"
        self.reward_mode = reward_mode
        self.reward_weights = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.mask_high_risk = bool(mask_high_risk)
        self.risk_threshold = float(risk_threshold)
        self.mask_high_cong_prob = bool(mask_high_cong_prob)
        self.cong_prob_threshold = float(cong_prob_threshold)
        self.rng = np.random.default_rng(seed)

        if self.reward_mode not in {"negative_cost", "relative_to_shortest"}:
            raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")
        if not self.action_impact_path.exists():
            raise FileNotFoundError(f"Missing action impact file: {self.action_impact_path}")
        if not self.feature_names_path.exists():
            raise FileNotFoundError(f"Missing action impact feature names file: {self.feature_names_path}")

        data = np.load(self.action_impact_path, allow_pickle=False)
        self.impact_features = data["impact_features"].astype(np.float32, copy=False)
        self.impact_feature_names = [str(name) for name in data["impact_feature_names"]]
        self.gw_pairs = data["gw_pairs"].astype(np.int16, copy=False)
        self.time_t = data["time_t"].astype(np.int32, copy=False)
        self.target_time = data["target_time"].astype(np.int32, copy=False)
        self.valid_mask = data["valid_mask"].astype(bool, copy=False)
        self.n_test, self.gw_pair_count, loaded_k, _ = self.impact_features.shape
        if loaded_k != self.k:
            raise ValueError(f"K mismatch: requested {self.k}, loaded {loaded_k}")

        self.feature_index = {name: idx for idx, name in enumerate(self.impact_feature_names)}
        missing = [name for name in OBS_FEATURE_NAMES if name not in self.feature_index]
        if missing:
            raise KeyError(f"Missing required observation features: {missing}")
        self.obs_feature_indices = np.asarray([self.feature_index[name] for name in OBS_FEATURE_NAMES], dtype=np.int32)

        self.scaler = self._build_or_load_scaler()
        self.obs_dim = self.k * len(OBS_FEATURE_NAMES) + self.k
        self.action_space = spaces.Discrete(self.k)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.sample_pos = 0
        self.gw_pair_pos = 0
        self.episode_start_sample = 0
        self.episode_end_sample = self.n_test

    def _build_or_load_scaler(self) -> dict:
        valid = self.valid_mask
        payload = {
            "k": self.k,
            "observation_feature_names": OBS_FEATURE_NAMES,
            "normalized_features": [],
            "unnormalized_features": sorted(UNNORMALIZED_FEATURES),
            "mean": {},
            "std": {},
            "source": str(self.action_impact_path),
            "note": "Scaler is computed from all valid actions in action_impact_features.",
        }
        for name in OBS_FEATURE_NAMES:
            values = self.impact_features[..., self.feature_index[name]]
            if name in UNNORMALIZED_FEATURES:
                payload["mean"][name] = 0.0
                payload["std"][name] = 1.0
                continue
            selected = values[valid]
            mean = float(selected.mean())
            std = float(selected.std())
            if std < 1e-8:
                std = 1.0
            payload["normalized_features"].append(name)
            payload["mean"][name] = mean
            payload["std"][name] = std
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with self.scaler_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return payload

    def _current_values(self) -> np.ndarray:
        return self.impact_features[self.sample_pos, self.gw_pair_pos]

    def _advance(self) -> None:
        self.gw_pair_pos += 1
        if self.gw_pair_pos >= self.gw_pair_count:
            self.gw_pair_pos = 0
            self.sample_pos += 1

    def action_masks(self) -> np.ndarray:
        if self.sample_pos >= self.episode_end_sample:
            return np.zeros(self.k, dtype=bool)

        mask = self.valid_mask[self.sample_pos, self.gw_pair_pos].copy()
        values = self._current_values()
        zero_hop = values[:, self.feature_index["is_zero_hop"]] > 0.5
        if bool(np.all(zero_hop)):
            zero_mask = np.zeros(self.k, dtype=bool)
            zero_mask[0] = bool(mask[0])
            return zero_mask

        if self.mask_high_risk:
            risk = values[:, self.feature_index["path_risk_score_max"]]
            mask &= risk <= self.risk_threshold
        if self.mask_high_cong_prob:
            cong = values[:, self.feature_index["path_cong_prob_max"]]
            mask &= cong <= self.cong_prob_threshold
        if not bool(mask.any()):
            cost = values[:, self.feature_index["action_cost"]].copy()
            base_valid = self.valid_mask[self.sample_pos, self.gw_pair_pos]
            cost[~base_valid] = np.inf
            fallback = int(np.argmin(cost))
            mask[fallback] = True
        return mask.astype(bool)

    def get_observation(self) -> np.ndarray:
        if self.sample_pos >= self.episode_end_sample:
            return np.zeros(self.obs_dim, dtype=np.float32)
        values = self._current_values()[:, self.obs_feature_indices].astype(np.float32, copy=True)
        for local_idx, name in enumerate(OBS_FEATURE_NAMES):
            if name in UNNORMALIZED_FEATURES:
                continue
            mean = float(self.scaler["mean"][name])
            std = float(self.scaler["std"][name])
            values[:, local_idx] = (values[:, local_idx] - mean) / (std + 1e-8)
        mask_float = self.action_masks().astype(np.float32)
        return np.concatenate([values.reshape(-1), mask_float], axis=0).astype(np.float32)

    def compute_raw_cost(self, action: int) -> float:
        values = self.impact_features[self.sample_pos, self.gw_pair_pos, int(action)]
        switch_penalty = 0.0
        return float(
            self.reward_weights["w_delay"] * values[self.feature_index["path_delay_ms_sum"]]
            + self.reward_weights["w_mlu"] * values[self.feature_index["post_mlu"]]
            + self.reward_weights["w_delta_mlu"] * values[self.feature_index["delta_mlu"]]
            + self.reward_weights["w_congestion"] * values[self.feature_index["delta_congestion_count"]]
            + self.reward_weights["w_risk"] * values[self.feature_index["path_risk_score_max"]]
            + self.reward_weights["w_cong_prob"] * values[self.feature_index["path_cong_prob_max"]]
            + self.reward_weights["w_switch"] * switch_penalty
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        max_samples = options.get("max_samples", None)
        random_start = bool(options.get("random_start", False))
        start_sample = int(options.get("start_sample", 0))

        if max_samples is None:
            episode_len = self.n_test - start_sample
        else:
            episode_len = int(max_samples)
        if episode_len <= 0:
            raise ValueError(f"Episode max_samples must be positive, got {episode_len}")

        if random_start:
            max_start = max(0, self.n_test - episode_len)
            start_sample = int(self.rng.integers(0, max_start + 1))

        self.episode_start_sample = max(0, min(start_sample, self.n_test - 1))
        self.episode_end_sample = min(self.n_test, self.episode_start_sample + episode_len)
        self.sample_pos = self.episode_start_sample
        self.gw_pair_pos = 0
        info = {
            "start_sample": self.episode_start_sample,
            "sample_pos": self.sample_pos,
            "gw_pair_pos": self.gw_pair_pos,
            "episode_end_sample": self.episode_end_sample,
            "N_test": self.n_test,
            "gw_pair_count": self.gw_pair_count,
            "K": self.k,
        }
        return self.get_observation(), info

    def step(self, action: int):
        action = int(action)
        if action < 0 or action >= self.k:
            raise ValueError(f"Action out of range: {action}")

        mask = self.action_masks()
        invalid = not bool(mask[action])
        values = self._current_values()[action]
        raw_cost = self.compute_raw_cost(action)
        shortest_raw_cost = self.compute_raw_cost(0)
        path_switch_penalty = 0.0

        if invalid:
            reward = self.invalid_action_penalty
        elif self.reward_mode == "negative_cost":
            reward = -raw_cost
        else:
            reward = shortest_raw_cost - raw_cost

        src_gateway, dst_gateway = self.gw_pairs[self.gw_pair_pos]
        info = {
            "sample_pos": int(self.sample_pos),
            "gw_pair_pos": int(self.gw_pair_pos),
            "time_t": int(self.time_t[self.sample_pos]),
            "src_gateway": int(src_gateway),
            "dst_gateway": int(dst_gateway),
            "action": action,
            "reward": float(reward),
            "raw_cost": float(raw_cost),
            "shortest_raw_cost": float(shortest_raw_cost),
            "path_switch_penalty": float(path_switch_penalty),
            "path_delay_ms_sum": float(values[self.feature_index["path_delay_ms_sum"]]),
            "post_mlu": float(values[self.feature_index["post_mlu"]]),
            "delta_mlu": float(values[self.feature_index["delta_mlu"]]),
            "delta_congestion_count": float(values[self.feature_index["delta_congestion_count"]]),
            "path_risk_score_max": float(values[self.feature_index["path_risk_score_max"]]),
            "path_cong_prob_max": float(values[self.feature_index["path_cong_prob_max"]]),
            "is_zero_hop": bool(values[self.feature_index["is_zero_hop"]] > 0.5),
            "invalid_action": bool(invalid),
            "action_mask": mask.copy(),
        }

        self._advance()
        terminated = self.sample_pos >= self.episode_end_sample
        truncated = False
        return self.get_observation(), float(reward), terminated, truncated, info
