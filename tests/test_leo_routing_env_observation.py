import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from leo_routing_env import LeoRoutingEnv, OBS_FEATURE_NAMES


FORBIDDEN_POLICY_FEATURES = {
    "action_cost",
    "post_mlu",
    "delta_mlu",
    "post_congestion_count",
    "delta_congestion_count",
    "post_avg_utilization",
    "path_post_util_max",
    "path_post_util_mean",
    "path_post_congestion_count",
    "path_new_congestion_count",
}


def test_policy_observation_excludes_action_outcome_features():
    assert FORBIDDEN_POLICY_FEATURES.isdisjoint(OBS_FEATURE_NAMES)


def test_observation_shape_matches_pre_decision_features_plus_mask():
    env = LeoRoutingEnv(k=5)
    obs, _ = env.reset(seed=42, options={"max_samples": 1})

    expected_dim = env.k * len(OBS_FEATURE_NAMES) + env.k
    assert env.obs_dim == expected_dim
    assert env.observation_space.shape == (expected_dim,)
    assert obs.shape == (expected_dim,)


def test_reward_can_still_use_action_impact_outcomes():
    env = LeoRoutingEnv(k=5)
    env.reset(seed=42, options={"max_samples": 1})
    _, _, _, _, info = env.step(0)

    assert "raw_cost" in info
    assert "post_mlu" in info
    assert "delta_congestion_count" in info
