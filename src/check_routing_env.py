import argparse
import json
from pathlib import Path

import numpy as np

from leo_routing_env import LeoRoutingEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

REQUIRED_INFO_FIELDS = {
    "sample_pos",
    "gw_pair_pos",
    "time_t",
    "src_gateway",
    "dst_gateway",
    "action",
    "reward",
    "raw_cost",
    "shortest_raw_cost",
    "path_delay_ms_sum",
    "post_mlu",
    "delta_mlu",
    "delta_congestion_count",
    "path_risk_score_max",
    "path_cong_prob_max",
    "is_zero_hop",
    "action_mask",
    "invalid_action",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LeoRoutingEnv action masks and rollout behavior.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--rollout_steps", type=int, default=1000)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = LeoRoutingEnv(k=args.k)
    obs, reset_info = env.reset(seed=42, options={"max_samples": max(10, args.rollout_steps // env.gw_pair_count + 2)})

    action_space_ok = getattr(env.action_space, "n", None) == args.k
    obs_shape_ok = tuple(obs.shape) == tuple(env.observation_space.shape)
    obs_finite = bool(np.isfinite(obs).all())
    initial_mask = env.action_masks()
    mask_shape_ok = initial_mask.shape == (args.k,)
    mask_dtype_ok = initial_mask.dtype == bool

    rng = np.random.default_rng(42)
    rewards_finite = True
    masks_have_valid = True
    info_fields_ok = True
    zero_hop_mask_ok = True
    rollout_steps_done = 0
    zero_hop_seen = 0

    for _ in range(args.rollout_steps):
        mask = env.action_masks()
        if not bool(mask.any()):
            masks_have_valid = False
            break
        current_values = env._current_values()
        if bool(np.all(current_values[:, env.feature_index["is_zero_hop"]] > 0.5)):
            zero_hop_seen += 1
            if not (mask[0] and not mask[1:].any()):
                zero_hop_mask_ok = False
        action = int(rng.choice(np.flatnonzero(mask)))
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_finite &= bool(np.isfinite(reward))
        rewards_finite &= bool(np.isfinite(obs).all())
        info_fields_ok &= REQUIRED_INFO_FIELDS.issubset(set(info.keys()))
        rollout_steps_done += 1
        if terminated or truncated:
            break

    invalid_action_tested = False
    invalid_action_ok = True
    env_invalid = LeoRoutingEnv(k=args.k)
    env_invalid.reset(seed=123, options={"max_samples": 50})
    for _ in range(50 * env_invalid.gw_pair_count):
        mask = env_invalid.action_masks()
        invalid_actions = np.flatnonzero(~mask)
        if len(invalid_actions):
            invalid_action_tested = True
            _, reward, _, _, info = env_invalid.step(int(invalid_actions[0]))
            invalid_action_ok = bool(info["invalid_action"] and reward == env_invalid.invalid_action_penalty)
            break
        env_invalid.step(int(np.flatnonzero(mask)[0]))

    report = {
        "action_space_n": int(env.action_space.n),
        "action_space_ok": bool(action_space_ok),
        "observation_shape": list(obs.shape),
        "observation_space_shape": list(env.observation_space.shape),
        "observation_shape_ok": bool(obs_shape_ok),
        "observation_finite": bool(obs_finite),
        "action_mask_shape_ok": bool(mask_shape_ok),
        "action_mask_dtype_ok": bool(mask_dtype_ok),
        "masks_have_at_least_one_valid_action": bool(masks_have_valid),
        "rollout_steps_requested": int(args.rollout_steps),
        "rollout_steps_done": int(rollout_steps_done),
        "rewards_and_observations_finite": bool(rewards_finite),
        "info_fields_ok": bool(info_fields_ok),
        "zero_hop_seen": int(zero_hop_seen),
        "zero_hop_mask_ok": bool(zero_hop_mask_ok),
        "invalid_action_tested": bool(invalid_action_tested),
        "invalid_action_ok": bool(invalid_action_ok),
        "reset_info": reset_info,
        "scaler_path": str(env.scaler_path),
    }
    report["validation_passed"] = bool(
        action_space_ok
        and obs_shape_ok
        and obs_finite
        and mask_shape_ok
        and mask_dtype_ok
        and masks_have_valid
        and rewards_finite
        and info_fields_ok
        and zero_hop_mask_ok
        and invalid_action_tested
        and invalid_action_ok
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.results_dir / f"routing_env_check_topo144_k{args.k}.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("Routing env check complete:")
    print(f"- report: {report_path}")
    print(f"- observation shape: {tuple(obs.shape)}")
    print(f"- action_space.n: {env.action_space.n}")
    print(f"- zero-hop seen / mask ok: {zero_hop_seen} / {zero_hop_mask_ok}")
    print(f"- invalid action tested / ok: {invalid_action_tested} / {invalid_action_ok}")
    print(f"- validation passed: {report['validation_passed']}")

    if not report["validation_passed"]:
        raise SystemExit("Routing env validation failed. See report JSON.")


if __name__ == "__main__":
    main()
