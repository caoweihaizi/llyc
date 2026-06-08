import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from leo_routing_env import LeoRoutingEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "runs" / "masked_ppo_topo144_k5_smoke"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MaskablePPO smoke training outputs.")
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def require(path: Path, missing: list[str]) -> bool:
    if not path.exists():
        missing.append(str(path))
        return False
    return True


def main() -> None:
    args = parse_args()
    model_path = RUN_DIR / "maskable_ppo_smoke_model.zip"
    train_config_path = RUN_DIR / "train_config.json"
    train_summary_path = RESULTS_DIR / f"masked_ppo_smoke_train_summary_topo144_k{args.k}.json"
    eval_csv_path = RESULTS_DIR / f"masked_ppo_smoke_eval_topo144_k{args.k}.csv"
    eval_json_path = RESULTS_DIR / f"masked_ppo_smoke_eval_topo144_k{args.k}.json"
    figure_paths = [
        RESULTS_DIR / "masked_ppo_smoke_policy_reward_comparison.png",
        RESULTS_DIR / "masked_ppo_smoke_policy_cost_comparison.png",
        RESULTS_DIR / "masked_ppo_smoke_action_distribution.png",
    ]

    missing: list[str] = []
    for path in [model_path, train_config_path, train_summary_path, eval_csv_path, eval_json_path, *figure_paths]:
        require(path, missing)

    report = {
        "missing_files": missing,
        "required_files_exist": not missing,
        "model_reload_ok": False,
        "env_mask_ok": False,
        "eval_contains_all_policies": False,
        "eval_values_finite": False,
        "maskable_ppo_invalid_action_count_zero": False,
        "maskable_ppo_vs_random_valid": None,
        "validation_passed": False,
    }

    if not missing:
        model = MaskablePPO.load(model_path)
        report["model_reload_ok"] = model is not None

        env = LeoRoutingEnv(k=args.k)
        obs, _ = env.reset(options={"max_samples": 10})
        mask = env.action_masks()
        report["env_mask_ok"] = bool(obs.shape == env.observation_space.shape and mask.shape == (args.k,) and mask.dtype == bool and mask.any())

        df = pd.read_csv(eval_csv_path)
        expected = {"maskable_ppo", "random_valid", "shortest", "min_raw_cost", "min_risk", "min_cong_prob"}
        report["eval_contains_all_policies"] = expected.issubset(set(df["policy"].astype(str)))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        report["eval_values_finite"] = bool(np.isfinite(df[numeric_cols].to_numpy()).all())
        ppo_row = df[df["policy"] == "maskable_ppo"].iloc[0]
        random_row = df[df["policy"] == "random_valid"].iloc[0]
        report["maskable_ppo_invalid_action_count_zero"] = int(ppo_row["invalid_action_count"]) == 0
        report["maskable_ppo_vs_random_valid"] = {
            "ppo_mean_reward": float(ppo_row["mean_reward"]),
            "random_valid_mean_reward": float(random_row["mean_reward"]),
            "ppo_reward_minus_random_valid": float(ppo_row["mean_reward"] - random_row["mean_reward"]),
            "ppo_mean_raw_cost": float(ppo_row["mean_raw_cost"]),
            "random_valid_mean_raw_cost": float(random_row["mean_raw_cost"]),
        }

    report["validation_passed"] = bool(
        report["required_files_exist"]
        and report["model_reload_ok"]
        and report["env_mask_ok"]
        and report["eval_contains_all_policies"]
        and report["eval_values_finite"]
        and report["maskable_ppo_invalid_action_count_zero"]
    )

    report_path = RESULTS_DIR / f"masked_ppo_smoke_check_topo144_k{args.k}.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("MaskedPPO smoke output check complete:")
    print(f"- report: {report_path}")
    print(f"- validation passed: {report['validation_passed']}")
    print(f"- missing files: {len(missing)}")
    if report["maskable_ppo_vs_random_valid"]:
        delta = report["maskable_ppo_vs_random_valid"]["ppo_reward_minus_random_valid"]
        print(f"- ppo reward minus random_valid: {delta:.6f}")


if __name__ == "__main__":
    main()
