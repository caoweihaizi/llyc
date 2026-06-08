import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from masked_ppo_utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MaskedPPO full/mid training outputs.")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = PROJECT_ROOT / "runs" / f"{args.run_name}_topo144_k{args.k}"
    paths = {
        "best_model": run_dir / "best_model.zip",
        "final_model": run_dir / "final_model.zip",
        "train_config": run_dir / "train_config.json",
        "train_summary": RESULTS_DIR / f"{args.run_name}_train_summary_topo144_k{args.k}.json",
        "eval_csv": RESULTS_DIR / f"{args.tag}_eval_topo144_k{args.k}.csv",
        "eval_json": RESULTS_DIR / f"{args.tag}_eval_topo144_k{args.k}.json",
    }
    figure_paths = [
        RESULTS_DIR / f"{args.tag}_policy_reward_comparison.png",
        RESULTS_DIR / f"{args.tag}_policy_cost_comparison.png",
        RESULTS_DIR / f"{args.tag}_policy_mlu_comparison.png",
        RESULTS_DIR / f"{args.tag}_policy_congestion_comparison.png",
        RESULTS_DIR / f"{args.tag}_action_distribution.png",
        RESULTS_DIR / f"{args.tag}_training_reward_curve.png",
        RESULTS_DIR / f"{args.tag}_training_loss_curve.png",
        RESULTS_DIR / f"{args.tag}_eval_reward_curve.png",
        RESULTS_DIR / f"{args.tag}_eval_raw_cost_curve.png",
    ]

    missing = [str(path) for path in [*paths.values(), *figure_paths] if not path.exists()]
    report = {
        "missing_files": missing,
        "required_files_exist": not missing,
        "model_reload_ok": False,
        "eval_contains_all_policies": False,
        "eval_values_finite": False,
        "maskable_ppo_invalid_action_count_zero": False,
        "ppo_better_than_random_valid": False,
        "ppo_better_than_shortest": False,
        "ppo_vs_min_raw_cost": {},
        "warnings": [],
        "validation_passed": False,
    }

    if paths["best_model"].exists():
        model = MaskablePPO.load(paths["best_model"])
        report["model_reload_ok"] = model is not None

    if paths["eval_csv"].exists():
        df = pd.read_csv(paths["eval_csv"])
        expected = {"maskable_ppo", "random_valid", "shortest", "min_raw_cost", "min_risk", "min_cong_prob"}
        report["eval_contains_all_policies"] = expected.issubset(set(df["policy"].astype(str)))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        report["eval_values_finite"] = bool(np.isfinite(df[numeric_cols].to_numpy()).all())
        if report["eval_contains_all_policies"]:
            ppo = df[df["policy"] == "maskable_ppo"].iloc[0]
            random_valid = df[df["policy"] == "random_valid"].iloc[0]
            shortest = df[df["policy"] == "shortest"].iloc[0]
            min_raw = df[df["policy"] == "min_raw_cost"].iloc[0]
            report["maskable_ppo_invalid_action_count_zero"] = int(ppo["invalid_action_count"]) == 0
            report["ppo_better_than_random_valid"] = float(ppo["mean_reward"]) > float(random_valid["mean_reward"])
            report["ppo_better_than_shortest"] = float(ppo["mean_reward"]) > float(shortest["mean_reward"])
            report["ppo_vs_min_raw_cost"] = {
                "ppo_mean_reward": float(ppo["mean_reward"]),
                "min_raw_cost_mean_reward": float(min_raw["mean_reward"]),
                "reward_gap_to_min_raw_cost": float(min_raw["mean_reward"] - ppo["mean_reward"]),
                "ppo_mean_raw_cost": float(ppo["mean_raw_cost"]),
                "min_raw_cost_mean_raw_cost": float(min_raw["mean_raw_cost"]),
                "raw_cost_gap_to_min_raw_cost": float(ppo["mean_raw_cost"] - min_raw["mean_raw_cost"]),
            }
            if not report["ppo_better_than_random_valid"]:
                report["warnings"].append("PPO did not outperform random_valid.")
            if not report["ppo_better_than_shortest"]:
                report["warnings"].append("PPO did not outperform shortest; keep result as a training diagnostic, not a hard failure.")

    report["validation_passed"] = bool(
        report["required_files_exist"]
        and report["model_reload_ok"]
        and report["eval_contains_all_policies"]
        and report["eval_values_finite"]
        and report["maskable_ppo_invalid_action_count_zero"]
        and report["ppo_better_than_random_valid"]
    )
    report_path = RESULTS_DIR / f"{args.tag}_check_topo144_k{args.k}.json"
    write_json(report_path, report)

    print("MaskedPPO output check complete:")
    print(f"- report: {report_path}")
    print(f"- validation passed: {report['validation_passed']}")
    print(f"- warnings: {len(report['warnings'])}")
    if report["ppo_vs_min_raw_cost"]:
        print(f"- reward gap to min_raw_cost: {report['ppo_vs_min_raw_cost']['reward_gap_to_min_raw_cost']:.6f}")


if __name__ == "__main__":
    main()
