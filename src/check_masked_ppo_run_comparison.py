import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from masked_ppo_utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MaskedPPO run comparison outputs.")
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = RESULTS_DIR / f"masked_ppo_run_comparison_topo144_k{args.k}.csv"
    json_path = RESULTS_DIR / f"masked_ppo_run_comparison_topo144_k{args.k}.json"
    figure_paths = [
        RESULTS_DIR / "masked_ppo_run_reward_comparison.png",
        RESULTS_DIR / "masked_ppo_run_cost_comparison.png",
        RESULTS_DIR / "masked_ppo_run_congestion_comparison.png",
        RESULTS_DIR / "masked_ppo_run_gap_to_min_raw_cost.png",
    ]
    missing = [str(path) for path in [csv_path, json_path, *figure_paths] if not path.exists()]
    report = {
        "missing_files": missing,
        "values_finite": False,
        "ppo_invalid_actions_zero": False,
        "ppo_runs_better_than_random_valid": {},
        "ppo_runs_better_than_shortest": {},
        "best_by_mean_reward": None,
        "best_by_mean_raw_cost": None,
        "best_by_delta_congestion_count": None,
        "continue100k_reward_improvement_vs_mid": None,
        "lr1e4_reward_improvement_vs_continue100k": None,
        "lr1e4_recommendation": None,
        "recommendation": None,
        "validation_passed": False,
    }

    if not csv_path.exists():
        write_json(RESULTS_DIR / f"masked_ppo_run_comparison_check_topo144_k{args.k}.json", report)
        raise FileNotFoundError(f"Missing comparison CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number]).columns
    report["values_finite"] = bool(np.isfinite(df[numeric].to_numpy()).all())
    ppo = df[df["policy"] == "maskable_ppo"].copy()
    report["ppo_invalid_actions_zero"] = bool((ppo["invalid_action_count"].astype(int) == 0).all()) if not ppo.empty else False

    random_reward = float(df[df["policy"] == "random_valid"].iloc[0]["mean_reward"]) if (df["policy"] == "random_valid").any() else None
    shortest_reward = float(df[df["policy"] == "shortest"].iloc[0]["mean_reward"]) if (df["policy"] == "shortest").any() else None
    for row in ppo.itertuples(index=False):
        label = str(row.model_label)
        if random_reward is not None:
            report["ppo_runs_better_than_random_valid"][label] = bool(row.mean_reward > random_reward)
        if shortest_reward is not None:
            report["ppo_runs_better_than_shortest"][label] = bool(row.mean_reward > shortest_reward)

    if not ppo.empty:
        report["best_by_mean_reward"] = str(ppo.loc[ppo["mean_reward"].idxmax(), "model_label"])
        report["best_by_mean_raw_cost"] = str(ppo.loc[ppo["mean_raw_cost"].idxmin(), "model_label"])
        report["best_by_delta_congestion_count"] = str(ppo.loc[ppo["mean_delta_congestion_count"].idxmin(), "model_label"])
        labels = set(ppo["model_label"].astype(str))
        if {"masked_ppo_mid", "masked_ppo_continue100k"}.issubset(labels):
            mid = float(ppo[ppo["model_label"] == "masked_ppo_mid"].iloc[0]["mean_reward"])
            cont = float(ppo[ppo["model_label"] == "masked_ppo_continue100k"].iloc[0]["mean_reward"])
            improvement = cont - mid
            report["continue100k_reward_improvement_vs_mid"] = improvement
            if improvement > 0.3:
                report["recommendation"] = "continue100k improved clearly over mid; continuing toward 200k is reasonable."
            elif improvement < 0.1:
                report["recommendation"] = "continue100k improvement is small; tune hyperparameters before a 200k run."
            else:
                report["recommendation"] = "continue100k improved modestly; consider ent_coef comparison before 200k."
        else:
            report["recommendation"] = "Missing mid or continue100k; compare available runs only."

        if {"masked_ppo_continue100k", "masked_ppo_lr1e4"}.issubset(labels):
            cont = float(ppo[ppo["model_label"] == "masked_ppo_continue100k"].iloc[0]["mean_reward"])
            lr1e4 = float(ppo[ppo["model_label"] == "masked_ppo_lr1e4"].iloc[0]["mean_reward"])
            lr1e4_improvement = lr1e4 - cont
            report["lr1e4_reward_improvement_vs_continue100k"] = lr1e4_improvement
            if lr1e4_improvement > 0.3:
                report["lr1e4_recommendation"] = "lr1e4 improved clearly over continue100k; continuing lr=1e-4 toward 200k is reasonable."
            elif lr1e4_improvement > 0.1:
                report["lr1e4_recommendation"] = "lr1e4 improved modestly; consider another 50k continuation before a direct 200k run."
            elif lr1e4_improvement >= 0:
                report["lr1e4_recommendation"] = "lr1e4 improvement is small; do not continue directly to 200k."
            else:
                report["lr1e4_recommendation"] = "lr1e4 is below continue100k; keep continue100k as the best PPO run."
            report["recommendation"] = report["lr1e4_recommendation"]

    report["validation_passed"] = bool(not missing and report["values_finite"] and report["ppo_invalid_actions_zero"])
    report_path = RESULTS_DIR / f"masked_ppo_run_comparison_check_topo144_k{args.k}.json"
    write_json(report_path, report)

    print("MaskedPPO run comparison check complete:")
    print(f"- report: {report_path}")
    print(f"- validation passed: {report['validation_passed']}")
    print(f"- best_by_mean_reward: {report['best_by_mean_reward']}")
    print(f"- recommendation: {report['recommendation']}")


if __name__ == "__main__":
    main()
