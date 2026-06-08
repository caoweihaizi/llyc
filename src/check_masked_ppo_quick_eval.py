import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from masked_ppo_utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


CONTINUE100K_REFERENCE = {
    "mean_reward": 2.994414,
    "mean_raw_cost": 182.261162,
    "mean_delta_congestion_count": 0.013183,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check quick MaskedPPO evaluation outputs.")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = RESULTS_DIR / f"{args.tag}_eval_topo144_k{args.k}.csv"
    json_path = RESULTS_DIR / f"{args.tag}_eval_topo144_k{args.k}.json"
    run_dir = PROJECT_ROOT / "runs" / f"{args.tag}_eval_topo144_k{args.k}"
    progress_log = run_dir / "progress.log"
    tensorboard_dir = run_dir / "tensorboard"

    missing = [str(path) for path in [csv_path, json_path, progress_log, tensorboard_dir] if not path.exists()]
    report = {
        "tag": args.tag,
        "missing_files": missing,
        "contains_all_policies": False,
        "values_finite": False,
        "maskable_ppo_invalid_action_count_zero": False,
        "maskable_ppo_metrics": {},
        "continue100k_reference": CONTINUE100K_REFERENCE,
        "recommendation": "unknown",
        "validation_passed": False,
    }

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        expected = {"maskable_ppo", "random_valid", "shortest", "min_raw_cost", "min_risk", "min_cong_prob"}
        report["contains_all_policies"] = expected.issubset(set(df["policy"].astype(str)))
        numeric = df.select_dtypes(include=[np.number]).columns
        report["values_finite"] = bool(np.isfinite(df[numeric].to_numpy()).all())
        if (df["policy"] == "maskable_ppo").any():
            ppo = df[df["policy"] == "maskable_ppo"].iloc[0]
            report["maskable_ppo_invalid_action_count_zero"] = int(ppo["invalid_action_count"]) == 0
            report["maskable_ppo_metrics"] = {
                "mean_reward": float(ppo["mean_reward"]),
                "mean_raw_cost": float(ppo["mean_raw_cost"]),
                "mean_post_mlu": float(ppo["mean_post_mlu"]),
                "mean_delta_mlu": float(ppo["mean_delta_mlu"]),
                "mean_delta_congestion_count": float(ppo["mean_delta_congestion_count"]),
                "invalid_action_count": int(ppo["invalid_action_count"]),
            }
            reward = float(ppo["mean_reward"])
            if reward < 2.7:
                report["recommendation"] = "skip_full_eval"
            elif reward <= 3.0:
                report["recommendation"] = "ask_user_for_full_eval"
            else:
                report["recommendation"] = "recommend_full_eval"

    report["validation_passed"] = bool(
        not missing
        and report["contains_all_policies"]
        and report["values_finite"]
        and report["maskable_ppo_invalid_action_count_zero"]
    )
    report_path = RESULTS_DIR / f"{args.tag}_check_topo144_k{args.k}.json"
    write_json(report_path, report)
    print("Quick eval check complete:")
    print(f"- report: {report_path}")
    print(f"- validation passed: {report['validation_passed']}")
    print(f"- recommendation: {report['recommendation']}")


if __name__ == "__main__":
    main()
