import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from masked_ppo_utils import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple MaskedPPO runs with heuristic baselines.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tags", nargs="+", default=["masked_ppo_mid", "masked_ppo_continue100k", "masked_ppo_ent002"])
    return parser.parse_args()


def load_eval(tag: str, k: int) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{tag}_eval_topo144_k{k}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["source_tag"] = tag
    return df


def save_bar(df: pd.DataFrame, x: str, y: str, path: Path, ylabel: str) -> None:
    plt.figure(figsize=(10, 4.8))
    plt.bar(df[x], df[y])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    ppo_rows = []
    baseline_rows = []

    for tag in args.tags:
        df = load_eval(tag, args.k)
        if df is None:
            missing.append(tag)
            continue
        ppo = df[df["policy"] == "maskable_ppo"].copy()
        if not ppo.empty:
            ppo["model_label"] = tag
            ppo_rows.append(ppo)
        if not baseline_rows:
            baseline = df[df["policy"] != "maskable_ppo"].copy()
            baseline["model_label"] = baseline["policy"]
            baseline_rows.append(baseline)

    if not ppo_rows and not baseline_rows:
        raise FileNotFoundError("No evaluation files found for requested tags.")

    combined = pd.concat([*ppo_rows, *baseline_rows], ignore_index=True)
    min_raw = combined[combined["policy"] == "min_raw_cost"].iloc[0]
    shortest = combined[combined["policy"] == "shortest"].iloc[0]
    combined["reward_gap_to_min_raw_cost"] = float(min_raw["mean_reward"]) - combined["mean_reward"]
    combined["raw_cost_gap_to_min_raw_cost"] = combined["mean_raw_cost"] - float(min_raw["mean_raw_cost"])
    combined["delta_congestion_count_gap_to_min_raw_cost"] = (
        combined["mean_delta_congestion_count"] - float(min_raw["mean_delta_congestion_count"])
    )
    combined["reward_improvement_vs_shortest"] = combined["mean_reward"] - float(shortest["mean_reward"])
    combined["raw_cost_reduction_vs_shortest"] = float(shortest["mean_raw_cost"]) - combined["mean_raw_cost"]
    combined["delta_mlu_reduction_vs_shortest"] = float(shortest["mean_delta_mlu"]) - combined["mean_delta_mlu"]
    combined["delta_congestion_count_reduction_vs_shortest"] = (
        float(shortest["mean_delta_congestion_count"]) - combined["mean_delta_congestion_count"]
    )

    ppo_temp = combined[combined["policy"] == "maskable_ppo"].set_index("model_label")
    run_pair_deltas = {}
    for left, right in [
        ("masked_ppo_lr1e4", "masked_ppo_continue100k"),
        ("masked_ppo_continue100k", "masked_ppo_mid"),
        ("masked_ppo_lr1e4", "masked_ppo_mid"),
    ]:
        if left in ppo_temp.index and right in ppo_temp.index:
            run_pair_deltas[f"{left}_vs_{right}"] = {
                "mean_reward_delta": float(ppo_temp.loc[left, "mean_reward"] - ppo_temp.loc[right, "mean_reward"]),
                "mean_raw_cost_delta": float(ppo_temp.loc[left, "mean_raw_cost"] - ppo_temp.loc[right, "mean_raw_cost"]),
                "mean_delta_congestion_count_delta": float(
                    ppo_temp.loc[left, "mean_delta_congestion_count"] - ppo_temp.loc[right, "mean_delta_congestion_count"]
                ),
            }
    comparison_cols = [
        "model_label",
        "source_tag",
        "policy",
        "mean_reward",
        "mean_raw_cost",
        "mean_post_mlu",
        "mean_delta_mlu",
        "mean_delta_congestion_count",
        "mean_risk_score",
        "mean_cong_prob",
        "reward_gap_to_min_raw_cost",
        "raw_cost_gap_to_min_raw_cost",
        "delta_congestion_count_gap_to_min_raw_cost",
        "reward_improvement_vs_shortest",
        "raw_cost_reduction_vs_shortest",
        "delta_mlu_reduction_vs_shortest",
        "delta_congestion_count_reduction_vs_shortest",
        "invalid_action_count",
    ]
    for col in comparison_cols:
        if col not in combined.columns:
            combined[col] = ""
    out = combined[comparison_cols].copy()

    csv_path = RESULTS_DIR / f"masked_ppo_run_comparison_topo144_k{args.k}.csv"
    json_path = RESULTS_DIR / f"masked_ppo_run_comparison_topo144_k{args.k}.json"
    out.to_csv(csv_path, index=False)
    payload = {
        "missing_tags": missing,
        "rows": out.to_dict(orient="records"),
        "available_tags": [tag for tag in args.tags if tag not in missing],
        "run_pair_deltas": run_pair_deltas,
        "skipped_full_eval": ["masked_ppo_ent002"] if "masked_ppo_ent002" not in args.tags else [],
    }
    write_json(json_path, payload)

    ppo_only = out[out["policy"] == "maskable_ppo"].copy()
    if not ppo_only.empty:
        save_bar(ppo_only, "model_label", "mean_reward", RESULTS_DIR / "masked_ppo_run_reward_comparison.png", "mean reward")
        save_bar(ppo_only, "model_label", "mean_raw_cost", RESULTS_DIR / "masked_ppo_run_cost_comparison.png", "mean raw cost")
        save_bar(
            ppo_only,
            "model_label",
            "mean_delta_congestion_count",
            RESULTS_DIR / "masked_ppo_run_congestion_comparison.png",
            "mean delta congestion count",
        )
        save_bar(
            ppo_only,
            "model_label",
            "reward_gap_to_min_raw_cost",
            RESULTS_DIR / "masked_ppo_run_gap_to_min_raw_cost.png",
            "reward gap to min_raw_cost",
        )

    print("MaskedPPO run comparison complete:")
    print(f"- csv: {csv_path}")
    print(f"- json: {json_path}")
    print(f"- missing tags: {missing}")


if __name__ == "__main__":
    main()
