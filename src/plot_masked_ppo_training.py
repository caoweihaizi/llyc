import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MaskablePPO training and evaluation diagnostics.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, required=True)
    return parser.parse_args()


def plot_line(df: pd.DataFrame, x: str, y: str, ylabel: str, path: Path) -> bool:
    if x not in df.columns or y not in df.columns:
        return False
    plt.figure(figsize=(9, 4.5))
    plt.plot(df[x], df[y], marker="o", linewidth=1.0)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return True


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_log = args.run_dir / "training_log.csv"
    eval_log = args.run_dir / "eval_log.csv"
    summary = {"run_dir": str(args.run_dir), "generated": [], "skipped": []}

    eval_df_for_fallback = pd.read_csv(eval_log) if eval_log.exists() else None

    if train_log.exists():
        train_df = pd.read_csv(train_log)
        reward_path = RESULTS_DIR / f"{args.tag}_training_reward_curve.png"
        if plot_line(train_df, "num_timesteps", "rollout_ep_rew_mean", "rollout episode reward mean", reward_path):
            summary["generated"].append(str(reward_path))
        elif eval_df_for_fallback is not None and "mean_reward" in eval_df_for_fallback.columns:
            plt.figure(figsize=(9, 4.5))
            plt.plot(eval_df_for_fallback["num_timesteps"], eval_df_for_fallback["mean_reward"], marker="o", linewidth=1.0)
            plt.xlabel("num_timesteps")
            plt.ylabel("periodic eval mean reward")
            plt.grid(True, linewidth=0.4, alpha=0.4)
            plt.tight_layout()
            plt.savefig(reward_path, dpi=200)
            plt.close()
            summary["generated"].append(str(reward_path))
            summary["skipped"].append("training_reward_curve: rollout_ep_rew_mean not found; used periodic eval mean_reward as proxy")
        else:
            summary["skipped"].append("training_reward_curve: rollout_ep_rew_mean not found")

        loss_candidates = ["train_loss", "train_value_loss", "train_policy_gradient_loss"]
        available = [col for col in loss_candidates if col in train_df.columns]
        loss_path = RESULTS_DIR / f"{args.tag}_training_loss_curve.png"
        if available:
            plt.figure(figsize=(9, 4.5))
            for col in available:
                plt.plot(train_df["num_timesteps"], train_df[col], marker="o", linewidth=1.0, label=col)
            plt.xlabel("num_timesteps")
            plt.ylabel("loss")
            plt.legend()
            plt.grid(True, linewidth=0.4, alpha=0.4)
            plt.tight_layout()
            plt.savefig(loss_path, dpi=200)
            plt.close()
            summary["generated"].append(str(loss_path))
        else:
            summary["skipped"].append("training_loss_curve: no train loss fields found")
    else:
        summary["skipped"].append(f"missing training log: {train_log}")

    if eval_log.exists():
        eval_df = pd.read_csv(eval_log)
        eval_reward_path = RESULTS_DIR / f"{args.tag}_eval_reward_curve.png"
        if plot_line(eval_df, "num_timesteps", "mean_reward", "eval mean reward", eval_reward_path):
            summary["generated"].append(str(eval_reward_path))
        else:
            summary["skipped"].append("eval_reward_curve: mean_reward not found")

        eval_cost_path = RESULTS_DIR / f"{args.tag}_eval_raw_cost_curve.png"
        if plot_line(eval_df, "num_timesteps", "mean_raw_cost", "eval mean raw cost", eval_cost_path):
            summary["generated"].append(str(eval_cost_path))
        else:
            summary["skipped"].append("eval_raw_cost_curve: mean_raw_cost not found")
    else:
        summary["skipped"].append(f"missing eval log: {eval_log}")

    summary_path = RESULTS_DIR / f"{args.tag}_training_plot_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("MaskedPPO training plot complete:")
    for path in summary["generated"]:
        print(f"- figure: {path}")
    for item in summary["skipped"]:
        print(f"- skipped: {item}")
    print(f"- summary: {summary_path}")


if __name__ == "__main__":
    main()
