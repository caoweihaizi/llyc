import argparse
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from masked_ppo_utils import choose_heuristic_action, make_env, parse_optional_int, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full evaluate MaskablePPO and heuristic routing policies.")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_samples", type=str, default="none")
    parser.add_argument("--tag", type=str, default="masked_ppo_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_progress", type=str_to_bool, default=True)
    parser.add_argument("--progress_print_freq", type=int, default=5000)
    parser.add_argument("--tensorboard_eval", type=str_to_bool, default=True)
    parser.add_argument("--progress_log", type=Path, default=None)
    return parser.parse_args()


def log_line(message: str, log_file, flush: bool = True) -> None:
    print(message, flush=flush)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def make_writer(enabled: bool, tb_dir: Path):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(tb_dir))
    except Exception as exc:
        print(f"Warning: TensorBoard eval writer failed to initialize: {exc}", flush=True)
        return None


def progress_rollout(
    env,
    policy_name: str,
    action_fn,
    seed: int,
    total_steps: int,
    progress_print_freq: int,
    show_progress: bool,
    log_file,
    writer,
) -> dict:
    obs, reset_info = env.reset(seed=seed)
    done = False
    start_time = time.time()
    last_print = 0
    steps = 0
    total_reward = 0.0
    invalid_action_count = 0
    zero_hop_count = 0
    action_counts = np.zeros(env.action_space.n, dtype=np.int64)
    sums = {
        "raw_cost": 0.0,
        "delay": 0.0,
        "post_mlu": 0.0,
        "delta_mlu": 0.0,
        "delta_congestion_count": 0.0,
        "risk_score": 0.0,
        "cong_prob": 0.0,
    }

    log_line(f"[Eval Policy Start] policy={policy_name}, total_steps={total_steps}", log_file)
    while not done:
        action = int(action_fn(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
        total_reward += float(reward)
        sums["raw_cost"] += float(info["raw_cost"])
        sums["delay"] += float(info["path_delay_ms_sum"])
        sums["post_mlu"] += float(info["post_mlu"])
        sums["delta_mlu"] += float(info["delta_mlu"])
        sums["delta_congestion_count"] += float(info["delta_congestion_count"])
        sums["risk_score"] += float(info["path_risk_score_max"])
        sums["cong_prob"] += float(info["path_cong_prob_max"])
        invalid_action_count += int(info["invalid_action"])
        zero_hop_count += int(info["is_zero_hop"])
        action_counts[action] += 1

        should_print = show_progress and progress_print_freq > 0 and (steps - last_print >= progress_print_freq or done)
        if should_print:
            last_print = steps
            elapsed = max(time.time() - start_time, 1e-9)
            fps = steps / elapsed
            percent = min(100.0, 100.0 * steps / max(total_steps, 1))
            eta = max(0.0, (total_steps - steps) / max(fps, 1e-9))
            running_mean_reward = total_reward / steps
            running_mean_raw_cost = sums["raw_cost"] / steps
            running_mean_delta_congestion = sums["delta_congestion_count"] / steps
            message = (
                "[Eval Progress] "
                f"policy={policy_name} steps={steps}/{total_steps} percent={percent:.2f}% "
                f"elapsed={elapsed:.1f}s fps={fps:.2f} eta={eta:.1f}s "
                f"running_mean_reward={running_mean_reward:.6f} "
                f"running_mean_raw_cost={running_mean_raw_cost:.6f} "
                f"running_mean_delta_congestion_count={running_mean_delta_congestion:.6f}"
            )
            log_line(message, log_file)
            if writer is not None:
                try:
                    prefix_progress = f"eval_progress/{policy_name}"
                    prefix_metrics = f"eval_metrics/{policy_name}"
                    writer.add_scalar(f"{prefix_progress}/steps_done", steps, steps)
                    writer.add_scalar(f"{prefix_progress}/percent", percent, steps)
                    writer.add_scalar(f"{prefix_progress}/fps", fps, steps)
                    writer.add_scalar(f"{prefix_progress}/elapsed_seconds", elapsed, steps)
                    writer.add_scalar(f"{prefix_progress}/eta_seconds", eta, steps)
                    writer.add_scalar(f"{prefix_metrics}/running_mean_reward", running_mean_reward, steps)
                    writer.add_scalar(f"{prefix_metrics}/running_mean_raw_cost", running_mean_raw_cost, steps)
                    writer.add_scalar(f"{prefix_metrics}/running_mean_delta_congestion_count", running_mean_delta_congestion, steps)
                    writer.add_scalar(f"{prefix_metrics}/running_mean_post_mlu", sums["post_mlu"] / steps, steps)
                    writer.add_scalar(f"{prefix_metrics}/running_mean_delta_mlu", sums["delta_mlu"] / steps, steps)
                    writer.add_scalar(f"{prefix_metrics}/running_mean_risk_score", sums["risk_score"] / steps, steps)
                    writer.add_scalar(f"{prefix_metrics}/running_mean_cong_prob", sums["cong_prob"] / steps, steps)
                except Exception as exc:
                    log_line(f"Warning: TensorBoard eval write failed: {exc}", log_file)

    elapsed = max(time.time() - start_time, 1e-9)
    row = {
        "policy": policy_name,
        "total_steps": steps,
        "steps": steps,
        "evaluated_samples": int(reset_info["episode_end_sample"] - reset_info["start_sample"]),
        "evaluated_od_pairs": steps,
        "mean_reward": total_reward / steps,
        "total_reward": total_reward,
        "mean_raw_cost": sums["raw_cost"] / steps,
        "mean_delay": sums["delay"] / steps,
        "mean_post_mlu": sums["post_mlu"] / steps,
        "mean_delta_mlu": sums["delta_mlu"] / steps,
        "mean_delta_congestion_count": sums["delta_congestion_count"] / steps,
        "mean_risk_score": sums["risk_score"] / steps,
        "mean_cong_prob": sums["cong_prob"] / steps,
        "invalid_action_count": int(invalid_action_count),
        "zero_hop_count": int(zero_hop_count),
        "action_id_distribution": {str(i): int(v) for i, v in enumerate(action_counts.tolist())},
    }
    log_line(
        "[Eval Policy Done] "
        f"policy={policy_name}, mean_reward={row['mean_reward']:.6f}, "
        f"mean_raw_cost={row['mean_raw_cost']:.6f}, "
        f"mean_delta_congestion_count={row['mean_delta_congestion_count']:.6f}, "
        f"elapsed={elapsed:.1f}s",
        log_file,
    )
    return row


def save_figures(df: pd.DataFrame, tag: str) -> list[Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    specs = [
        ("mean_reward", "mean reward", f"{tag}_policy_reward_comparison.png"),
        ("mean_raw_cost", "mean raw cost", f"{tag}_policy_cost_comparison.png"),
        ("mean_post_mlu", "mean post MLU", f"{tag}_policy_mlu_comparison.png"),
        ("mean_delta_congestion_count", "mean delta congestion count", f"{tag}_policy_congestion_comparison.png"),
    ]
    for column, ylabel, filename in specs:
        path = RESULTS_DIR / filename
        plt.figure(figsize=(9, 4.5))
        plt.bar(df["policy"], df[column])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)

    action_rows = []
    for row in df.itertuples(index=False):
        dist = row.action_id_distribution
        if isinstance(dist, str):
            dist = json.loads(dist.replace("'", '"'))
        total = sum(dist.values()) or 1
        for action_id, count in dist.items():
            action_rows.append({"policy": row.policy, "action": int(action_id), "ratio": count / total})
    action_df = pd.DataFrame(action_rows)
    pivot = action_df.pivot(index="policy", columns="action", values="ratio").fillna(0.0)
    path = RESULTS_DIR / f"{tag}_action_distribution.png"
    pivot.plot(kind="bar", stacked=True, figsize=(10, 4.8))
    plt.ylabel("action ratio")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Missing model: {args.model_path}")
    max_samples = parse_optional_int(args.max_samples)
    model = MaskablePPO.load(args.model_path)
    rng = np.random.default_rng(args.seed)
    policies = ["maskable_ppo", "random_valid", "shortest", "min_raw_cost", "min_risk", "min_cong_prob"]
    rows = []
    eval_run_dir = PROJECT_ROOT / "runs" / f"{args.tag}_eval_topo144_k{args.k}"
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = args.progress_log or eval_run_dir / "progress.log"
    tb_dir = eval_run_dir / "tensorboard"
    writer = make_writer(args.tensorboard_eval, tb_dir)
    overall_start = time.time()

    probe_env = make_env(args.k, max_samples=max_samples, seed=args.seed, random_start=False)
    _, probe_info = probe_env.reset(seed=args.seed)
    total_steps_per_policy = int((probe_info["episode_end_sample"] - probe_info["start_sample"]) * probe_info["gw_pair_count"])
    estimated_total_steps = total_steps_per_policy * len(policies)

    with progress_log_path.open("w", encoding="utf-8") as log_file:
        log_line(
            "[Eval Start] "
            f"tag={args.tag}, max_samples={max_samples}, total_policies={len(policies)}, "
            f"estimated_total_steps={estimated_total_steps}, tensorboard_eval={args.tensorboard_eval}",
            log_file,
        )
        for policy in policies:
            env = make_env(args.k, max_samples=max_samples, seed=args.seed, random_start=False)
            if policy == "maskable_ppo":
                def action_fn(obs, env=env, model=model):
                    action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
                    return int(action)
            else:
                def action_fn(obs, env=env, policy=policy, rng=rng):
                    return choose_heuristic_action(env, policy, rng)
            row = progress_rollout(
                env=env,
                policy_name=policy,
                action_fn=action_fn,
                seed=args.seed,
                total_steps=total_steps_per_policy,
                progress_print_freq=args.progress_print_freq,
                show_progress=args.show_progress,
                log_file=log_file,
                writer=writer,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / f"{args.tag}_eval_topo144_k{args.k}.csv"
    json_path = RESULTS_DIR / f"{args.tag}_eval_topo144_k{args.k}.json"
    df.to_csv(csv_path, index=False)
    write_json(json_path, rows)
    fig_paths = save_figures(df, args.tag)

    ppo = df[df["policy"] == "maskable_ppo"].iloc[0]
    random_valid = df[df["policy"] == "random_valid"].iloc[0]
    shortest = df[df["policy"] == "shortest"].iloc[0]
    if ppo["mean_reward"] < shortest["mean_reward"]:
        print("Warning: PPO mean_reward is below shortest. This can happen with limited training or reward exploration.")
    if ppo["mean_reward"] < random_valid["mean_reward"]:
        print("Warning: PPO mean_reward is below random_valid; inspect training stability and reward scaling.")

    if writer is not None:
        writer.flush()
        writer.close()
    total_elapsed = time.time() - overall_start
    with progress_log_path.open("a", encoding="utf-8") as log_file:
        log_line(
            "[Eval Done] "
            f"tag={args.tag}, total_elapsed={total_elapsed:.1f}s, output_csv={csv_path}, output_json={json_path}",
            log_file,
        )
    print("Full MaskablePPO evaluation complete:", flush=True)
    print(f"- csv: {csv_path}", flush=True)
    print(f"- json: {json_path}", flush=True)
    print(f"- progress_log: {progress_log_path}", flush=True)
    if args.tensorboard_eval:
        print(f"- TensorBoard: tensorboard --logdir {tb_dir}", flush=True)
    for path in fig_paths:
        print(f"- figure: {path}", flush=True)


if __name__ == "__main__":
    main()
