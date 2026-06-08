from __future__ import annotations

import argparse
import csv
import importlib.util
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn

from masked_ppo_utils import make_env, parse_optional_int, rollout_maskable_ppo, write_json


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


class TrainAndEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq: int,
        best_model_path: Path,
        train_log_path: Path,
        eval_log_path: Path,
        requested_timesteps: int,
        progress_print_freq: int,
        show_progress: bool,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.best_model_path = best_model_path
        self.train_log_path = train_log_path
        self.eval_log_path = eval_log_path
        self.requested_timesteps = int(requested_timesteps)
        self.progress_print_freq = int(progress_print_freq)
        self.show_progress = bool(show_progress)
        self.best_eval_mean_reward = -np.inf
        self.latest_eval_reward: float | None = None
        self.latest_eval_raw_cost: float | None = None
        self.latest_eval_delta_congestion: float | None = None
        self.train_rows: list[dict] = []
        self.eval_rows: list[dict] = []
        self.last_eval_timestep = 0
        self.last_progress_print_step = 0
        self.start_num_timesteps = 0
        self.start_time = time.time()

    def _write_rows(self, path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        fields = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _on_training_start(self) -> None:
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.start_num_timesteps = int(self.num_timesteps)
        self.last_progress_print_step = self.start_num_timesteps
        self.start_time = time.time()

    def _on_rollout_end(self) -> None:
        row = {"num_timesteps": int(self.num_timesteps)}
        for key, value in self.logger.name_to_value.items():
            if key.startswith("train/") or key.startswith("rollout/"):
                try:
                    row[key.replace("/", "_")] = float(value)
                except (TypeError, ValueError):
                    pass
        self.train_rows.append(row)
        self._write_rows(self.train_log_path, self.train_rows)

    def _run_eval(self) -> None:
        row = rollout_maskable_ppo(self.model, self.eval_env, seed=42, policy_name="maskable_ppo")
        row["num_timesteps"] = int(self.num_timesteps)
        self.eval_rows.append(row)
        self._write_rows(self.eval_log_path, self.eval_rows)
        self.latest_eval_reward = float(row["mean_reward"])
        self.latest_eval_raw_cost = float(row["mean_raw_cost"])
        self.latest_eval_delta_congestion = float(row["mean_delta_congestion_count"])
        if row["mean_reward"] > self.best_eval_mean_reward:
            self.best_eval_mean_reward = float(row["mean_reward"])
            self.model.save(self.best_model_path)
        print(
            "[PPO Eval] "
            f"timesteps={self.num_timesteps}, "
            f"mean_reward={row['mean_reward']:.6f}, "
            f"mean_raw_cost={row['mean_raw_cost']:.6f}, "
            f"mean_delta_congestion_count={row['mean_delta_congestion_count']:.6f}, "
            f"best={self.best_eval_mean_reward:.6f}"
        )

    def _on_step(self) -> bool:
        if self.show_progress and self.progress_print_freq > 0:
            progressed = int(self.num_timesteps) - self.start_num_timesteps
            if int(self.num_timesteps) - self.last_progress_print_step >= self.progress_print_freq:
                self.last_progress_print_step = int(self.num_timesteps)
                elapsed = max(time.time() - self.start_time, 1e-9)
                fps = progressed / elapsed
                percent = min(100.0, 100.0 * progressed / max(self.requested_timesteps, 1))
                latest_eval = "NA" if self.latest_eval_reward is None else f"{self.latest_eval_reward:.6f}"
                print(
                    "[PPO Progress] "
                    f"steps={progressed}/{self.requested_timesteps}, "
                    f"percent={percent:.1f}%, "
                    f"elapsed={elapsed:.1f}s, "
                    f"fps={fps:.2f}, "
                    f"latest_eval_reward={latest_eval}, "
                    f"best_eval_reward={self.best_eval_mean_reward:.6f}"
                )
        if self.eval_freq > 0 and self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self.last_eval_timestep = int(self.num_timesteps)
            self._run_eval()
        return True

    def _on_training_end(self) -> None:
        if not self.eval_rows:
            self._run_eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on LeoRoutingEnv.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--max_samples", type=str, default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--eval_max_samples", type=int, default=500)
    parser.add_argument("--run_name", type=str, default="masked_ppo_full")
    parser.add_argument("--resume_model", type=Path, default=None)
    parser.add_argument("--reward_mode", type=str, default="relative_to_shortest")
    parser.add_argument("--show_progress", type=str_to_bool, default=True)
    parser.add_argument("--progress_bar", type=str_to_bool, default=True)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--progress_print_freq", type=int, default=5000)
    return parser.parse_args()


def version_info() -> dict:
    import gymnasium
    import stable_baselines3
    import sb3_contrib

    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
    }
    print("Training environment:")
    for key, value in info.items():
        print(f"- {key}: {value}")
    if torch.cuda.is_available():
        print(f"- GPU: {torch.cuda.get_device_name(0)}")
    return info


def dependency_status() -> dict[str, bool]:
    status = {
        "tqdm": importlib.util.find_spec("tqdm") is not None,
        "rich": importlib.util.find_spec("rich") is not None,
        "tensorboard": importlib.util.find_spec("tensorboard") is not None,
    }
    print("Progress/logging dependencies:")
    for name, available in status.items():
        print(f"- {name}: {available}")
    missing = [name for name, available in status.items() if not available]
    if missing:
        print(f"Warning: missing optional progress/logging packages: {missing}")
    return status


def main() -> None:
    args = parse_args()
    max_samples = parse_optional_int(args.max_samples)
    run_dir = PROJECT_ROOT / "runs" / f"{args.run_name}_topo144_k{args.k}"
    run_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    versions = version_info()
    dep_status = dependency_status()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_env = make_env(args.k, max_samples=max_samples, seed=args.seed, random_start=True, reward_mode=args.reward_mode)
    eval_env = make_env(args.k, max_samples=args.eval_max_samples, seed=args.seed, random_start=False, reward_mode=args.reward_mode)
    obs, reset_info = train_env.reset(seed=args.seed)
    print(f"- observation shape: {obs.shape}")
    print(f"- action_space: Discrete({train_env.action_space.n})")
    print(f"- train reset info: {reset_info}")

    tensorboard_arg = str(run_dir / "tensorboard") if dep_status["tensorboard"] else None
    if tensorboard_arg is None:
        print("Warning: tensorboard is not installed; tensorboard logging disabled.")

    policy_kwargs = {"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "activation_fn": torch.nn.Tanh}
    if args.resume_model is not None:
        if not args.resume_model.exists():
            raise FileNotFoundError(f"Missing resume model: {args.resume_model}")
        model = MaskablePPO.load(args.resume_model, env=train_env, device=args.device, tensorboard_log=tensorboard_arg)
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
        model.ent_coef = args.ent_coef
        model.clip_range = get_schedule_fn(args.clip_range)
        model.vf_coef = args.vf_coef
        model.max_grad_norm = args.max_grad_norm
        model.verbose = args.verbose
        print(f"Resumed model: {args.resume_model}")
        print(f"Applied resumed training overrides: learning_rate={args.learning_rate}, ent_coef={args.ent_coef}")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=args.verbose,
            seed=args.seed,
            device=args.device,
            tensorboard_log=tensorboard_arg,
        )

    actual_device = str(model.device)
    progress_bar_enabled = bool(args.progress_bar and dep_status["tqdm"] and dep_status["rich"])
    print("MaskablePPO run settings:")
    print(f"- run_name: {args.run_name}")
    print(f"- resume_model: {args.resume_model}")
    print(f"- total_timesteps: {args.timesteps}")
    print(f"- max_samples: {max_samples}")
    print(f"- device request / used: {args.device} / {actual_device}")
    print(f"- learning_rate: {args.learning_rate}")
    print(f"- ent_coef: {args.ent_coef}")
    print(f"- n_steps: {args.n_steps}")
    print(f"- batch_size: {args.batch_size}")
    print(f"- n_epochs: {args.n_epochs}")
    print(f"- progress_bar: {progress_bar_enabled}")
    print(f"- tqdm/rich/tensorboard: {dep_status}")
    print(f"- TensorBoard log dir: {tensorboard_arg}")
    print(f"- progress_print_freq: {args.progress_print_freq}")
    print(f"- SB3 selected device: {actual_device}")
    start_time = time.time()
    callback = TrainAndEvalCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        best_model_path=run_dir / "best_model.zip",
        train_log_path=run_dir / "training_log.csv",
        eval_log_path=run_dir / "eval_log.csv",
        requested_timesteps=args.timesteps,
        progress_print_freq=args.progress_print_freq,
        show_progress=args.show_progress,
    )
    reset_num_timesteps = args.resume_model is None
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        tb_log_name=args.run_name,
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=progress_bar_enabled,
    )
    wall_time = time.time() - start_time

    final_model_path = run_dir / "final_model.zip"
    model.save(final_model_path)
    if not (run_dir / "best_model.zip").exists():
        model.save(run_dir / "best_model.zip")

    config = {
        **vars(args),
        "max_samples_parsed": max_samples,
        "device_used": actual_device,
        "observation_shape": list(obs.shape),
        "action_space_n": train_env.action_space.n,
        "version_info": versions,
        "tensorboard_log": tensorboard_arg,
        "dependency_status": dep_status,
        "progress_bar_enabled": progress_bar_enabled,
        "progress_print_freq": args.progress_print_freq,
        "policy": "MlpPolicy",
        "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "activation_fn": "torch.nn.Tanh"},
    }
    write_json(run_dir / "train_config.json", config)

    summary = {
        "run_name": args.run_name,
        "wall_time_seconds": wall_time,
        "total_timesteps_requested": args.timesteps,
        "total_timesteps_actual": int(model.num_timesteps),
        "device": actual_device,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "ent_coef": args.ent_coef,
        "clip_range": args.clip_range,
        "best_eval_mean_reward": callback.best_eval_mean_reward,
        "best_model_path": str(run_dir / "best_model.zip"),
        "final_model_path": str(final_model_path),
        "training_log_path": str(run_dir / "training_log.csv"),
        "eval_log_path": str(run_dir / "eval_log.csv"),
        "tensorboard_log": tensorboard_arg,
        "tensorboard_command": f"tensorboard --logdir {run_dir / 'tensorboard'}" if tensorboard_arg else None,
    }
    summary_path = RESULTS_DIR / f"{args.run_name}_train_summary_topo144_k{args.k}.json"
    write_json(summary_path, summary)

    print("MaskablePPO training complete:")
    print(f"- actual timesteps: {int(model.num_timesteps)}")
    print(f"- wall_time_seconds: {wall_time:.2f}")
    print(f"- best eval mean reward: {callback.best_eval_mean_reward:.6f}")
    print(f"- best model: {run_dir / 'best_model.zip'}")
    print(f"- final model: {final_model_path}")
    if tensorboard_arg:
        print(f"- TensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}")
    print(f"- summary: {summary_path}")


if __name__ == "__main__":
    main()
