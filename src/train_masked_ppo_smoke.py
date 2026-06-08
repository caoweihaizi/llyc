import argparse
import csv
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from leo_routing_env import LeoRoutingEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "runs" / "masked_ppo_topo144_k5_smoke"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


class FixedEpisodeOptionsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_samples: int | None, random_start: bool):
        super().__init__(env)
        self.max_samples = max_samples
        self.random_start = random_start

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        merged_options = dict(options or {})
        if self.max_samples is not None:
            merged_options.setdefault("max_samples", self.max_samples)
        merged_options.setdefault("random_start", self.random_start)
        return self.env.reset(seed=seed, options=merged_options)

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()


class CsvTrainingCallback(BaseCallback):
    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = log_path
        self.rows: list[dict[str, float]] = []

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_rollout_end(self) -> None:
        row = {"num_timesteps": int(self.num_timesteps)}
        for key in ["train/approx_kl", "train/entropy_loss", "train/policy_gradient_loss", "train/value_loss", "train/loss"]:
            value = self.logger.name_to_value.get(key)
            if value is not None:
                row[key.replace("/", "_")] = float(value)
        self.rows.append(row)
        with self.log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=sorted({k for r in self.rows for k in r}))
            writer.writeheader()
            writer.writerows(self.rows)

    def _on_step(self) -> bool:
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke train MaskablePPO on LeoRoutingEnv.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--clip_range", type=float, default=0.2)
    return parser.parse_args()


def print_versions() -> dict[str, Any]:
    import gymnasium
    import stable_baselines3
    import sb3_contrib

    version_info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
    }
    print("Environment versions:")
    for key, value in version_info.items():
        print(f"- {key}: {value}")
    if torch.cuda.is_available():
        print(f"- GPU: {torch.cuda.get_device_name(0)}")
    return version_info


def make_env(k: int, max_samples: int, seed: int) -> gym.Env:
    env = LeoRoutingEnv(k=k, seed=seed)
    return FixedEpisodeOptionsWrapper(env, max_samples=max_samples, random_start=True)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    args = parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    version_info = print_versions()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = make_env(args.k, args.max_samples, args.seed)
    obs, info = env.reset(seed=args.seed)
    print("LeoRoutingEnv smoke settings:")
    print(f"- observation shape: {obs.shape}")
    print(f"- action_space: Discrete({env.action_space.n})")
    print(f"- initial action mask: {env.action_masks()}")
    print(f"- reset info: {info}")

    policy_kwargs = {"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "activation_fn": torch.nn.Tanh}
    tensorboard_log = RUN_DIR / "tensorboard"
    if importlib.util.find_spec("tensorboard") is None:
        print("Warning: tensorboard is not installed; SB3 tensorboard logging is disabled for this smoke run.")
        tensorboard_log_arg = None
    else:
        tensorboard_log_arg = str(tensorboard_log)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=tensorboard_log_arg,
    )

    actual_device = str(model.device)
    print(f"- SB3 selected device: {actual_device}")
    if actual_device == "cuda":
        print("Note: SB3 may warn that MlpPolicy can be faster on CPU; this smoke run keeps the requested device.")

    callback = CsvTrainingCallback(RUN_DIR / "training_log.csv")
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)

    model_path = RUN_DIR / "maskable_ppo_smoke_model.zip"
    model.save(model_path)

    train_config = {
        **vars(args),
        "device_used": actual_device,
        "observation_shape": list(obs.shape),
        "action_space_n": env.action_space.n,
        "version_info": version_info,
        "policy": "MlpPolicy",
        "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "activation_fn": "torch.nn.Tanh"},
        "tensorboard_log": tensorboard_log_arg,
        "model_path": str(model_path),
    }
    config_path = RUN_DIR / "train_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(train_config), file, indent=2, ensure_ascii=False)

    summary = {
        "success": True,
        "timesteps": args.timesteps,
        "max_samples": args.max_samples,
        "device_used": actual_device,
        "model_path": str(model_path),
        "train_config_path": str(config_path),
        "training_log_path": str(RUN_DIR / "training_log.csv"),
        "note": "Smoke training only; this is not a formal PPO experiment.",
    }
    summary_path = RESULTS_DIR / f"masked_ppo_smoke_train_summary_topo144_k{args.k}.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(summary), file, indent=2, ensure_ascii=False)

    print("MaskablePPO smoke training complete:")
    print(f"- model: {model_path}")
    print(f"- train_config: {config_path}")
    print(f"- training_log: {RUN_DIR / 'training_log.csv'}")
    print(f"- summary: {summary_path}")


if __name__ == "__main__":
    main()
