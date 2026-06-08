import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare prediction models and baselines.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {path}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def add_row(rows: list[dict[str, Any]], model: str, metrics: dict[str, Any], note: str) -> None:
    rows.append(
        {
            "model": model,
            "MAE_util": float(metrics["MAE_util"]),
            "RMSE_util": float(metrics["RMSE_util"]),
            "MAE_load_norm": float(metrics["MAE_load_norm"]),
            "RMSE_load_norm": float(metrics["RMSE_load_norm"]),
            "Precision": float(metrics["Precision"]),
            "Recall": float(metrics["Recall"]),
            "F1": float(metrics["F1"]),
            "note": note,
        }
    )


def save_metric_bars(df: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(df))
    width = 0.36
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, df["MAE_util"], width, label="MAE_util")
    plt.bar(x + width / 2, df["RMSE_util"], width, label="RMSE_util")
    plt.xticks(x, df["model"], rotation=30, ha="right")
    plt.ylabel("error")
    plt.title("Prediction model utilization error comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_f1_bars(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    plt.bar(df["model"], df["F1"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("F1")
    plt.title("Congestion prediction F1 comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_markdown(df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Prediction Model Comparison",
        "",
        "| Model | MAE_util | RMSE_util | MAE_load_norm | RMSE_load_norm | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.MAE_util:.6f} | {row.RMSE_util:.6f} | "
            f"{row.MAE_load_norm:.6f} | {row.RMSE_load_norm:.6f} | "
            f"{row.Precision:.6f} | {row.Recall:.6f} | {row.F1:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Last 和 HA 是无训练朴素基线。",
            "- GRU-only 和 LSTM-only 使用时间序列，但不使用 edge graph 图结构。",
            "- UGGRU 同时使用 edge graph 图结构和时间序列。",
            "- UGGRU + MC Dropout 在 UGGRU 基础上额外提供不确定性和风险排序能力。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    inputs = {
        "simple": results_dir / "simple_baselines_metrics.json",
        "gru": results_dir / "gru_baseline_test_metrics.json",
        "lstm": results_dir / "lstm_baseline_test_metrics.json",
        "summary": results_dir / "prediction_summary_metrics.json",
        "mc": results_dir / "mc_dropout_metrics.json",
    }
    for path in inputs.values():
        require_file(path)

    simple = load_json(inputs["simple"])
    gru = load_json(inputs["gru"])
    lstm = load_json(inputs["lstm"])
    summary = load_json(inputs["summary"])
    mc = load_json(inputs["mc"])

    rows: list[dict[str, Any]] = []
    add_row(rows, "Last", simple["Last"], "No-training persistence baseline")
    add_row(rows, "HA", simple["HA"], "No-training historical average baseline")
    add_row(rows, "GRU-only", gru, "Temporal baseline without edge graph")
    add_row(rows, "LSTM-only", lstm, "Temporal baseline without edge graph")

    uggru_metrics = {}
    uggru_metrics.update(summary["uggru_test_regression"])
    uggru_metrics.update(summary["val_to_test_threshold"])
    add_row(rows, "UGGRU", uggru_metrics, "Graph and temporal model, threshold selected on validation")
    add_row(rows, "UGGRU + MC Dropout", mc, "UGGRU with MC Dropout uncertainty")

    df = pd.DataFrame(rows)
    csv_path = results_dir / "prediction_model_comparison.csv"
    json_path = results_dir / "prediction_model_comparison.json"
    md_path = results_dir / "prediction_model_comparison.md"
    mae_rmse_path = results_dir / "prediction_model_comparison_mae_rmse.png"
    f1_path = results_dir / "prediction_model_comparison_f1.png"

    df.to_csv(csv_path, index=False, float_format="%.9g")
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin({"models": rows}), file, indent=2, ensure_ascii=False)
    write_markdown(df, md_path)
    save_metric_bars(df, mae_rmse_path)
    save_f1_bars(df, f1_path)

    print("Prediction model comparison:")
    for row in rows:
        print(
            f"- {row['model']}: MAE_util={row['MAE_util']:.9g}, "
            f"RMSE_util={row['RMSE_util']:.9g}, F1={row['F1']:.9g}"
        )
    print("Output files:")
    for path in [csv_path, json_path, md_path, mae_rmse_path, f1_path]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
