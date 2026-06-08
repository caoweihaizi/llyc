import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "runs" / "uggru_topo144_seq12"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

DEFAULT_TRAIN_LOG = RUN_DIR / "train_log.csv"
DEFAULT_TRAIN_CONFIG = RUN_DIR / "train_config.json"
DEFAULT_TEST_METRICS = RESULTS_DIR / "uggru_test_metrics.json"
DEFAULT_VAL_THRESHOLD = RESULTS_DIR / "uggru_val_selected_threshold_test_metrics.json"
DEFAULT_MC_METRICS = RESULTS_DIR / "mc_dropout_metrics.json"
DEFAULT_RISK_TOPK = RESULTS_DIR / "mc_dropout_risk_topk.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize UGGRU prediction and MC Dropout results for thesis reporting."
    )
    parser.add_argument("--train_log", type=Path, default=DEFAULT_TRAIN_LOG)
    parser.add_argument("--train_config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--test_metrics", type=Path, default=DEFAULT_TEST_METRICS)
    parser.add_argument("--val_threshold_metrics", type=Path, default=DEFAULT_VAL_THRESHOLD)
    parser.add_argument("--mc_metrics", type=Path, default=DEFAULT_MC_METRICS)
    parser.add_argument("--risk_topk", type=Path, default=DEFAULT_RISK_TOPK)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def extract_training_summary(train_log: pd.DataFrame) -> dict[str, Any]:
    if train_log.empty:
        raise ValueError("train_log.csv is empty.")
    if "epoch" not in train_log.columns or "val_loss" not in train_log.columns:
        raise ValueError("train_log.csv must contain epoch and val_loss columns.")

    best_idx = train_log["val_loss"].idxmin()
    best_row = train_log.loc[best_idx]
    final_row = train_log.iloc[-1]
    return {
        "total_epochs": int(len(train_log)),
        "best_epoch": int(best_row["epoch"]),
        "best_val_loss": float(best_row["val_loss"]),
        "final_train_loss": float(final_row["train_loss"]),
        "final_val_loss": float(final_row["val_loss"]),
    }


def build_summary_rows(
    config_summary: dict[str, Any],
    train_summary: dict[str, Any],
    uggru_test_summary: dict[str, Any],
    threshold_summary: dict[str, Any],
    mc_summary: dict[str, Any],
    risk_lift_summary: dict[str, Any],
) -> pd.DataFrame:
    groups = [
        ("training_config", config_summary),
        ("training_curve", train_summary),
        ("uggru_test_regression", uggru_test_summary),
        ("val_to_test_threshold", threshold_summary),
        ("mc_dropout_uncertainty", mc_summary),
        ("risk_ranking_lift", risk_lift_summary),
    ]
    rows = []
    for group, metrics in groups:
        for metric, value in metrics.items():
            rows.append({"group": group, "metric": metric, "value": value})
    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    config_summary: dict[str, Any],
    train_summary: dict[str, Any],
    uggru_test_summary: dict[str, Any],
    threshold_summary: dict[str, Any],
    mc_summary: dict[str, Any],
    risk_lift_summary: dict[str, Any],
) -> None:
    lines = [
        "# UGGRU Prediction Result Summary",
        "",
        "## 模型训练配置摘要",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
    ]
    for key, value in config_summary.items():
        lines.append(f"| {key} | {fmt(value)} |")

    lines.extend(
        [
            f"| total_epochs | {train_summary['total_epochs']} |",
            f"| best_epoch | {train_summary['best_epoch']} |",
            f"| best_val_loss | {fmt(train_summary['best_val_loss'])} |",
            f"| final_train_loss | {fmt(train_summary['final_train_loss'])} |",
            f"| final_val_loss | {fmt(train_summary['final_val_loss'])} |",
            "",
            "## 预测性能摘要",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| UGGRU MAE_util | {fmt(uggru_test_summary['MAE_util'])} |",
            f"| UGGRU RMSE_util | {fmt(uggru_test_summary['RMSE_util'])} |",
            f"| UGGRU MAE_load_norm | {fmt(uggru_test_summary['MAE_load_norm'])} |",
            f"| UGGRU RMSE_load_norm | {fmt(uggru_test_summary['RMSE_load_norm'])} |",
            f"| selected_threshold | {fmt(threshold_summary['selected_threshold'])} |",
            f"| calibrated Precision | {fmt(threshold_summary['Precision'])} |",
            f"| calibrated Recall | {fmt(threshold_summary['Recall'])} |",
            f"| calibrated F1 | {fmt(threshold_summary['F1'])} |",
            f"| predicted_positive_ratio | {pct(threshold_summary['predicted_positive_ratio'])} |",
            f"| true_positive_ratio | {pct(threshold_summary['true_positive_ratio'])} |",
            "",
            "## 不确定性与风险排序摘要",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| MC Dropout MAE_util | {fmt(mc_summary['MAE_util'])} |",
            f"| MC Dropout RMSE_util | {fmt(mc_summary['RMSE_util'])} |",
            f"| coverage_1std | {pct(mc_summary['coverage_1std'])} |",
            f"| coverage_2std | {pct(mc_summary['coverage_2std'])} |",
            f"| uncertainty_error_corr | {fmt(mc_summary['uncertainty_error_corr'])} |",
            f"| MC Precision | {fmt(mc_summary['Precision'])} |",
            f"| MC Recall | {fmt(mc_summary['Recall'])} |",
            f"| MC F1 | {fmt(mc_summary['F1'])} |",
            f"| risk_top1_congestion_rate | {pct(mc_summary['risk_top1_congestion_rate'])} |",
            f"| risk_top5_congestion_rate | {pct(mc_summary['risk_top5_congestion_rate'])} |",
            f"| risk_top10_congestion_rate | {pct(mc_summary['risk_top10_congestion_rate'])} |",
            f"| top1_lift | {fmt(risk_lift_summary['top1_lift'])}x |",
            f"| top5_lift | {fmt(risk_lift_summary['top5_lift'])}x |",
            f"| top10_lift | {fmt(risk_lift_summary['top10_lift'])}x |",
            "",
            "说明：风险排序提升倍数以 test set 真实拥塞比例作为随机基线，"
            "即 top-k 高风险位置真实拥塞比例 / 全体位置真实拥塞比例。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    for path in [
        args.train_log,
        args.train_config,
        args.test_metrics,
        args.val_threshold_metrics,
        args.mc_metrics,
        args.risk_topk,
    ]:
        require_file(path)

    results_dir = ensure_dir(args.results_dir)
    train_log = pd.read_csv(args.train_log)
    train_config = load_json(args.train_config)
    test_metrics = load_json(args.test_metrics)
    val_threshold_metrics = load_json(args.val_threshold_metrics)
    mc_metrics = load_json(args.mc_metrics)
    risk_topk = pd.read_csv(args.risk_topk)

    if risk_topk.empty:
        raise ValueError(f"Risk top-k file is empty: {args.risk_topk}")

    config_summary = {
        "model": train_config.get("model"),
        "seq_len": train_config.get("seq_len"),
        "batch_size": train_config.get("batch_size"),
        "lr": train_config.get("lr"),
        "gcn_hidden": train_config.get("gcn_hidden"),
        "gru_hidden": train_config.get("gru_hidden"),
        "dropout": train_config.get("dropout"),
        "pos_weight": train_config.get("pos_weight"),
        "device": train_config.get("device"),
    }
    train_summary = extract_training_summary(train_log)
    uggru_test_summary = {
        "MAE_util": float(test_metrics["MAE_util"]),
        "RMSE_util": float(test_metrics["RMSE_util"]),
        "MAE_load_norm": float(test_metrics["MAE_load_norm"]),
        "RMSE_load_norm": float(test_metrics["RMSE_load_norm"]),
    }

    threshold_metrics = val_threshold_metrics["applied_to_test"]["best_f1_threshold"]
    threshold_summary = {
        "selected_threshold": float(threshold_metrics["threshold"]),
        "Precision": float(threshold_metrics["Precision"]),
        "Recall": float(threshold_metrics["Recall"]),
        "F1": float(threshold_metrics["F1"]),
        "predicted_positive_ratio": float(threshold_metrics["predicted_positive_ratio"]),
        "true_positive_ratio": float(threshold_metrics["true_positive_ratio"]),
    }

    mc_summary = {
        "MAE_util": float(mc_metrics["MAE_util"]),
        "RMSE_util": float(mc_metrics["RMSE_util"]),
        "coverage_1std": float(mc_metrics["coverage_1std"]),
        "coverage_2std": float(mc_metrics["coverage_2std"]),
        "uncertainty_error_corr": float(mc_metrics["uncertainty_error_corr"]),
        "Precision": float(mc_metrics["Precision"]),
        "Recall": float(mc_metrics["Recall"]),
        "F1": float(mc_metrics["F1"]),
        "risk_top1_congestion_rate": float(mc_metrics["risk_top1_congestion_rate"]),
        "risk_top5_congestion_rate": float(mc_metrics["risk_top5_congestion_rate"]),
        "risk_top10_congestion_rate": float(mc_metrics["risk_top10_congestion_rate"]),
    }

    true_positive_ratio = float(mc_metrics["true_positive_ratio"])
    if true_positive_ratio <= 0:
        raise ValueError("true_positive_ratio must be positive to compute risk lift.")
    risk_lift_summary = {
        "top1_lift": mc_summary["risk_top1_congestion_rate"] / true_positive_ratio,
        "top5_lift": mc_summary["risk_top5_congestion_rate"] / true_positive_ratio,
        "top10_lift": mc_summary["risk_top10_congestion_rate"] / true_positive_ratio,
    }

    summary = {
        "inputs": {
            "train_log": str(args.train_log),
            "train_config": str(args.train_config),
            "uggru_test_metrics": str(args.test_metrics),
            "val_threshold_metrics": str(args.val_threshold_metrics),
            "mc_dropout_metrics": str(args.mc_metrics),
            "risk_topk": str(args.risk_topk),
        },
        "training_config": config_summary,
        "training_curve": train_summary,
        "uggru_test_regression": uggru_test_summary,
        "val_to_test_threshold": threshold_summary,
        "mc_dropout_uncertainty": mc_summary,
        "risk_ranking_lift": risk_lift_summary,
    }

    csv_path = results_dir / "prediction_summary_metrics.csv"
    json_path = results_dir / "prediction_summary_metrics.json"
    report_path = results_dir / "prediction_summary_report.md"

    summary_rows = build_summary_rows(
        config_summary=config_summary,
        train_summary=train_summary,
        uggru_test_summary=uggru_test_summary,
        threshold_summary=threshold_summary,
        mc_summary=mc_summary,
        risk_lift_summary=risk_lift_summary,
    )
    summary_rows.to_csv(csv_path, index=False, float_format="%.9g")
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(summary), file, indent=2, ensure_ascii=False)
    write_report(
        output_path=report_path,
        config_summary=config_summary,
        train_summary=train_summary,
        uggru_test_summary=uggru_test_summary,
        threshold_summary=threshold_summary,
        mc_summary=mc_summary,
        risk_lift_summary=risk_lift_summary,
    )

    print("Prediction result summary complete:")
    print(f"- best_epoch: {train_summary['best_epoch']}")
    print(f"- best_val_loss: {train_summary['best_val_loss']:.9g}")
    print(f"- selected_threshold: {threshold_summary['selected_threshold']:.9g}")
    print(f"- test_F1: {threshold_summary['F1']:.9g}")
    print(f"- coverage_2std: {mc_summary['coverage_2std']:.9g}")
    print(f"- uncertainty_error_corr: {mc_summary['uncertainty_error_corr']:.9g}")
    print(f"- top1_lift: {risk_lift_summary['top1_lift']:.9g}")
    print(f"- top5_lift: {risk_lift_summary['top5_lift']:.9g}")
    print(f"- top10_lift: {risk_lift_summary['top10_lift']:.9g}")
    print("Output files:")
    print(f"- {csv_path}")
    print(f"- {json_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
