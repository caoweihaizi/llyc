import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

REQUIRED_STRATEGIES = {
    "shortest_path",
    "min_delay_path",
    "min_risk_path",
    "min_cong_prob_path",
    "min_action_cost_path",
}

IMPROVEMENT_COLUMNS = {
    "post_mlu_improvement_pct_vs_shortest",
    "delta_mlu_improvement_pct_vs_shortest",
    "delta_congestion_count_improvement_pct_vs_shortest",
    "action_cost_improvement_pct_vs_shortest",
    "risk_score_improvement_pct_vs_shortest",
    "cong_prob_improvement_pct_vs_shortest",
    "delay_change_pct_vs_shortest",
}

REQUIRED_REPORT_SECTIONS = [
    "## 1. 本阶段目的",
    "## 3. 策略定义",
    "## 4. 主要对比结果",
    "## 5. 相对 shortest_path 的改善",
    "## 8. 当前局限",
    "## 9. 下一步工作",
]

REQUIRED_FIGURES = [
    "routing_baseline_post_mlu_comparison.png",
    "routing_baseline_delta_mlu_comparison.png",
    "routing_baseline_delta_congestion_comparison.png",
    "routing_baseline_action_cost_comparison.png",
    "routing_baseline_risk_delay_tradeoff.png",
    "routing_baseline_improvement_vs_shortest.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check routing baseline summary outputs.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required output file does not exist: {path}")


def expected_best(df: pd.DataFrame, metric: str) -> str:
    return str(df.loc[df[metric].idxmin(), "strategy"])


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir

    comparison_path = results_dir / f"routing_baseline_comparison_topo144_k{args.k}.csv"
    summary_path = results_dir / f"routing_baseline_summary_topo144_k{args.k}.json"
    report_path = results_dir / f"routing_baseline_report_topo144_k{args.k}.md"
    check_path = results_dir / f"routing_baseline_summary_check_topo144_k{args.k}.json"

    for path in [comparison_path, summary_path, report_path]:
        require_file(path)

    comparison = pd.read_csv(comparison_path)
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    report = report_path.read_text(encoding="utf-8")

    strategies = set(comparison["strategy"].astype(str))
    strategies_ok = REQUIRED_STRATEGIES == strategies
    improvement_cols_ok = IMPROVEMENT_COLUMNS.issubset(set(comparison.columns))

    numeric = comparison.select_dtypes(include=[np.number])
    finite_ok = bool(np.isfinite(numeric.to_numpy()).all())

    best_action_cost_ok = summary.get("best_action_cost_strategy") == expected_best(comparison, "action_cost")
    best_post_mlu_ok = summary.get("best_post_mlu_strategy") == expected_best(comparison, "post_mlu")
    best_delta_mlu_ok = summary.get("best_delta_mlu_strategy") == expected_best(comparison, "delta_mlu")
    best_delta_cong_ok = summary.get("best_delta_congestion_count_strategy") == expected_best(
        comparison, "delta_congestion_count"
    )

    report_sections_ok = all(section in report for section in REQUIRED_REPORT_SECTIONS)
    figure_paths = [results_dir / name for name in REQUIRED_FIGURES]
    figures_ok = all(path.exists() for path in figure_paths)

    validation_passed = bool(
        strategies_ok
        and improvement_cols_ok
        and finite_ok
        and best_action_cost_ok
        and best_post_mlu_ok
        and best_delta_mlu_ok
        and best_delta_cong_ok
        and report_sections_ok
        and figures_ok
    )

    payload = {
        "comparison_csv": str(comparison_path),
        "summary_json": str(summary_path),
        "report_md": str(report_path),
        "strategies_ok": strategies_ok,
        "strategies_found": sorted(strategies),
        "improvement_columns_ok": improvement_cols_ok,
        "missing_improvement_columns": sorted(IMPROVEMENT_COLUMNS - set(comparison.columns)),
        "finite_numeric_values": finite_ok,
        "best_action_cost_strategy_ok": best_action_cost_ok,
        "best_post_mlu_strategy_ok": best_post_mlu_ok,
        "best_delta_mlu_strategy_ok": best_delta_mlu_ok,
        "best_delta_congestion_count_strategy_ok": best_delta_cong_ok,
        "expected_best_action_cost_strategy": expected_best(comparison, "action_cost"),
        "expected_best_post_mlu_strategy": expected_best(comparison, "post_mlu"),
        "expected_best_delta_mlu_strategy": expected_best(comparison, "delta_mlu"),
        "expected_best_delta_congestion_count_strategy": expected_best(comparison, "delta_congestion_count"),
        "report_sections_ok": report_sections_ok,
        "missing_report_sections": [section for section in REQUIRED_REPORT_SECTIONS if section not in report],
        "figures_ok": figures_ok,
        "missing_figures": [str(path) for path in figure_paths if not path.exists()],
        "validation_passed": validation_passed,
    }

    with check_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print("Routing baseline summary check complete:")
    print(f"- report: {check_path}")
    print(f"- strategies ok: {strategies_ok}")
    print(f"- improvement columns ok: {improvement_cols_ok}")
    print(f"- finite numeric values: {finite_ok}")
    print(f"- best action_cost ok: {best_action_cost_ok}")
    print(f"- best post_mlu ok: {best_post_mlu_ok}")
    print(f"- best delta_mlu ok: {best_delta_mlu_ok}")
    print(f"- best delta congestion ok: {best_delta_cong_ok}")
    print(f"- report sections ok: {report_sections_ok}")
    print(f"- figures ok: {figures_ok}")
    print(f"- validation passed: {validation_passed}")

    if not validation_passed:
        raise SystemExit("Routing baseline summary validation failed. See check JSON.")


if __name__ == "__main__":
    main()
