import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

STRATEGY_INFO = {
    "shortest_path": {
        "chinese_name": "最短跳数路径",
        "description": "选择 path_id=0 的 hop-based shortest path，作为基础最短路径 baseline。",
        "rule": "path_id == 0",
    },
    "min_delay_path": {
        "chinese_name": "最小时延路径",
        "description": "选择 path_delay_ms_sum 最小的候选路径，代表纯时延优先策略。",
        "rule": "min(path_delay_ms_sum)，并列时优先 hop_count 和 path_id 更小",
    },
    "min_risk_path": {
        "chinese_name": "最低预测风险路径",
        "description": "选择 path_risk_score_max 最小的候选路径，代表预测风险优先策略。",
        "rule": "min(path_risk_score_max)，并列时优先 hop_count 和 path_id 更小",
    },
    "min_cong_prob_path": {
        "chinese_name": "最低拥塞概率路径",
        "description": "选择 path_cong_prob_max 最小的候选路径，代表拥塞概率优先策略。",
        "rule": "min(path_cong_prob_max)，并列时优先 hop_count 和 path_id 更小",
    },
    "min_action_cost_path": {
        "chinese_name": "最低综合代价路径",
        "description": "选择启发式 action_cost 最小的候选路径，综合 delay、MLU、拥塞增量和预测风险。",
        "rule": "min(action_cost)，并列时优先 hop_count 和 path_id 更小",
    },
}

ORDER = list(STRATEGY_INFO.keys())

IMPROVEMENT_COLUMNS = [
    "post_mlu_improvement_pct_vs_shortest",
    "delta_mlu_improvement_pct_vs_shortest",
    "delta_congestion_count_improvement_pct_vs_shortest",
    "action_cost_improvement_pct_vs_shortest",
    "risk_score_improvement_pct_vs_shortest",
    "cong_prob_improvement_pct_vs_shortest",
    "delay_change_pct_vs_shortest",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize action-impact routing baseline comparisons.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DIR)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {path}")


def pct_improve(shortest_value: float, strategy_value: float) -> float:
    denom = abs(float(shortest_value))
    if denom < 1e-12:
        return 0.0
    return (float(shortest_value) - float(strategy_value)) / denom * 100.0


def fmt(value: float, digits: int = 6) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float) -> str:
    return f"{float(value):.2f}%"


def make_comment(row: pd.Series) -> str:
    strategy = str(row["strategy"])
    if strategy == "shortest_path":
        return "基础最短路径 baseline，时延和风险均未显式优化。"
    if strategy == "min_delay_path":
        return "时延最低，但风险和 post_mlu 相对较高，说明纯时延优先不等于负载均衡最优。"
    if strategy == "min_risk_path":
        return "显著降低预测风险、delta_mlu 和新增拥塞，但平均路径更长，综合 action_cost 上升。"
    if strategy == "min_cong_prob_path":
        return "与最低风险路径类似，降低拥塞概率和新增拥塞，但牺牲一定时延。"
    if strategy == "min_action_cost_path":
        return "综合 delay、MLU、拥塞增量和风险后 action_cost 最低，是当前最强启发式 baseline。"
    return ""


def build_comparison(strategy_df: pd.DataFrame) -> pd.DataFrame:
    strategy_df = strategy_df.copy()
    strategy_df["strategy"] = pd.Categorical(strategy_df["strategy"], categories=ORDER, ordered=True)
    strategy_df = strategy_df.sort_values("strategy").reset_index(drop=True)
    shortest = strategy_df[strategy_df["strategy"].astype(str) == "shortest_path"].iloc[0]

    rows = []
    for row in strategy_df.itertuples(index=False):
        item = row._asdict()
        strategy = str(item["strategy"])
        out = {
            "strategy": strategy,
            "chinese_name": STRATEGY_INFO[strategy]["chinese_name"],
            "description": STRATEGY_INFO[strategy]["description"],
            "hop_count": item["hop_count"],
            "path_delay_ms_sum": item["path_delay_ms_sum"],
            "post_mlu": item["post_mlu"],
            "delta_mlu": item["delta_mlu"],
            "post_congestion_count": item["post_congestion_count"],
            "delta_congestion_count": item["delta_congestion_count"],
            "path_post_util_max": item["path_post_util_max"],
            "path_new_congestion_count": item["path_new_congestion_count"],
            "path_risk_score_max": item["path_risk_score_max"],
            "path_cong_prob_max": item["path_cong_prob_max"],
            "action_cost": item["action_cost"],
            "post_mlu_improvement_pct_vs_shortest": pct_improve(shortest["post_mlu"], item["post_mlu"]),
            "delta_mlu_improvement_pct_vs_shortest": pct_improve(shortest["delta_mlu"], item["delta_mlu"]),
            "delta_congestion_count_improvement_pct_vs_shortest": pct_improve(
                shortest["delta_congestion_count"], item["delta_congestion_count"]
            ),
            "action_cost_improvement_pct_vs_shortest": pct_improve(shortest["action_cost"], item["action_cost"]),
            "risk_score_improvement_pct_vs_shortest": pct_improve(
                shortest["path_risk_score_max"], item["path_risk_score_max"]
            ),
            "cong_prob_improvement_pct_vs_shortest": pct_improve(
                shortest["path_cong_prob_max"], item["path_cong_prob_max"]
            ),
            "delay_change_pct_vs_shortest": pct_improve(shortest["path_delay_ms_sum"], item["path_delay_ms_sum"]),
        }
        out["final_comment"] = make_comment(pd.Series(out))
        rows.append(out)
    return pd.DataFrame(rows)


def best_strategy(df: pd.DataFrame, metric: str) -> str:
    return str(df.loc[df[metric].idxmin(), "strategy"])


def metric_dict(df: pd.DataFrame, strategy: str) -> dict:
    row = df[df["strategy"] == strategy].iloc[0]
    fields = [
        "hop_count",
        "path_delay_ms_sum",
        "post_mlu",
        "delta_mlu",
        "post_congestion_count",
        "delta_congestion_count",
        "path_post_util_max",
        "path_risk_score_max",
        "path_cong_prob_max",
        "action_cost",
    ]
    return {field: float(row[field]) for field in fields}


def build_summary_json(comparison: pd.DataFrame, stats: dict) -> dict:
    shortest = comparison[comparison["strategy"] == "shortest_path"].iloc[0]
    min_risk = comparison[comparison["strategy"] == "min_risk_path"].iloc[0]
    min_cost = comparison[comparison["strategy"] == "min_action_cost_path"].iloc[0]

    key_findings = [
        "min_risk_path 相比 shortest_path 明显降低 risk_score、delta_mlu 和新增拥塞，但会增加时延。",
        "min_delay_path 虽然时延最低，但 risk_score 和 post_mlu 较高，说明纯时延最短并不等价于负载均衡最优。",
        "min_action_cost_path 在综合 delay、MLU、拥塞增量和风险后取得最低 action_cost。",
        "预测风险信息能够为路由选择提供有效依据，特别是在降低新增拥塞和 MLU 增量方面。",
        "这一步仍然是单 OD 增量加载 proxy，不是完整多流全局重路由。",
    ]
    limitations = [
        "当前只评估单个 OD demand 的增量加载影响，没有全网多 OD 同时重路由。",
        "当前没有长期多步状态演化，也没有模拟后续时刻的网络反馈闭环。",
        "当前没有训练强化学习策略，只是传统/启发式 baseline 汇总。",
        "action_cost 权重是启发式设置，后续奖励函数仍需根据实验目标调参。",
        "zero-hop 场景下多条候选路径等价，后续 RL action mask 应只保留 path_id=0。",
    ]

    return {
        "best_post_mlu_strategy": best_strategy(comparison, "post_mlu"),
        "best_delta_mlu_strategy": best_strategy(comparison, "delta_mlu"),
        "best_delta_congestion_count_strategy": best_strategy(comparison, "delta_congestion_count"),
        "best_action_cost_strategy": best_strategy(comparison, "action_cost"),
        "best_risk_score_strategy": best_strategy(comparison, "path_risk_score_max"),
        "best_delay_strategy": best_strategy(comparison, "path_delay_ms_sum"),
        "shortest_path_metrics": metric_dict(comparison, "shortest_path"),
        "min_risk_path_metrics": metric_dict(comparison, "min_risk_path"),
        "min_action_cost_path_metrics": metric_dict(comparison, "min_action_cost_path"),
        "min_risk_vs_shortest_summary": {
            "delta_mlu_improvement_pct": float(min_risk["delta_mlu_improvement_pct_vs_shortest"]),
            "delta_congestion_count_improvement_pct": float(
                min_risk["delta_congestion_count_improvement_pct_vs_shortest"]
            ),
            "risk_score_improvement_pct": float(min_risk["risk_score_improvement_pct_vs_shortest"]),
            "delay_change_pct": float(min_risk["delay_change_pct_vs_shortest"]),
            "action_cost_improvement_pct": float(min_risk["action_cost_improvement_pct_vs_shortest"]),
            "different_ratio": stats.get("min_risk_vs_shortest_different_ratio"),
        },
        "min_action_cost_vs_shortest_summary": {
            "delta_mlu_improvement_pct": float(min_cost["delta_mlu_improvement_pct_vs_shortest"]),
            "delta_congestion_count_improvement_pct": float(
                min_cost["delta_congestion_count_improvement_pct_vs_shortest"]
            ),
            "risk_score_improvement_pct": float(min_cost["risk_score_improvement_pct_vs_shortest"]),
            "delay_change_pct": float(min_cost["delay_change_pct_vs_shortest"]),
            "action_cost_improvement_pct": float(min_cost["action_cost_improvement_pct_vs_shortest"]),
            "different_ratio": stats.get("min_action_cost_vs_shortest_different_ratio"),
        },
        "key_findings": key_findings,
        "limitations": limitations,
        "source_action_impact_stats": stats,
    }


def save_bar(df: pd.DataFrame, metric: str, path: Path, ylabel: str) -> None:
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(df["strategy"], df[metric])
    plt.ylabel(ylabel)
    plt.xticks(rotation=18, ha="right")
    plt.grid(axis="y", linewidth=0.4, alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_figures(comparison: pd.DataFrame, results_dir: Path) -> list[Path]:
    paths = [
        results_dir / "routing_baseline_post_mlu_comparison.png",
        results_dir / "routing_baseline_delta_mlu_comparison.png",
        results_dir / "routing_baseline_delta_congestion_comparison.png",
        results_dir / "routing_baseline_action_cost_comparison.png",
        results_dir / "routing_baseline_risk_delay_tradeoff.png",
        results_dir / "routing_baseline_improvement_vs_shortest.png",
    ]
    save_bar(comparison, "post_mlu", paths[0], "post_mlu")
    save_bar(comparison, "delta_mlu", paths[1], "delta_mlu")
    save_bar(comparison, "delta_congestion_count", paths[2], "delta_congestion_count")
    save_bar(comparison, "action_cost", paths[3], "action_cost")

    plt.figure(figsize=(7, 5.2))
    plt.scatter(comparison["path_delay_ms_sum"], comparison["path_risk_score_max"], s=80)
    for row in comparison.itertuples(index=False):
        plt.annotate(str(row.strategy), (row.path_delay_ms_sum, row.path_risk_score_max), xytext=(5, 4), textcoords="offset points")
    plt.xlabel("path_delay_ms_sum")
    plt.ylabel("path_risk_score_max")
    plt.title("Risk-delay tradeoff")
    plt.grid(linewidth=0.4, alpha=0.35)
    plt.tight_layout()
    plt.savefig(paths[4], dpi=200)
    plt.close()

    plot_df = comparison[comparison["strategy"] != "shortest_path"].copy()
    metrics = [
        "delta_mlu_improvement_pct_vs_shortest",
        "delta_congestion_count_improvement_pct_vs_shortest",
        "action_cost_improvement_pct_vs_shortest",
        "risk_score_improvement_pct_vs_shortest",
    ]
    labels = ["delta_mlu", "delta_congestion", "action_cost", "risk_score"]
    x = np.arange(len(plot_df))
    width = 0.18
    plt.figure(figsize=(10, 5.2))
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        plt.bar(x + (idx - 1.5) * width, plot_df[metric], width=width, label=label)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(x, plot_df["strategy"], rotation=18, ha="right")
    plt.ylabel("improvement vs shortest (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths[5], dpi=200)
    plt.close()
    return paths


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: fmt(x, 6))
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = []
    for row in view.itertuples(index=False):
        rows.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join([header, sep, *rows])


def write_report(
    comparison: pd.DataFrame,
    summary: dict,
    stats: dict,
    output_path: Path,
    figure_paths: list[Path],
) -> None:
    main_cols = [
        "strategy",
        "chinese_name",
        "path_delay_ms_sum",
        "post_mlu",
        "delta_mlu",
        "delta_congestion_count",
        "path_risk_score_max",
        "path_cong_prob_max",
        "action_cost",
    ]
    improve_cols = [
        "strategy",
        "delta_mlu_improvement_pct_vs_shortest",
        "delta_congestion_count_improvement_pct_vs_shortest",
        "action_cost_improvement_pct_vs_shortest",
        "risk_score_improvement_pct_vs_shortest",
        "delay_change_pct_vs_shortest",
    ]
    improve_df = comparison[comparison["strategy"].isin(["min_risk_path", "min_cong_prob_path", "min_action_cost_path"])]

    strategy_rows = []
    for strategy in ORDER:
        strategy_rows.append(
            f"| `{strategy}` | {STRATEGY_INFO[strategy]['chinese_name']} | {STRATEGY_INFO[strategy]['rule']} | {STRATEGY_INFO[strategy]['description']} |"
        )
    strategy_table = "\n".join(
        ["| 策略 | 中文名 | 选择规则 | 作用 |", "|---|---|---|---|", *strategy_rows]
    )

    text = f"""# 基于 Action Impact 的传统路由 Baseline 对比报告

## 1. 本阶段目的

本阶段不是训练强化学习模型，也不是实现 PPO，而是基于第 10 阶段规则/仿真型 action impact 结果，对不同传统和启发式路由策略进行对比。其目标是回答：在单 OD demand 增量加载的 proxy 设置下，最短路径、最小时延路径、预测风险路径、拥塞概率路径和综合代价路径分别会对 `post_mlu`、新增拥塞和动作代价造成什么影响，并为后续 Masked PPO 路由提供 baseline。

## 2. 输入数据

- `data/processed/action_impact_features_test_topo144_k5.npz`
- `data/results/action_impact_strategy_comparison_topo144_k5.csv`
- `data/results/action_impact_strategy_delta_topo144_k5.csv`
- `data/results/action_impact_stats_topo144_k5.json`

本阶段沿用第 10 阶段假设：当前 `link_state` 是 shortest-delay Dijkstra 生成的基础网络状态；每次只把一个 OD demand 增量加载到候选路径上，不移除原有流量，也不做全网重路由。因此这里的结果是动作影响 proxy，不是完整路由策略执行后的全局闭环状态。

## 3. 策略定义

{strategy_table}

## 4. 主要对比结果

{markdown_table(comparison, main_cols)}

`shortest_path` 是基础最短路径 baseline；`min_delay_path` 代表纯时延优先；`min_risk_path` 代表预测风险优先；`min_cong_prob_path` 代表拥塞概率优先；`min_action_cost_path` 代表综合代价优先。

## 5. 相对 shortest_path 的改善

{markdown_table(improve_df, improve_cols)}

从结果看，`min_risk_path` 相比 `shortest_path` 的 `delta_mlu` 改善为 {fmt_pct(summary['min_risk_vs_shortest_summary']['delta_mlu_improvement_pct'])}，新增拥塞改善为 {fmt_pct(summary['min_risk_vs_shortest_summary']['delta_congestion_count_improvement_pct'])}，预测风险改善为 {fmt_pct(summary['min_risk_vs_shortest_summary']['risk_score_improvement_pct'])}；但其 delay change 为 {fmt_pct(summary['min_risk_vs_shortest_summary']['delay_change_pct'])}，说明它平均路径更长。`min_action_cost_path` 的综合 action_cost 改善为 {fmt_pct(summary['min_action_cost_vs_shortest_summary']['action_cost_improvement_pct'])}，是当前最强的启发式综合 baseline。

## 6. 关键图表

![Post MLU comparison](routing_baseline_post_mlu_comparison.png)

图中横坐标为策略，纵坐标为平均 `post_mlu`，用于比较执行单 OD action 后的全网最大利用率。

![Delta MLU comparison](routing_baseline_delta_mlu_comparison.png)

图中横坐标为策略，纵坐标为平均 `delta_mlu`，用于比较不同策略对 MLU 增量的影响。

![Delta congestion comparison](routing_baseline_delta_congestion_comparison.png)

图中横坐标为策略，纵坐标为平均 `delta_congestion_count`，用于比较新增拥塞链路数量。

![Action cost comparison](routing_baseline_action_cost_comparison.png)

图中横坐标为策略，纵坐标为平均 `action_cost`，用于比较综合启发式代价。

![Risk delay tradeoff](routing_baseline_risk_delay_tradeoff.png)

图中横坐标为 `path_delay_ms_sum`，纵坐标为 `path_risk_score_max`，每个点代表一种策略，用于观察风险和时延之间的折中。

![Improvement vs shortest](routing_baseline_improvement_vs_shortest.png)

图中横坐标为策略，纵坐标为相对 `shortest_path` 的改善百分比，展示 MLU、拥塞、代价和风险四类指标的变化。

## 7. 阶段性结论

1. 最短路径并不总是负载均衡意义上的最优路径，它没有显式考虑链路利用率、拥塞和预测风险。
2. 最小时延路径虽然 delay 最低，但 `risk_score` 和 `post_mlu` 较高，说明纯时延优先可能带来更高拥塞风险。
3. 风险感知路径能够降低预测风险、MLU 增量和新增拥塞，但通常需要牺牲一定路径时延。
4. 综合 `action_cost` 路径在 delay、MLU、拥塞增量和风险之间取得更稳健折中，是当前最强启发式 baseline。
5. 后续强化学习需要在 delay、MLU、congestion、risk_score 之间学习动态权衡，而不是固定使用单一启发式。

## 8. 当前局限

1. 当前仍然是单 OD 增量加载 proxy。
2. 当前没有全网多 OD 同时重路由。
3. 当前没有长期多步状态演化。
4. 当前没有训练强化学习策略。
5. `action_cost` 权重是启发式设置，后续奖励函数需要进一步调参。

## 9. 下一步工作

下一步进入第 12 阶段：强化学习环境设计与 action mask 构建。后续将使用 K=5 候选路径作为动作空间，使用 action impact features 作为环境反馈基础，并结合 `risk_score`、`cong_prob`、`post_mlu`、`delta_congestion_count` 等指标设计奖励函数。对 zero-hop 场景和不可行动作需要设计 action mask，最后再实现 Masked PPO。
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir

    strategy_path = results_dir / f"action_impact_strategy_comparison_topo144_k{args.k}.csv"
    delta_path = results_dir / f"action_impact_strategy_delta_topo144_k{args.k}.csv"
    stats_path = results_dir / f"action_impact_stats_topo144_k{args.k}.json"
    feature_names_path = args.processed_dir / f"action_impact_feature_names_topo144_k{args.k}.json"

    for path in [strategy_path, delta_path, stats_path, feature_names_path]:
        require_file(path)

    strategy_df = pd.read_csv(strategy_path)
    pd.read_csv(delta_path)
    with stats_path.open("r", encoding="utf-8") as file:
        stats = json.load(file)
    with feature_names_path.open("r", encoding="utf-8") as file:
        json.load(file)

    comparison = build_comparison(strategy_df)
    summary = build_summary_json(comparison, stats)

    comparison_path = results_dir / f"routing_baseline_comparison_topo144_k{args.k}.csv"
    summary_path = results_dir / f"routing_baseline_summary_topo144_k{args.k}.json"
    report_path = results_dir / f"routing_baseline_report_topo144_k{args.k}.md"

    comparison.to_csv(comparison_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    figure_paths = save_figures(comparison, results_dir)
    write_report(comparison, summary, stats, report_path, figure_paths)

    print("Routing baseline summary complete:")
    print(f"- comparison csv: {comparison_path}")
    print(f"- summary json: {summary_path}")
    print(f"- report md: {report_path}")
    print(f"- best post_mlu strategy: {summary['best_post_mlu_strategy']}")
    print(f"- best delta_mlu strategy: {summary['best_delta_mlu_strategy']}")
    print(f"- best delta_congestion_count strategy: {summary['best_delta_congestion_count_strategy']}")
    print(f"- best action_cost strategy: {summary['best_action_cost_strategy']}")
    print("- figures:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
