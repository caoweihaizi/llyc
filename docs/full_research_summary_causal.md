# 基于流量预测与动作影响评估的低轨卫星网络负载均衡路由研究总结

副标题：从 Abilene 流量建模、LEO 链路状态预测到 MaskedPPO 路由决策的完整实验闭环

> 本文档基于项目中已有脚本、配置、CSV、JSON、NPZ 元信息和 PNG 图像整理；没有重新训练模型，没有重新生成大规模数据，也没有改动既有实验结果。此前总结文档在部分环境中曾出现中文编码显示问题，本详细版已用正常 UTF-8 中文重新组织并扩展。

## 1. 研究背景与总体目标

低地球轨道 LEO（Low Earth Orbit，低地球轨道）卫星网络具有拓扑时变、星地接入快速变化、星间链路 ISL（Inter-Satellite Link，星间链路）资源有限、业务流量时空分布不均衡等特点。传统最短路径或最小时延路由容易反复选择低跳数、低时延链路，导致局部链路拥塞。

本研究围绕“基于流量预测与强化学习的低轨卫星网络负载均衡路由算法研究”展开：先使用流量预测模型提前识别未来链路风险，再将预测风险聚合到候选路径层面，进一步通过动作影响评估和强化学习进行路径选择。当前强化学习环境是 **offline action-impact routing environment**，不是完整在线多流动态重路由仿真器；`min_raw_cost` 是基于已知动作代价的强启发式上界，不应表述为普通在线路由算法。

```mermaid
flowchart TD
    A[Abilene 真实 OD 流量] --> B[topo144 LEO 星座与固定 4-ISL 拓扑]
    B --> C[Gateway 动态接入]
    C --> D[Shortest-delay Dijkstra 链路状态生成]
    D --> E[link_state 链路状态数据集]
    E --> F[Edge Graph 与 seq12 预测样本]
    F --> G[UGGRU 链路状态预测]
    G --> H[Validation 阈值校准]
    G --> I[MC Dropout 不确定性与 risk_score]
    I --> J[K=5 候选路径生成]
    J --> K[路径级风险特征聚合]
    K --> L[Action Impact 单 OD 动作影响模型]
    L --> M[传统启发式 Routing Baseline]
    L --> N[LeoRoutingEnv + Action Mask]
    N --> O[MaskablePPO 训练与评估]
    O --> P[多 run 汇总比较]
```

### 1.1 研究问题分解与技术闭环逻辑

本项目真正要解决的问题不是单纯“预测下一时刻链路负载”，也不是单纯“训练一个强化学习智能体”。在低轨卫星网络中，路由决策面对的是一个连续变化的资源分配问题：地面业务流量在时间和空间上不断变化，卫星相对地面高速运动，gateway 接入卫星随时间切换，星间链路容量有限且局部区域容易出现热点。如果只在当前时刻基于最短跳数或最小时延做路径选择，算法往往会忽略未来几分钟内即将发生的拥塞风险，因此可能把流量继续压到已经接近瓶颈的链路上。

因此，本研究首先把问题拆成“看见未来”和“做出动作”两个层面。UGGRU 预测模块负责从历史链路状态中估计下一时刻链路利用率、链路负载和拥塞概率；MC Dropout 进一步给出预测不确定性，使系统不仅知道“哪里可能拥塞”，也知道“这个判断有多不确定”。这一步输出的是链路级信息，即每条 ISL 的 `util_pred_mean`、`util_pred_std`、`cong_prob_mean`、`risk_score` 等。

但是路由动作不是选择一条链路，而是为某个 OD demand 选择一条端到端候选路径。因此，链路级预测不能直接作为路由动作，需要进一步沿着候选路径的 `edge_path` 聚合为路径级特征。路径风险特征回答的问题是：“如果候选路径由若干条链路组成，那么这条路径整体风险有多高？”其中 max 聚合关注路径瓶颈链路，mean 聚合关注路径平均状态，sum 聚合关注路径累计代价。这一步把预测模块的输出转化为路由可理解的候选动作属性。

仅有路径风险仍然不够，因为选择一条路径会对当前网络状态产生影响。两个路径即使预测风险相近，也可能因为承载的 OD demand 大小、路径长度、当前链路剩余容量不同，导致完全不同的后果。因此本研究设计了 action impact 动作影响模型：在当前 link_state 基础上，模拟将某个 OD demand 增量加载到候选路径上，计算 post_mlu、delta_mlu、delta_congestion_count、action_cost 等指标。它回答的问题是：“如果现在选择这条路径，会让网络状态发生什么变化？”

最后，强化学习环境 LeoRoutingEnv 将上述 action impact 特征封装成 State / Action / Reward / Transition。智能体每一步面对一个 gateway OD pair，从 K=5 条候选路径中选择一个动作。action mask 保证无效动作和 zero-hop 等价动作不会被错误选择。MaskedPPO 则学习一个从状态到动作的策略，目标是在不直接手写每种策略规则的情况下，学习在 delay、MLU、拥塞增量和风险之间做权衡。至此，系统形成了从“历史流量与链路状态”到“未来风险预测”，再到“路径级风险”，再到“动作影响”，最后到“路由决策”的闭环。

需要注意的是，这个闭环当前仍是离线实验闭环：action impact 是预计算 proxy，LeoRoutingEnv 读取这些预计算结果，而不是在线重新仿真每一步全网多流重路由。这个限制并不削弱当前阶段的意义，因为它先验证了预测信息能否被转化为可操作的路由决策信号；后续若扩展到在线多步动态仿真器，可以沿用当前的状态设计、动作空间、风险特征和 reward 设计经验。

## 2. 技术路线与已完成模块总览

| 模块 | 输入 | 处理 | 输出 | 当前状态 | 局限 |
| --- | --- | --- | --- | --- | --- |
| Abilene 数据解析 | Abilene traffic matrix 与拓扑文件 | 解析 OD 矩阵、PoP/gateway、链路 | `od_matrices_full.npy` 等 | 完成 | 尚未扩展多场景业务流量 |
| LEO topo144 星座 | `configs/base.yaml` | 生成 144 星、8 轨道面、固定 4-ISL | 卫星位置、ISL edge CSV | 完成 | ISL 边集合固定 |
| Gateway 动态接入 | gateway 经纬度、卫星位置 | 按 15° 最小仰角选择接入卫星 | `gateway_access_topo144.npy` | 完成 | fallback 是工程近似 |
| link_state 仿真 | OD、gateway 接入、LEO edge | shortest-delay Dijkstra 映射流量 | `link_state` CSV | 完成 | Dijkstra 是数据生成基线 |
| edge graph 与 seq12 样本 | link_state、edge CSV | 构建链路图与监督样本 | samples NPZ、splits JSON | 完成 | 固定窗口 seq_len=12 |
| UGGRU 预测 | seq12 样本、edge graph | GraphConv + GRU 多任务预测 | best_model、metrics | 完成 | 主要是单 seed |
| MC Dropout | UGGRU best model | 多次随机推理，计算均值/方差/risk | 不确定性指标与图 | 完成 | 主要提供风险排序 |
| baseline 对比 | samples、训练结果 | Last/HA/GRU/LSTM/UGGRU 对比 | 模型对比表与图 | 完成 | 未做复杂调参 |
| candidate paths | topo144 graph | hop-based K-shortest paths | K=5 候选路径 | 完成 | 只按跳数生成 |
| path risk features | MC 预测、candidate paths | 链路风险聚合到路径 | 路径风险 NPZ/CSV | 完成 | 只处理 test split |
| action impact | path risk、link_state | 单 OD 增量加载 | 动作影响特征 | 完成 | proxy，非全网重路由 |
| routing baseline | action impact | 启发式策略汇总 | baseline 报告 | 完成 | min_action_cost 是强启发式 |
| LeoRoutingEnv | action impact features | Gymnasium 环境与 mask | 环境检查、heuristic eval | 完成 | 离线环境 |
| MaskedPPO | LeoRoutingEnv | smoke/mid/continue/lr1e4 训练 | PPO run comparison | 阶段性完成 | 仍低于 min_raw_cost 上界 |

## 3. 数据与 LEO 网络建模

**本节为什么需要。**所有后续预测和路由实验都依赖一个可信的数据与网络基础。上一节只是给出了总体技术路线，但还没有说明业务流量从哪里来、卫星网络如何构造、gateway 如何接入卫星。如果没有真实或相对可信的 OD 流量输入，后续链路状态和拥塞标签就会失去研究意义；如果没有明确的 LEO 拓扑与接入模型，路由动作空间也无法定义。本节解决的是“实验对象和基础数据是什么”的问题。

本研究使用 Abilene 真实骨干网 OD（Origin-Destination，源宿）流量作为业务流量来源。`od_matrices_full.npy` 的 shape 为 `(48384, 12, 12)`，表示 48384 个时间片、12 个 PoP/gateway、12×12 有序 OD 流量矩阵。Abilene raw 值的单位已经修正为 `100 bytes / 5 minutes`，换算公式为 `Mbps = raw * 100 * 8 / 300 / 1e6`。

| 项目 | 数值/说明 |
| --- | --- |
| OD shape | (48384, 12, 12) |
| sat_positions shape | (48384, 144, 3) |
| gateway_access shape | (48384, 12) |
| topo_name | topo144 |
| 卫星数量 | 144 |
| 轨道面数量 | 8 |
| 每轨卫星数量 | 18 |
| 轨道高度 | 550 km |
| 轨道倾角 | 53° |
| ISL 无向边数量 | 288 |
| 每颗卫星度数 | 4 |
| 最小仰角阈值 | 15° |
| fallback_rate 平均值 | 0.005232 |
| fallback_rate 最大值 | 0.062789 |

![Abilene total traffic curve](../data/results/abilene_total_traffic_curve.png)

> 图注：展示 Abilene 总流量随时间变化，用于说明输入业务流量具有真实时序波动。横坐标为time，纵坐标为total traffic。

![LEO topo144 snapshot](../data/results/leo_topo144_snapshot.png)

> 图注：展示 topo144 星座某一时刻的空间分布和链路结构。横坐标为空间位置/经纬度，纵坐标为卫星与链路位置。

![Gateway access example](../data/results/gateway_access_example.png)

> 图注：展示地面 gateway 到可见卫星的动态接入示例。横坐标为time 或 gateway，纵坐标为接入卫星/可见关系。

当前卫星位置和 gateway 接入是动态的，但 ISL 边集合固定，因此当前星间拓扑是“固定 4-ISL 边集合 + 动态节点位置”的简化实验拓扑。`remain_visible_time` 目前仍是 9999 占位特征。

**本节如何连接下一阶段。**本节输出了 OD 流量、卫星位置、ISL 边集合和 gateway 接入关系。下一阶段会把这些输入组合起来：每个时间片根据 gateway 接入确定 OD 流量的源/目的接入卫星，再通过基线路由映射到 ISL，从而生成 link_state 链路状态数据集。

## 4. 链路状态仿真与诊断

**本节为什么需要。**上一节给出了业务流量和 LEO 网络结构，但预测模型不能直接从 OD 矩阵学习路由效果；它需要监督学习标签，例如每条链路在每个时间片的负载、利用率、时延和拥塞状态。因此，本节要解决的是“如何把地面 OD 流量变成链路级训练数据”的问题。只有生成 link_state，后续 UGGRU 才有可学习的输入与标签。

链路状态数据集使用 shortest-delay Dijkstra 将 Abilene OD 流量映射到 LEO 星间链路。这里的 Dijkstra 只是数据生成阶段基线，用于形成可学习的链路状态样本，不是最终提出的负载均衡路由算法。

| 项目 | 数值 |
| --- | --- |
| link_state 文件大小 | 718.35 MB |
| 总行数 | 13934304 |
| time 数量 | 48383 |
| edge_id 数量 | 288 |
| 总拥塞样本数 | 189456 |
| Top 10 拥塞边占比 | 0.161298666 |
| 有拥塞 time 占比 | 0.844490834 |
| 最大同时拥塞链路数 | 15 |
| 平均每 time 拥塞链路数 | 3.91575553 |
| utilization 公式误差 | 1.7763568394e-15 |
| next 标签抽样检查 | True |

MLU（Maximum Link Utilization，最大链路利用率）表示同一时刻所有链路中最高的 utilization。p95/p99 反映尾部高负载风险。当前有拥塞 time 占比较高，但总体正样本比例低，是因为每个 time 有 288 条链路，通常只有少数链路拥塞。


![Average utilization curve](../data/results/leo_average_utilization_curve.png)

> 图注：展示全网平均链路利用率随时间变化。横坐标为time，纵坐标为average utilization。

![MLU curve](../data/results/leo_mlu_curve.png)

> 图注：展示每个时间片的最大链路利用率。横坐标为time，纵坐标为maximum link utilization。

![Top congested edges](../data/results/top_congested_edges.png)

> 图注：展示拥塞次数最多的链路。横坐标为edge_id，纵坐标为congestion count。

![Congestion count by time](../data/results/congestion_count_by_time.png)

> 图注：展示每个时间片同时拥塞链路数量。横坐标为time，纵坐标为congested edge count。

**本节如何连接下一阶段。**link_state 数据集把“OD 流量 + LEO 拓扑”转化为了“链路状态时序”。下一阶段会从 link_state 中提取过去 12 个时间片的链路特征，并构建 edge graph 邻接矩阵，形成 UGGRU 可直接训练的监督学习样本。

## 5. 预测样本构建与 UGGRU 模型

**本节为什么需要。**上一节已经得到逐时间片、逐链路的 link_state，但它仍是扁平 CSV 数据，不适合直接输入图时序模型。预测任务需要固定长度历史窗口、固定 edge 顺序、统一归一化方式和 train/val/test 时间划分。因此，本节要解决的是“如何把链路状态数据集转成模型训练样本”，并设计能够利用链路拓扑和时间序列的 UGGRU 模型。

预测对象是链路 edge，而不是卫星节点。edge graph 的节点对应 288 条 ISL；如果两条 ISL 共享同一颗卫星，则它们在 edge graph 中相邻。GCN（Graph Convolutional Network，图卷积网络）提取链路空间相关性，GRU（Gated Recurrent Unit，门控循环单元）提取时间依赖。

| 项目 | 数值/说明 |
| --- | --- |
| edge adjacency shape | (288, 288) |
| edge graph 平均度 | 6.000000 |
| X shape | [48372, 12, 288, 6] |
| y_utilization shape | [48372, 288] |
| y_load_mbps_norm shape | [48372, 288] |
| y_congestion shape | [48372, 288] |
| feature_names | utilization, load_mbps_norm, delay_ms_norm, queue_len_norm, remain_visible_time_norm, congestion_label |
| train/val/test | [0,33860) / [33860,41115) / [41115,48372) |
| 样本文件大小 | 487.62 MB |
| scaler | 基于 train split；remain_visible_time raw_std=0 时安全处理 |
| y_congestion dtype | int8 |

UGGRU 输入为 `X=[B,12,288,6]`，输出下一时间片 utilization、load_mbps_norm 和 congestion logit。损失为 `1.0*MSE(util)+0.3*MSE(load)+0.5*BCE(congestion)`，并用训练集 `pos_weight` 缓解拥塞类别不平衡。

| 训练配置 | 数值 |
| --- | --- |
| model | uggru |
| seq_len | 12 |
| batch_size | 16 |
| lr | 0.001000 |
| gcn_hidden | 32 |
| gru_hidden | 64 |
| dropout | 0.200000 |
| pos_weight | 66.294270 |
| device | cuda |
| best_epoch | 43 |
| best_val_loss | 0.186405 |

![Sample utilization distribution](../data/results/sample_y_utilization_distribution.png)

> 图注：展示标签 utilization 分布。横坐标为utilization，纵坐标为frequency。

![Sample congestion ratio split](../data/results/sample_y_congestion_ratio_split.png)

> 图注：展示 train/val/test 拥塞正样本比例。横坐标为split，纵坐标为positive ratio。

![UGGRU train val loss](../data/results/uggru_train_val_loss.png)

> 图注：展示 UGGRU 训练和验证损失曲线。横坐标为epoch，纵坐标为loss。

### 5.1 为什么 edge graph 的节点是链路而不是卫星

本项目的预测目标是每一条星间链路的下一时刻状态，包括 utilization、load 和 congestion_label。若直接把卫星作为图节点，则模型天然更适合预测“卫星节点状态”；但本研究真正关心的是链路是否拥塞、链路负载是否升高，以及后续路由路径是否会经过高风险链路。因此，第 4 阶段构建的是 edge-to-edge graph：每条 ISL 是一个 edge-node，两条 ISL 如果共享同一颗卫星，就认为它们相邻。

这种建模方式的好处是，空间依赖关系直接发生在“被预测对象”之间。例如某颗卫星连接的 4 条 ISL 会共享转发压力，某条链路负载升高可能意味着相邻链路也承受相近方向的流量变化。edge graph 可以让 GraphConvLayer 在链路之间传播局部拓扑上下文，而不是先预测卫星状态再间接推断链路状态。对于后续路径风险聚合来说，这种链路级预测结果也更自然，因为候选路径本身就是 edge_path 的序列。

### 5.2 为什么 GraphConv + GRU 适合链路状态预测

链路状态预测同时包含空间依赖和时间依赖。空间依赖来自 LEO 网络拓扑：共享卫星的 ISL 之间存在转发耦合，某些拓扑位置更容易成为 Abilene OD 流量映射后的瓶颈。时间依赖来自业务流量本身：Abilene OD 矩阵具有连续时序变化，短时间内链路负载存在惯性，同时也会在高峰期或接入变化时出现突变。UGGRU 的结构正是对这两类依赖的组合建模。

GraphConvLayer 先在每个时间片上用 edge adjacency 聚合相邻链路特征，得到包含局部拓扑上下文的链路表示；随后 GRU 对每条链路的 12 个历史时间片序列建模，提取时间趋势。这样，模型既不会退化为完全独立地预测每条链路，也不会只依赖当前时刻的静态拓扑，而是把“链路邻域”和“历史趋势”合在一起用于下一时刻预测。

### 5.3 从 baseline 对比看 UGGRU 的作用

Last baseline 的 MAE_util 最低，表明链路利用率具有很强的短时惯性：很多普通链路在相邻时间片变化很小，直接复制最后一个历史值就能在平均绝对误差上取得不错结果。但 Last 的 RMSE_util 明显较高，说明它在峰值、突变和拥塞附近的误差更大；RMSE 对大误差更敏感，因此能暴露短时复制策略对异常变化的不足。

HA baseline 使用历史平均，会进一步平滑掉短时变化，因此拥塞识别几乎失效。GRU-only 和 LSTM-only 能学习时间序列，因此 recall 较高，但 Precision 和 F1 较低，说明它们在不使用链路图结构的情况下容易产生较多误报。UGGRU 的 F1 明显高于 GRU-only 和 LSTM-only，说明 edge graph 提供的空间结构信息确实有助于区分真实拥塞链路和仅有时间趋势相似的普通链路。

从论文写作角度看，这一部分可以作为“预测模型有效性验证”小节：Last 证明问题存在强时序惯性，GRU/LSTM 证明时间模型有价值，UGGRU 进一步证明链路拓扑结构能提升拥塞识别能力。

**本节如何连接下一阶段。**UGGRU 产生的是下一时间片的链路级预测，包括 utilization、load 和 congestion logit。下一阶段会先对拥塞概率进行阈值校准，再通过 MC Dropout 获得预测不确定性，把普通点预测扩展为带风险意识的链路级预测结果。

## 6. 阈值校准、MC Dropout 与预测模型对比

**本节为什么需要。**UGGRU 已经能输出链路级预测，但路由决策需要的不只是“一个预测值”，还需要知道预测是否可靠，以及哪些链路未来更值得避开。特别是在拥塞样本稀少的情况下，分类阈值会显著影响 Precision、Recall 和 F1。因此，本节解决两个问题：第一，如何用 validation set 选择更合理的拥塞阈值；第二，如何用 MC Dropout 为链路预测补充不确定性和风险排序能力。

由于拥塞标签高度不平衡，默认阈值 0.5 并不适合直接用于拥塞分类。当前在 validation set 上选择阈值 `0.95`，再固定应用到 test set，避免 test set 调参。

| 指标 | UGGRU test/val-to-test | MC Dropout |
| --- | --- | --- |
| MAE_util | 0.033072 | 0.033984 |
| RMSE_util | 0.160971 | 0.161991 |
| Precision | 0.460489 | 0.477283 |
| Recall | 0.625566 | 0.598702 |
| F1 | 0.530482 | 0.531142 |
| coverage_1std | - | 0.764586 |
| coverage_2std | - | 0.899498 |
| uncertainty_error_corr | - | 0.555298 |

MC Dropout（Monte Carlo Dropout，蒙特卡洛 Dropout）不是为了显著降低 MAE/RMSE，而是提供不确定性和风险排序。当前 `risk_score = util_pred_mean + lambda * util_pred_std`；Top 1%/5%/10% 高风险位置真实拥塞率分别为 0.574614、0.222687、0.123023，lift 分别为 43.54、16.87、9.32。

| Model | MAE_util | RMSE_util | Precision | Recall | F1 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| Last | 0.031954 | 0.211180 | 0.447329 | 0.447232 | 0.447281 | No-training persistence baseline |
| HA | 0.057850 | 0.221878 | 0.002793 | 0.000036 | 0.000072 | No-training historical average baseline |
| GRU-only | 0.033540 | 0.170046 | 0.145799 | 0.852047 | 0.248991 | Temporal baseline without edge graph |
| LSTM-only | 0.035003 | 0.169367 | 0.151694 | 0.848856 | 0.257391 | Temporal baseline without edge graph |
| UGGRU | 0.033072 | 0.160971 | 0.460489 | 0.625566 | 0.530482 | Graph and temporal model, threshold selected on validation |
| UGGRU + MC Dropout | 0.033984 | 0.161991 | 0.477283 | 0.598702 | 0.531142 | UGGRU with MC Dropout uncertainty |

Last baseline 的 MAE_util 最低但 RMSE 高，说明短时惯性强而峰值/突变预测弱；HA 平滑短时变化，拥塞识别几乎失效；GRU-only 与 LSTM-only 能建模时间序列但缺少链路图结构；UGGRU 引入 edge graph 后在 RMSE 和拥塞 F1 上更优。


![UGGRU threshold PRF curve](../data/results/uggru_threshold_prf_curve.png)

> 图注：展示不同阈值下 Precision、Recall、F1 的变化。横坐标为threshold，纵坐标为precision/recall/F1。

![Val-to-test threshold PRF curve](../data/results/uggru_val_to_test_threshold_prf_curve.png)

> 图注：展示 validation 阈值选择及 test 应用逻辑。横坐标为threshold，纵坐标为precision/recall/F1。

![MC uncertainty error scatter](../data/results/mc_dropout_uncertainty_error_scatter.png)

> 图注：展示预测标准差与绝对误差关系。横坐标为util_pred_std，纵坐标为absolute error。

![MC risk top-k](../data/results/mc_dropout_risk_topk.png)

> 图注：展示高风险 Top-k 位置真实拥塞率。横坐标为top-k ratio，纵坐标为true congestion rate。

![Prediction model comparison MAE RMSE](../data/results/prediction_model_comparison_mae_rmse.png)

> 图注：比较不同预测模型 MAE/RMSE。横坐标为model，纵坐标为MAE/RMSE。

![Prediction model comparison F1](../data/results/prediction_model_comparison_f1.png)

> 图注：比较不同预测模型拥塞 F1。横坐标为model，纵坐标为F1。

### 6.1 MC Dropout 的研究意义

如果只看 MAE/RMSE，MC Dropout 相比普通 UGGRU 并没有带来明显提升，甚至会因为随机采样平均而略有差异。但这不是 MC Dropout 在本项目中的主要价值。本项目需要的不只是一个点估计，而是一个能服务于路由决策的风险信号。对于路由而言，一条链路预测利用率较高但模型非常确定，与一条链路预测利用率中等但不确定性很高，可能都应被视为需要谨慎避开的对象。

因此，本研究将 `util_pred_mean` 和 `util_pred_std` 组合为 `risk_score = mean + lambda * std`。这个定义并不试图替代 utilization 预测，而是把预测均值和不确定性共同转化为保守风险估计。`coverage_2std=0.899498` 表明 mean ± 2std 对真实 utilization 有较高覆盖率；`uncertainty_error_corr=0.555298` 表明不确定性与预测误差存在中等正相关。换句话说，模型“不确定”的地方确实更容易出错，这使不确定性可以作为路由风险的一部分。

Top-k 风险排序结果对后续路由尤其关键。真实拥塞样本整体比例约为 1.32%，但按 risk_score 排序的 Top 1% 位置真实拥塞率达到 57.46%，lift 达到 43.54 倍。这说明 risk_score 能把少量最危险的链路位置显著富集出来。对于路由模块而言，这意味着不需要把所有链路都同等看待，而可以优先避开路径中包含高 risk_score 的链路，从而把预测模块的输出真正转化为动作选择依据。

### 6.2 阈值校准为什么必须在 validation set 上完成

拥塞分类是高度不平衡任务，默认 0.5 阈值通常会偏向高 recall 或高误报，无法满足后续路由评估需要。第 5 阶段先在 validation set 上扫描阈值，选择 best F1 threshold=0.95，再固定应用到 test set。这样做的关键不是让 test 指标最大化，而是避免把测试集用于调参，保证评估流程严谨。

从结果看，threshold=0.95 在 test set 上取得 Precision=0.460489、Recall=0.625566、F1=0.530482。这个 F1 不应孤立理解为“分类任务最终完美完成”，而应理解为“预测模块能够提供有可用精度的拥塞风险信号”。后续路径风险、action impact 和 RL 环境并不只依赖二值拥塞标签，而是更多使用连续的 utilization、cong_prob 和 risk_score，因此阈值校准只是预测模块输出体系中的一部分。

**本节如何连接下一阶段。**本节最终输出的是链路级风险：每条 edge 在每个 test sample 下的预测均值、预测标准差、拥塞概率和 risk_score。下一阶段不会直接把这些链路分数当作动作，而是先生成端到端候选路径，再把路径上的链路风险聚合成 path-level features。

## 7. 候选路径生成

**本节为什么需要。**上一节已经知道每条链路的预测风险，但路由必须在源卫星到目的卫星之间选择一条可行路径。如果没有候选路径集合，后续既无法聚合路径风险，也无法定义强化学习动作空间。因此，本节要解决“智能体有哪些可选动作”的问题：为每个有序卫星对生成固定数量 K=5 的候选路径。

第 8 阶段为 topo144 所有有序卫星对生成 K=5 候选路径，作为后续动作空间基础。候选路径生成只使用拓扑和 hop-based K-shortest paths，不使用 risk_score。

| 项目 | 数值 |
| --- | --- |
| K | 5 |
| 有序卫星 pair 数量 | 20592 |
| 总路径数 | 102960 |
| 平均每对候选路径数 | 5.000000 |
| hop_count mean | 6.931469 |
| hop_count p50 | 7.000000 |
| hop_count p95 | 11.000000 |
| hop_count max | 13 |
| 无路径 pair | False |
| 重复路径 | False |
| 路径连续性检查 | True |

![Candidate path hop count distribution](../data/results/candidate_path_hop_count_distribution.png)

> 图注：展示 K=5 候选路径跳数分布。横坐标为hop count，纵坐标为path count。

**本节如何连接下一阶段。**候选路径生成后，每个 OD 决策都有 K=5 个 path_id，每个 path_id 都有对应的 sat_path 和 edge_path。下一阶段会沿着这些 edge_path 读取链路级 risk_score、cong_prob 和 utilization 预测，并聚合为路径级风险特征。

## 8. 路径风险特征构建

**本节为什么需要。**上一阶段 UGGRU 与 MC Dropout 已经解决了“如何预测每条星间链路未来风险”的问题，输出的是链路级结果，例如 `util_pred_mean`、`util_pred_std`、`cong_prob_mean` 和 `risk_score`。但是路由决策不是选择单条链路，而是为一个 gateway OD 流选择一整条候选路径。也就是说，上一阶段已经知道“哪些链路危险”，但还不知道“哪条端到端路径危险”。本节出现的原因正是填补这个缺口：把链路级预测结果沿着候选路径的 `edge_path` 聚合为路径级风险特征，使预测结果能够被后续路由动作直接使用。

路径风险特征将链路级预测结果聚合到候选路径级别。每个 test sample 有 12 个 gateway 的 132 个有序 OD pair，每个 OD pair 有 K=5 条候选路径。

| 项目 | 数值 |
| --- | --- |
| features shape | [7257, 132, 5, 32] |
| N_test | 7257 |
| 每 sample OD pair 数 | 132 |
| K | 5 |
| 特征数 F | 32 |
| CSV 行数 | 4789620 |
| valid_mask 全 True | True |
| NaN/Inf | False / False |
| 抽样聚合检查 | True |
| rank_by_risk 1~K | True |
| shortest 与 min_risk 不同比例 | 0.494511 |
| min_risk 风险降低 | 0.196659 |
| min_risk hop_count 增加 | 0.389626 |
| 策略 | hop_count | delay_sum | current_util_max | pred_util_max | cong_prob_max | risk_score_max |
| --- | --- | --- | --- | --- | --- | --- |
| shortest_path | 3.474900 | 35.560780 | 0.587009 | 0.417917 | 0.652342 | 0.504526 |
| min_risk_path | 3.864526 | 40.023478 | 0.372471 | 0.245551 | 0.527992 | 0.307867 |
| min_cong_prob_path | 3.870629 | 39.830225 | 0.380285 | 0.251241 | 0.522778 | 0.316059 |
| min_delay_path | 3.474900 | 32.945477 | 0.670434 | 0.485564 | 0.675506 | 0.584401 |

zero-hop 场景表示源 gateway 和目的 gateway 同时接入同一颗卫星，此时 `edge_path=[]`，路径不经过 ISL。后续 RL 环境通过 action mask 将等价 zero-hop 动作压缩为只允许 path_id=0。


![Path risk score distribution](../data/results/path_risk_score_distribution.png)

> 图注：展示候选路径风险分数分布。横坐标为risk score，纵坐标为frequency。

![Path congestion probability distribution](../data/results/path_cong_prob_distribution.png)

> 图注：展示路径最大拥塞概率分布。横坐标为congestion probability，纵坐标为frequency。

![Path delay vs risk scatter](../data/results/path_delay_vs_risk_scatter.png)

> 图注：展示路径时延与风险之间的关系。横坐标为path_delay_ms_sum，纵坐标为path_risk_score_max。

![Shortest vs min risk comparison](../data/results/shortest_vs_min_risk_comparison.png)

> 图注：比较 shortest_path 与 min_risk_path。横坐标为strategy，纵坐标为metric。

![Min risk path id distribution](../data/results/min_risk_path_id_distribution.png)

> 图注：展示最低风险路径对应 path_id 分布。横坐标为path_id，纵坐标为count。

### 8.1 为什么链路级 risk_score 不能直接作为路由动作

UGGRU 和 MC Dropout 输出的是链路级风险，即每条 edge 在某个 test sample 下的风险状态。但路由动作选择的是一条从源接入卫星到目的接入卫星的路径，这条路径由多个 edge 组成。只看单条链路的 risk_score 无法判断整条路径是否安全：一条路径可能大部分链路很安全，但包含一条极高风险瓶颈链路；另一条路径可能每条链路风险都中等，但路径很长，累计风险和时延都较高。因此，链路级预测必须通过候选路径的 edge_path 聚合为路径级特征。

路径级聚合的关键是保留不同风险视角。max 聚合表示瓶颈风险，例如 `path_risk_score_max` 表示路径上最危险链路的风险；mean 聚合表示路径平均风险或平均负载状态；sum 聚合表示累计代价，适合描述长路径带来的总风险和总资源占用。对于 delay，sum 更符合端到端路径时延；对于 congestion probability，max 更适合表示“路径是否包含高拥塞概率链路”；对于 uncertainty，mean 和 max 分别表示整体不确定性和最不确定瓶颈。

### 8.2 shortest 与 min_risk 近半数不同的意义

结果显示，`shortest_path` 与 `min_risk_path` 的不同比例为 49.4511%。这说明在近一半的真实 gateway OD 决策中，按照跳数最短选择的路径并不是预测风险最低的路径。这个结果非常重要，因为它证明预测模块并不是只给出一个“附属指标”，而是真正会改变路径选择。

`min_risk_path` 相比 shortest_path 平均 `path_risk_score_max` 降低 0.196659，但平均 hop_count 增加 0.389626。这一现象符合路由直觉：为了避开高风险瓶颈链路，路径可能需要绕行，因此 hop_count 或 delay 会有所增加。也就是说，预测感知路由本质上不是追求最短，而是在“路径长度/时延”和“未来拥塞风险”之间做权衡。

这一步把流量预测模块和路由模块第一次真正连接起来。没有 path risk features，预测结果只能停留在链路层；有了路径级风险，后续 action impact 和 RL 环境才能把每条候选路径作为一个可评估、可比较、可选择的动作。

**本节如何连接下一阶段。**路径风险特征解决了“路径本身风险如何表示”的问题，但仍然没有回答“如果把当前 OD demand 真正加载到这条路径上，会让网络状态变成什么样”。例如两条路径的 `path_risk_score_max` 可能接近，但其中一条路径承载的 OD demand 更大、当前链路利用率更高，执行后可能造成更大的 MLU 增量。因此，本节输出的 path-level features 会进入下一阶段 Action Impact，进一步被转化为动作执行后的 post_mlu、delta_mlu、delta_congestion_count 和 action_cost。

## 9. 动作影响模型 Action Impact

**本节为什么需要。**上一节 path risk features 已经把链路级风险聚合到了路径级，能够说明“这条候选路径看起来有多危险”。但是这仍然只是路径静态属性，不能说明当前 OD demand 一旦选择这条路径后，会如何改变网络状态。两条路径的风险可能相近，但如果路径长度、当前链路负载、OD demand 大小不同，动作后果就可能完全不同。因此，仅有 path risk 还不足以支持路由决策，必须进一步评估“选择某条路径这个动作”会造成什么影响。

Action Impact 的本质是把“候选路径”变成“可比较的候选动作”。它不仅看路径自身的 delay 和风险，还模拟将当前 OD demand 增量加载到路径上的链路，计算执行动作后的 post_mlu、delta_mlu、delta_congestion_count、path_post_util_max 和 action_cost 等指标。这样，后续模块比较的不再只是路径属性，而是动作后果。

需要强调的是，Action Impact 是规则/仿真型单 OD 增量加载 proxy。给定当前网络状态、某个 gateway OD demand 和一条候选路径，模型将该 demand 增量加载到路径上的每条 ISL，计算 post-action 的局部与全网 proxy 指标。它不从 link_state 中移除原有流量，也不做全网多 OD 重新分配。因此它不是完整在线多流重路由仿真器，而是为后续 routing baseline 和 LeoRoutingEnv 提供动作后果表。

| 项目 | 数值 |
| --- | --- |
| impact_features shape | [7257, 132, 5, 49] |
| N_test | 7257 |
| gw_pair_count | 132 |
| K | 5 |
| F_impact | 49 |
| NaN/Inf | False / False |
| zero-hop action count | 830290 |
| zero-hop ratio | 0.173352 |
| post_mlu >= pre_mlu | True |
| delta_total_load 校验 | True |
| 抽样重算检查 | True |
| 策略 | post_mlu | delta_mlu | delta_congestion_count | action_cost | path_delay_ms_sum | path_risk_score_max |
| --- | --- | --- | --- | --- | --- | --- |
| shortest_path | 1.406086 | 0.005090 | 0.027019 | 181.484869 | 35.560780 | 0.504526 |
| min_delay_path | 1.406568 | 0.005571 | 0.026159 | 179.707853 | 32.945477 | 0.584401 |
| min_risk_path | 1.403249 | 0.002253 | 0.014972 | 183.576781 | 40.023478 | 0.307867 |
| min_cong_prob_path | 1.403328 | 0.002331 | 0.015762 | 183.481210 | 39.830225 | 0.316059 |
| min_action_cost_path | 1.402751 | 0.001755 | 0.009029 | 177.890029 | 33.623256 | 0.390138 |

`min_risk_path` 能明显降低预测风险和新增拥塞，但平均 delay 更高；`min_delay_path` 虽然时延最低，但风险与 post_mlu 较高；`min_action_cost_path` 综合 delay、MLU、拥塞增量和预测风险后 action_cost 最低。


![Action impact strategy post MLU](../data/results/action_impact_strategy_post_mlu.png)

> 图注：比较不同策略执行后的平均 post_mlu。横坐标为strategy，纵坐标为post_mlu。

![Action impact strategy delta MLU](../data/results/action_impact_strategy_delta_mlu.png)

> 图注：比较不同策略造成的 MLU 增量。横坐标为strategy，纵坐标为delta_mlu。

![Action impact strategy congestion delta](../data/results/action_impact_strategy_congestion_delta.png)

> 图注：比较不同策略造成的新增拥塞数量。横坐标为strategy，纵坐标为delta_congestion_count。

![Action impact strategy cost](../data/results/action_impact_strategy_cost.png)

> 图注：比较不同策略的 action_cost。横坐标为strategy，纵坐标为action_cost。

![Action impact shortest vs min risk scatter](../data/results/action_impact_shortest_vs_min_risk_scatter.png)

> 图注：比较 shortest_path 和 min_risk_path 的 action_cost 关系。横坐标为shortest action_cost，纵坐标为min_risk action_cost。

![Action impact selected path id distribution](../data/results/action_impact_selected_path_id_distribution.png)

> 图注：展示 min_action_cost_path 选择 path_id 分布。横坐标为path_id，纵坐标为count。

### 9.1 Action Impact 的特征体系

Action Impact 特征可以大致分为五类。第一类是基础标识特征，包括 sample_idx、time_t、src_gateway、dst_gateway、src_sat、dst_sat、path_id、hop_count、od_demand_mbps、is_zero_hop 和 is_valid，用于定位当前动作对应的时刻、业务流和候选路径。第二类是 pre-action 全网状态，包括 pre_mlu、pre_congestion_count、pre_avg_utilization、pre_total_load_mbps，用于描述动作执行前网络是否已经接近拥塞。

第三类是路径局部状态与动作增量，包括 path_pre_load_sum、path_pre_util_max、affected_edge_count、added_load_edge_sum 等。这些特征刻画“该 OD demand 加到哪些链路上、会加多少负载”。第四类是 post-action 指标，包括 path_post_util_max、post_mlu、delta_mlu、delta_congestion_count、delta_total_load_mbps 等，用于衡量动作造成的即时后果。第五类是预测风险相关特征，包括 path_pred_util_max、path_cong_prob_max、path_risk_score_max、path_util_uncertainty_mean 等，用于把预测模块的风险感知能力注入动作评估。

在这些指标中，post_mlu 表示执行动作后的全网最大链路利用率，越低越有利于负载均衡；delta_mlu 表示动作使全网 MLU 增加了多少；delta_congestion_count 表示动作新增了多少拥塞链路；action_cost 是启发式综合代价，由 delay、post_mlu、delta_congestion_count 和 path_risk_score_max 等分项组成。它不是最终 RL reward 的唯一形式，但为 routing baseline 和 RL reward 设计提供了直接参考。

### 9.2 为什么 Action Impact 是单 OD 增量加载 proxy

真实在线路由系统中，当某条 OD 流改变路径后，全网多个 OD 流可能同时重分配，链路状态会随时间滚动演化，后续时刻的 gateway 接入和卫星位置也会改变。当前 Action Impact 没有模拟这些复杂反馈，而是在已有 link_state 基础上，把单个 OD demand 增量加载到候选路径上，观察局部和全网 proxy 指标如何变化。

这种设计的局限是显然的：它不是完整全局 post-routing state，也不能直接表示长期多步效果。但它的优势是可控、可解释、可大规模预计算。对于每个 test sample、每个 gateway OD pair、每条候选路径，系统都能得到一组一致的动作后果特征。这使得后续 LeoRoutingEnv 可以快速读取这些结果进行 RL 训练，而不必在每个 step 中重新跑昂贵的网络仿真。

### 9.3 min_risk 与 min_action_cost 的差异

`min_risk_path` 选择 path_risk_score_max 最低的路径，因此它显著降低风险、delta_mlu 和新增拥塞。但它不直接优化 delay，也不直接优化 action_cost 的所有分项。结果中 min_risk_path 的 path_delay_ms_sum 高于 shortest_path，action_cost 反而比 shortest_path 更差。这说明“最低预测风险”并不等于“综合路由代价最低”，它更像是一个保守避险策略。

`min_action_cost_path` 则把 delay、post_mlu、delta_congestion_count 和 path_risk_score_max 放在同一个启发式代价中综合考虑，因此在 routing baseline 中表现最强：它既降低 delta_mlu 和新增拥塞，又降低 action_cost，delay 也没有明显恶化。后续强化学习的 reward 设计正是从这个思想出发，让策略学习在多个目标之间自动权衡，而不是固定使用某一个单指标规则。

因此，Action Impact 在研究链条中的位置非常关键：它把“预测风险”转化为“动作后果”，再把“动作后果”转化为“reward 和 observation”。没有这一步，RL 环境只能看到静态路径属性；有了这一步，智能体才能学习不同路径选择对 MLU、拥塞和风险的实际影响。

**本节如何连接下一阶段。**Action Impact 输出后，系统已经拥有每个 `sample_idx + gateway OD pair + path_id` 的动作后果。下一步可以有两种使用方式：第一，不训练模型，直接根据这些动作后果设计传统启发式策略，例如选择最低风险、最低拥塞概率或最低综合代价路径；第二，把这些动作后果封装为强化学习环境的 observation 和 reward，让智能体学习状态到动作的映射。因此，第 10 阶段会先基于 Action Impact 做传统 routing baseline，第 11 阶段再把它封装进 LeoRoutingEnv。

## 10. 传统启发式路由 Baseline 汇总

**本节为什么需要。**上一节 Action Impact 已经为每条候选路径计算了执行后的代价和影响。既然每个动作的后果已经可比较，就可以先不训练任何模型，而是用一组简单规则直接选路径。这一步的目的不是替代强化学习，而是回答一个更基础的问题：预测风险和动作代价是否真的能改善传统最短路径？如果基于风险或综合代价的简单规则都无法优于 shortest_path，那么后续训练 PPO 的意义就会不足。

第 11 阶段不训练新模型，而是基于 action impact 汇总五种传统/启发式策略：shortest_path、min_delay_path、min_risk_path、min_cong_prob_path 和 min_action_cost_path。

| 最优维度 | 策略 |
| --- | --- |
| best_post_mlu_strategy | min_action_cost_path |
| best_delta_mlu_strategy | min_action_cost_path |
| best_delta_congestion_count_strategy | min_action_cost_path |
| best_action_cost_strategy | min_action_cost_path |
| best_risk_score_strategy | min_risk_path |
| best_delay_strategy | min_delay_path |
| 策略 | post_mlu | delta_mlu | delta_congestion_count | action_cost | risk_score | delay | final_comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shortest_path | 1.406086 | 0.005090 | 0.027019 | 181.484869 | 0.504526 | 35.560780 | 基础最短路径 baseline，时延和风险均未显式优化。 |
| min_delay_path | 1.406568 | 0.005571 | 0.026159 | 179.707853 | 0.584401 | 32.945477 | 时延最低，但风险和 post_mlu 相对较高，说明纯时延优先不等于负载均衡最优。 |
| min_risk_path | 1.403249 | 0.002253 | 0.014972 | 183.576781 | 0.307867 | 40.023478 | 显著降低预测风险、delta_mlu 和新增拥塞，但平均路径更长，综合 action_cost 上升。 |
| min_cong_prob_path | 1.403328 | 0.002331 | 0.015762 | 183.481210 | 0.316059 | 39.830225 | 与最低风险路径类似，降低拥塞概率和新增拥塞，但牺牲一定时延。 |
| min_action_cost_path | 1.402751 | 0.001755 | 0.009029 | 177.890029 | 0.390138 | 33.623256 | 综合 delay、MLU、拥塞增量和风险后 action_cost 最低，是当前最强启发式 baseline。 |
| 相对 shortest_path | delta_mlu 改善 | 新增拥塞改善 | risk_score 改善 | action_cost 改善 | delay change |
| --- | --- | --- | --- | --- | --- |
| min_risk_path | 55.742142 | 44.586972 | 38.978936 | -1.152665 | -12.549494 |
| min_action_cost_path | 65.528223 | 66.582953 | 22.672318 | 1.980793 | 5.448486 |

`min_risk_path` 相比 shortest_path 显著降低 delta_mlu、新增拥塞和 risk_score，但 action_cost 上升；`min_action_cost_path` 在综合代价上最优，是当前最强启发式 baseline。

这些启发式策略分别代表不同决策偏好：`min_delay_path` 代表纯时延优先，`min_risk_path` 代表预测风险优先，`min_cong_prob_path` 代表拥塞概率优先，`min_action_cost_path` 代表综合 delay、MLU、拥塞增量和风险后的代价优先。它们共同构成 PPO 的对照组：PPO 至少应该优于 random_valid 和 shortest，并尽量接近 min_action_cost 或 min_raw_cost 这类强启发式策略。


![Routing baseline post MLU comparison](../data/results/routing_baseline_post_mlu_comparison.png)

> 图注：比较五种启发式策略的 post_mlu。横坐标为strategy，纵坐标为post_mlu。

![Routing baseline delta MLU comparison](../data/results/routing_baseline_delta_mlu_comparison.png)

> 图注：比较五种策略对 MLU 的增量影响。横坐标为strategy，纵坐标为delta_mlu。

![Routing baseline delta congestion comparison](../data/results/routing_baseline_delta_congestion_comparison.png)

> 图注：比较五种策略新增拥塞数量。横坐标为strategy，纵坐标为delta_congestion_count。

![Routing baseline action cost comparison](../data/results/routing_baseline_action_cost_comparison.png)

> 图注：比较五种策略综合 action_cost。横坐标为strategy，纵坐标为action_cost。

![Routing baseline risk delay tradeoff](../data/results/routing_baseline_risk_delay_tradeoff.png)

> 图注：展示策略在 delay 与 risk_score 之间的权衡。横坐标为path_delay_ms_sum，纵坐标为path_risk_score_max。

![Routing baseline improvement vs shortest](../data/results/routing_baseline_improvement_vs_shortest.png)

> 图注：展示各策略相对 shortest_path 的改善百分比。横坐标为strategy，纵坐标为improvement percentage。

**本节如何连接下一阶段。**传统 routing baseline 证明了 action impact 表确实包含有用决策信息：不同规则选出的路径在 post_mlu、delta_mlu、delta_congestion_count 和 action_cost 上差异明显。下一阶段要做的是把这种“规则选路径”升级为“智能体学策略”：不再固定写死某一个规则，而是定义 RL 环境，让 PPO 根据 observation 学习在不同状态下应该偏向 delay、risk、congestion 还是综合 cost。

## 11. 强化学习环境 LeoRoutingEnv

**本节为什么需要。**Action Impact 生成的是一个静态动作后果表，routing baseline 则是在这个表上手写规则选路径。要训练强化学习智能体，仅有表格还不够，必须把问题形式化为 State / Action / Reward / Transition：智能体在什么状态下观察什么信息、可以选择哪些动作、选择后得到什么 reward、环境如何推进到下一步。LeoRoutingEnv 的作用就是把 action impact 表封装成 Gymnasium 风格环境，使后续 MaskedPPO 可以直接训练。

LeoRoutingEnv 是 Gymnasium 风格的离线 action-impact 路由环境。每一步决策对应一个 `sample_idx + gateway OD pair`，动作是从 K=5 条候选路径中选择一个 path_id。它读取预计算的 action impact features，而不是在线重新计算全网链路状态。

| 项目 | 数值/说明 |
| --- | --- |
| observation shape | [140] |
| action_space | Discrete(5) |
| observation 构成 | 5 × 27 selected features + 5 action mask = 140 |
| action_masks shape | (5,) bool |
| zero-hop mask | zero-hop 时只允许 path_id=0 |
| invalid action penalty | invalid 动作不崩溃，返回惩罚并推进 |
| reward_mode | relative_to_shortest |
| reward 公式 | shortest_path_raw_cost - selected_action_raw_cost |
| rollout 检查 | True |
| invalid action 测试 | True |

需要 action mask 的原因有两点：过滤无效动作，以及在 zero-hop 场景下屏蔽 path_id=1~4 的等价动作。当前环境可被 MaskablePPO 调用，但仍是离线环境。

| Policy | mean_reward | mean_raw_cost | mean_post_mlu | mean_delta_congestion_count | mean_risk_score | invalid_action_count |
| --- | --- | --- | --- | --- | --- | --- |
| random_valid | -6.249836 | 191.505412 | 1.405960 | 0.026508 | 0.548193 | 0 |
| shortest | 0.000000 | 185.255577 | 1.406086 | 0.027019 | 0.504526 | 0 |
| min_raw_cost | 4.256782 | 180.998794 | 1.402522 | 0.009177 | 0.379573 | 0 |
| min_risk | -1.186437 | 186.442014 | 1.403249 | 0.014972 | 0.307867 | 0 |
| min_cong_prob | -1.072660 | 186.328236 | 1.403328 | 0.015762 | 0.316059 | 0 |

![Routing env policy reward comparison](../data/results/routing_env_policy_reward_comparison.png)

> 图注：比较环境中 heuristic policy 的平均 reward。横坐标为policy，纵坐标为mean_reward。

![Routing env policy cost comparison](../data/results/routing_env_policy_cost_comparison.png)

> 图注：比较 heuristic policy 的 raw_cost。横坐标为policy，纵坐标为mean_raw_cost。

![Routing env policy action distribution](../data/results/routing_env_policy_action_distribution.png)

> 图注：展示不同策略选择 action_id 的分布。横坐标为action_id，纵坐标为count。

### 11.1 State / Action / Reward / Transition 设计

LeoRoutingEnv 的 state 不是原始链路状态矩阵，而是当前 OD 决策对应的 K=5 条候选路径动作影响特征。对于当前 `sample_pos` 和 `gw_pair_pos`，环境读取 `impact_features[sample_pos, gw_pair_pos, :, :]`，从 49 个 action impact 特征中选择 27 个与路由决策最相关的特征，再将 5 条路径的特征 flatten 成一维向量，最后拼接 5 维 action mask。因此 observation shape 为 `5×27+5=140`。

这种 flatten 设计是为了兼容 Stable-Baselines3 / sb3-contrib 中常用的 MlpPolicy。虽然从结构上看候选路径集合更适合用 set encoder 或 attention 处理，但第一版环境优先保证稳定、可训练和可检查。每个 observation 中同时包含所有候选动作的信息，因此 MLP 仍然可以学习不同 path_id 之间的相对优劣。

Action 是 `Discrete(5)`，表示选择当前 OD pair 的第几条候选路径。Reward 默认使用 `relative_to_shortest`：`reward = shortest_path_raw_cost - selected_action_raw_cost`。如果选择的动作比 shortest_path 代价低，reward 为正；如果更差，reward 为负。这样 reward 具有清晰基准含义：智能体不是直接追求绝对 cost，而是在学习相对最短路径能改善多少。

Transition 在当前环境中是离线顺序推进：每一步从当前 gateway OD pair 推进到下一个 OD pair；132 个 OD pair 结束后进入下一个 test sample。环境并不根据智能体动作更新未来 link_state，因此它不是在线动态仿真器。这一点必须在论文中明确说明，否则容易让读者误以为 PPO 已经在完整动态网络中闭环运行。

### 11.2 Action mask 的必要性

action mask 在本环境中不是可选装饰，而是保证动作空间语义正确的关键机制。首先，候选路径生成和后续处理虽然大多数情况下 valid_mask 全 True，但环境仍保留基础 valid mask，以便未来扩展到断链、不可见链路或候选路径不足时使用。其次，zero-hop 场景下源 gateway 和目的 gateway 接入同一颗卫星，5 条候选路径在 ISL 层面都是空路径或等价动作。如果不 mask，PPO 会在这些无意义 path_id 之间学习随机差异，增加训练噪声。

zero-hop 时只允许 path_id=0，可以把等价动作压缩为唯一合法动作。检查结果中 zero-hop mask 生效，invalid action 测试通过。后续 MaskedPPO 训练和评估中 invalid_action_count=0，说明模型确实通过 mask 避免了非法动作。这也是选择 MaskablePPO 而不是普通 PPO 的直接原因之一。

### 11.3 环境结果如何理解

heuristic policy 评估表明，random_valid 明显差于 shortest，说明动作空间中不同候选路径质量差异显著，随机选择不可取。shortest 的 reward 为 0，因为 reward 定义就是相对 shortest 的改善。min_raw_cost reward 最高，因为它每一步都直接选择 raw_cost 最小动作。min_risk 和 min_cong_prob 在 risk 或 cong_prob 上更低，但 mean_reward 为负，说明单纯避险可能牺牲 delay 或其他 cost 分项。

这些结果为 PPO 训练提供了三类参照：random_valid 是下界，shortest 是传统基线，min_raw_cost 是 oracle-like greedy upper-bound。PPO 的合理目标不是超过 min_raw_cost，而是显著优于 random_valid 和 shortest，并尽量缩小与 min_raw_cost 的差距。

**本节如何连接下一阶段。**LeoRoutingEnv 给出了标准化的 observation、action_space、reward 和 action_masks。下一阶段 MaskablePPO 将直接使用这个环境进行训练：每一步读取 observation，结合 action mask 选择一个 path_id，并根据 relative_to_shortest reward 更新策略。换句话说，本节把“可比较的动作后果表”正式变成了“可训练的强化学习问题”。

## 12. MaskablePPO 训练过程与调参

**本节为什么需要。**上一节已经把路由问题封装成 LeoRoutingEnv，但普通 PPO（Proximal Policy Optimization，近端策略优化）并不知道哪些动作在当前状态下不可选，也不能自动处理 zero-hop 场景中的等价动作。由于当前动作空间虽然固定为 K=5，但不同状态下有效动作语义可能不同，因此需要使用能够读取 `action_masks()` 的 MaskablePPO。MaskablePPO 学习的不是直接读取 `min_raw_cost`，而是从 140 维 observation 中学习“当前状态下应该选择哪个 path_id”。

PPO 训练使用 sb3-contrib 的 MaskablePPO，使策略在采样和评估时都能读取 `action_masks()`。本阶段只训练路由策略，不重新训练流量预测模型。

### 12.1 Smoke training
smoke training 使用约 10000 timesteps、max_samples=500，目标是验证 Gymnasium 环境、action mask、模型保存/加载和评估流程。结果优于 random_valid 和 shortest，invalid_action_count=0，但不追求最优性能。

### 12.2 masked_ppo_mid
`masked_ppo_mid` 使用 timesteps=50000、max_samples=2000，实际训练步数 51200。全量 test mean_reward=2.852388，优于 random_valid 和 shortest，但低于 min_raw_cost。

### 12.3 masked_ppo_continue100k
`masked_ppo_continue100k` 从 mid best_model 继续训练 additional 50000，实际累计步数 101200。全量 test mean_reward=2.994414，相比 mid 提升 +0.142026。

### 12.4 masked_ppo_ent002
`masked_ppo_ent002` 将 ent_coef 提高到 0.02 并从头训练 50000。quick2000 mean_reward=2.458092，低于继续评估阈值，因此未做全量评估。

### 12.5 masked_ppo_lr1e4
`masked_ppo_lr1e4` 从 continue100k best_model 继续训练，learning_rate=1e-4，additional 50000，实际累计步数 151201。全量 test mean_reward=3.135387，是当前最优 PPO run。

| Run | resume/设置 | actual timesteps | wall_time_seconds | best_eval_mean_reward | full/quick 结果 |
| --- | --- | --- | --- | --- | --- |
| masked_ppo_mid | from scratch, lr=3e-4, ent=0.01 | 51200 | 1276.199473 | 2.390852 | full mean_reward=2.852388 |
| masked_ppo_continue100k | resume mid, lr=3e-4, ent=0.01 | 101200 | 1282.100000 | 2.524202 | full mean_reward=2.994414 |
| masked_ppo_ent002 | from scratch, lr=3e-4, ent=0.02 | 51200 | 1644.484630 | 2.366886 | quick2000 mean_reward=2.458092，未全量 |
| masked_ppo_lr1e4 | resume continue100k, lr=1e-4, ent=0.01 | 151201 | 2289.006529 | 2.639668 | full mean_reward=3.135387 |

![MaskedPPO mid policy reward comparison](../data/results/masked_ppo_mid_policy_reward_comparison.png)

> 图注：展示 mid run 与 heuristic policies 的 reward 对比。横坐标为policy，纵坐标为mean_reward。

![MaskedPPO continue100k policy reward comparison](../data/results/masked_ppo_continue100k_policy_reward_comparison.png)

> 图注：展示 continue100k run 的 reward 对比。横坐标为policy，纵坐标为mean_reward。

![MaskedPPO lr1e4 policy reward comparison](../data/results/masked_ppo_lr1e4_policy_reward_comparison.png)

> 图注：展示 lr1e4 run 的 reward 对比。横坐标为policy，纵坐标为mean_reward。

![MaskedPPO run reward comparison](../data/results/masked_ppo_run_reward_comparison.png)

> 图注：比较多个 PPO run 的平均 reward。横坐标为run/policy，纵坐标为mean_reward。

![MaskedPPO run cost comparison](../data/results/masked_ppo_run_cost_comparison.png)

> 图注：比较多个 PPO run 的 raw_cost。横坐标为run/policy，纵坐标为mean_raw_cost。

![MaskedPPO run congestion comparison](../data/results/masked_ppo_run_congestion_comparison.png)

> 图注：比较多个 PPO run 的新增拥塞指标。横坐标为run/policy，纵坐标为mean_delta_congestion_count。

![MaskedPPO gap to min raw cost](../data/results/masked_ppo_run_gap_to_min_raw_cost.png)

> 图注：展示各 run 到 min_raw_cost 上界的差距。横坐标为run，纵坐标为gap。

### 12.6 为什么使用 MaskablePPO 而不是普通 PPO

普通 PPO 只能在固定离散动作空间中采样动作，无法天然理解某些动作在当前状态下不可选。LeoRoutingEnv 中虽然 action_space 固定为 Discrete(5)，但不同 OD pair 的合法动作集合可能不同，尤其 zero-hop 场景下 path_id=1~4 是等价无意义动作，应当被屏蔽。如果使用普通 PPO，模型仍可能采样这些动作，导致 invalid penalty、训练噪声和策略解释混乱。

MaskablePPO 在采样和评估时都读取 `action_masks()`，使策略概率分布只定义在合法动作上。这不仅提高训练稳定性，也让评估结果更可信：最终多个 PPO run 的 invalid_action_count 均为 0，说明动作掩码在训练和测试中都生效。对于后续扩展动态 ISL 或链路故障场景，action mask 会更重要，因为候选路径可能因为链路断开或可见性不足而临时不可用。

### 12.7 各个 PPO run 的实验目的

smoke training 的目的不是追求性能，而是验证工程链条：环境能否被 MaskablePPO 接收、action mask 是否能传入模型、模型能否保存和加载、评估脚本能否统计所有指标。smoke 成功后，才进入中等规模训练。

`masked_ppo_mid` 是第一轮中等规模正式训练，用于确认 PPO 在 50000 timesteps、max_samples=2000 下能否超过 random_valid 和 shortest。结果显示 mean_reward=2.852388，证明 PPO 可以利用 observation 中的动作影响特征学习到优于传统最短路径的策略。

`masked_ppo_continue100k` 从 mid 的 best model 延续训练，目的是观察训练曲线是否仍有提升空间。它将 full test mean_reward 提升到 2.994414，说明原 mid run 尚未完全收敛，继续训练有效。但提升幅度约 +0.142，属于中等偏小，因此不宜盲目直接扩大到 200k。

`masked_ppo_ent002` 提高 ent_coef 到 0.02，意图增强探索，观察是否能跳出已有策略模式。但 quick2000 mean_reward=2.458092，明显低于 continue100k 和后续评估阈值，说明在当前环境中更高探索强度没有收益，可能导致策略在已经较明确的 action_cost 结构中探索过度，降低稳定性。

`masked_ppo_lr1e4` 从 continue100k best model 继续训练，同时把 learning_rate 降到 1e-4，目的是在已有策略基础上做更细粒度微调。它最终 full test mean_reward=3.135387，成为当前最优 PPO run。这个结果说明在策略已经具备基本能力后，较低学习率有利于继续细化决策边界，而不是像高探索设置那样破坏已有策略。

### 12.8 PPO 结果能说明什么，不能说明什么

PPO 结果说明：当前 observation 设计、reward 设计和 action mask 是有效的，智能体能够从 action impact 特征中学习到优于 shortest_path 的动作选择规律。`masked_ppo_lr1e4` 相比 shortest_path 显著降低 mean_raw_cost 和 mean_delta_congestion_count，且 invalid_action_count=0，证明它不是靠非法动作或评估漏洞取得提升。

PPO 结果不能说明：当前策略已经是最终最优在线路由算法。首先，训练和评估都在离线 action-impact 环境中进行；其次，min_raw_cost 仍然更强，说明 PPO 还没有完全逼近单步 oracle-like 上界；再次，当前主要是单 seed 结果，正式论文应补充多随机种子和方差。更重要的是，在线多步动态环境中，策略动作会改变未来状态，这种长期反馈目前还没有被建模。

因此，当前更稳妥的结论是：MaskedPPO 已经验证了“动作影响特征 + action mask + 相对最短路径 reward”这一路由学习框架可行，并在离线环境中显著优于 shortest_path；但它仍处于从离线 proxy 环境向真实在线路由系统过渡的阶段。

**本节如何连接下一阶段。**MaskedPPO 训练结果提供了最终路由策略评估对象：`masked_ppo_mid`、`masked_ppo_continue100k`、`masked_ppo_ent002` 和 `masked_ppo_lr1e4`。下一节将把这些 PPO run 与 random_valid、shortest 和 min_raw_cost 放在同一张表中比较，回答“最终 PPO 到底处在什么水平”这一核心问题。

## 13. 最终 PPO 与 Baseline 对比

| policy | mean_reward | mean_raw_cost | mean_post_mlu | mean_delta_mlu | mean_delta_congestion_count | mean_risk_score | mean_cong_prob | invalid_action_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_valid | -6.249836 | 191.505412 | 1.405960 | 0.004964 | 0.026508 | 0.548193 | 0.677753 | 0 |
| shortest | 0.000000 | 185.255577 | 1.406086 | 0.005090 | 0.027019 | 0.504526 | 0.652342 | 0 |
| masked_ppo_mid | 2.852388 | 182.403188 | 1.403941 | 0.002944 | 0.014061 | 0.420950 | 0.608047 | 0 |
| masked_ppo_continue100k | 2.994414 | 182.261162 | 1.403640 | 0.002643 | 0.013183 | 0.410776 | 0.595185 | 0 |
| masked_ppo_lr1e4 | 3.135387 | 182.120189 | 1.403644 | 0.002648 | 0.012428 | 0.403261 | 0.594860 | 0 |
| min_raw_cost | 4.256782 | 180.998794 | 1.402522 | 0.001525 | 0.009177 | 0.379573 | 0.582809 | 0 |

当前最终链条可以概括为：`random_valid << shortest_path << MaskedPPO(lr1e4) < min_raw_cost`。`masked_ppo_lr1e4` 相比 shortest_path 将 mean_raw_cost 从 185.255577 降至 182.120189，将 mean_delta_congestion_count 从 0.027019 降至 0.012428，invalid_action_count=0。它仍低于 min_raw_cost，reward_gap=1.121395。

## 14. 关于 min_raw_cost 为什么这么强，以及为什么仍需要强化学习

**本节为什么需要。**上一节最终对比显示，`masked_ppo_lr1e4` 已经明显优于 shortest_path，但仍低于 `min_raw_cost`。如果不解释 `min_raw_cost` 的性质，读者很容易产生疑问：既然 min_raw_cost 最强，为什么还需要 PPO？因此，本节专门讨论 `min_raw_cost` 的定位，说明它为什么强、为什么不能当作普通在线算法、以及它在本文实验中的作用。

`min_raw_cost` 直接读取每个候选动作的 raw_cost，并在当前 step 选择 raw_cost 最小路径。由于当前 reward_mode 是 `relative_to_shortest`，它等价于每一步直接选择当前 reward 最大动作。在离线 action-impact 环境中，它天然很强，应定义为 **oracle-like greedy upper-bound baseline**，而不是普通在线路由算法。

答辩可用表述：虽然 `min_raw_cost` 的单步指标最好，但它依赖已知每个候选动作的完整动作影响代价，等价于访问了动作后果表，因此更适合作为上界参考而不是可部署的在线路由策略。PPO 的意义在于学习从状态到动作的映射；在后续更接近真实的在线多步环境中，完整动作影响不是免费可得，单步贪心也未必长期最优。当前 MaskablePPO 已经显著优于 shortest_path 和 random_valid，说明状态特征与 action mask 能够支持策略学习。

### 14.1 min_raw_cost 的“强”来自哪里

`min_raw_cost` 的强势表现并不是偶然的。当前环境中，每个候选动作的 action impact 已经被预先计算出来，包括 delay、post_mlu、delta_congestion_count、risk_score 等分项，raw_cost 也是由这些分项加权得到的。`min_raw_cost` 在每个 step 直接遍历 K=5 个候选动作，选择 raw_cost 最小者。因此它不是通过学习得到策略，而是直接访问了当前 reward 所依赖的目标函数。

换句话说，在 `relative_to_shortest` reward 下，`min_raw_cost` 的选择等价于最大化当前 step reward。它天然是单步贪心上界，类似 oracle-like baseline。它的作用是告诉我们：“如果每一步都能免费知道所有候选动作的完整代价，那么最好能做到什么程度？”因此它非常适合作为 PPO 的性能上界参照。

### 14.2 为什么不能把 min_raw_cost 当成普通在线路由算法

真实在线低轨卫星路由中，动作后果并不一定能被免费、准确、即时地得到。要获得每条候选路径的真实 post_mlu、delta_congestion_count 和长期影响，系统需要知道当前全网所有流量分布、未来接入变化、其他 OD 流的同步决策以及链路状态演化。当前 `min_raw_cost` 使用的是第 10 阶段预计算的 action impact 表，因此它依赖离线数据和已知后果。

此外，`min_raw_cost` 是单步贪心策略。即使在某一步选择 raw_cost 最小动作，也不代表长期多步最优。真实路由可能需要为了未来状态保留容量，或者在多个 OD 流之间做协调。单步最小代价可能在短期内最优，但在多步动态环境中导致后续拥塞或路径振荡。因此，在论文中必须把 `min_raw_cost` 定位为“强启发式上界”或“oracle-like greedy upper-bound”，而不是本文最终算法。

### 14.3 为什么仍然需要强化学习

强化学习的目标不是在当前离线表格环境中机械地超过 oracle-like 上界，而是学习一个可泛化的状态到动作映射。PPO 策略在部署时只需要 observation，而不必显式枚举和手写每个动作的完整代价规则。随着后续环境扩展到在线动态仿真，动作后果可能不再被预先计算，reward 也可能包含长期回报、稳定性、切换成本、安全约束等因素，此时 RL 的优势会更明显。

当前 MaskedPPO 结果已经说明：在有限训练步数下，智能体能从 action impact 特征中学习到显著优于 shortest_path 的策略，并保持 invalid_action_count=0。这一结果并不意味着 PPO 已经完成最终路由优化，而是证明了使用 action mask 强化学习处理候选路径动作空间是可行的。后续如果加入在线状态转移、多步 reward 和多流交互，PPO 或更高级的 masked RL 方法才有机会体现超过单步启发式的长期决策能力。

答辩中如果被问到“既然 min_raw_cost 最强，为什么不用它”，可以回答：`min_raw_cost` 是基于已知动作后果表的单步 oracle-like 上界，用来衡量当前 reward 下的理论贪心水平；真实在线路由中动作后果不是免费可得，且单步最优不等于长期最优。本文保留 min_raw_cost 作为强 baseline，是为了诚实展示 PPO 与上界的差距，同时证明 PPO 已经显著优于传统 shortest_path，并为后续在线多步环境奠定策略学习基础。

**本节如何连接下一阶段。**明确 min_raw_cost 的定位后，论文价值分析才不会过度宣称 PPO 已经超过所有方法，也不会因为 PPO 低于 min_raw_cost 而否定研究意义。下一节将从硕士论文角度说明：当前工作的价值不在于声称 PPO 已达最终最优，而在于构建了从预测、风险、动作影响到 masked RL 的完整可验证框架。

## 15. 当前成果是否支撑硕士论文

从工作量看，项目已经覆盖数据解析、星座建模、链路状态生成、预测模型、不确定性评估、路径风险聚合、动作影响模型、传统 routing baseline、Gymnasium 环境和 MaskablePPO 训练评估，整体闭环较完整，足以支撑硕士论文实验部分。创新性更偏系统集成型和方法应用改进型：将 UGGRU 预测、不确定性风险分数、动作影响 proxy 和 action mask 强化学习串联到 LEO 路由问题中。理论创新中等偏弱，因此正式论文中应加强问题定义、指标设计、消融实验和局限性表述。

建议题目：1）基于流量预测与动作影响评估的低轨卫星网络负载均衡路由方法研究；2）面向低轨卫星网络的预测感知负载均衡路由方法研究；3）基于流量预测和动作掩码强化学习的低轨卫星网络路由优化研究。

### 15.1 面向硕士论文的价值分析

当前工作的工作量主要体现在完整链条的工程实现和实验闭环上。从原始 Abilene 数据下载解析，到 LEO topo144 星座与 gateway 动态接入，再到 link_state 链路状态数据集、UGGRU 预测模型、MC Dropout 不确定性、baseline 对比、候选路径、路径风险、动作影响模型、Gymnasium 环境和 MaskedPPO 训练评估，已经形成了一个覆盖数据、模型、路由与强化学习的较完整实验系统。这种完整性本身就是硕士论文中很重要的工程与实验贡献。

可以凝练的创新点至少有三个。第一，面向链路预测任务构建 edge graph，并用 UGGRU 进行链路级时空预测，使预测对象与路由路径中的 edge_path 一致。第二，将 MC Dropout 不确定性转化为 risk_score，并进一步聚合到路径级风险特征，实现从链路预测到路径选择的桥接。第三，提出基于 action impact 的离线路由环境，将候选路径动作、预测风险、动作后果和 action mask 结合起来，为 MaskedPPO 路由策略训练提供了可解释的状态与 reward。

可以直接进入论文的结果包括：Abilene 单位换算与数据规模、topo144 星座与接入统计、link_state 数据集诊断、UGGRU 与 baseline 对比、MC Dropout 风险排序 lift、path risk 中 shortest 与 min_risk 的差异比例、action impact 中 min_action_cost 的优势、LeoRoutingEnv 检查结果、MaskedPPO 多 run 对比结果。这些内容可以支撑论文第 3 章系统建模、第 4 章预测模型、第 5 章路由决策与实验评估。

需要谨慎处理的结果包括：PPO 当前仍低于 min_raw_cost；RL 环境是离线 action-impact proxy；当前主要是单 seed；ISL 边集合固定；remain_visible_time 是占位；业务流量还没有多场景扩展。这些不应被隐藏，而应在论文“实验设置”和“局限性”中主动说明。主动说明这些限制反而能增强论文可信度，避免被认为过度宣称。

正式论文还建议补充：多随机种子下的 PPO 结果均值和标准差；UGGRU 的更规范消融实验；PPO reward 权重敏感性分析；若时间允许，加入动态 ISL 或至少模拟链路不可用场景；整理更符合论文风格的图表，包括统一字体、颜色和坐标标签；补充复杂度分析，说明预测、候选路径评估和 PPO 决策的计算开销。

## 16. 当前简化假设与不足

| 不足/假设 | 说明 |
| --- | --- |
| 固定 ISL 边集合 | 当前不是完整动态断链/重连星间拓扑。 |
| remain_visible_time 占位 | 当前为 9999，占位而非真实链路剩余可见时间。 |
| 业务流量单一 | 主要使用 Abilene OD，尚未构造均匀/人口/周期/突发多场景。 |
| link_state 生成基线 | 使用 shortest-delay Dijkstra，不代表最终路由策略。 |
| action impact proxy | 只做单 OD 增量加载，不做多 OD 全局重路由。 |
| RL 离线环境 | 读取预计算 action impact，不是在线动态仿真器。 |
| PPO 单 seed 为主 | 论文阶段需要补充多随机种子统计。 |
| min_raw_cost 优于 PPO | 需要明确其 oracle-like upper-bound 定位。 |
| 无真实 TLE/SGP4 | 当前星座运动是简化建模。 |
| 无故障/断链场景 | 尚未评估链路故障、卫星故障或业务突发扰动。 |

## 17. 后续工作计划

| 阶段 | 计划 |
| --- | --- |
| 短期 | 整理最终图表；补充随机种子；输出论文材料包；生成最终实验表；明确 min_raw_cost 和 offline environment 定位。 |
| 中期 | 引入动态 ISL；计算真实 remain_visible_time；扩展多场景流量；补充多随机种子统计和消融实验。 |
| 长期 | 构建在线多步动态路由环境；支持多流联合重路由；探索分布式多智能体；接入 TLE/SGP4；加入故障和业务突发场景。 |

## 18. 可直接用于论文/汇报的阶段性结论

- 已经形成从 Abilene 真实 OD 流量到 LEO 链路状态预测、路径风险聚合和路由决策的完整实验闭环。
- UGGRU 同时利用 edge graph 空间结构和 GRU 时间依赖，相比纯时间序列 baseline 在 RMSE 和拥塞 F1 上表现更优。
- MC Dropout 提供了有意义的不确定性估计，risk_score 能显著富集真实拥塞链路。
- 路径风险特征表明最低风险路径与最短路径约 49.45% 情况下不同，说明预测风险会改变路径选择。
- Action impact 模型能够量化候选路径动作对 MLU、新增拥塞、delay 和综合 cost 的影响。
- min_action_cost 是强启发式 baseline，MaskedPPO(lr1e4) 则学习状态到动作的映射。
- MaskedPPO(lr1e4) 显著优于 shortest_path，但仍低于 min_raw_cost 上界。
- 当前系统是完整实验闭环，但不是最终真实在线多流动态系统。

## 18.1 十张关键图的深度解释与论文放置建议

1. `uggru_train_val_loss.png`：该图主要现象是训练损失和验证损失整体下降并趋于稳定，说明 UGGRU 在当前样本构建方式下可以有效收敛。它支持“预测模型训练过程稳定”的结论，适合放在论文预测模型实验章节。局限是训练曲线只能说明收敛，不足以单独证明模型泛化能力，还需结合 test 指标和 baseline 对比。

2. `prediction_model_comparison_f1.png`：该图突出 UGGRU 与 UGGRU+MC Dropout 的 F1 明显高于 Last、HA、GRU-only 和 LSTM-only。它支持“图结构 + 时间序列建模有助于拥塞识别”的结论，适合放在预测模型对比实验章节。局限是 F1 依赖阈值，论文中必须说明 UGGRU 阈值来自 validation set。

3. `mc_dropout_risk_topk.png`：该图展示风险排序前 1%、5%、10% 位置的真实拥塞率明显高于随机基线。它支持“MC Dropout 的主要价值是风险排序而非降低 MAE”的结论，适合放在不确定性评估章节。局限是该图只评估 test set 上的排序能力，还没有直接证明在线路由长期收益。

4. `candidate_path_hop_count_distribution.png`：该图展示 K=5 候选路径的 hop_count 分布，说明动作空间中既有短路径，也有一定绕行路径。它适合放在候选路径生成章节，用于解释动作空间规模和路径长度范围。局限是候选路径只按 hop-based K-shortest 生成，没有考虑动态链路可见性。

5. `path_delay_vs_risk_scatter.png`：该图展示路径 delay 与 risk_score 的关系，能直观看到低风险路径不一定是最低 delay 路径。它支持“预测风险会引入 delay-risk tradeoff”的结论，适合放在路径风险特征章节。局限是散点图点数较多，论文中可能需要抽样或使用透明度优化可读性。

6. `shortest_vs_min_risk_comparison.png`：该图比较 shortest_path 与 min_risk_path 的差异，结合 49.4511% 不同比例，说明预测风险确实会改变路径选择。它适合放在预测感知路径选择章节。局限是 min_risk 只优化风险，不代表综合最优。

7. `action_impact_strategy_cost.png`：该图比较不同策略 action_cost，突出 min_action_cost_path 的综合优势。它支持“动作影响模型能区分不同路径策略后果”的结论，适合放在 action impact 或传统 baseline 章节。局限是 action_cost 权重是启发式设置，论文中应说明权重可调。

8. `routing_baseline_improvement_vs_shortest.png`：该图展示 min_risk、min_cong_prob、min_action_cost 相对 shortest 的改善百分比。它适合放在 routing baseline 汇总章节，用于说明预测风险和综合代价策略对最短路径的改进。局限是百分比改善对分母较小的指标敏感，需配合原始指标表一起解读。

9. `routing_env_policy_reward_comparison.png`：该图展示 random_valid、shortest、min_raw_cost、min_risk、min_cong_prob 在 LeoRoutingEnv 中的 reward 对比。它支持“环境 reward 定义合理，能区分策略优劣”的结论，适合放在强化学习环境设计章节。局限是这些策略仍是启发式，不代表学习策略效果。

10. `masked_ppo_run_reward_comparison.png`：该图展示 mid、continue100k、lr1e4 等 PPO run 与 baseline 的 reward 差异，是最终强化学习结果的核心图。它支持“lr1e4 是当前最优 PPO run，且明显优于 shortest”的结论，适合放在最终路由实验章节。局限是 PPO 仍低于 min_raw_cost，因此图中必须明确 min_raw_cost 是 oracle-like 上界。

## 19. 图表索引

| 图编号 | 路径 | 含义 | 横坐标 | 纵坐标 |
| --- | --- | --- | --- | --- |
| 图1 | ../data/results/abilene_total_traffic_curve.png | Abilene total traffic curve | time | total traffic |
| 图2 | ../data/results/leo_topo144_snapshot.png | LEO topo144 snapshot | 空间位置/经纬度 | 卫星与链路位置 |
| 图3 | ../data/results/gateway_access_example.png | Gateway access example | time 或 gateway | 接入卫星/可见关系 |
| 图4 | ../data/results/leo_average_utilization_curve.png | Average utilization curve | time | average utilization |
| 图5 | ../data/results/leo_mlu_curve.png | MLU curve | time | maximum link utilization |
| 图6 | ../data/results/top_congested_edges.png | Top congested edges | edge_id | congestion count |
| 图7 | ../data/results/congestion_count_by_time.png | Congestion count by time | time | congested edge count |
| 图8 | ../data/results/sample_y_utilization_distribution.png | Sample utilization distribution | utilization | frequency |
| 图9 | ../data/results/sample_y_congestion_ratio_split.png | Sample congestion ratio split | split | positive ratio |
| 图10 | ../data/results/uggru_train_val_loss.png | UGGRU train val loss | epoch | loss |
| 图11 | ../data/results/uggru_threshold_prf_curve.png | UGGRU threshold PRF curve | threshold | precision/recall/F1 |
| 图12 | ../data/results/uggru_val_to_test_threshold_prf_curve.png | Val-to-test threshold PRF curve | threshold | precision/recall/F1 |
| 图13 | ../data/results/mc_dropout_uncertainty_error_scatter.png | MC uncertainty error scatter | util_pred_std | absolute error |
| 图14 | ../data/results/mc_dropout_risk_topk.png | MC risk top-k | top-k ratio | true congestion rate |
| 图15 | ../data/results/prediction_model_comparison_mae_rmse.png | Prediction model comparison MAE RMSE | model | MAE/RMSE |
| 图16 | ../data/results/prediction_model_comparison_f1.png | Prediction model comparison F1 | model | F1 |
| 图17 | ../data/results/candidate_path_hop_count_distribution.png | Candidate path hop count distribution | hop count | path count |
| 图18 | ../data/results/path_risk_score_distribution.png | Path risk score distribution | risk score | frequency |
| 图19 | ../data/results/path_cong_prob_distribution.png | Path congestion probability distribution | congestion probability | frequency |
| 图20 | ../data/results/path_delay_vs_risk_scatter.png | Path delay vs risk scatter | path_delay_ms_sum | path_risk_score_max |
| 图21 | ../data/results/shortest_vs_min_risk_comparison.png | Shortest vs min risk comparison | strategy | metric |
| 图22 | ../data/results/min_risk_path_id_distribution.png | Min risk path id distribution | path_id | count |
| 图23 | ../data/results/action_impact_strategy_post_mlu.png | Action impact strategy post MLU | strategy | post_mlu |
| 图24 | ../data/results/action_impact_strategy_delta_mlu.png | Action impact strategy delta MLU | strategy | delta_mlu |
| 图25 | ../data/results/action_impact_strategy_congestion_delta.png | Action impact strategy congestion delta | strategy | delta_congestion_count |
| 图26 | ../data/results/action_impact_strategy_cost.png | Action impact strategy cost | strategy | action_cost |
| 图27 | ../data/results/action_impact_shortest_vs_min_risk_scatter.png | Action impact shortest vs min risk scatter | shortest action_cost | min_risk action_cost |
| 图28 | ../data/results/action_impact_selected_path_id_distribution.png | Action impact selected path id distribution | path_id | count |
| 图29 | ../data/results/routing_baseline_post_mlu_comparison.png | Routing baseline post MLU comparison | strategy | post_mlu |
| 图30 | ../data/results/routing_baseline_delta_mlu_comparison.png | Routing baseline delta MLU comparison | strategy | delta_mlu |
| 图31 | ../data/results/routing_baseline_delta_congestion_comparison.png | Routing baseline delta congestion comparison | strategy | delta_congestion_count |
| 图32 | ../data/results/routing_baseline_action_cost_comparison.png | Routing baseline action cost comparison | strategy | action_cost |
| 图33 | ../data/results/routing_baseline_risk_delay_tradeoff.png | Routing baseline risk delay tradeoff | path_delay_ms_sum | path_risk_score_max |
| 图34 | ../data/results/routing_baseline_improvement_vs_shortest.png | Routing baseline improvement vs shortest | strategy | improvement percentage |
| 图35 | ../data/results/routing_env_policy_reward_comparison.png | Routing env policy reward comparison | policy | mean_reward |
| 图36 | ../data/results/routing_env_policy_cost_comparison.png | Routing env policy cost comparison | policy | mean_raw_cost |
| 图37 | ../data/results/routing_env_policy_action_distribution.png | Routing env policy action distribution | action_id | count |
| 图38 | ../data/results/masked_ppo_mid_policy_reward_comparison.png | MaskedPPO mid policy reward comparison | policy | mean_reward |
| 图39 | ../data/results/masked_ppo_continue100k_policy_reward_comparison.png | MaskedPPO continue100k policy reward comparison | policy | mean_reward |
| 图40 | ../data/results/masked_ppo_lr1e4_policy_reward_comparison.png | MaskedPPO lr1e4 policy reward comparison | policy | mean_reward |
| 图41 | ../data/results/masked_ppo_run_reward_comparison.png | MaskedPPO run reward comparison | run/policy | mean_reward |
| 图42 | ../data/results/masked_ppo_run_cost_comparison.png | MaskedPPO run cost comparison | run/policy | mean_raw_cost |
| 图43 | ../data/results/masked_ppo_run_congestion_comparison.png | MaskedPPO run congestion comparison | run/policy | mean_delta_congestion_count |
| 图44 | ../data/results/masked_ppo_run_gap_to_min_raw_cost.png | MaskedPPO gap to min raw cost | run | gap |

## 20. 结果文件索引

| 文件 | 作用 |
| --- | --- |
| data/results/prediction_model_comparison.csv | 预测模型对比表 |
| data/results/prediction_summary_metrics.json | UGGRU 与 MC Dropout 汇总指标 |
| data/results/mc_dropout_metrics.json | MC Dropout 不确定性与风险排序指标 |
| data/processed/candidate_paths_topo144_k5.pkl | K=5 候选路径主文件 |
| data/processed/candidate_paths_topo144_k5_stats.json | 候选路径生成统计 |
| data/processed/path_risk_features_test_topo144_k5.npz | 路径级风险特征 |
| data/results/path_risk_feature_stats_topo144_k5.json | 路径风险特征统计 |
| data/processed/action_impact_features_test_topo144_k5.npz | 动作影响特征 |
| data/results/action_impact_strategy_comparison_topo144_k5.csv | 动作影响策略平均指标 |
| data/results/routing_baseline_comparison_topo144_k5.csv | 传统路由 baseline 汇总 |
| data/results/routing_env_heuristic_policy_eval_topo144_k5.csv | RL 环境 heuristic policy 评估 |
| data/results/masked_ppo_run_comparison_topo144_k5.csv | 多个 MaskedPPO run 与 baseline 对比 |

## 21. 脚本索引

| 阶段 | 关键脚本 |
| --- | --- |
| 数据处理 | `src/download_abilene.py`, `src/parse_abilene.py`, `src/parse_abilene_topology.py`, `src/check_abilene_data.py` |
| 星座与接入 | `src/build_leo_constellation.py`, `src/ground_sat_access.py`, `src/check_leo_access.py` |
| link_state | `src/simulate_link_state.py`, `src/check_link_state.py`, `src/diagnose_link_state.py` |
| prediction samples | `src/build_edge_graph.py`, `src/build_prediction_samples.py`, `src/check_prediction_samples.py` |
| UGGRU | `src/dataset.py`, `src/models.py`, `src/train_predictor.py`, `src/evaluate_predictor.py` |
| MC Dropout | `src/evaluate_uncertainty_mc_dropout.py`, `src/check_uncertainty_outputs.py` |
| prediction baseline | `src/evaluate_simple_baselines.py`, `src/baseline_models.py`, `src/train_rnn_baseline.py`, `src/evaluate_rnn_baseline.py`, `src/compare_prediction_models.py` |
| candidate paths | `src/build_candidate_paths.py`, `src/check_candidate_paths.py` |
| path risk | `src/build_path_risk_features.py`, `src/check_path_risk_features.py` |
| action impact | `src/simulate_action_impact.py`, `src/check_action_impact.py` |
| routing baseline | `src/summarize_routing_baselines.py`, `src/check_routing_baseline_summary.py` |
| RL env | `src/leo_routing_env.py`, `src/check_routing_env.py`, `src/evaluate_env_heuristic_policies.py` |
| MaskedPPO | `src/train_masked_ppo_smoke.py`, `src/train_masked_ppo.py`, `src/evaluate_masked_ppo_full.py`, `src/plot_masked_ppo_training.py` |
| run comparison | `src/compare_masked_ppo_runs.py`, `src/check_masked_ppo_run_comparison.py` |

## 22. 缺失文件列表

以下推荐文件或图片未找到。注意：`data/results/candidate_paths_topo144_k5_stats.json` 未找到，但实际候选路径统计文件存在于 `data/processed/candidate_paths_topo144_k5_stats.json`。
- `data/results/candidate_paths_topo144_k5_stats.json`

## 当前最重要的结论

1. 预测模块、路径风险模块、动作影响模块和 MaskedPPO 路由模块已经形成完整实验闭环。
2. `masked_ppo_lr1e4` 是当前最优 PPO run，显著优于 shortest_path，但仍低于 min_raw_cost 上界。
3. risk_score 与 action impact 将预测结果转化为路由决策可用的风险/代价特征，是当前系统最关键的连接环节。

## 最需要谨慎表述的地方

1. `min_raw_cost` 必须表述为 oracle-like greedy upper-bound baseline，不能写成普通在线路由算法或最终算法。
2. LeoRoutingEnv 是 offline action-impact routing environment，不是完整在线多流动态重路由仿真器。
3. 当前 ISL 边集合固定，`remain_visible_time` 仍为占位，正式论文中必须作为简化假设说明。

## 还需要补充的实验/文件清单

1. 多随机种子 PPO 训练结果：建议至少补 3 个 seed，统计 `mean_reward`、`mean_raw_cost`、`mean_delta_congestion_count` 的均值和标准差。
2. UGGRU 消融实验：建议补充不使用 edge graph、不同 seq_len、不同 dropout 或不同 hidden size 的对比，以增强模型设计说服力。
3. reward 权重敏感性分析：当前 action_cost 和 RL reward 权重是启发式设置，建议补充不同 `w_delay / w_mlu / w_risk / w_congestion` 下的策略表现。
4. 动态 ISL 或链路不可用实验：即便不能完整实现真实动态拓扑，也可构造部分链路失效场景，验证 action mask 和候选路径机制。
5. 多场景业务流量：除 Abilene 外，建议构造均匀、周期、热点突发、人口加权等 synthetic scenarios，用于验证方法稳健性。
6. 论文级图表重绘：统一字体、字号、颜色、图例位置和中英文标注，避免直接使用调试阶段图片。
7. 复杂度分析文件：补充候选路径生成、path risk 聚合、action impact 构建、PPO 单步决策的时间/空间复杂度说明。
8. 在线多步环境设计草案：说明如何从当前 offline action-impact environment 过渡到真正会更新链路状态的 online simulator。
