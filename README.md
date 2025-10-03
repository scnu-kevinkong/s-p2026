# Pro²Guard — MDP+STL 运行时安全强化

Pro²Guard 是基于概率模型检验的 LLM Agent 主动式运行时安全强化系统。该版本引入并优化了 MDP+STL 算法：在离散时间马尔可夫链（DTMC）上对 STL 的时窗算子进行概率化评估，并以此驱动实时干预与离线评测。

核心更新（MDP+STL）：
- 支持窗口化时序算子 `F[a,b]` 与 `G[a,b]` 的概率评估（product-style 概率）。
- 双通道评估：基线通道（逐步调用 `runtime_monitor`）与优化通道（基于 `model.json` 的动态规划/蒙特卡洛概率）。
- 可选蒙特卡洛交叉验证与 95% 置信区间输出。
- 与 Pro²Guard 既有指标打通：Unsafe%、Completion%、Prediction%、Time-Ahead。

## 快速开始

- 安装 PRISM
```
cd prism
./install.sh
```
PRISM 可执行位于 `./prism/bin/prism`。也可通过设置环境变量 `PRISM_HOME` 来使用系统安装的 PRISM。

- 安装依赖
```
pip install -r requirement.txt
```

## 学习 DTMC 模型
用于 Embodied 与自动驾驶（AV）场景的 DTMC 学习：
```
cd src/
python3 -m safereach.embodied_build
# 或
python3 -m safereach.av_build
```
生成的模型包含：`dtmc.prism`（PRISM 模型）、`model.json`（状态索引与转移概率）、`abstraction.json`（抽象映射）。

## MDP+STL 离线评测（AV）
MDP+STL 的 AV 评测入口位于 `src/safereach/time_stl.py`，提供基线与优化双通道。

- 单次评测示例（环境变量定义规格）：
```
cd src/
export STL_FOLDER="Law53_0_1"     # 待评测模式（示例）
export STL_KIND="F"               # F 或 G
export STL_A="0.0"               # 左窗（步）
export STL_B="2.0"               # 右窗（步）
export STL_THETA="0.5"           # 基线通道阈值
export STL_PRED_KEY="collision"  # 谓词键（如 collision）
export STL_PRED_VAL="1"          # 谓词值（如 1）
python3 -m safereach.time_stl
```

- 批量评测（多规格/多模式）：
```
cd src/
export STL_BATCH=on
export STL_SPECS="F,0,2,collision,1,0.5;G,0,5,collision,1,0.5"
# 可选：蒙特卡洛交叉验证
export STL_MC=on
export STL_MC_TRIALS=1000
python3 -m safereach.time_stl
```
输出目录：`src/log_safereach/offline_stl/`，包含 JSON/CSV/SVG 对比报表与汇总。

关键实现要点：
- 基线通道：逐步调用 `runtime_monitor`，在 `[a,b]` 窗内以阈值判断满足性（见 `STLEvaluator.baseline_check`）。
- 优化通道：基于 `model.json`，通过分布推进与窗口存留概率计算 `F/G` 的 product-style 概率（见 `STLEvaluator.prob_from_model`）。
- 监控退化与回退：若 PRISM 不可用，则 `runtime_monitor` 回退到纯 Python 的 `model.json` 路径计算。

## MDP+STL 离线评测（Embodied）
Embodied 版本入口位于 `src/safereach/embodied_time_stl.py`，接口与 AV 版本一致，文件组织位于 `src/safereach/embodied/`。

- 单次评测：
```
cd src/
export STL_FOLDER="log_raw_t20"
export STL_KIND="F"
export STL_A="0.0"
export STL_B="2.0"
export STL_THETA="0.5"
python3 -m safereach.embodied_time_stl
```

- 批量评测：
```
cd src/
export STL_BATCH=on
python3 -m safereach.embodied_time_stl
```
输出目录：`src/log_safereach/offline_stl_embodied/`。

Embodied 特性：
- 规格文件 `spec`（位于样本文件夹）决定不安全集合，且 `SAFE_REACH_STEP_TIME_S` 控制步长聚合。
- 报表自动附带 Pro²Guard 指标（Prediction%、Time-Ahead、Unsafe%、Completion%）。

## 运行时强化（Agent 集成）
在 LangChain + AgentSpec 的 Agent 中集成 `runtime_monitor`，当达到阈值（如 `SAFE_REACH_THETA`）触发主动式干预。
示例（`src/embodied_agent.py`）：
```python
from agentspec.controlled_agent_excector import initialize_controlled_agent

tool = Tool(name="robotic controller", description="High-level controller", func=planner.llm_skill_interact)

agent = initialize_controlled_agent(
    tools=[tool], llm=llm, agent="zero-shot-react-description", rules=[],
    abs=kwargs["abs"], dtmc_path=kwargs["dtmc_path"], unsafe_states=kwargs["unsafe_states"],
    cache=kwargs["cache"], task_obj=kwargs["task_obj"], threshold=kwargs["threshold"]
)
```

## 诊断工具（优先级建议）
`src/safereach/mdp_stl_diagnostic.py` 提供跨平台（AV/Embodied）诊断汇总：
```
cd src/
python3 -m safereach.mdp_stl_diagnostic
```
输出：`src/log_safereach/mdp_stl_diagnostic_report.json` 与 `.csv`，包含 AII（Action Influence Index）与关键指标的期望改进区间，辅助 MDP+STL 的实施与调参优先级。

## 环境变量配置摘要
- `PRISM_HOME`：PRISM 安装目录（可选）。
- `STL_BATCH`、`STL_SPECS`、`STL_MODES`：批量评测控制与规格定义。
- `STL_MC`、`STL_MC_TRIALS`：蒙特卡洛交叉验证与次数。
- `SAFE_REACH_THETA`、`AV_THRESHOLDS`：预测与拦截阈值。
- `SAFE_REACH_STEP_TIME_S`：Embodied 步长（秒）。
- `PRECOMPUTE_VALUES`、`SAFE_REACH_UCB`、`UCB_COEFF`：运行时概率的预计算与不确定性上界偏置。

## PRISM 与回退说明
- 优先使用 `PRISM_HOME` 下的 `bin/prism`；若不可用，`runtime_monitor` 会使用 Java 方式调用 PRISM。
- 如 PRISM 调用失败，将自动回退为基于 `model.json` 的纯 Python 概率计算，确保评测与运行时强化不中断。

## 参考与致谢
- PRISM: http://www.prismmodelchecker.org/
- AgentSpec: https://arxiv.org/abs/2503.18666
- SafeAgentBench: https://github.com/shengyin1224/SafeAgentBench