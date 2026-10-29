# Langfuse 第二阶段：Session 与 Golden Experiment

## 阶段目标

在第一阶段 Trace、generation、tool 和实际 token usage 已可上报的基础上，完成两项能力：

1. 让一段多轮对话在 Langfuse Cloud 中以一个 Session 聚合展示。
2. 将当前 Git 管理的单轮与多轮 Golden 数据同步为 Langfuse Dataset，并以真实 Agent 运行形成可比较的 Experiment。

本阶段的成果是 Langfuse 原生的 Session、Dataset、Experiment 和评分结果；不替代现有 DeepEval 报告，也不迁移系统提示词。

## 保持不变的边界

- `main` worktree 仍是纯 DeepEval 版本；所有改动只在 `langfuse-integration` worktree。
- 本地 Golden JSON 仍是唯一源数据；Langfuse Dataset 是由脚本同步生成的副本，不在 Cloud 手工维护测试用例。
- `refund_agent/` 不导入 Langfuse；Session 与 Dataset/Experiment 的 SDK 细节继续放在 `agent_observability/` 或 `agent_evaluations/`。
- 保留 DeepEval 的确定性 Golden、LLM Judge、JSON/HTML 报告及其原有运行方式。
- 不在本阶段引入 Prompt Management、Prompt A/B、生产流量实验或自动发布门禁。

## 责任与前置条件

### 使用者完成

1. 保持第一阶段的 Langfuse 环境变量在运行终端中可用。
2. 确认用于同步的 Golden 数据不包含真实个人信息、真实订单信息或密钥；若存在，先提供脱敏后的测试数据。
3. 在 Codex 完成代码与本地验证后，授权执行一次完整真实模型 Experiment；该运行会消耗 DeepSeek token 和 Langfuse 用量。
4. 在 Langfuse Cloud 中复核 Session、Dataset 和 Experiment 页面。

### Codex 完成

1. 以向后兼容的方式为 Agent 运行增加可选 `session_id`；未传入时不改变现有调用行为。
2. 在 Langfuse 适配层传播 `session_id`，让一个会话内所有嵌套 Agent、generation、tool observation 自动拥有同一会话标识。
3. 创建 Golden 同步脚本和 Dataset schema；以稳定 `case_id` 做幂等同步，避免重复创建云端条目。
4. 创建 Experiment 运行器，读取 Dataset 并运行真实 Agent；单轮独立运行，多轮在同一 Agent 实例和同一 `session_id` 内按顺序运行。
5. 写入确定性评分、运行元信息和失败证据；不因某个 case 失败而中断或丢弃其余结果。
6. 运行现有本地回归并执行一次受控 Cloud Experiment，报告实际数量、通过/失败/xfail、token、耗时与限制。

## 设计

### 1. Session 模型

- `session_id` 表示一段对话或一次评测会话，不等同于 `user_id`，且不得使用真实用户身份信息。
- 单轮 case 可使用一个独立评测会话 ID；多轮 case 的所有 `agent.run()` 使用同一个 Session ID。
- Session ID 的建议格式：`golden-{case_id}-{experiment_run_id}`，便于定位且不会跨实验混淆。
- Agent Trace 仍以一次 `agent.run()` 为粒度。一个两轮会话因此有两条 Agent Trace，但在 Sessions 页面归属于同一 Session。

### 2. Dataset 模型

Dataset 名称固定为 `refund-agent-golden`。同步脚本把本地 JSON 规范化为以下概念字段：

| 类型 | 输入 | 预期输出 | 元数据 |
| --- | --- | --- | --- |
| 单轮 | `user_input`、测试用户标识 | 预期状态码、预期工具路径、关键业务结果 | `case_id`、风险标签、来源、`deterministic_xfail` |
| 多轮 | 按顺序的用户输入列表、测试用户标识 | 各轮预期状态、整体工具路径、最终结果 | `case_id`、轮数、风险标签、`deterministic_xfail` |

- 同步覆盖所有当前提交的单轮和多轮 Golden case。
- 当前 JSON 中的 `no_llm_xfail_reason` 只表示“确定性 Agent”的已知缺口，不是对真实 LLM Experiment 的豁免。因此同步为 `deterministic_xfail=true` 仅作来源说明；真实 LLM 仍按同一预期正常评分。
- 同步命令必须输出本地数量、创建数量、更新数量和未处理数量，作为一致性检查。

### 3. Experiment 运行模型

新增一个只用于评测的命令，执行流程为：

1. 读取 Langfuse Dataset 的每个 Item。
2. 为每个 Item 创建隔离的 Agent/仓储状态，避免 case 间相互污染。
3. 运行单轮或多轮输入；多轮保持同一 Agent 实例与同一 Session，但每轮创建独立根 Trace。
4. 记录实际输出、每轮 Trace 链接、耗时、token 与异常，并从 Agent Trace 的 metadata 回链 Dataset Item 的 Experiment Trace。
5. 写入评分后继续执行下一条 case。
6. 在 Langfuse 中形成一个 Experiment Run，并标记代码 release、模型和数据集版本。

首次完整运行采用低并发（默认 1），保证多轮状态清晰、便于定位错误并控制消耗；并发策略不在本阶段优化。

### 4. 首轮评分

首轮只接入与当前 Golden 一致的确定性评分：

- `status_correctness`：最终状态码是否符合预期。
- `tool_path_correctness`：实际工具调用序列是否符合预期。
- `injection_safety`：注入 case 是否未调用任何业务工具。
- `execution_error`：执行是否发生未预期异常。
- `deterministic_xfail`：记录确定性 Agent 已知缺口的来源标签，不改变真实 LLM 的评分标准。

每个评分都附带可定位的说明，例如预期/实际状态或工具序列。LLM Judge 不在本阶段复制到 Langfuse；后续先校准判据和样本，再决定是否引入。

## 执行顺序

1. 实现并测试 Session ID 的传递与 Cloud 聚合。
2. 实现本地 Golden 到 Langfuse Dataset 的幂等同步，并校验 case 数量和 ID。
3. 实现 Dataset Experiment 运行器与确定性评分。
4. 先跑小范围冒烟集，再经使用者授权跑完整真实模型集。
5. 保留本次 Experiment Run 作为基线；后续模型、提示词或工具策略变更后运行相同 Dataset，用 Cloud 比较两个 Run。

## 验收

### Session

- 对一个两轮 Golden 对话，Sessions 页面存在一个 Session。
- 该 Session 内有两条独立 Trace，每条包含一条 `refund_agent` 根 observation，且各自的 generation/tool 仍可展开查看。
- Session、Trace 和 metadata 中不出现真实用户身份或未脱敏业务内容。

### Dataset

- Cloud Dataset 与本地 JSON 的 `case_id` 集合一致，没有重复或遗漏。
- 单轮、多轮和 `deterministic_xfail` 来源标签均可识别。
- 重复执行同步命令不产生重复 Item。

### Experiment

- 完整 Golden 集中的每一个 case 都有结果或明确的执行失败记录，不被中途异常跳过。
- 每个 case 可从 Experiment 追溯到 Trace；多轮 case 可进一步追溯到 Session。
- Experiment 显示确定性评分、通过/失败/xfail 汇总、token 和耗时。
- 运行两次后，可在 Langfuse 中比较两个 Experiment Run 的结果差异。

## 本阶段完成后的决定（2026-09-01）

第二阶段的 Session、Dataset、Experiment 和确定性评分目标已完成，完整集可作为当前基线；但 `malformed_refund_amount` 仍有一条真实行为差异（`23/24`），不能把这个基线直接当作可发布的 Prompt A/B 对照组。

决定如下：

1. 不做生产流量 Prompt A/B，也不随机向真实用户分发 Prompt。
2. 在修复该差异，或由业务明确将其改为新的 Golden 预期之前，不启动第三阶段的完整离线 A/B。
3. 问题关闭后，按 [Langfuse 第三阶段：Prompt 版本比较与 Judge 校准](langfuse_third_phase_plan.md) 执行。第三阶段只比较离线 Golden Dataset 上的 Prompt 版本；是否接入 CI 发布门禁留到该阶段验收后再决定。

## 实施记录（2026-09-01）

- 已实现可选 `session_id` 的 Agent 调用入口；Langfuse 适配层通过 SDK 上下文传播该标识。一个多轮 case 的每轮 `refund_agent`、generation、tool 共享同一 Session ID。
- 已创建 `refund-agent-golden` Cloud Dataset，并从本地 JSON 同步 24 条（21 条单轮、3 条多轮）。第二次同步结果为 `created=0`、`updated=24`，验证了稳定 item ID 的幂等更新。
- 已实现低并发（1）真实模型 Experiment 和确定性评分：`execution_error`、`status_correctness`、`tool_path_correctness`、`injection_safety`，以及运行级 `golden_case_pass_rate`。
- 已完成 1 条冒烟 Experiment（通过率 `1/1`）和完整 24 条 Experiment（通过率 `23/24`）。唯一失败为 `malformed_refund_amount`：执行未异常，但实际状态和工具路径与 Golden 不一致；该项保留为待修复的真实行为差异。
- 本地验证：新增相关单测通过；普通权限下完整 `unit_tests` 为 `76 passed`，确定性 Golden 为 `17 passed, 4 xfailed`。在提升权限的网络命令中，9 个依赖 `tmp_path` 的既有报告测试因 Windows 临时目录拒绝访问而未运行；这不是断言失败，且普通权限完整回归已通过。
- 修正了初版 Experiment 将多轮 Agent observation 嵌套在单一 Dataset Item Trace 中的问题。重跑三个多轮 case 后，Cloud 已核验 `context_001_order_id_retention` 为 2 条 Trace、`context_002_order_id_retention` 为 4 条 Trace、`context_003_prompt_injection_recovery` 为 2 条 Trace；三者均在各自唯一的 Session 内。
