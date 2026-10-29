# Langfuse 首个落地目标

## 目标

在不改变退款 Agent 业务行为和现有 DeepEval 评测体系的前提下，引入 Langfuse SDK，记录真实 LLM Agent 的运行、模型调用、工具调用，以及模型返回的实际 token usage。

本阶段只验证可观测数据是否完整、准确、可关联；不进行 Prompt 管理或 Prompt 优化。

## 责任与前置条件

### 使用者完成

1. 注册或登录 Langfuse Cloud，创建一个仅用于本项目开发验证的 Project。
2. 在 Project Settings → API Keys 创建项目级 `Public Key` 和 `Secret Key`。
3. 在本机环境变量或 CI Secret 中配置以下值，不提交到 Git，也不发送到对话中：

   ```text
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_BASE_URL=https://cloud.langfuse.com
   LANGFUSE_TRACING_ENVIRONMENT=development
   LANGFUSE_RELEASE=langfuse-integration
   ```

   如选择 US、Japan、HIPAA 区域或自托管实例，`LANGFUSE_BASE_URL` 改为该项目实际所在实例地址。
4. 完成后只需告知“Langfuse 环境变量已配置”；无需提供任何密钥值。
5. 在验收阶段登录 Langfuse Cloud，确认本项目的 trace 可见。模型调用会消耗既有 DeepSeek 额度，Langfuse Cloud 可能产生相应服务用量。

### Codex 完成

1. 仅在 `langfuse-integration` worktree 中修改代码；不修改纯 DeepEval 的 `main` worktree。
2. 在独立 `agent_observability/` 包中增加 Langfuse SDK 依赖、非敏感配置读取和必要的 `.gitignore` 规则；`refund_agent/` 不导入 Langfuse。
3. 实现 Agent、generation、tool 三层 observation，并复用现有供应商 usage，不伪造 token 或成本数据。
4. 对可记录的业务字段进行最小化和脱敏处理；不读取、展示或提交密钥。
5. 运行现有 DeepEval 回归测试，验证其命令、指标和报告结构未被 Langfuse 改动破坏。
6. 在环境变量已配置的前提下，运行约定的三类验证请求，检查 SDK 是否成功上报并报告验证结果与限制。

### 无法代办的边界

- Codex 不能替使用者注册 Langfuse 账户、创建其项目或生成项目密钥。
- 在没有有效 Langfuse 凭证前，Codex 可以完成本地代码和无凭证回归验证，但不能完成 Cloud trace 验收。
- 不将 Langfuse 作为退款业务判断依据；网络故障、凭证缺失或上报失败不得改变退款 Agent 的业务结果。
- 正常 Agent 请求不强制 flush Langfuse，避免遥测网络请求进入用户响应关键路径；短生命周期验证脚本在结束时显式 flush。

## 执行顺序

1. 使用者完成 Langfuse 项目与本机环境变量配置。
2. Codex 在本 worktree 实现 SDK 与三层 observation。
3. Codex 先运行不依赖 Cloud 的现有 DeepEval 回归，确认原评测未退化。
4. Codex 执行三类真实 LLM 请求并确认 SDK 上报结果。
5. 使用者在 Cloud 中复核 trace；双方确认后，首个落地目标完成。

验证命令：在已配置环境变量的新终端中，运行 `run_langfuse_validation.cmd`。该命令只执行约定的三类真实模型请求，不运行 LLM Judge。

## 保持不变的部分

- 保留 DeepEval 的 `@observe`、Golden、确定性指标、LLM Judge 和 JSON/HTML 报告。
- 不改退款业务规则、工具 schema、工具调度、状态码、模型选择和用户可见回复。
- 不迁移 `SYSTEM_PROMPT`，不建立 Prompt A/B、Dataset 或 Experiment。

## 实施内容

### 1. SDK 与配置

- 在 Python 依赖中增加 `langfuse`。
- 仅从本机环境变量或 CI Secret 读取 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_BASE_URL`。
- 配置 `LANGFUSE_TRACING_ENVIRONMENT` 和 `LANGFUSE_RELEASE`，以区分开发、测试与后续发布版本。
- 不把 Langfuse 凭证写入 `auth.json`、源码、测试数据或报告。

### 2. Agent 根观测

在 `LLMRefundAgent.run()` 创建一条 Langfuse Agent observation，表达一次用户请求对应的一次 Agent 执行。

记录：匿名化用户标识、会话标识、输入摘要、最终状态码、最终回复摘要、总耗时、模型调用轮数。

### 3. 模型调用观测

在 `_model_completion()` 的每次真实模型调用创建一条 generation，并嵌套在对应 Agent observation 下。

记录：模型名、消息输入、工具 schema、模型输出、耗时、异常，以及供应商响应中的实际 token usage。

`usage_details` 只写互斥的 `input` 和 `output` token bucket，不把 `total_tokens` 作为第三个可计费 bucket 上传，避免重复统计。

### 4. 工具调用观测

在 `_call_tool()` 的每次工具执行创建一条 tool observation，并嵌套在当前 Agent observation 下。

记录：工具名、脱敏参数摘要、工具结果状态码、耗时与异常。不得默认上传完整订单对象、用户身份信息或其他敏感业务数据。

### 5. 与 DeepEval 的关系

Langfuse 与 DeepEval 并存，导入时使用别名避免覆盖：

```python
from deepeval.tracing import observe as deepeval_observe
from langfuse import observe as langfuse_observe
```

DeepEval 继续负责测试执行、业务硬断言、Judge 与现有报告；Langfuse 只负责观测、token/cost 与运行对比。

`refund_agent/` 仅接受一个可选的通用观测对象，不知道其具体是 Langfuse 还是其他平台。应用启动或验证脚本在外部创建 `agent_observability.langfuse_adapter` 并注入该对象。

## 验收

在 Langfuse Cloud 中核对以下三类真实运行：

1. 单轮正常退款：存在 Agent、generation、工具调用；工具顺序和模型调用轮数可见。
2. 多轮补订单号：每个 `agent.run()` 都是独立 Agent observation，且能通过会话标识关联。
3. Prompt Injection 拦截：当前实现仍需一次模型调用识别攻击，因此会产生一条 generation 并记录实际 token usage；不得产生业务工具 observation。

同时满足：

- 现有 DeepEval 测试仍可按原命令运行，指标与报告结构不变。
- Langfuse generation 的 input/output token 与 `AgentResult.model_calls` 已提取的供应商 usage 一致。
- 可在 Langfuse 中查看单次请求总耗时、模型耗时、工具耗时、模型调用轮数和 token/cost。

## 完成本阶段后再决定

只有本阶段通过验收后，才讨论将系统提示词迁入 Prompt Management，以及基于现有 Golden 运行完整 Agent 的 Prompt Experiment。
