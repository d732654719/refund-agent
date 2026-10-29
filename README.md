# Minimal Refund Agent

当前分支包含 DeepEval 评测和 Langfuse 观测、数据集与实验；基础版见 `main` 分支。Langfuse 配置与验收入口见 [集成说明](langfuse_plan/langfuse_first_landing_plan.md) 和 [评测模块](agent_evaluations/AGENTS.md)。

开发入口见 [AGENTS.md](AGENTS.md)：按任务定位模块、运行和验证。文档随代码维护的流程见 [documentation.md](docs/documentation.md)。

一个用于评估实验的最小电商退款 Agent。默认使用 DeepSeek 的
OpenAI-compatible Tool Calling，由 LLM 理解用户意图、选择工具并生成回复；
退款规则、用户身份和状态修改仍由可信代码控制。

项目同时保留不调用外部模型的确定性基线，方便对比 LLM Agent 和固定工作流。

## 已实现

- 查询单个订单和当前用户的全部订单；
- 校验当前用户是否拥有订单；
- 校验 30 天退款期限；
- 全额和部分退款；
- 每次退款成功后固定展示原订单金额、历史退款金额（本次前）、本次退款金额和未退款金额；
- 防止超额退款、重复退款和越权访问；
- LLM 多轮工具调用与统一工具调度；
- 模型调用的业务工具经过 `_call_tool()` 写入 Trace；
- 输出区分 Agent 回答与执行 Trace，并标记模型总轮数和每次工具调用所在轮次；
- LLM Trace 按时间记录每轮模型响应、模型请求的工具以及工具返回结果；
- 对 Prompt Injection 仍执行确定性退款策略；
- 返回可序列化的 Agent 执行轨迹；
- 内置单轮 Golden、多轮 Conversation 和不调用真实模型的单元测试。

关键安全边界位于 `RefundTools` 和 `RefundPolicy`，不是 LLM 的 System
Prompt。模型不能控制当前登录用户，也不能绕过退款期限、金额和权限检查。
退款成功后的金额明细同样由程序根据 `create_refund` 的结构化结果统一生成，
不依赖 LLM 自行记忆或计算金额。
同一次请求中，退款成功后继续查询订单，不会覆盖退款成功说明。后续模型超时或步骤
超限时，仍会展示已退款金额，并保留异常提示。订单只保存在内存中，退出后重新
运行会恢复样例订单。

## LLM 模式的意图与状态

LLM 模式不再用“输入中是否出现退款关键字”推断无工具请求的状态。本分支的 System Prompt
要求模型在非空、且不需要业务工具时调用 `final_response` 提交 `status + message`。
程序在尚无业务工具结果且该轮仅请求一个 `final_response` 时解析它，同时兼容无工具响应中的 JSON 文本；只接受
`needs_order_id`、`out_of_scope`、`responded` 和 `PROMPT_INJECTION_BLOCKED` 四个
白名单状态；格式或状态不合法时会要求模型重答。空输入不请求 LLM，由程序直接返回
`needs_input`，因此模型不能把“批量退款但缺少单个订单号”等有效输入误标为
`needs_input`。

`final_response` 属于结束协议，不属于退款业务工具，也不作为业务工具 Trace 记录。未知工具会先经过名称白名单检查和有次数上限的纠正流程，具体约束见 [运行模块](refund_agent/AGENTS.md)。

一旦调用了业务工具，业务状态仍由可信工具结果产生，例如 `REFUND_CREATED` 对应
`completed`，越权查询对应 `ACCESS_DENIED`。因此，LLM 负责理解自然语言语义，程序
负责校验输出契约，工具与业务代码负责最终事实和执行结果。

“查询我的所有订单”和“退款我的所有订单”在 System Prompt 中有明确优先级：前者
调用 `list_user_orders`；后者属于不支持的批量退款，不调用工具，并要求用户提供一个
明确订单号。`list_user_orders` 的工具描述也明确禁止用于批量退款请求。

完整的状态决策表、状态来源和新增状态流程见 [`agent_status_guide.md`](agent_status_guide.md)。
代码中的唯一状态目录是 `refund_agent/agent_status_catalog.py`，构造 `AgentResult` 时会拒绝未登记
状态，避免 `expected_status`、Agent 和工具之间出现无法追踪的新字符串。

检测到 Prompt Injection 时，Agent 不调用业务工具，程序把回复统一为固定安全文案；
该状态只阻止当前轮，下一轮不含攻击指令的正常退款请求仍可继续处理。

## LLM 如何调用工具

```text
用户输入
  → LLM 返回 tool_calls(name, arguments)
  → LLMRefundAgent._dispatch_tool()
  → RefundAgent._call_tool()
  → RefundTools 中的真正工具
  → ToolResult 写入 Trace 并返回给 LLM
  → LLM 决定下一步或生成最终回复
```

## 示例订单

| 订单 | 用户 | 状态 | 用途 |
|---|---|---|---|
| `A1001` | `user-001` | 5 天前送达 | 可正常退款，金额 299 元 |
| `A1002` | `user-001` | 45 天前送达 | 超过退款期限 |
| `A1003` | `user-001` | 已退款 | 验证重复退款 |
| `A2001` | `user-002` | 2 天前送达 | 验证越权访问 |

## 运行

首次使用时，在仓库根目录创建虚拟环境并安装依赖（报告包通过 Git 获取，需安装 Git 和 Git LFS）：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e . -r requirements-report.txt
```

默认使用真实 LLM。在本目录执行：

```powershell
python -m refund_agent.command_line_interface "帮我退款订单 A1001" --show-trace
```

API Key 优先读取环境变量：

```powershell
$env:DEEPSEEK_API_KEY="你的 API Key"
```

如果没有环境变量，可通过 `REFUND_AGENT_AUTH_FILE` 指定凭证文件；默认读取仓库父目录的 `auth.json`：

```json
{
  "deepseek": {
    "api_key": "..."
  }
}
```

离线确定性模式：

```powershell
python -m refund_agent.command_line_interface "帮我退款订单 A1001" --mode deterministic --show-trace
```

交互模式：

```powershell
python -m refund_agent.command_line_interface --show-trace
```

交互模式会保留同一进程内的对话上下文，因此可以先提到订单号，再用“那查询下吧”
这样的省略表达继续交流。输入 `/reset` 可清空消息上下文；内存订单和已经产生的退款
状态不会被重置。

不调用外部模型的单元测试：

```powershell
.venv\Scripts\python.exe -m pytest unit_tests -q
```

使用 `single_turn_golden_cases.json` 评估确定性 Agent（不调用 Target LLM 和 Judge LLM）：

```powershell
.\run_deterministic_golden_cases.cmd
```

先安装固定版本的报告模板，再安装 Agent 命令：

```powershell
python -m pip install -r requirements-report.txt
pip install -e .
refund-agent "订单 A1001 退款 100 元" --show-trace
```

## 结构

```text
refund_agent/
├── refund_agent/
│   ├── deterministic_refund_agent.py  # 不调用模型的退款流程基线
│   ├── llm_agent.py                   # LLM 退款流程与安全调度
│   ├── model_runtime.py               # 模型、认证和客户端连接
│   ├── refund_business_tools.py       # 可信退款工具与硬约束
│   ├── refund_policy.py               # 退款资格规则
│   ├── sample_order_repository.py     # 内存订单仓库与样例订单
│   ├── refund_domain_models.py        # Order、Trace、AgentResult 数据结构
│   ├── agent_status_catalog.py        # 所有 Agent 状态码与说明
│   └── command_line_interface.py      # 命令行入口
├── agent_evaluations/
│   ├── single_turn_golden_cases.json  # 单轮业务样例
│   ├── multi_turn_conversation_cases.json # 多轮对话样例
│   ├── test_deterministic_golden_cases.py # 离线确定性基线评测
│   ├── test_llm_single_turn_cases.py  # 真实模型单轮评测
│   ├── test_llm_multi_turn_cases.py   # 真实模型多轮评测
│   └── evaluation_report_session.py   # 运行级数据汇总并调用共享报告包
├── requirements-report.txt            # 固定版本的报告模板依赖
├── report/                            # 每次运行保留的 JSON 与 HTML
├── report-assets/                     # 报告生成时由模板包提供的字体资源
└── unit_tests/                        # 不调用真实模型的代码验证
```

报告的 Schema、校验、JSON 写入和 Palm HTML 渲染来自独立的 [报告模板仓库](https://github.com/d732654719/deepeval-report-template)，通过 `requirements-report.txt` 安装固定版本。

## DeepEval 评估

DeepEval 第一版已经接入，完整指南见：

- [`llm_evaluation_guide.md`](llm_evaluation_guide.md)
- [`refund_evaluation_case_matrix.md`](refund_evaluation_case_matrix.md)：退款业务场景、组合策略与数据集扩充计划

一次运行全部单轮 Golden 和多轮 Conversation：

```powershell
.\run_llm_evaluations.cmd
```

报告根据实际选中的 `llm_eval` 测试生成；直接指定 LLM 测试文件也会生成报告。
`-m no_llm_eval`、`-m "not llm_eval"` 和 `--collect-only` 不会初始化或重置报告。

这是完整的真实 LLM 评估入口，底层使用 DeepEval 的 pytest 插件统一收集
`agent_evaluations/test_llm_single_turn_cases.py` 和
`agent_evaluations/test_llm_multi_turn_cases.py`。

两个评估入口的边界：

| 命令 | 被测 Agent | 是否调用真实 LLM |
|---|---|---|
| `.\run_deterministic_golden_cases.cmd` | `RefundAgent` 确定性基线 | 否 |
| `.\run_llm_evaluations.cmd` | `LLMRefundAgent` + DeepEval Judge | 是 |

两者读取同一份 `single_turn_golden_cases.json`。`expected_response_contains` 只由无 LLM
确定性评测使用；LLM 评测不会对自然语言回复做逐字断言。

这个 Agent 提供 Judge 所需字段：

- `result.input`
- `result.response`
- `result.status`
- `result.trace[*].tool_name`
- `result.trace[*].arguments`
- `result.trace[*].result`
- `result.trace[*].arguments_raw` / `result_raw`
- `result.duration_ms` / `result.model_duration_ms`
- `result.usage` / `result.model_calls`

单轮用例固定五项指标，多轮用例在同一数据结构上增加 `Knowledge Retention`，共六项：

1. 写入 DeepEval 报告的确定性 `Status Correctness` 与工具路径；
2. `ToolCorrectnessMetric`；
3. 自定义退款业务 `GEval`；
4. `KnowledgeRetentionMetric` 多轮上下文评估；
5. 基于 `@observe` 完整 Trace 的 `TaskCompletionMetric`；
6. 退款规则感知的 `RefundStepEfficiencyMetric`。

Case 只有在全部适用指标都达到各自阈值时才通过。每次真实运行会保留：

```text
report/yyyy-mm-dd-001.json
report/yyyy-mm-dd-001.html
```

同一天按 `001、002、003...` 递增。JSON 使用 UTF-8 与两空格缩进，便于快速检查；
HTML 为 Palm · 椰林绿离线报告，单轮和多轮都使用 `cases[].turns[].trace[]`。

`agent_evaluations/test_llm_single_turn_cases.py` 先运行被测 Agent，再通过 DeepEval 的
`assert_test(golden=..., metrics=...)` 将 pytest 当前用例与捕获到的 Trace 关联。
状态和工具路径不再使用前置 Python `assert`，所以校验失败也会进入同一份
DeepEval 报告，不会出现 pytest 已运行但报告缺少该案例的情况。
Target 执行异常会记录为 `Agent Execution`；Judge 内部异常会记录为 metric error。
所有指标的解释性 `reason` 面向中文使用者输出简体中文；`score`、`threshold`、
`evaluation model`、工具名和状态码等技术字段保持原文。
`Task Completion` 低于 `0.7` 或 `Refund Step Efficiency` 低于 `0.8` 时，
该 pytest 用例会失败。

其中 `Refund Step Efficiency` 以 `0.8` 为通过线，并明确把查订单、查退款
政策和可信退款工具校验视为必要步骤。容易误判安全检查为多余步骤的 DeepEval
原生 `Step Efficiency` 已不再执行。
