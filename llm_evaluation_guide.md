# Refund Agent DeepEval 接入指南

## 1. 这次接入完成了什么

项目现在有两套互补的测试：

```text
确定性测试
└── pytest unit_tests/：验证 Python 业务逻辑

DeepEval
├── 单轮 Golden
│   ├── 确定性状态码与工具路径
│   ├── Tool Correctness 和退款业务回复质量 G-Eval
│   ├── Task Completion
│   └── Refund Step Efficiency
└── 多轮 Conversation
    ├── Knowledge Retention：是否记住订单号等上下文
    └── Conversation Turn Contract：逐轮确定性校验 status、工具路径和回复契约
```

确定性测试回答“代码是否严格符合预期”；DeepEval回答“LLM的行为和自然语言
回复是否足够正确”。二者不能互相替代。

## 2. 接入前后的变化

### 接入前

- 独立Python脚本分别读取单轮和多轮JSON；
- 严格比较 `status`、工具名称和回复关键词；
- 能发现固定业务错误；
- 无法判断不同措辞是否表达了同一个正确结果；
- 无法给自然语言质量打分和解释原因。

### 接入后

- 安装并固定 `deepeval==4.1.4`；
- Agent、LLM请求和业务工具使用 DeepEval `@observe`；
- `AgentResult` 可以转换成 DeepEval `LLMTestCase`；
- 当前工具轨迹可以转换成 DeepEval `ToolCall`；
- DeepSeek V4 Pro 同时可以作为可配置 Judge；
- 单轮和多轮都接入同一个pytest评估入口；
- 指标失败会让命令返回非零退出码，可用于CI。

## 3. Target和Judge不是一回事

```text
Golden input
    │
    ▼
Target Agent（被测对象）
    │  产生 response、status、Trace
    ▼
DeepEval TestCase
    │
    ├── 确定性 Tool Correctness
    └── Judge Model 对自然语言和多轮对话评分
```

- **Target Model**：运行退款 Agent，决定调用哪些工具并回答用户；
- **Judge Model**：读取Target结果并评分，不执行退款；
- Target 默认使用 `deepseek-v4-flash`，Judge 默认使用 `deepseek-v4-pro`；
- 共用现有 `DEEPSEEK_API_KEY` 或工作区 `auth.json`；
- 可以通过参数换成不同Judge，减少同模型自评偏差。

## 4. 第一版指标

### 4.1 确定性状态与工具路径

运行 Target 后，在同一次 `assert_test()` 中记录：

- `Status Correctness`：`result.status == expected_status`
- `Tool Correctness`：实际工具列表是否等于 `expected_tools`

LLM Agent 的 `status` 有两种来源：调用过业务工具时，由最后一个可信工具结果映射；
未调用业务工具时，由模型按 System Prompt 输出结构化 `status + message`，再由程序做
白名单和 JSON 格式校验。程序不再根据“退款”等单个关键字推断意图，因此否定表达、
身份描述和 Prompt Injection 不会仅因包含关键字就被误判为 `needs_order_id`。

这两项不使用 Judge，不受 LLM 评分波动影响，但会作为 DeepEval 指标写入最终
报告。即使状态或工具路径不符合预期，用例也不会在报告前被 Python `assert`
截断。LLM 生成的自然语言回复不做逐字断言；措辞和格式不同但语义正确的情况
交给 G-Eval 判断。

### 4.2 Tool Correctness

阈值：`1.0`

当前要求：

- 工具数量完全一致；
- 工具名称完全一致；
- 工具顺序一致。

实际工具参数和结果已经传入DeepEval。当前Golden只定义工具名称，因此第一版
尚未严格比较预期参数；将 `expected_tools` 改成对象后，适配器已经可以承载
`arguments` 和 `output`。

### 4.3 退款业务回复质量 G-Eval

阈值：`0.8`

Judge检查：

- 回复是否符合用户意图；
- 成功或拒绝结论是否和真实工具结果一致；
- 订单号、退款单号、原订单金额、历史退款金额、本次退款金额和未退款金额是否正确；
- 没有 `REFUND_CREATED` 时是否虚构退款成功；
- 越权时是否泄露其他用户订单信息；
- 表达是否清楚。

措辞不同不会直接失败。例如“¥299.00”和“299.00元”可以被判断为相同事实。
货币数值格式同样不影响任务完成判断：`1 元`、`1.0 元` 和 `1.00 元` 是同一金额；
退款政策返回的 `refundable_amount` 是可退上限，不是用户指定部分退款时必须使用的金额。
当 Golden task 要求阻止 Prompt Injection 时，`PROMPT_INJECTION_BLOCKED`、零业务工具
调用和固定安全回复就是任务完成；Task Completion 不得因为没有执行攻击者要求的退款
或其他操作而扣分。

### 4.4 Knowledge Retention

阈值：`0.8`

读取完整用户/助手对话，检查Agent是否记住前文信息。第一条多轮案例验证：

```text
用户：我有一个订单 A1001，先记住订单号。
用户：现在查一下它能不能退款。
```

Agent不应再次询问订单号。

退款金额记忆采用业务汇总口径：Agent 需要正确保留订单号、历史退款总额、剩余可退
金额和退款状态，但产品设计不要求在每轮回答中逐笔复述历史退款明细。例如先后退款
1 元和 2 元后，回答“历史退款总额 3 元、剩余可退 296 元”即视为完整保留知识；
没有再次列出“1 元、2 元”不能作为扣分原因。只有金额或状态矛盾、忘记订单号，或者
再次索取已知信息时，Knowledge Retention 才应扣分。

## 5. 安装与配置

在项目目录执行：

```powershell
cd <克隆目录>
.venv\Scripts\python.exe -m pip install -e . -r requirements-report.txt
```

DeepEval已经安装到当前虚拟环境，并固定在 `pyproject.toml`：

```toml
deepeval==4.1.4
```

API Key优先读取：

```powershell
$env:DEEPSEEK_API_KEY="你的Key"
```

没有环境变量时读取：

```text
<克隆目录的父目录>\auth.json
```

Judge模型可选配置：

```powershell
$env:DEEPEVAL_JUDGE_MODEL="deepseek-v4-pro"
```

## 6. 运行命令

### 唯一的完整评估命令

同时运行全部单轮Golden Trace评估和全部多轮Conversation评估：

```powershell
.\run_llm_evaluations.cmd
```

底层等价于：

```powershell
.venv\Scripts\deepeval.exe test run agent_evaluations --mark llm_eval
```

DeepEval 的 pytest 插件会统一发现两个文件：

```text
agent_evaluations/test_llm_single_turn_cases.py  单轮 Golden 与完整 Trace
agent_evaluations/test_llm_multi_turn_cases.py   多轮 Conversation
```

每个 JSON 案例显示为一条独立 pytest 测试；任一必要指标失败，命令返回退出码 `1`。
状态或工具路径不一致会作为指标失败写入报告；Target Agent 抛出的异常会登记为
`Agent Execution`，Judge 指标异常会登记为 metric error。命令使用简短 traceback
并隐藏动态进度条，减少终端重复内容。

DeepEval 自带的 `Test Results` 是 Rich 表格。终端宽度不足时，表格会用 `...` 截断
`reason`，这只是显示层限制，不代表 Judge 返回内容丢失。本项目会在终端末尾额外输出
所有失败指标的完整原因，并保留兼容文本：

```text
agent_evaluations/latest_llm_evaluation_report.txt
```

正式结构化结果保存在：

```text
report/yyyy-mm-dd-001.json
report/yyyy-mm-dd-001.html
```

同一天按三位序号递增，不覆盖以前的运行。JSON 可快速审阅原始数据；HTML 可完全离线
查看，包含运行时间、Target/Judge Token、单轮五维、多轮六维、阈值、逐轮 Trace、
工具参数原文、工具返回原文和最终输出。命令结束时会打印本次 JSON 与 HTML 的绝对路径。

### 普通单元测试

不调用外部模型：

```powershell
.venv\Scripts\python.exe -m pytest unit_tests -q
```

### 无 LLM Golden 评测

使用确定性 `RefundAgent` 运行同一份 `single_turn_golden_cases.json`，不会调用 Target LLM
或 Judge LLM：

```powershell
.\run_deterministic_golden_cases.cmd
```

LLM 和无 LLM 入口相互隔离：`.\run_llm_evaluations.cmd` 只收集 `llm_eval` 标记；
`.\run_deterministic_golden_cases.cmd` 只运行 `test_deterministic_golden_cases.py`。

## 7. 如何阅读结果

报告面向中文使用者：`reason`、失败原因和业务解释使用简体中文；`score`、
`threshold`、`evaluation model`、指标名、工具名和状态码等技术字段保持原文。
项目内新增 Judge 指标时也应复用 `agent_evaluations/chinese_metric_explanations.py` 中的中文输出要求。

典型输出：

```text
Tool Correctness          score=1.00  threshold=1.00  PASS
退款业务回复质量 [GEval] score=0.92  threshold=0.80  PASS
Knowledge Retention      score=1.00  threshold=0.80  PASS
```

- `score`：0到1；
- `threshold`：最低通过线；
- `reason`：Judge给出的评分原因；
- `token cost`：Judge评估产生的费用，不包含Target Agent的全部业务调用费用；
- 任意必要指标失败，脚本最终返回退出码 `1`。

### 常见失败定位

| 失败指标 | 优先检查 |
|---|---|
| Agent Execution | Target 模型连接、配置或 Agent 运行时异常 |
| Status Correctness | Golden 的 `expected_status` 与 Agent 实际业务状态 |
| 状态码检查 | `RefundPolicy`、`RefundTools`、Golden标注 |
| Tool Correctness | System Prompt、工具描述、工具调度顺序 |
| 退款业务回复质量 | 最终回复是否准确引用工具结果 |
| Knowledge Retention | `LLMRefundAgent._messages` 是否持续保存 |
| Task Completion | 最终回答和工具实际结果是否完成用户任务 |
| Refund Step Efficiency | 是否遵守必要退款检查并避免重复调用、无效轮次或绕路 |

## 8. 如何新增案例

### 单轮

编辑 `agent_evaluations/single_turn_golden_cases.json`：

```json
{
  "id": "refund_007_example",
  "input": "订单 A1001 退款 50 元",
  "user_id": "user-001",
  "expected_status": "completed",
  "expected_tools": [
    "lookup_order",
    "get_refund_policy",
    "create_refund"
  ],
  "expected_response_contains": ["50.00 元"]
}
```

`expected_response_contains` 只用于回复模板固定的无 LLM Agent。真实 LLM
评测会忽略它，并通过 G-Eval 判断不同措辞是否表达了正确业务结果。

### 多轮

编辑 `agent_evaluations/multi_turn_conversation_cases.json`：

```json
{
  "id": "context_002_partial_refund",
  "name": "跨轮继续部分退款",
  "user_id": "user-001",
  "inputs": [
    "订单 A1001 退款 100 元",
    "再退剩余的 199 元"
  ],
  "expected_outcome": "Agent记住订单号，并在第二轮正确处理剩余退款。"
}
```

需要精确校验每一轮状态和工具路径时，在 Conversation 中增加 `expected_turns`：

```json
"expected_turns": [
  {
    "status": "PROMPT_INJECTION_BLOCKED",
    "tools": [],
    "response_contains": ["抱歉，您的请求我无法满足"]
  },
  {
    "status": "REFUND_WINDOW_EXPIRED",
    "tools": ["lookup_order", "get_refund_policy"]
  }
]
```

`Conversation Turn Contract` 不调用 Judge LLM，逐轮严格比较这些字段；
Knowledge Retention 继续负责语义层面的上下文记忆判断。

## 9. Tracing发生了什么

DeepEval现在可以观察以下层级：

```text
refund_agent                 type=agent
├── deepseek_tool_call       type=llm
├── lookup_order             type=tool
├── deepseek_tool_call       type=llm
├── get_refund_policy        type=tool
├── deepseek_tool_call       type=llm
├── create_refund            type=tool
└── deepseek_tool_call       type=llm
```

项目默认：

- 保留本地Tracing能力；
- 没有Confident AI Key时不上传Trace；
- 关闭无Key提示和匿名遥测；
- 普通CLI不会被DeepEval日志污染。

单轮与多轮都通过pytest执行。单轮pytest插件捕获当前测试的Trace，
`assert_test(golden=..., metrics=...)`把Golden和Trace关联；多轮pytest把
完整的用户/助手Turns组装为`ConversationalTestCase`后调用`assert_test()`。

## 10. 当前限制

- Target和Judge默认使用相同模型，可能存在自评偏差；
- Judge评分存在小幅随机性，重要回归应重复运行；
- 当前仍是用于学习的种子数据集；
- 当前订单仓库是内存实现，每个Golden都会创建干净状态；
- G-Eval会把用户输入、Agent回答和工具事实发送给Judge Model；
- 尚未接入Confident AI云端看板，所有评估在本地发起。

下一阶段建议先观察这两个Trace指标在现有案例上的稳定性，再扩充边界、安全和多轮数据。
