# 退款运行模块

## 职责与接口

本目录从用户输入生成 `AgentResult`，包含 LLM 调度、确定性基线和可信业务工具。

| 入口 | 责任与输出 |
| --- | --- |
| [llm_agent.py](llm_agent.py) | `LLMRefundAgent.run(user_input, user_id)` 执行一轮用户请求；模型可在该请求内调用多轮工具 |
| [deterministic_refund_agent.py](deterministic_refund_agent.py) | `RefundAgent.run(...)` 提供不调用模型的基线；`_call_tool()` 记录工具 Trace |
| [refund_business_tools.py](refund_business_tools.py) | `RefundTools` 查询订单、检查资格和创建退款，返回 `ToolResult` |
| [refund_policy.py](refund_policy.py) | `RefundPolicy.evaluate(order, now)` 返回资格决定；不修改订单 |
| [sample_order_repository.py](sample_order_repository.py) | 内存订单读写与样例数据；不是持久化存储 |
| [refund_domain_models.py](refund_domain_models.py) | `Order`、`ToolResult`、`TraceStep`、`AgentResult` 的数据契约 |
| [model_runtime.py](model_runtime.py) | 模型默认配置、认证读取与客户端构造 |
| [command_line_interface.py](command_line_interface.py) | CLI 参数、模式选择和输出；不实现退款规则 |

## 决策约束

- 身份由调用方传入，模型工具参数不能决定当前用户。`RefundTools.create_refund()` 必须再次检查订单归属、资格和金额，不能信任先前查单或 Prompt 的结论。
- 金额使用 `Decimal`；期限和资格判断以 `RefundPolicy.evaluate()` 为准，说明文档不另抄一份规则数值表。
- `content` 是模型文本，`tool_calls` 才是工具请求。`ToolResult.code`、`AgentResult.status`、`Order.status` 各有职责；状态常量和映射的唯一入口是 [agent_status_catalog.py](agent_status_catalog.py)。
- `_complete_result()` 根据本次 Trace 中成功的 `create_refund` 生成金额说明；后续查询不能抹掉已退款事实，后续异常也不能被伪装成正常完成。
- `reset_conversation()` 清理对话上下文，不撤销订单修改。一个多轮用例复用 Agent；不同用例重新创建仓库和 Agent。
- 认证来源顺序由 `load_deepseek_api_key()` 实现；不在文档或日志中复制实际密钥。

## 本分支协议

无业务工具结果时，当前实现用 `_parse_no_tool_response()` 解析模型 `content` 中的 `status + message` JSON，并校验状态白名单。不要将另一个 worktree 的 `final_response` 协议当成本分支已实现的接口。

## 修改后检查

从项目根目录按改动选择 [运行单元测试](../unit_tests/AGENTS.md)。状态或工具接口变化时，同时检查 [评测模块](../agent_evaluations/AGENTS.md) 的预期及 [状态指南](../agent_status_guide.md)。模型替身测试只能验证程序协议，真实模型行为另按项目入口执行评测。
