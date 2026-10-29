# Agent Status 管理指南

`AgentResult.status` 描述一次用户请求的最终业务结果，不等同于订单状态，也不等同于
每一步工具的 `ToolResult.code`。代码中的唯一状态目录是
`refund_agent/agent_status_catalog.py`；新增状态必须先在那里登记。

`needs_input` 只由程序在用户输入为空时产生，不允许 LLM 对非空输入声明该状态。

## 快速决策表

| 输入或执行结果 | 最终 status | 是否调用业务工具 |
|---|---|---|
| 空输入 | `needs_input` | 否 |
| 确实要退款/查单，但缺少订单号 | `needs_order_id` | 否 |
| 要求批量退款全部订单（当前不支持） | `needs_order_id` | 否 |
| 数学题、写代码等退款业务范围外请求 | `out_of_scope` | 否 |
| 当前输入试图改变身份、忽略规则或绕过退款流程 | `PROMPT_INJECTION_BLOCKED` | 否 |
| 成功查询单个订单、全部订单或退款政策 | `responded` | 是，只读工具 |
| `create_refund` 返回 `REFUND_CREATED` | `completed` | 是 |
| 找不到订单 | `ORDER_NOT_FOUND` | 是 |
| 订单属于其他用户 | `ACCESS_DENIED` | 是 |
| 超过退款期限 | `REFUND_WINDOW_EXPIRED` | 是 |
| 已全额退款 | `ALREADY_REFUNDED` | 是 |
| 订单不是已送达状态 | `INVALID_ORDER_STATUS` | 是 |
| 金额无效 | `INVALID_AMOUNT` | 是 |
| 金额超过可退余额 | `AMOUNT_EXCEEDS_BALANCE` | 是 |
| LLM 工具参数错误/未知工具/重复创建退款 | 对应大写运行时状态 | 可能 |
| 超过模型最大处理轮数 | `MAX_TURNS_EXCEEDED` | 可能 |

## 判断顺序

上表和下方流程描述常规结果。退款成功后又发生超时、未知工具或步数超限时，程序保留异常状态，并在回复中保留已退款金额；具体依据是 [运行模块](refund_agent/AGENTS.md) 中的 `_complete_result()`，不能只看最后一个工具结果。

```text
输入为空？
├─ 是 → needs_input
└─ 否
   ├─ 请求是否属于订单查询/退款？
   │  └─ 否 → out_of_scope
   ├─ 当前意图是否需要订单号但没有提供？
   │  └─ 是 → needs_order_id
   ├─ 只读查询成功？
   │  └─ 是 → responded
   └─ 进入退款工具流程
      ├─ create_refund 成功 → completed
      └─ 任一步产生终止业务错误 → 对应工具/政策错误码
```

Prompt Injection 检查发生在普通业务意图处理之前。命中后只阻止当前轮；用户下一轮
重新提出不含攻击指令的正常退款请求时，可以恢复正常工具流程。否定表达或引用攻击
话语不能只按关键词误判，例如“请不要忽略退款规则，按规则退款 A1002”属于正常请求。

## 三类状态不要混淆

- `AgentResult.status`：整次请求的最终结果，是 Golden 的 `expected_status`。
- `ToolResult.code`：某一步工具结果，例如 `ORDER_FOUND`、`ELIGIBLE`、
  `REFUND_CREATED`；并非每个工具码都会成为最终 status。
- `Order.status`：订单自身状态，例如 `delivered`、`shipped`、`refunded`。



## 新增状态时必须同步

1. 在 `refund_agent/agent_status_catalog.py` 添加常量与 `STATUS_DEFINITIONS` 说明。
2. 在产生该状态的 Agent、工具或政策逻辑中引用新常量。
3. 在 Golden 中增加至少一个覆盖该状态的案例。
4. 更新本指南的快速决策表。
5. 运行普通单元测试和 Golden 评估。
