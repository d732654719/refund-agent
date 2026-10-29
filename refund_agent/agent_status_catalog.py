from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# Conversation and orchestration statuses.
NEEDS_INPUT: Final = "needs_input"
NEEDS_ORDER_ID: Final = "needs_order_id"
OUT_OF_SCOPE: Final = "out_of_scope"
RESPONDED: Final = "responded"
COMPLETED: Final = "completed"
MAX_TURNS_EXCEEDED: Final = "MAX_TURNS_EXCEEDED"
TASK_TIMEOUT: Final = "TASK_TIMEOUT"
PROMPT_INJECTION_BLOCKED: Final = "PROMPT_INJECTION_BLOCKED"

# Terminal business/tool failures that are exposed as AgentResult.status.
ORDER_NOT_FOUND: Final = "ORDER_NOT_FOUND"
ACCESS_DENIED: Final = "ACCESS_DENIED"
REFUND_WINDOW_EXPIRED: Final = "REFUND_WINDOW_EXPIRED"
ALREADY_REFUNDED: Final = "ALREADY_REFUNDED"
INVALID_ORDER_STATUS: Final = "INVALID_ORDER_STATUS"
INVALID_AMOUNT: Final = "INVALID_AMOUNT"
AMOUNT_EXCEEDS_BALANCE: Final = "AMOUNT_EXCEEDS_BALANCE"
DUPLICATE_REFUND_ATTEMPT: Final = "DUPLICATE_REFUND_ATTEMPT"
INVALID_TOOL_ARGUMENTS: Final = "INVALID_TOOL_ARGUMENTS"
UNKNOWN_TOOL: Final = "UNKNOWN_TOOL"


NO_TOOL_FINAL_STATUSES: Final = frozenset(
    {
        NEEDS_ORDER_ID,
        OUT_OF_SCOPE,
        RESPONDED,
        PROMPT_INJECTION_BLOCKED,
    }
)
TERMINAL_TOOL_STATUSES: Final = frozenset(
    {
        ORDER_NOT_FOUND,
        ACCESS_DENIED,
        REFUND_WINDOW_EXPIRED,
        ALREADY_REFUNDED,
        INVALID_ORDER_STATUS,
        INVALID_AMOUNT,
        AMOUNT_EXCEEDS_BALANCE,
        DUPLICATE_REFUND_ATTEMPT,
        INVALID_TOOL_ARGUMENTS,
        UNKNOWN_TOOL,
    }
)


@dataclass(frozen=True)
class AgentStatusDefinition:
    category: str
    meaning: str
    generated_when: str
    source: str


STATUS_DEFINITIONS: Final[dict[str, AgentStatusDefinition]] = {
    NEEDS_INPUT: AgentStatusDefinition(
        category="input",
        meaning="没有足够输入形成有效请求。",
        generated_when="输入为空或只有空白字符。",
        source="程序直接判断，不调用 LLM 或工具。",
    ),
    NEEDS_ORDER_ID: AgentStatusDefinition(
        category="input",
        meaning="意图属于退款业务，但需要一个明确订单号才能继续。",
        generated_when="用户要退款/查单/查政策但未提供订单号，或请求批量退款。",
        source="无工具回答由 LLM 声明，程序做白名单校验。",
    ),
    OUT_OF_SCOPE: AgentStatusDefinition(
        category="scope",
        meaning="用户当前请求不属于电商订单查询或退款能力。",
        generated_when="例如要求回答数学题、写代码或执行其他范围外操作。",
        source="无工具回答由 LLM 根据完整语义判断，不能只匹配‘退款’关键字。",
    ),
    PROMPT_INJECTION_BLOCKED: AgentStatusDefinition(
        category="security_rejection",
        meaning="当前请求试图改变身份、忽略规则或绕过可信退款流程，已阻止本轮请求。",
        generated_when="用户当前输入明确企图覆盖指令、绕过权限/政策/工具检查，或要求 Agent 伪造执行结果。",
        source="无工具回答由 LLM 做语义识别，程序校验状态并应用固定安全回复。",
    ),
    RESPONDED: AgentStatusDefinition(
        category="success",
        meaning="请求已回答，但没有创建退款。",
        generated_when="成功查询订单/订单列表/政策，或直接回答无需执行退款的问题。",
        source="只读工具成功后由程序映射，或无工具回答由 LLM 声明。",
    ),
    COMPLETED: AgentStatusDefinition(
        category="success",
        meaning="退款已经真实创建。",
        generated_when="create_refund 返回 REFUND_CREATED。",
        source="只能由可信工具结果映射，LLM 无权自行声明。",
    ),
    ORDER_NOT_FOUND: AgentStatusDefinition(
        category="business_failure",
        meaning="订单不存在。",
        generated_when="lookup_order 找不到订单号。",
        source="可信工具结果。",
    ),
    ACCESS_DENIED: AgentStatusDefinition(
        category="business_failure",
        meaning="订单不属于当前登录用户。",
        generated_when="订单存在，但 requester_id 与订单用户不一致。",
        source="可信工具结果。",
    ),
    REFUND_WINDOW_EXPIRED: AgentStatusDefinition(
        category="business_failure",
        meaning="订单已超过退款期限。",
        generated_when="退款政策判断送达时间超过 30 天。",
        source="可信退款政策结果。",
    ),
    ALREADY_REFUNDED: AgentStatusDefinition(
        category="business_failure",
        meaning="订单已全额退款。",
        generated_when="可退余额为 0 或订单状态为 refunded。",
        source="可信退款政策结果。",
    ),
    INVALID_ORDER_STATUS: AgentStatusDefinition(
        category="business_failure",
        meaning="当前订单状态不允许退款。",
        generated_when="订单不是 delivered，例如 shipped。",
        source="可信退款政策结果。",
    ),
    INVALID_AMOUNT: AgentStatusDefinition(
        category="business_failure",
        meaning="退款金额格式错误或不大于 0。",
        generated_when="create_refund 校验金额失败。",
        source="可信工具结果。",
    ),
    AMOUNT_EXCEEDS_BALANCE: AgentStatusDefinition(
        category="business_failure",
        meaning="申请退款金额超过当前可退余额。",
        generated_when="create_refund 的 amount 大于 refundable_amount。",
        source="可信工具结果。",
    ),
    DUPLICATE_REFUND_ATTEMPT: AgentStatusDefinition(
        category="runtime_failure",
        meaning="同一用户请求中重复尝试创建退款。",
        generated_when="一个 Agent run 中第二次调用 create_refund。",
        source="Agent 安全调度器。",
    ),
    INVALID_TOOL_ARGUMENTS: AgentStatusDefinition(
        category="runtime_failure",
        meaning="LLM 生成的工具参数无法解析或缺少必要参数。",
        generated_when="工具参数不是合法 JSON 或缺少订单号/金额。",
        source="Agent 工具调度器。",
    ),
    UNKNOWN_TOOL: AgentStatusDefinition(
        category="runtime_failure",
        meaning="LLM 请求了未注册工具。",
        generated_when="tool_call.name 不在允许的工具集合中。",
        source="Agent 工具调度器。",
    ),
    MAX_TURNS_EXCEEDED: AgentStatusDefinition(
        category="runtime_failure",
        meaning="模型处理轮数达到上限仍未产生最终回答。",
        generated_when="LLM tool-calling 循环超过 max_turns。",
        source="Agent 循环控制器。",
    ),
    TASK_TIMEOUT: AgentStatusDefinition(
        category="runtime_failure",
        meaning="目标模型单次调用超过时限，任务已停止。",
        generated_when="一次 OpenAI/DeepSeek 模型请求超过 60 秒。",
        source="Agent 模型调用边界。",
    ),
}

ALL_AGENT_STATUSES: Final = frozenset(STATUS_DEFINITIONS)


def validate_agent_status(status: str) -> str:
    if status not in ALL_AGENT_STATUSES:
        raise ValueError(
            f"未登记的 Agent status：{status!r}。请先在 refund_agent/agent_status_catalog.py 中登记。"
        )
    return status


def agent_status_from_tool_code(tool_code: str) -> str:
    """Map the last trusted tool result to the public AgentResult status."""
    if tool_code == "REFUND_CREATED":
        return COMPLETED
    if tool_code in TERMINAL_TOOL_STATUSES:
        return tool_code
    return RESPONDED
