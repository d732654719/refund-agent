from __future__ import annotations

import re
from decimal import Decimal
from time import perf_counter
from typing import Any, Callable

from .refund_domain_models import AgentResult, ToolResult, TraceStep
from .refund_policy import RefundPolicy
from .sample_order_repository import InMemoryOrderRepository
from .agent_status_catalog import (
    COMPLETED,
    NEEDS_INPUT,
    NEEDS_ORDER_ID,
    OUT_OF_SCOPE,
    RESPONDED,
)
from .refund_business_tools import RefundTools


# Use ASCII-alphanumeric boundaries instead of ``\b``. In Python regex,
# Chinese characters and ASCII letters are both "word" characters, so ``\b``
# cannot find an order ID in natural input such as "退款A2001".
ORDER_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]\d{4,})(?![A-Za-z0-9])"
)
MONEY_PATTERNS = (
    re.compile(r"([0-9]+(?:\.[0-9]{1,2})?)\s*元"),
    re.compile(r"[¥￥$]\s*([0-9]+(?:\.[0-9]{1,2})?)"),
)
REFUND_KEYWORDS = ("退款", "退钱", "退货", "refund")
LIST_ORDERS_KEYWORDS = ("我的订单", "所有订单", "全部订单", "订单列表")


class RefundAgent:
    """
    Minimal tool-using agent.

    It is deterministic on purpose: the target behavior is repeatable while
    the evaluation pipeline is being built. The orchestration layer can later
    be replaced by an LLM without changing the trusted tools or policy.
    """

    def __init__(
        self,
        repository: InMemoryOrderRepository,
        policy: RefundPolicy | None = None,  
    ) -> None:
        self.tools = RefundTools(repository=repository, policy=policy)

    def run(self, user_input: str, user_id: str) -> AgentResult:
        trace: list[TraceStep] = []
        normalized = user_input.strip()

        if not normalized:
            return self._result(
                user_input, user_id, "请告诉我需要处理的退款问题。", NEEDS_INPUT, trace
            )

        has_refund_intent = any(
            keyword in normalized.lower() for keyword in REFUND_KEYWORDS
        )
        wants_order_list = any(
            keyword in normalized.lower() for keyword in LIST_ORDERS_KEYWORDS
        )
        if wants_order_list and not has_refund_intent:
            listed = self._call_tool(
                trace,
                "list_user_orders",
                {},
                lambda: self.tools.list_user_orders(user_id),
            )
            orders = listed.data["orders"]
            if not orders:
                return self._result(
                    user_input,
                    user_id,
                    "当前账户下没有订单。",
                    RESPONDED,
                    trace,
                )
            lines = [f"查询到 {len(orders)} 个订单："]
            lines.extend(
                f"- {order['order_id']}：状态 {order['status']}，"
                f"金额 {order['amount']} 元，可退余额 {order['refundable_balance']} 元"
                for order in orders
            )
            return self._result(
                user_input,
                user_id,
                "\n".join(lines),
                RESPONDED,
                trace,
            )

        if not has_refund_intent:
            return self._result(
                user_input,
                user_id,
                "我目前只能协助处理电商订单退款。请提供订单号，例如 A1001。",
                OUT_OF_SCOPE,
                trace,
            )

        order_match = ORDER_ID_PATTERN.search(normalized)
        if not order_match:
            return self._result(
                user_input,
                user_id,
                "可以帮您处理退款，请先提供订单号。",
                NEEDS_ORDER_ID,
                trace,
            )
        order_id = order_match.group(1).upper()

        lookup = self._call_tool(
            trace,
            "lookup_order",
            {"order_id": order_id},
            lambda: self.tools.lookup_order(user_id, order_id),
        )
        if not lookup.ok:
            response = (
                "无法访问该订单，请确认订单号属于当前登录账户。"
                if lookup.code == "ACCESS_DENIED"
                else "没有找到这个订单，请检查订单号后重试。"
            )
            return self._result(user_input, user_id, response, lookup.code, trace)

        policy = self._call_tool(
            trace,
            "get_refund_policy",
            {"order_id": order_id},
            lambda: self.tools.get_refund_policy(user_id, order_id),
        )
        decision = policy.data["decision"]
        if not decision["eligible"]:
            return self._result(
                user_input,
                user_id,
                f"订单 {order_id} 暂不能退款：{decision['reason']}",
                decision["code"],
                trace,
            )

        requested_amount = self._extract_amount(normalized)
        if requested_amount is None:
            requested_amount = Decimal(decision["refundable_amount"])

        refund = self._call_tool(
            trace,
            "create_refund",
            {
                "order_id": order_id,
                "amount": str(requested_amount),
                "reason": "用户申请退款",
            },
            lambda: self.tools.create_refund(
                requester_id=user_id,
                order_id=order_id,
                amount=str(requested_amount),
                reason="用户申请退款",
            ),
        )
        if not refund.ok:
            return self._result(
                user_input,
                user_id,
                f"退款未创建：{refund.message}",
                refund.code,
                trace,
            )

        response = self._format_refund_success_response(refund.data)
        return self._result(user_input, user_id, response, COMPLETED, trace)

    @staticmethod
    def _format_refund_success_response(data: dict[str, Any]) -> str:
        """Render the required, code-controlled refund amount summary."""
        return (
            f"退款已成功创建。订单：{data['order_id']}，"
            f"退款单号：{data['refund_id']}。\n"
            f"- 原订单金额：{data['original_order_amount']} 元\n"
            f"- 历史退款金额（本次前）：{data['historical_refunded_amount']} 元\n"
            f"- 本次退款金额：{data['current_refund_amount']} 元\n"
            f"- 未退款金额：{data['remaining_refundable_amount']} 元"
        )

    @staticmethod
    def _extract_amount(text: str) -> Decimal | None:
        for pattern in MONEY_PATTERNS:
            match = pattern.search(text)
            if match:
                return Decimal(match.group(1)).quantize(Decimal("0.01"))
        return None

    @staticmethod
    def _call_tool(
        trace: list[TraceStep],
        tool_name: str,
        arguments: dict[str, Any],
        call: Callable[[], ToolResult],
        turn: int | None = None,
    ) -> ToolResult:
        started = perf_counter()
        result = call()
        trace.append(
            TraceStep(
                tool_name=tool_name,
                arguments=arguments,
                result=result.to_dict(),
                duration_ms=(perf_counter() - started) * 1000,
                turn=turn,
            )
        )
        return result

    @staticmethod
    def _result(
        user_input: str,
        user_id: str,
        response: str,
        status: str,
        trace: list[TraceStep],
    ) -> AgentResult:
        return AgentResult(
            input=user_input,
            user_id=user_id,
            response=response,
            status=status,
            trace=trace,
        )
