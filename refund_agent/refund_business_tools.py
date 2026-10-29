from __future__ import annotations

from decimal import Decimal, InvalidOperation

from deepeval.tracing import observe

from .refund_domain_models import Order, ToolResult
from .refund_policy import RefundPolicy
from .sample_order_repository import InMemoryOrderRepository


class RefundTools:
    """Trusted tools. The caller identity is injected by the runtime."""

    def __init__(
        self,
        repository: InMemoryOrderRepository,
        policy: RefundPolicy | None = None,
    ) -> None:
        self.repository = repository
        self.policy = policy or RefundPolicy()

    def _owned_order(
        self, requester_id: str, order_id: str
    ) -> tuple[Order | None, ToolResult | None]:
        order = self.repository.get(order_id)
        if order is None:
            return None, ToolResult(
                ok=False,
                code="ORDER_NOT_FOUND",
                message="未找到该订单。",
            )
        if order.user_id != requester_id:
            return None, ToolResult(
                ok=False,
                code="ACCESS_DENIED",
                message="无权访问该订单。",
            )
        return order, None

    @observe(name="lookup_order", type="tool", description="查询当前用户自己的订单")
    def lookup_order(self, requester_id: str, order_id: str) -> ToolResult:
        order, error = self._owned_order(requester_id, order_id)
        if error is not None:
            return error
        assert order is not None
        return ToolResult(
            ok=True,
            code="ORDER_FOUND",
            message="订单查询成功。",
            data={"order": order.public_view()},
        )

    @observe(
        name="list_user_orders",
        type="tool",
        description="查询当前登录用户自己的全部订单",
    )
    def list_user_orders(self, requester_id: str) -> ToolResult:
        orders = self.repository.list_by_user(requester_id)
        return ToolResult(
            ok=True,
            code="ORDERS_FOUND" if orders else "NO_ORDERS",
            message=(
                f"查询到 {len(orders)} 个订单。"
                if orders
                else "当前用户没有订单。"
            ),
            data={"orders": [order.public_view() for order in orders]},
        )

    @observe(
        name="get_refund_policy",
        type="tool",
        description="检查订单退款资格和可退金额",
    )
    def get_refund_policy(self, requester_id: str, order_id: str) -> ToolResult:
        order, error = self._owned_order(requester_id, order_id)
        if error is not None:
            return error
        assert order is not None
        decision = self.policy.evaluate(order)
        return ToolResult(
            ok=True,
            code=decision.code,
            message=decision.reason,
            data={"decision": decision.to_dict()},
        )

    @observe(name="create_refund", type="tool", description="创建一笔订单退款")
    def create_refund(
        self,
        requester_id: str,
        order_id: str,
        amount: str,
        reason: str,
    ) -> ToolResult:
        order, error = self._owned_order(requester_id, order_id)
        if error is not None:
            return error
        assert order is not None
        decision = self.policy.evaluate(order)
        if not decision.eligible:
            return ToolResult(
                ok=False,
                code=decision.code,
                message=decision.reason,
                data={"decision": decision.to_dict()},
            )

        try:
            requested_amount = Decimal(str(amount)).quantize(Decimal("0.01"))
        except (InvalidOperation, ValueError):
            return ToolResult(
                ok=False,
                code="INVALID_AMOUNT",
                message="退款金额格式不正确。",
            )

        if requested_amount <= 0:
            return ToolResult(
                ok=False,
                code="INVALID_AMOUNT",
                message="退款金额必须大于 0。",
            )
        if requested_amount > decision.refundable_amount:
            return ToolResult(
                ok=False,
                code="AMOUNT_EXCEEDS_BALANCE",
                message=f"退款金额超过可退余额 {decision.refundable_amount} 元。",
                data={"refundable_amount": str(decision.refundable_amount)},
            )

        original_order_amount = order.amount
        historical_refunded_amount = order.refunded_amount
        order.refunded_amount += requested_amount
        if order.refundable_balance == 0:
            order.status = "refunded"
        self.repository.save(order)

        refund_id = self.repository.next_refund_id()
        return ToolResult(
            ok=True,
            code="REFUND_CREATED",
            message="退款已创建。",
            data={
                "refund_id": refund_id,
                "order_id": order.order_id,
                "amount": format(requested_amount, ".2f"),
                "original_order_amount": format(original_order_amount, ".2f"),
                "historical_refunded_amount": format(
                    historical_refunded_amount, ".2f"
                ),
                "current_refund_amount": format(requested_amount, ".2f"),
                "total_refunded_amount": format(order.refunded_amount, ".2f"),
                "reason": reason,
                "remaining_refundable_amount": format(
                    order.refundable_balance, ".2f"
                ),
            },
        )
