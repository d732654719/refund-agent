from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from .refund_domain_models import Order, PolicyDecision


class RefundPolicy:
    """Deterministic policy layer that cannot be bypassed by the agent."""

    def __init__(self, physical_window_days: int = 30) -> None:
        self.physical_window_days = physical_window_days

    def evaluate(
        self, order: Order, now: datetime | None = None
    ) -> PolicyDecision:
        now = now or datetime.now(timezone.utc)

        if order.refundable_balance <= 0 or order.status == "refunded":
            return PolicyDecision(
                eligible=False,
                code="ALREADY_REFUNDED",
                reason="订单已完成全额退款，不能重复退款。",
                refundable_amount=Decimal("0"),
            )

        if order.status != "delivered" or order.delivered_at is None:
            return PolicyDecision(
                eligible=False,
                code="INVALID_ORDER_STATUS",
                reason="只有已送达的订单可以申请退款。",
                refundable_amount=Decimal("0"),
            )

        age_days = (now - order.delivered_at).days
        if age_days > self.physical_window_days:
            return PolicyDecision(
                eligible=False,
                code="REFUND_WINDOW_EXPIRED",
                reason=f"订单已超过 {self.physical_window_days} 天退款期限。",
                refundable_amount=Decimal("0"),
            )

        return PolicyDecision(
            eligible=True,
            code="ELIGIBLE",
            reason="订单处于退款期限内，可以退款。",
            refundable_amount=order.refundable_balance,
        )
