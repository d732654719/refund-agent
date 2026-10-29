from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from .refund_domain_models import Order


class InMemoryOrderRepository:
    """Small mutable repository used by the demo agent and tests."""

    def __init__(self, orders: list[Order] | None = None) -> None:
        self._orders = {order.order_id.upper(): order for order in orders or []}
        self._refund_sequence = 0

    def get(self, order_id: str) -> Order | None:
        return self._orders.get(order_id.upper())

    def list_by_user(self, user_id: str) -> list[Order]:
        return sorted(
            (order for order in self._orders.values() if order.user_id == user_id),
            key=lambda order: order.order_id,
        )

    def save(self, order: Order) -> None:
        self._orders[order.order_id.upper()] = order

    def next_refund_id(self) -> str:
        self._refund_sequence += 1
        return f"RF-{self._refund_sequence:06d}"


def create_sample_repository(now: datetime | None = None) -> InMemoryOrderRepository:
    now = now or datetime.now(timezone.utc)
    return InMemoryOrderRepository(
        [
            Order(
                order_id="A1001",
                user_id="user-001",
                amount=Decimal("299.00"),
                status="delivered",
                delivered_at=now - timedelta(days=5),
            ),
            Order(
                order_id="A1002",
                user_id="user-001",
                amount=Decimal("99.00"),
                status="delivered",
                delivered_at=now - timedelta(days=45),
            ),
            Order(
                order_id="A1003",
                user_id="user-001",
                amount=Decimal("59.00"),
                status="refunded",
                delivered_at=now - timedelta(days=3),
                refunded_amount=Decimal("59.00"),
            ),
            Order(
                order_id="A2001",
                user_id="user-002",
                amount=Decimal("499.00"),
                status="delivered",
                delivered_at=now - timedelta(days=2),
            ),
            Order(
                order_id="A1004",
                user_id="user-001",
                amount=Decimal("79.00"),
                status="delivered",
                delivered_at=now - timedelta(days=1),
            ),
            Order(
                order_id="A1005",
                user_id="user-001",
                amount=Decimal("59.00"),
                status="shipped",
                delivered_at=now - timedelta(days=3),
               
            )
        ]
    )
