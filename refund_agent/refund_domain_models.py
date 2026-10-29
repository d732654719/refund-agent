from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from .agent_status_catalog import validate_agent_status


def unavailable_target_usage() -> dict[str, Any]:
    """Return an explicit, report-friendly token capture placeholder."""
    return {
        "source": "target",
        "capture_status": "unavailable",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "note": "模型响应未提供 usage。",
    }


@dataclass
class Order:
    order_id: str
    user_id: str
    amount: Decimal
    status: str
    delivered_at: datetime | None
    product_type: str = "physical"
    refunded_amount: Decimal = Decimal("0")

    @property
    def refundable_balance(self) -> Decimal:
        return max(Decimal("0"), self.amount - self.refunded_amount)

    def public_view(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "amount": str(self.amount),
            "status": self.status,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "product_type": self.product_type,
            "refunded_amount": str(self.refunded_amount),
            "refundable_balance": str(self.refundable_balance),
        }


@dataclass(frozen=True)
class PolicyDecision:
    eligible: bool
    code: str
    reason: str
    refundable_amount: Decimal

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "code": self.code,
            "reason": self.reason,
            "refundable_amount": str(self.refundable_amount),
        }


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    code: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "code": self.code,
            "message": self.message,
            "data": self.data,
        }


@dataclass(frozen=True)
class TraceStep:
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    duration_ms: float
    turn: int | None = None
    tool_call_id: str | None = None
    arguments_raw: str | None = None
    result_raw: str | None = None
    raw_capture_status: str = "unavailable"

    def to_dict(self) -> dict[str, Any]:
        data = {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "duration_ms": round(self.duration_ms, 3),
        }
        if self.turn is not None:
            data["turn"] = self.turn
        data.update(
            {
                "tool_call_id": self.tool_call_id,
                "arguments_raw": self.arguments_raw,
                "result_raw": self.result_raw,
                "raw_capture_status": self.raw_capture_status,
            }
        )
        return data


@dataclass(frozen=True)
class AgentResult:
    input: str
    user_id: str
    response: str
    status: str
    trace: list[TraceStep]
    model_turns: int = 0
    conversation_turn: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    model_duration_ms: float = 0.0
    target_model: str | None = None
    usage: dict[str, Any] = field(default_factory=unavailable_target_usage)
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, str] | None = None

    def __post_init__(self) -> None:
        validate_agent_status(self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "user_id": self.user_id,
            "response": self.response,
            "status": self.status,
            "conversation_turn": self.conversation_turn,
            "model_turns": self.model_turns,
            "duration_ms": round(self.duration_ms, 3),
            "model_duration_ms": round(self.model_duration_ms, 3),
            "target_model": self.target_model,
            "usage": self.usage,
            "model_calls": self.model_calls,
            "error": self.error,
            "trace": [step.to_dict() for step in self.trace],
            "events": self.events,
        }
