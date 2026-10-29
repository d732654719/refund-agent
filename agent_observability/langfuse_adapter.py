"""Best-effort Langfuse adapter for applications that run the refund Agent.

This package is intentionally outside ``refund_agent``. It can observe a run,
but it has no business decision authority and is injected by application code.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from hashlib import sha256
import logging
import os
from typing import Any, ContextManager

from langfuse import Langfuse, propagate_attributes


LOGGER = logging.getLogger(__name__)


def _text_summary(value: Any) -> dict[str, Any]:
    """Return a stable, non-reversible summary instead of raw user content."""
    text = "" if value is None else str(value)
    return {
        "length": len(text),
        "sha256_12": sha256(text.encode("utf-8")).hexdigest()[:12],
    }


def summarize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Keep message structure useful while excluding prompt and order contents."""
    summary: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            summary.append({"message_type": type(message).__name__})
            continue

        item: dict[str, Any] = {"role": message.get("role", "unknown")}
        if "content" in message:
            item["content"] = _text_summary(message.get("content"))
        if "tool_call_id" in message:
            item["tool_call_id_present"] = bool(message["tool_call_id"])
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            item["tool_calls"] = [
                {
                    "name": getattr(getattr(call, "function", None), "name", None),
                    "arguments": _text_summary(
                        getattr(getattr(call, "function", None), "arguments", None)
                    ),
                }
                for call in tool_calls
            ]
        summary.append(item)
    return summary


def summarize_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Describe tool argument shape without sending identifiers or amounts."""
    return {
        key: _text_summary(value) if value is not None else {"is_null": True}
        for key, value in arguments.items()
    }


def summarize_model_message(message: Any) -> dict[str, Any]:
    """Describe a provider response without persisting its raw content."""
    tool_calls = getattr(message, "tool_calls", None)
    return {
        "content": _text_summary(getattr(message, "content", None)),
        "tool_calls": [
            {
                "name": getattr(getattr(call, "function", None), "name", None),
                "arguments": _text_summary(
                    getattr(getattr(call, "function", None), "arguments", None)
                ),
            }
            for call in tool_calls or []
        ],
    }


def create_langfuse_client() -> Langfuse:
    """Create a Langfuse client for explicit Cloud operations.

    Unlike best-effort runtime telemetry, Dataset and Experiment commands must
    fail fast when credentials are absent so that a partial local run is not
    mistaken for a Cloud evaluation.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        raise RuntimeError(
            "Langfuse 环境变量未在当前终端生效："
            "LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY"
        )
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=os.getenv("LANGFUSE_BASE_URL"),
        environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT"),
        release=os.getenv("LANGFUSE_RELEASE"),
    )


class LangfuseTelemetry:
    """Failure-isolated adapter around the Langfuse Python SDK."""

    def __init__(self, client: Langfuse | None = None) -> None:
        self._client: Langfuse | None = client
        if self._client is not None:
            return
        try:
            self._client = create_langfuse_client()
        except RuntimeError:
            # Missing credentials are an expected local-development condition.
            return
        except Exception:
            LOGGER.exception("Langfuse initialization failed; telemetry disabled.")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def start_agent(
        self,
        *,
        user_id: str,
        conversation_turn: int,
        user_input: str,
        session_id: str | None = None,
        trace_id: str | None = None,
        experiment_trace_id: str | None = None,
    ) -> ContextManager[Any]:
        if self._client is None:
            return nullcontext(None)
        return self._agent_observation_context(
            user_id=user_id,
            conversation_turn=conversation_turn,
            user_input=user_input,
            session_id=session_id,
            trace_id=trace_id,
            experiment_trace_id=experiment_trace_id,
        )

    @contextmanager
    def _agent_observation_context(
        self,
        *,
        user_id: str,
        conversation_turn: int,
        user_input: str,
        session_id: str | None,
        trace_id: str | None,
        experiment_trace_id: str | None,
    ) -> Any:
        """Keep Session propagation outside the Agent business package."""
        assert self._client is not None
        observation_started = False
        try:
            with propagate_attributes(session_id=session_id):
                with self._client.start_as_current_observation(
                    trace_context={"trace_id": trace_id} if trace_id else None,
                    name="refund_agent",
                    as_type="agent",
                    input={"user_input": _text_summary(user_input)},
                    metadata={
                        "user_id_hash": _text_summary(user_id)["sha256_12"],
                        "conversation_turn": conversation_turn,
                        "experiment_trace_id": experiment_trace_id,
                    },
                ) as observation:
                    observation_started = True
                    yield observation
        except Exception:
            if observation_started:
                raise
            LOGGER.exception("Unable to start Langfuse Agent observation.")
            yield None

    def start_generation(
        self, *, messages: list[Any], model: str
    ) -> ContextManager[Any]:
        if self._client is None:
            return nullcontext(None)
        try:
            return self._client.start_as_current_observation(
                name="deepseek_completion",
                as_type="generation",
                input={"messages": summarize_messages(messages)},
                model=model,
                model_parameters={"tool_choice": "auto", "stream": False},
                metadata={"tool_names": list(TOOL_NAMES)},
            )
        except Exception:
            LOGGER.exception("Unable to start Langfuse generation observation.")
            return nullcontext(None)

    def start_tool(
        self, *, tool_name: str, arguments: dict[str, Any], turn: int | None
    ) -> ContextManager[Any]:
        if self._client is None:
            return nullcontext(None)
        try:
            return self._client.start_as_current_observation(
                name=tool_name,
                as_type="tool",
                input={"arguments": summarize_tool_arguments(arguments)},
                metadata={"turn": turn},
            )
        except Exception:
            LOGGER.exception("Unable to start Langfuse tool observation.")
            return nullcontext(None)

    @staticmethod
    def update(observation: Any, **values: Any) -> None:
        if observation is None:
            return
        try:
            observation.update(**values)
        except Exception:
            LOGGER.exception("Unable to update Langfuse observation.")

    def flush(self) -> None:
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception:
            LOGGER.exception("Unable to flush Langfuse observations.")


def create_langfuse_telemetry(client: Langfuse | None = None) -> LangfuseTelemetry:
    """Create the optional adapter at the application composition boundary."""
    return LangfuseTelemetry(client=client)


TOOL_NAMES = (
    "list_user_orders",
    "lookup_order",
    "get_refund_policy",
    "create_refund",
)
