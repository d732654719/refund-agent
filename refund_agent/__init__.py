"""Minimal e-commerce refund agent."""

import os

# Keep local tracing available for DeepEval without polluting normal CLI/tests
# or sending anonymous usage telemetry.
os.environ.setdefault("CONFIDENT_TRACE_VERBOSE", "0")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from .deterministic_refund_agent import RefundAgent
from .llm_agent import LLMRefundAgent
from .sample_order_repository import InMemoryOrderRepository, create_sample_repository
from .agent_status_catalog import STATUS_DEFINITIONS

__all__ = [
    "RefundAgent",
    "LLMRefundAgent",
    "InMemoryOrderRepository",
    "create_sample_repository",
    "STATUS_DEFINITIONS",
]
