from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from refund_agent import RefundAgent, create_sample_repository


GOLDENS_PATH = Path(__file__).with_name("single_turn_golden_cases.json")
GOLDENS: list[dict[str, Any]] = json.loads(
    GOLDENS_PATH.read_text(encoding="utf-8")
)


def _expected_tool_names(golden: dict[str, Any]) -> list[str]:
    return [
        tool if isinstance(tool, str) else tool["name"]
        for tool in golden.get("expected_tools", [])
    ]


def _golden_params() -> list[Any]:
    params: list[Any] = []
    for case in GOLDENS:
        reason = case.get("no_llm_xfail_reason")
        marks = (
            pytest.mark.xfail(reason=reason, strict=True)
            if reason
            else ()
        )
        params.append(
            pytest.param(
                case,
                id=f"single_turn_golden_cases.json(no_llm)::{case['id']}",
                marks=marks,
            )
        )
    return params


@pytest.mark.no_llm_eval
@pytest.mark.parametrize("case", _golden_params())
def test_deterministic_golden_cases(case: dict[str, Any]) -> None:
    """Evaluate the deterministic RefundAgent without calling any LLM."""
    result = RefundAgent(create_sample_repository()).run(
        case["input"],
        case["user_id"],
    )

    assert result.status == case["expected_status"]
    assert [step.tool_name for step in result.trace] == _expected_tool_names(case)

    # The no-LLM agent renders fixed response templates, so exact key phrases
    # are stable and appropriate here. The LLM evaluation ignores this field.
    for expected_text in case.get("expected_response_contains", []):
        assert expected_text in result.response
