from __future__ import annotations

import json
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase

from refund_agent import LLMRefundAgent, create_sample_repository
from agent_evaluations.code_contract_metrics import (
    AgentExecutionErrorMetric,
    StatusCorrectnessMetric,
    TaskTimeoutMetric,
)
from agent_evaluations.deepeval_adapters import (
    build_deepseek_judge,
    build_single_turn_metrics,
    build_trace_metrics,
    evaluation_timeout_from_error,
    expected_tool_calls,
    judge_timeout_from_metrics,
    result_to_test_case,
    target_result_timeout_error,
)
from agent_evaluations.evaluation_report_session import record_case_metrics


GOLDENS_PATH = Path(__file__).with_name("single_turn_golden_cases.json")
GOLDENS: list[dict[str, Any]] = json.loads(
    GOLDENS_PATH.read_text(encoding="utf-8")
)


@pytest.mark.llm_eval
@pytest.mark.parametrize(
    "case",
    GOLDENS,
    ids=lambda case: f"single_turn_golden_cases.json::{case['id']}",
)
def test_llm_goldens_json(case: dict[str, Any]) -> None:
    """Run one real agent trace and fail when business or trace metrics fail."""
    try:
        agent = LLMRefundAgent.from_environment(create_sample_repository())
        result = agent.run(case["input"], case["user_id"])
    except Exception as error:
        error_case = LLMTestCase(
            name=case["id"],
            input=case["input"],
            actual_output=f"Target Agent 执行异常：{type(error).__name__}: {error}",
            metadata={"user_id": case["user_id"]},
        )
        metrics = [AgentExecutionErrorMetric(error)]
        try:
            with redirect_stderr(StringIO()):
                assert_test(
                    test_case=error_case,
                    metrics=metrics,
                    run_async=False,
                )
        finally:
            record_case_metrics(
                source="single_turn_golden_cases.json",
                case=case,
                metrics=metrics,
                results=[],
                target_output=error_case.actual_output,
                execution_error=error,
            )
        return

    target_timeout = target_result_timeout_error(result)
    if target_timeout is not None:
        timeout_case = result_to_test_case(result, case)
        metrics = [TaskTimeoutMetric(target_timeout)]
        try:
            with redirect_stderr(StringIO()):
                assert_test(
                    test_case=timeout_case,
                    metrics=metrics,
                    run_async=False,
                )
        finally:
            record_case_metrics(
                source="single_turn_golden_cases.json",
                case=case,
                metrics=metrics,
                results=[result],
                target_output=result.response,
                execution_error=target_timeout,
            )
        return

    # DeepEval's pytest plugin associates the observed trace above with this
    # Golden. Status and tool-path mismatches are metrics too, so every Agent
    # result reaches the final DeepEval report instead of failing beforehand.
    golden = Golden(
        name=case["id"],
        input=case["input"],
        additional_metadata={"user_id": case["user_id"]},
        context=[
            json.dumps(
                {
                    "status": result.status,
                    "tool_results": [
                        {
                            "name": step.tool_name,
                            "arguments": step.arguments,
                            "result": step.result,
                        }
                        for step in result.trace
                    ],
                },
                ensure_ascii=False,
            )
        ],
        expected_tools=expected_tool_calls(case.get("expected_tools", [])),
    )
    judge = build_deepseek_judge()
    metrics = (
        [
            StatusCorrectnessMetric(
                actual_status=result.status,
                expected_status=case["expected_status"],
            )
        ]
        + build_trace_metrics(judge, case["task"])
        + build_single_turn_metrics(judge)
    )
    judge_timeout = None
    try:
        with redirect_stderr(StringIO()):
            assert_test(golden=golden, metrics=metrics)
    except Exception as error:
        judge_timeout = evaluation_timeout_from_error(error, stage="Judge")
        if judge_timeout is None:
            raise
    finally:
        judge_timeout = judge_timeout or judge_timeout_from_metrics(metrics)
        if judge_timeout is not None:
            timeout_metric = TaskTimeoutMetric(judge_timeout)
            timeout_metric.measure(None)
            metrics.append(timeout_metric)
        record_case_metrics(
            source="single_turn_golden_cases.json",
            case=case,
            metrics=metrics,
            results=[result],
            target_output=result.response,
            judge_usage=judge.report_usage(),
            execution_error=judge_timeout,
        )
    if judge_timeout is not None:
        pytest.fail(str(judge_timeout))
