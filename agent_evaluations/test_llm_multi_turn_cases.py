from __future__ import annotations

import json
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from deepeval import assert_test
from deepeval.test_case import ConversationalTestCase, Turn

from agent_evaluations.deepeval_adapters import (
    build_multi_turn_metrics,
    build_deepseek_judge,
    conversation_execution_context,
    evaluation_timeout_from_error,
    judge_timeout_from_metrics,
    result_to_assistant_turn,
    target_result_timeout_error,
    validate_multi_turn_case_contract,
)
from agent_evaluations.code_contract_metrics import (
    ConversationDataIntegrityMetric,
    ConversationExecutionErrorMetric,
    ConversationTaskTimeoutMetric,
)
from agent_evaluations.evaluation_report_session import record_case_metrics
from refund_agent import LLMRefundAgent, create_sample_repository


CONVERSATIONS_PATH = Path(__file__).with_name("multi_turn_conversation_cases.json")
CONVERSATIONS: list[dict[str, Any]] = json.loads(
    CONVERSATIONS_PATH.read_text(encoding="utf-8")
)


@pytest.mark.llm_eval
@pytest.mark.parametrize(
    "case",
    CONVERSATIONS,
    ids=lambda case: f"multi_turn_conversation_cases.json::{case['id']}",
)
def test_conversations_json(case: dict[str, Any]) -> None:
    """Run a real multi-turn conversation as one DeepEval pytest case."""
    contract_errors = validate_multi_turn_case_contract(case)
    if contract_errors:
        invalid_case = ConversationalTestCase(
            name=case.get("id", "invalid-multi-turn-case"),
            turns=[
                Turn(role="user", content=user_input)
                for user_input in case.get("inputs", ["Golden 数据不完整"])
            ],
            scenario=case.get("task"),
            expected_outcome=case.get("expected_outcome"),
        )
        metrics = [ConversationDataIntegrityMetric(contract_errors)]
        try:
            with redirect_stderr(StringIO()):
                assert_test(
                    test_case=invalid_case,
                    metrics=metrics,
                    run_async=False,
                )
        finally:
            record_case_metrics(
                source="multi_turn_conversation_cases.json",
                case=case,
                metrics=metrics,
                results=[],
                target_output="未调用 Target Agent：多轮 Golden 数据不完整。",
            )
        return

    turns: list[Turn] = []
    results = []
    target_timeout = None

    try:
        agent = LLMRefundAgent.from_environment(create_sample_repository())
        for user_input in case["inputs"]:
            turns.append(Turn(role="user", content=user_input))
            result = agent.run(user_input, case["user_id"])
            results.append(result)
            turns.append(result_to_assistant_turn(result))
            target_timeout = target_result_timeout_error(result)
            if target_timeout is not None:
                break
    except Exception as error:
        if not turns:
            turns.append(Turn(role="user", content=case["inputs"][0]))
        turns.append(
            Turn(
                role="assistant",
                content=(
                    f"Target Agent 执行异常：{type(error).__name__}: {error}"
                ),
            )
        )
        error_case = ConversationalTestCase(
            name=case["id"],
            turns=turns,
            scenario=case["task"],
            expected_outcome=case["expected_outcome"],
        )
        metrics = [ConversationExecutionErrorMetric(error)]
        try:
            with redirect_stderr(StringIO()):
                assert_test(
                    test_case=error_case,
                    metrics=metrics,
                    run_async=False,
                )
        finally:
            record_case_metrics(
                source="multi_turn_conversation_cases.json",
                case=case,
                metrics=metrics,
                results=results,
                target_output=turns[-1].content,
                execution_error=error,
            )
        return

    if target_timeout is not None:
        timeout_case = ConversationalTestCase(
            name=case["id"],
            turns=turns,
            scenario=case["task"],
            context=conversation_execution_context(results),
            expected_outcome=case["expected_outcome"],
        )
        metrics = [ConversationTaskTimeoutMetric(target_timeout)]
        try:
            with redirect_stderr(StringIO()):
                assert_test(
                    test_case=timeout_case,
                    metrics=metrics,
                    run_async=False,
                )
        finally:
            record_case_metrics(
                source="multi_turn_conversation_cases.json",
                case=case,
                metrics=metrics,
                results=results,
                target_output=results[-1].response,
                execution_error=target_timeout,
            )
        return

    test_case = ConversationalTestCase(
        name=case["id"],
        turns=turns,
        scenario=case["task"],
        metadata={
            "case_id": case["id"],
            "turn_count": len(results),
            "evidence_location": "assistant turn metadata.trace 与 context",
            "execution_context": conversation_execution_context(results),
            "expected_turns": case["expected_turns"],
        },
        context=[
            *conversation_execution_context(results),
            json.dumps(
                {"expected_turns": case["expected_turns"]},
                ensure_ascii=False,
            ),
        ],
        expected_outcome=case["expected_outcome"],
        chatbot_role=(
            "只处理电商订单查询与退款，并遵守权限和退款政策的客服 Agent"
        ),
    )
    judge = build_deepseek_judge()
    metrics = build_multi_turn_metrics(
        judge,
        actual_results=results,
        expected_turns=case["expected_turns"],
    )
    judge_timeout = None
    try:
        with redirect_stderr(StringIO()):
            assert_test(
                test_case=test_case,
                metrics=metrics,
                run_async=False,
            )
    except Exception as error:
        judge_timeout = evaluation_timeout_from_error(error, stage="Judge")
        if judge_timeout is None:
            raise
    finally:
        judge_timeout = judge_timeout or judge_timeout_from_metrics(metrics)
        if judge_timeout is not None:
            timeout_metric = ConversationTaskTimeoutMetric(judge_timeout)
            timeout_metric.measure(None)
            metrics.append(timeout_metric)
        record_case_metrics(
            source="multi_turn_conversation_cases.json",
            case=case,
            metrics=metrics,
            results=results,
            target_output=turns[-1].content,
            judge_usage=judge.report_usage(),
            execution_error=judge_timeout,
        )
    if judge_timeout is not None:
        pytest.fail(str(judge_timeout))
