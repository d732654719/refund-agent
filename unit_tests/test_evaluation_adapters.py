from __future__ import annotations

import unittest
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from deepeval.test_case import MultiTurnParams

from refund_agent import RefundAgent, create_sample_repository
from refund_agent.agent_status_catalog import TASK_TIMEOUT
from agent_evaluations.chinese_metric_explanations import (
    CHINESE_REPORT_REQUIREMENT,
    REFUND_KNOWLEDGE_RETENTION_POLICY,
    REFUND_TASK_COMPLETION_POLICY,
    ChineseKnowledgeRetentionMetric,
    ChineseTaskCompletionMetric,
)
from agent_evaluations.code_contract_metrics import (
    AgentExecutionErrorMetric,
    ConversationExecutionErrorMetric,
    ConversationStatusCorrectnessMetric,
    ConversationTaskTimeoutMetric,
    ConversationToolCorrectnessMetric,
    ConversationTurnContractMetric,
    StatusCorrectnessMetric,
    TaskTimeoutMetric,
    ToolPathCorrectnessMetric,
)
from agent_evaluations.deepeval_adapters import (
    CONVERSATIONAL_JUDGE_PARAMS,
    JUDGE_REQUEST_TIMEOUT_SECONDS,
    MULTI_TURN_METRIC_NAMES,
    EvaluationTaskTimeoutError,
    ReportingDeepSeekModel,
    all_applicable_metrics_passed,
    build_deepseek_judge,
    conversation_execution_context,
    judge_timeout_from_metrics,
    result_to_assistant_turn,
    result_to_test_case,
    target_result_timeout_error,
    trace_to_tool_calls,
    validate_multi_turn_case_contract,
)
from agent_evaluations.refund_workflow_efficiency_metric import build_refund_efficiency_prompt


class DeepEvalAdapterTests(unittest.TestCase):
    def test_judge_has_one_60_second_attempt(self) -> None:
        with patch(
            "agent_evaluations.deepeval_adapters.load_deepseek_api_key",
            return_value="test-key",
        ):
            judge = build_deepseek_judge()

        self.assertEqual(JUDGE_REQUEST_TIMEOUT_SECONDS, 60)
        self.assertEqual(judge.generation_kwargs["timeout"], 60)
        self.assertEqual(judge.kwargs["timeout"], 60)
        self.assertEqual(judge.kwargs["max_retries"], 0)

    def test_reporting_judge_bypasses_deepeval_retry_wrapper(self) -> None:
        judge = object.__new__(ReportingDeepSeekModel)
        judge._report_usage_calls = []
        cost = SimpleNamespace(input_tokens=3, output_tokens=2)

        with patch(
            "agent_evaluations.deepeval_adapters._BASE_DEEPSEEK_GENERATE",
            return_value=("ok", cost),
        ) as generate:
            output, observed_cost = judge.generate("prompt")

        self.assertEqual(output, "ok")
        self.assertIs(observed_cost, cost)
        generate.assert_called_once_with(judge, "prompt", schema=None)
        self.assertEqual(judge.report_usage()["calls"], 1)

    def test_judge_timeout_is_normalized_as_task_timeout(self) -> None:
        judge = object.__new__(ReportingDeepSeekModel)
        judge._report_usage_calls = []

        with patch(
            "agent_evaluations.deepeval_adapters._BASE_DEEPSEEK_GENERATE",
            side_effect=TimeoutError("request timed out"),
        ) as generate:
            with self.assertRaisesRegex(
                EvaluationTaskTimeoutError,
                "任务超时：Judge 请求超过 60 秒",
            ):
                judge.generate("prompt")

        generate.assert_called_once_with(judge, "prompt", schema=None)
        self.assertEqual(judge.report_usage()["calls"], 0)

    def test_target_timeout_result_skips_judge_and_keeps_trace(self) -> None:
        observed = RefundAgent(create_sample_repository()).run(
            "订单 A1002 退款",
            "user-001",
        )
        timed_out = replace(
            observed,
            status=TASK_TIMEOUT,
            response="目标模型请求超时。",
            error={"type": "APITimeoutError", "message": "request timed out"},
        )

        timeout = target_result_timeout_error(timed_out)
        turn = result_to_assistant_turn(timed_out)

        self.assertIsInstance(timeout, EvaluationTaskTimeoutError)
        self.assertIn("任务超时：Target Agent 请求超过 60 秒", str(timeout))
        self.assertEqual(turn.metadata["status"], TASK_TIMEOUT)
        self.assertEqual(len(turn.metadata["trace"]), len(observed.trace))

    def test_timeout_metrics_mark_single_and_multi_case_as_error(self) -> None:
        timeout = EvaluationTaskTimeoutError(stage="Judge")
        single = TaskTimeoutMetric(timeout)
        multi = ConversationTaskTimeoutMetric(timeout)

        self.assertEqual(single.measure(None), 0.0)
        self.assertEqual(multi.measure(None), 0.0)
        self.assertEqual(single.error, "任务超时")
        self.assertEqual(multi.error, "任务超时")
        self.assertFalse(single.success)
        self.assertFalse(multi.success)

    def test_metric_stored_judge_timeout_is_normalized(self) -> None:
        metric = SimpleNamespace(
            error="APITimeoutError: request timed out",
            reason=None,
            score=None,
            success=None,
        )

        timeout = judge_timeout_from_metrics([metric])

        self.assertIsInstance(timeout, EvaluationTaskTimeoutError)
        self.assertIn("任务超时", metric.error)
        self.assertEqual(metric.reason, metric.error)
        self.assertEqual(metric.score, 0.0)
        self.assertFalse(metric.success)

    def test_conversational_judge_avoids_unsupported_context_parameter(self) -> None:
        self.assertNotIn(MultiTurnParams.CONTEXT, CONVERSATIONAL_JUDGE_PARAMS)
        self.assertIn(MultiTurnParams.METADATA, CONVERSATIONAL_JUDGE_PARAMS)

    def test_all_multi_turn_goldens_have_explicit_per_turn_contracts(self) -> None:
        cases_path = (
            Path(__file__).parents[1]
            / "agent_evaluations"
            / "multi_turn_conversation_cases.json"
        )
        cases = json.loads(cases_path.read_text(encoding="utf-8"))

        self.assertEqual(
            {
                case["id"]: validate_multi_turn_case_contract(case)
                for case in cases
            },
            {case["id"]: [] for case in cases},
        )

    def test_multi_turn_contract_validation_refuses_to_infer_expectations(self) -> None:
        errors = validate_multi_turn_case_contract(
            {
                "id": "missing-contract",
                "inputs": ["第一轮", "第二轮"],
            }
        )

        self.assertIn("缺少 expected_turns", errors[0])

    def test_multi_turn_contract_validation_checks_each_dimension(self) -> None:
        errors = validate_multi_turn_case_contract(
            {
                "inputs": ["第一轮"],
                "expected_turns": [{"status": "", "tools": [3]}],
            }
        )

        self.assertTrue(any("status" in error for error in errors))
        self.assertTrue(any("tools" in error for error in errors))

    def test_multi_turn_result_keeps_trace_and_metadata_on_same_turn(self) -> None:
        result = RefundAgent(create_sample_repository()).run(
            "订单 A1002 退款",
            "user-001",
        )

        turn = result_to_assistant_turn(result)
        context = conversation_execution_context([result])

        self.assertEqual(turn.role, "assistant")
        self.assertEqual(turn.metadata["status"], "REFUND_WINDOW_EXPIRED")
        self.assertFalse(turn.tools_called)
        self.assertEqual(turn.metadata["trace"][0]["tool_name"], "lookup_order")
        self.assertIn('"status": "REFUND_WINDOW_EXPIRED"', context[0])

    def test_multi_turn_status_and_tool_metrics_are_separate_dimensions(self) -> None:
        result = RefundAgent(create_sample_repository()).run(
            "订单 A1002 退款",
            "user-001",
        )
        expected = [
            {
                "status": "REFUND_WINDOW_EXPIRED",
                "tools": ["lookup_order", "get_refund_policy"],
            }
        ]
        status_metric = ConversationStatusCorrectnessMetric(
            actual_results=[result],
            expected_turns=expected,
        )
        tool_metric = ConversationToolCorrectnessMetric(
            actual_results=[result],
            expected_turns=expected,
        )

        self.assertEqual(status_metric.measure(None), 1.0)
        self.assertEqual(status_metric.__name__, "Status Correctness")
        self.assertEqual(tool_metric.measure(None), 1.0)
        self.assertEqual(tool_metric.__name__, "Tool Correctness")
        self.assertEqual(
            MULTI_TURN_METRIC_NAMES,
            (
                "Status Correctness",
                "Task Completion",
                "Refund Step Efficiency",
                "Tool Correctness",
                "G-Eval",
                "Knowledge Retention",
            ),
        )

    def test_case_pass_requires_every_applicable_threshold(self) -> None:
        passing = [
            SimpleNamespace(score=1.0, threshold=1.0, error=None),
            SimpleNamespace(score=0.8, threshold=0.8, error=None),
        ]
        one_below_threshold = [
            *passing,
            SimpleNamespace(score=0.79, threshold=0.8, error=None),
        ]

        self.assertTrue(all_applicable_metrics_passed(passing))
        self.assertFalse(all_applicable_metrics_passed(one_below_threshold))
        self.assertFalse(all_applicable_metrics_passed([]))

    def test_judge_prompts_require_chinese_reasons(self) -> None:
        task_metric = object.__new__(ChineseTaskCompletionMetric)
        retention_metric = object.__new__(ChineseKnowledgeRetentionMetric)

        task_prompt = task_metric._get_prompt(
            "generate_verdict",
            task="处理退款",
            actual_outcome="退款完成",
        )
        retention_prompt = retention_metric._get_prompt(
            "generate_reason",
            attritions=[],
            score="1.00",
        )

        self.assertIn(CHINESE_REPORT_REQUIREMENT, task_prompt)
        self.assertIn(CHINESE_REPORT_REQUIREMENT, retention_prompt)
        self.assertIn(REFUND_TASK_COMPLETION_POLICY, task_prompt)
        self.assertIn("1 元、1.0 元和 1.00 元", task_prompt)
        self.assertIn("最多可以退款", task_prompt)
        self.assertIn("PROMPT_INJECTION_BLOCKED", task_prompt)
        self.assertIn("正确拒绝并给出下一步引导", task_prompt)
        self.assertIn(REFUND_KNOWLEDGE_RETENTION_POLICY, retention_prompt)
        self.assertIn("不要求列出每一笔历史退款明细", retention_prompt)
        self.assertIn("verdict 应为 no", retention_prompt)

    def test_tool_path_correctness_has_chinese_reason(self) -> None:
        result = RefundAgent(create_sample_repository()).run(
            "订单 A1001 退款 100 元",
            "user-001",
        )
        matching_case = result_to_test_case(
            result,
            {
                "id": "matching_tools",
                "expected_tools": [
                    "lookup_order",
                    "get_refund_policy",
                    "create_refund",
                ],
            },
        )
        wrong_case = result_to_test_case(
            result,
            {"id": "wrong_tools", "expected_tools": ["lookup_order"]},
        )

        matching = ToolPathCorrectnessMetric()
        wrong = ToolPathCorrectnessMetric()

        self.assertEqual(matching.measure(matching_case), 1.0)
        self.assertIn("工具路径符合预期", matching.reason)
        self.assertEqual(wrong.measure(wrong_case), 0.0)
        self.assertIn("工具路径不符合预期", wrong.reason)

    def test_execution_errors_become_failed_metrics(self) -> None:
        result = RefundAgent(create_sample_repository()).run(
            "我要退款",
            "user-001",
        )
        test_case = result_to_test_case(
            result,
            {"id": "execution_error", "expected_tools": []},
        )
        error = RuntimeError("model unavailable")

        single_metric = AgentExecutionErrorMetric(error)
        conversation_metric = ConversationExecutionErrorMetric(error)

        self.assertEqual(single_metric.measure(test_case), 0.0)
        self.assertFalse(single_metric.success)
        self.assertIn("model unavailable", single_metric.reason)
        self.assertEqual(conversation_metric.error_detail, single_metric.error_detail)

    def test_conversation_turn_contract_checks_status_and_tools(self) -> None:
        result = RefundAgent(create_sample_repository()).run(
            "订单 A1002 退款",
            "user-001",
        )
        expected = [
            {
                "status": "REFUND_WINDOW_EXPIRED",
                "tools": ["lookup_order", "get_refund_policy"],
                "response_contains": ["超过"],
            }
        ]
        metric = ConversationTurnContractMetric(
            actual_results=[result],
            expected_turns=expected,
        )

        self.assertEqual(metric.measure(None), 1.0)
        self.assertTrue(metric.success)

        wrong_metric = ConversationTurnContractMetric(
            actual_results=[result],
            expected_turns=[{"status": "completed", "tools": []}],
        )
        self.assertEqual(wrong_metric.measure(None), 0.0)
        self.assertIn("第 1 轮 status 不符合预期", wrong_metric.reason)
        self.assertIn("第 1 轮工具路径不符合预期", wrong_metric.reason)

    def test_status_correctness_is_recorded_as_a_metric(self) -> None:
        result = RefundAgent(create_sample_repository()).run(
            "我要退款",
            "user-001",
        )
        test_case = result_to_test_case(
            result,
            {"id": "missing_order", "expected_tools": []},
        )

        passing = StatusCorrectnessMetric(
            actual_status=result.status,
            expected_status="needs_order_id",
        )
        failing = StatusCorrectnessMetric(
            actual_status=result.status,
            expected_status="completed",
        )

        self.assertEqual(passing.measure(test_case), 1.0)
        self.assertTrue(passing.success)
        self.assertEqual(failing.measure(test_case), 0.0)
        self.assertFalse(failing.success)
        self.assertIn("expected=completed", failing.reason)
        self.assertIn("退款已经真实创建", failing.reason)
        self.assertIn("actual=needs_order_id", failing.reason)
        self.assertIn("需要一个明确订单号", failing.reason)

    def test_result_is_converted_to_deepeval_test_case(self) -> None:
        result = RefundAgent(create_sample_repository()).run(
            "订单 A1001 退款 100 元",
            "user-001",
        )
        golden = {
            "id": "partial_refund",
            "expected_tools": [
                "lookup_order",
                "get_refund_policy",
                "create_refund",
            ],
        }

        test_case = result_to_test_case(result, golden)

        self.assertEqual(test_case.name, "partial_refund")
        self.assertEqual(test_case.actual_output, result.response)
        self.assertEqual(
            [tool.name for tool in test_case.tools_called],
            ["lookup_order", "get_refund_policy", "create_refund"],
        )
        self.assertEqual(
            [tool.name for tool in test_case.expected_tools],
            ["lookup_order", "get_refund_policy", "create_refund"],
        )
        self.assertEqual(
            test_case.tools_called[-1].input_parameters["amount"],
            "100.00",
        )

    def test_trace_conversion_includes_tool_output(self) -> None:
        result = RefundAgent(create_sample_repository()).run(
            "退款订单 A2001",
            "user-001",
        )

        calls = trace_to_tool_calls(result.trace)

        self.assertEqual(calls[0].name, "lookup_order")
        self.assertEqual(calls[0].output["code"], "ACCESS_DENIED")

    def test_refund_efficiency_prompt_declares_mandatory_workflow(self) -> None:
        prompt = build_refund_efficiency_prompt(
            "按照退款规则处理订单",
            '{"name": "refund_agent"}',
        )

        self.assertIn("必须首先调用 lookup_order", prompt)
        self.assertIn("随后必须调用 get_refund_policy", prompt)
        self.assertIn("create_refund 工具完成", prompt)
        self.assertIn("不得因为跳过权限", prompt)
        self.assertIn("reason 的值必须使用", prompt)


if __name__ == "__main__":
    unittest.main()
