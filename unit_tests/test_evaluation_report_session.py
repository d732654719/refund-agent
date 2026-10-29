from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent_evaluations import evaluation_report_session as session
from agent_evaluations.deepeval_adapters import ReportingDeepSeekModel
from agent_evaluations.evaluation_report_writer import write_evaluation_report
from refund_agent.refund_domain_models import AgentResult, ToolResult, TraceStep


def _metrics(*, multi_turn: bool) -> list[SimpleNamespace]:
    definitions = [
        ("Status Correctness", 1.0),
        ("Task Completion", 0.7),
        ("Refund Step Efficiency", 0.8),
        ("Tool Correctness", 1.0),
        ("G-Eval", 0.8),
    ]
    if multi_turn:
        definitions.append(("Knowledge Retention", 0.8))
    return [
        SimpleNamespace(
            __name__=name,
            score=1.0,
            threshold=threshold,
            success=True,
            reason="达到阈值。",
            error=None,
            evaluation_model="deterministic-code",
        )
        for name, threshold in definitions
    ]


def _result(user_input: str, *, turn: int = 1) -> AgentResult:
    tool_result = ToolResult(ok=True, code="ORDER_FOUND", message="查到订单")
    return AgentResult(
        input=user_input,
        user_id="user-001",
        response="订单可以退款。",
        status="responded",
        trace=[
            TraceStep(
                tool_name="lookup_order",
                arguments={"order_id": "A1001"},
                result=tool_result.to_dict(),
                duration_ms=0.2,
                turn=1,
                tool_call_id=f"call-{turn}",
                arguments_raw='{"order_id":"A1001"}',
                result_raw=json.dumps(tool_result.to_dict(), ensure_ascii=False),
                raw_capture_status="captured",
            )
        ],
        model_turns=2,
        conversation_turn=turn,
        duration_ms=120.0,
        target_model="deepseek-v4-flash",
        usage={
            "source": "target",
            "capture_status": "captured",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "note": None,
        },
        model_calls=[
            {
                "turn": 1,
                "duration_ms": 80.0,
                "usage": {"total_tokens": 15},
                "outcome": "completed",
            }
        ],
        events=[
            {
                "event": "model_response",
                "turn": 1,
                "content": "准备查询订单。",
            }
        ],
    )


def test_session_writes_real_schema_and_combines_target_judge_tokens(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        session,
        "write_evaluation_report",
        lambda payload: write_evaluation_report(payload, report_dir=tmp_path),
    )
    monkeypatch.setattr(session, "record_text_case_metrics", lambda **kwargs: None)
    session.start_evaluation_session()
    single = {
        "id": "query_all_my_orders",
        "input": "查询订单 A1001",
        "task": "查询自己的订单",
        "user_id": "user-001",
        "expected_status": "responded",
        "expected_tools": ["lookup_order"],
    }
    judge_usage = {
        "source": "judge",
        "capture_status": "captured",
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
        "note": None,
    }
    session.record_case_metrics(
        source="single_turn_golden_cases.json",
        case=single,
        metrics=_metrics(multi_turn=False),
        results=[_result(single["input"])],
        target_output="订单可以退款。",
        judge_usage=judge_usage,
    )
    multi = {
        "id": "context_001_order_id_retention",
        "task": "跨轮记住订单号",
        "user_id": "user-001",
        "inputs": ["记住 A1001", "查询它"],
        "expected_outcome": "记住订单号",
        "expected_turns": [
            {"status": "responded", "tools": ["lookup_order"]},
            {"status": "responded", "tools": ["lookup_order"]},
        ],
    }
    session.record_case_metrics(
        source="multi_turn_conversation_cases.json",
        case=multi,
        metrics=_metrics(multi_turn=True),
        results=[_result("记住 A1001", turn=1), _result("查询它", turn=2)],
        target_output="订单可以退款。",
        judge_usage=judge_usage,
    )

    paths = session.finalize_evaluation_session(exit_status=0)
    assert paths is not None
    document = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert document["summary"]["case_counts"] == {
        "total": 2,
        "passed": 2,
        "failed": 0,
        "error": 0,
        "single_turn": 1,
        "multi_turn": 1,
    }
    summary_metrics = {metric["id"]: metric for metric in document["summary"]["metrics"]}
    assert summary_metrics["task_completion"]["score"] == 2.0
    assert summary_metrics["task_completion"]["max_score"] == 2
    assert summary_metrics["knowledge_retention"]["score"] == 1.0
    assert summary_metrics["knowledge_retention"]["max_score"] == 1
    assert document["summary"]["composite"] == {
        "earned_points": 11.0,
        "max_points": 11,
        "percentage": 100.0,
        "gap": 0.0,
        "method": "累计所有适用 Case 的原始得分；综合得分为六个维度累计得分之和",
    }
    assert document["run"]["tokens"]["target"] == 45
    assert document["run"]["tokens"]["judge"] == 60
    assert document["run"]["tokens"]["total"] == 105
    assert len(document["cases"][0]["metrics"]) == 5
    assert len(document["cases"][1]["metrics"]) == 6
    assert len(document["cases"][1]["turns"]) == 2
    assert document["cases"][0]["turns"][0]["trace"][0]["arguments_raw"] == '{"order_id":"A1001"}'
    assert paths.html_path.exists()


def test_reporting_judge_usage_retains_tokens_without_network() -> None:
    model = object.__new__(ReportingDeepSeekModel)
    model._report_usage_calls = []
    model._capture_report_usage(
        SimpleNamespace(input_tokens=12, output_tokens=7)
    )

    assert model.report_usage() == {
        "source": "judge",
        "capture_status": "captured",
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "total_tokens": 19,
        "calls": 1,
        "note": None,
    }


def test_summary_uses_cumulative_points_instead_of_case_average() -> None:
    metric_ids = [item["id"] for item in session.METRIC_CATALOG[:-1]]
    cases = []
    for index, task_score in enumerate((1.0, 0.5), start=1):
        cases.append(
            {
                "status": "passed" if index == 1 else "failed",
                "kind": "single_turn",
                "metrics": [
                    {
                        "id": metric_id,
                        "score": task_score if metric_id == "task_completion" else 1.0,
                        "passed": not (metric_id == "task_completion" and task_score < 0.7),
                    }
                    for metric_id in metric_ids
                ],
            }
        )

    summary = session._build_summary(cases)
    task = next(
        metric
        for metric in summary["metrics"]
        if metric["id"] == "task_completion"
    )

    assert task["score"] == 1.5
    assert task["max_score"] == 2
    assert summary["composite"]["earned_points"] == 9.5
    assert summary["composite"]["max_points"] == 10


def test_metric_error_forces_structured_score_to_zero() -> None:
    metric = SimpleNamespace(
        __name__="Knowledge Retention",
        score=1.0,
        threshold=0.8,
        error=TypeError("序列化失败"),
        reason="模型先返回了分数，但评估随后报错。",
        evaluation_model="deepseek-v4-pro",
    )

    normalized = session._normalize_metrics(
        [metric],
        applicable_ids=["knowledge_retention"],
        results=[],
        expected_turns=[],
        fallback_reason="不可用",
    )

    assert normalized[0]["score"] == 0.0
    assert normalized[0]["passed"] is False
    assert normalized[0]["error"] == {
        "type": "TypeError",
        "category": "evaluation",
        "message": "序列化失败",
    }


def test_timeout_keeps_partial_evidence_and_renders_public_chinese_error(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        session,
        "write_evaluation_report",
        lambda payload: write_evaluation_report(payload, report_dir=tmp_path),
    )
    monkeypatch.setattr(session, "record_text_case_metrics", lambda **kwargs: None)
    session.start_evaluation_session()
    case = {
        "id": "context_001_order_id_retention",
        "task": "跨轮记住订单号",
        "user_id": "user-001",
        "inputs": ["记住 A1001", "继续退款"],
        "expected_outcome": "完成两轮退款",
        "expected_turns": [
            {"status": "responded", "tools": ["lookup_order"]},
            {"status": "responded", "tools": []},
        ],
    }
    completed_metric = SimpleNamespace(
        __name__="Task Completion",
        score=0.94,
        threshold=0.7,
        success=True,
        reason="超时前已经完成的评分必须保留。",
        error=None,
        evaluation_model="deepseek-v4-pro",
    )
    session.record_case_metrics(
        source="multi_turn_conversation_cases.json",
        case=case,
        metrics=[completed_metric],
        results=[_result("记住 A1001", turn=1)],
        target_output="Traceback: internal.py:42 secret timeout stack",
        judge_usage=None,
        execution_error=TimeoutError("upstream request timed out at internal.py:42"),
    )

    paths = session.finalize_evaluation_session(exit_status=1)
    assert paths is not None
    document = json.loads(paths.json_path.read_text(encoding="utf-8"))
    reported = document["cases"][0]
    assert reported["status"] == "error"
    assert reported["error"] == {
        "type": "TimeoutError",
        "category": "timeout",
        "message": "任务超时：执行超过时限，评估未能完成。",
    }
    assert len(reported["turns"]) == 2
    assert reported["turns"][0]["trace"][0]["tool_name"] == "lookup_order"
    assert reported["turns"][0]["usage"]["total_tokens"] == 15
    assert reported["turns"][0]["model_calls"][0]["outcome"] == "completed"
    assert reported["turns"][0]["events"][0]["event"] == "model_response"
    assert reported["turns"][1]["user_input"] == "继续退款"
    assert reported["turns"][1]["agent_status"] == "execution_timeout"
    assert reported["turns"][1]["usage"]["capture_status"] == "unavailable"
    task_metric = next(
        metric for metric in reported["metrics"] if metric["id"] == "task_completion"
    )
    assert task_metric["score"] == 0.94
    assert task_metric["error"] is None
    missing_metric = next(
        metric for metric in reported["metrics"] if metric["id"] == "geval"
    )
    assert missing_metric["error"]["category"] == "timeout"

    html = paths.html_path.read_text(encoding="utf-8")
    assert "任务超时" in html
    assert "internal.py:42" not in html
    assert "Traceback" not in html
