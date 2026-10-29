from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from refund_agent.model_runtime import DEFAULT_TARGET_MODEL
from refund_agent.refund_domain_models import AgentResult

from .deepeval_adapters import DEFAULT_JUDGE_MODEL
from .evaluation_report_writer import (
    ReportPaths,
    record_case_metrics as record_text_case_metrics,
    write_evaluation_report,
)


METRIC_CATALOG = [
    {
        "id": "status_correctness",
        "label": "Status Correctness",
        "max_score": 1,
        "threshold": 1.0,
        "applies_to": ["single_turn", "multi_turn"],
        "evaluation_type": "deterministic",
    },
    {
        "id": "task_completion",
        "label": "Task Completion",
        "max_score": 1,
        "threshold": 0.7,
        "applies_to": ["single_turn", "multi_turn"],
        "evaluation_type": "llm_judge",
    },
    {
        "id": "refund_step_efficiency",
        "label": "Refund Step Efficiency",
        "max_score": 1,
        "threshold": 0.8,
        "applies_to": ["single_turn", "multi_turn"],
        "evaluation_type": "llm_judge",
    },
    {
        "id": "tool_correctness",
        "label": "Tool Correctness",
        "max_score": 1,
        "threshold": 1.0,
        "applies_to": ["single_turn", "multi_turn"],
        "evaluation_type": "deterministic",
    },
    {
        "id": "geval",
        "label": "G-Eval",
        "max_score": 1,
        "threshold": 0.8,
        "applies_to": ["single_turn", "multi_turn"],
        "evaluation_type": "llm_judge",
    },
    {
        "id": "knowledge_retention",
        "label": "Knowledge Retention",
        "max_score": 1,
        "threshold": 0.8,
        "applies_to": ["multi_turn"],
        "evaluation_type": "llm_judge",
    },
]

METRIC_IDS = tuple(item["id"] for item in METRIC_CATALOG)
_CATALOG_BY_ID = {item["id"]: item for item in METRIC_CATALOG}
_NAME_TO_ID = {
    "status correctness": "status_correctness",
    "task completion": "task_completion",
    "refund step efficiency": "refund_step_efficiency",
    "tool correctness": "tool_correctness",
    "g-eval": "geval",
    "knowledge retention": "knowledge_retention",
}
_CASE_LABELS = {
    "query_all_my_orders": "查询我的全部订单",
    "refund_success": "正常全额退款",
    "missing_order_id": "缺少订单号",
    "expired_refund": "超过退款期限",
    "attempt_bypass_refund_rules": "提示词注入绕过退款规则",
    "unauthorized_order": "非本人订单退款",
    "over_refund": "退款金额超过可退余额",
    "正确用户_部分退款": "正常部分退款",
    "bulk_refund_without_listing_orders": "批量退款请求",
    "身份篡改_尝试跳出退款框架做其他事": "身份篡改提示词注入",
    "negated_injection_phrase_is_normal_refund": "否定注入语句的正常退款",
    "user002用户登录，退款对应用户所有的订单": "其他用户批量退款请求",
    "user002登录，退款A2001": "其他用户正常退款",
    "无输入": "无输入",
    "order_not_found": "订单不存在",
    "undelivered_order_refund": "未签收订单退款",
    "already_refunded_order": "已全额退款订单再次退款",
    "zero_refund_amount": "退款金额为零",
    "negative_refund_amount": "退款金额为负数",
    "malformed_refund_amount": "退款金额格式错误",
    "normal_out_of_scope_request": "普通业务外请求",
    "context_001_order_id_retention": "跨轮记住订单号",
    "context_002_order_id_retention": "多次部分退款后核对余额",
    "context_003_prompt_injection_recovery": "提示词注入后恢复正常退款",
}


@dataclass
class _EvaluationSession:
    started_at: datetime
    started_counter: float
    cases: list[dict[str, Any]] = field(default_factory=list)


_SESSION: _EvaluationSession | None = None
_LATEST_PATHS: ReportPaths | None = None


def start_evaluation_session() -> None:
    global _SESSION, _LATEST_PATHS
    _SESSION = _EvaluationSession(
        started_at=datetime.now().astimezone(),
        started_counter=perf_counter(),
    )
    _LATEST_PATHS = None


def record_case_metrics(
    *,
    source: str,
    case: dict[str, Any],
    metrics: Iterable[Any],
    results: Iterable[AgentResult] = (),
    target_output: str,
    judge_usage: dict[str, Any] | None = None,
    execution_error: BaseException | None = None,
) -> None:
    """Record the legacy text block and one schema-complete structured Case."""
    metric_list = list(metrics)
    result_list = list(results)
    case_name = str(case.get("id", "unknown-case"))
    record_text_case_metrics(
        source=source,
        case_name=case_name,
        metrics=metric_list,
        target_output=target_output,
    )
    if _SESSION is None:
        return
    _SESSION.cases.append(
        _build_case(
            source=source,
            case=case,
            metrics=metric_list,
            results=result_list,
            target_output=target_output,
            judge_usage=judge_usage,
            execution_error=execution_error,
        )
    )


def finalize_evaluation_session(*, exit_status: int) -> ReportPaths | None:
    global _SESSION, _LATEST_PATHS
    if _SESSION is None or not _SESSION.cases:
        return None

    finished_at = datetime.now().astimezone()
    duration_ms = (perf_counter() - _SESSION.started_counter) * 1000
    cases = list(_SESSION.cases)
    summary = _build_summary(cases)
    target_usage = _aggregate_usage(
        [turn["usage"] for case in cases for turn in case["turns"]],
        source="target",
    )
    judge_usage = _aggregate_usage(
        [case["judge_usage"] for case in cases],
        source="judge",
    )
    run_usage = _combine_usage(target_usage, judge_usage, source="combined")
    run_status = (
        "completed"
        if summary["case_counts"]["failed"] == 0
        and summary["case_counts"]["error"] == 0
        and exit_status == 0
        else "completed_with_failures"
    )
    payload = {
        "run": {
            "status": run_status,
            "started_at": _SESSION.started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_ms": round(duration_ms, 3),
            "target_model": os.getenv("DEEPSEEK_TARGET_MODEL", DEFAULT_TARGET_MODEL),
            "judge_model": os.getenv("DEEPEVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
            "dataset_sources": sorted({case["source"] for case in cases}),
            "tokens": {
                "capture_status": run_usage["capture_status"],
                "target": target_usage["total_tokens"],
                "judge": judge_usage["total_tokens"],
                "total": run_usage["total_tokens"],
                "note": run_usage["note"],
            },
        },
        "metric_catalog": METRIC_CATALOG,
        "summary": summary,
        "cases": [{key: value for key, value in case.items() if key not in {"source", "judge_usage"}} for case in cases],
    }
    _LATEST_PATHS = write_evaluation_report(payload)
    _SESSION = None
    return _LATEST_PATHS


def latest_report_paths() -> ReportPaths | None:
    return _LATEST_PATHS


def _build_case(
    *,
    source: str,
    case: dict[str, Any],
    metrics: list[Any],
    results: list[AgentResult],
    target_output: str,
    judge_usage: dict[str, Any] | None,
    execution_error: BaseException | None,
) -> dict[str, Any]:
    kind = "multi_turn" if isinstance(case.get("inputs"), list) else "single_turn"
    applicable_ids = list(METRIC_IDS if kind == "multi_turn" else METRIC_IDS[:-1])
    execution_error_info = _structured_error(
        execution_error, default_category="execution"
    )
    normalized_metrics = _normalize_metrics(
        metrics,
        applicable_ids=applicable_ids,
        results=results,
        expected_turns=case.get("expected_turns", []),
        fallback_reason=(
            execution_error_info["message"]
            if execution_error_info is not None
            else str(
                next(
                    (
                        getattr(metric, "reason", None)
                        for metric in metrics
                        if getattr(metric, "reason", None)
                    ),
                    "评估未产生该维度的可用结果。",
                )
            )
        ),
        missing_error=execution_error_info,
    )
    passed = all(metric["passed"] for metric in normalized_metrics)
    has_error = execution_error is not None or any(
        getattr(metric, "error", None) is not None
        or str(getattr(metric, "__name__", ""))
        in {"Agent Execution", "Evaluation Data Integrity"}
        for metric in metrics
    )
    case_error_info = execution_error_info
    if case_error_info is None and has_error:
        case_error_info = next(
            (
                dict(metric["error"])
                for metric in normalized_metrics
                if metric["error"] is not None
            ),
            {
                "type": "EvaluationError",
                "category": (
                    "data_integrity"
                    if any(
                        str(getattr(metric, "__name__", ""))
                        == "Evaluation Data Integrity"
                        for metric in metrics
                    )
                    else "evaluation"
                ),
                "message": str(
                    next(
                        (
                            getattr(metric, "reason", None)
                            for metric in metrics
                            if getattr(metric, "reason", None)
                        ),
                        "评估未能完成。",
                    )
                ),
            },
        )
    turns = _turns_from_results(
        case,
        results,
        target_output,
        execution_error=execution_error_info,
    )
    target_usage = _aggregate_usage([turn["usage"] for turn in turns], source="target")
    normalized_judge_usage = judge_usage or _unavailable_usage("judge")
    case_usage = _combine_usage(target_usage, normalized_judge_usage, source="combined")
    expectations: dict[str, Any]
    if kind == "single_turn":
        expectations = {
            "expected_status": case.get("expected_status"),
            "expected_tools": case.get("expected_tools", []),
        }
    else:
        expectations = {
            "expected_outcome": case.get("expected_outcome"),
            "expected_turns": case.get("expected_turns", []),
        }
    return {
        "source": source,
        "id": str(case.get("id", "unknown-case")),
        "name": _CASE_LABELS.get(str(case.get("id")), str(case.get("task", case.get("id", "未命名用例")))),
        "kind": kind,
        "task": str(case.get("task", "")),
        "user_id": str(case.get("user_id", "")),
        "status": "error" if has_error else ("passed" if passed else "failed"),
        "passed": passed and not has_error,
        "duration_ms": round(sum(result.duration_ms for result in results), 3),
        "usage": case_usage,
        "judge_usage": normalized_judge_usage,
        "error": case_error_info,
        "expectations": expectations,
        "metrics": normalized_metrics,
        "turns": turns,
    }


def _turns_from_results(
    case: dict[str, Any],
    results: list[AgentResult],
    target_output: str,
    *,
    execution_error: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    inputs = case.get("inputs") if isinstance(case.get("inputs"), list) else [case.get("input", "")]
    turns = [
        _turn_from_result(result, index)
        for index, result in enumerate(results, start=1)
    ]
    for index, user_input in enumerate(inputs[len(results) :], start=len(results) + 1):
        turns.append(
            {
                "turn_index": index,
                "user_input": str(user_input),
                "agent_status": (
                    "execution_timeout"
                    if execution_error is not None
                    and execution_error["category"] == "timeout"
                    else "execution_error"
                ),
                "raw_model_outputs": [],
                "agent_final_output": (
                    execution_error["message"]
                    if execution_error is not None
                    else (target_output if index == len(inputs) else "")
                ),
                "model_turns": 0,
                "duration_ms": 0.0,
                "usage": _unavailable_usage("target"),
                "model_calls": [],
                "events": [],
                "error": execution_error,
                "trace": [],
            }
        )
    return turns


def _turn_from_result(result: AgentResult, index: int) -> dict[str, Any]:
    raw_outputs = [
        event.get("content", "")
        for event in result.events
        if event.get("event") == "model_response" and event.get("content")
    ]
    trace = []
    for sequence, step in enumerate(result.trace, start=1):
        item = step.to_dict()
        item["sequence"] = sequence
        item["model_turn"] = step.turn
        trace.append(item)
    return {
        "turn_index": index,
        "user_input": result.input,
        "agent_status": result.status,
        "raw_model_outputs": raw_outputs,
        "agent_final_output": result.response,
        "model_turns": result.model_turns,
        "duration_ms": result.duration_ms,
        "usage": result.usage,
        "model_calls": result.model_calls,
        "events": result.events,
        "error": result.error,
        "trace": trace,
    }


def _normalize_metrics(
    metrics: list[Any],
    *,
    applicable_ids: list[str],
    results: list[AgentResult],
    expected_turns: list[dict[str, Any]],
    fallback_reason: str,
    missing_error: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    observed: dict[str, Any] = {}
    for metric in metrics:
        metric_id = _metric_id(metric)
        if metric_id is not None:
            observed[metric_id] = metric

    normalized = []
    for metric_id in applicable_ids:
        catalog = _CATALOG_BY_ID[metric_id]
        metric = observed.get(metric_id)
        score = _score(metric)
        threshold = float(getattr(metric, "threshold", catalog["threshold"])) if metric is not None else float(catalog["threshold"])
        observed_error = getattr(metric, "error", None) if metric is not None else None
        error = (
            _structured_error(observed_error, default_category="evaluation")
            if observed_error is not None
            else (dict(missing_error) if metric is None and missing_error is not None else None)
        )
        passed = error is None and score >= threshold
        normalized.append(
            {
                "id": metric_id,
                "label": catalog["label"],
                "score": score,
                "max_score": 1,
                "threshold": threshold,
                "passed": passed,
                "evaluation_model": str(
                    getattr(metric, "evaluation_model", None) or "not_evaluated"
                ),
                "reason": str(getattr(metric, "reason", fallback_reason) or fallback_reason),
                "error": error,
                "evidence_scope": "turns" if metric_id in {"status_correctness", "tool_correctness"} and len(results) > 1 else ("conversation" if len(results) > 1 else "case"),
                "turn_components": _turn_components(metric_id, results, expected_turns),
            }
        )
    return normalized


def _metric_id(metric: Any) -> str | None:
    name = str(getattr(metric, "__name__", "")).strip().lower()
    if name in _NAME_TO_ID:
        return _NAME_TO_ID[name]
    if "geval" in name or "g-eval" in name or "回复质量" in name:
        return "geval"
    return None


def _score(metric: Any | None) -> float:
    if metric is None:
        return 0.0
    if getattr(metric, "error", None) is not None:
        return 0.0
    value = getattr(metric, "score", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _structured_error(
    error: Any | None, *, default_category: str
) -> dict[str, str] | None:
    """把内部异常压缩为可读、可分类且不泄露堆栈的报告错误。"""
    if error is None:
        return None
    error_type = type(error).__name__ if isinstance(error, BaseException) else "EvaluationError"
    raw = str(error)
    timeout = _is_timeout_error(error, raw)
    if timeout:
        return {
            "type": error_type,
            "category": "timeout",
            "message": "任务超时：执行超过时限，评估未能完成。",
        }
    first_line = next((line.strip() for line in raw.splitlines() if line.strip()), "执行失败。")
    if len(first_line) > 300:
        first_line = first_line[:297] + "..."
    return {
        "type": error_type,
        "category": default_category,
        "message": first_line,
    }


def _is_timeout_error(error: Any, raw: str) -> bool:
    if isinstance(error, TimeoutError):
        return True
    fingerprint = f"{type(error).__name__} {raw}".lower()
    return any(token in fingerprint for token in ("timeout", "timed out", "超时"))


def _turn_components(
    metric_id: str,
    results: list[AgentResult],
    expected_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if metric_id not in {"status_correctness", "tool_correctness"} or len(results) <= 1:
        return []
    components = []
    for index, (result, expected) in enumerate(zip(results, expected_turns), start=1):
        if metric_id == "status_correctness":
            actual = result.status
            wanted = expected.get("status")
            label = f"status：expected={wanted}, actual={actual}"
        else:
            actual = [step.tool_name for step in result.trace]
            wanted = expected.get("tools")
            label = f"工具路径：expected={wanted}, actual={actual}"
        matches = actual == wanted
        components.append(
            {
                "turn_index": index,
                "score": 1.0 if matches else 0.0,
                "passed": matches,
                "reason": label,
            }
        )
    return components


def _build_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "total": len(cases),
        "passed": sum(case["status"] == "passed" for case in cases),
        "failed": sum(case["status"] == "failed" for case in cases),
        "error": sum(case["status"] == "error" for case in cases),
        "single_turn": sum(case["kind"] == "single_turn" for case in cases),
        "multi_turn": sum(case["kind"] == "multi_turn" for case in cases),
    }
    metrics = []
    for catalog in METRIC_CATALOG:
        items = [
            metric
            for case in cases
            for metric in case["metrics"]
            if metric["id"] == catalog["id"]
        ]
        applicable = len(items)
        metrics.append(
            {
                "id": catalog["id"],
                "label": catalog["label"],
                "score": round(sum(item["score"] for item in items), 4),
                "max_score": applicable,
                "threshold": catalog["threshold"],
                "applicable_cases": applicable,
                "passed_cases": sum(item["passed"] for item in items),
                "failed_cases": sum(not item["passed"] for item in items),
                "available": applicable > 0,
            }
        )
    available_metrics = [metric for metric in metrics if metric["available"]]
    earned = round(sum(metric["score"] for metric in available_metrics), 4)
    max_points = sum(metric["max_score"] for metric in available_metrics)
    return {
        "case_counts": counts,
        "composite": {
            "earned_points": earned,
            "max_points": max_points,
            "percentage": round(earned / max_points * 100, 2),
            "gap": round(max_points - earned, 4),
            "method": "累计所有适用 Case 的原始得分；综合得分为六个维度累计得分之和",
        },
        "metrics": metrics,
    }


def _unavailable_usage(source: str) -> dict[str, Any]:
    return {
        "source": source,
        "capture_status": "unavailable",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "note": f"{source} 未提供 token usage。",
    }


def _aggregate_usage(usages: Iterable[dict[str, Any]], *, source: str) -> dict[str, Any]:
    values = list(usages)
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    totals = {
        field: sum(int(item[field]) for item in values if item.get(field) is not None)
        if any(item.get(field) is not None for item in values)
        else None
        for field in fields
    }
    captured = bool(values) and all(
        item.get("capture_status") == "captured" and item.get("total_tokens") is not None
        for item in values
    )
    available = any(item.get("total_tokens") is not None for item in values)
    status = "captured" if captured else ("partial" if available else "unavailable")
    return {
        "source": source,
        "capture_status": status,
        **totals,
        "note": None if status == "captured" else ("部分调用缺少 token usage。" if available else f"{source} 未提供 token usage。"),
    }


def _combine_usage(
    first: dict[str, Any], second: dict[str, Any], *, source: str
) -> dict[str, Any]:
    combined = _aggregate_usage([first, second], source=source)
    combined["note"] = (
        "Target 与 Judge token usage 均完整采集。"
        if combined["capture_status"] == "captured"
        else "仅汇总接口实际返回的 token；缺失值不估算、不补零。"
    )
    return combined
