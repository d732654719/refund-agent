from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_evaluations.evaluation_report_writer import write_evaluation_report
from deepeval_report_template.report_models import ReportValidationError


METRICS = (
    ("status_correctness", "Status Correctness", 1.0),
    ("task_completion", "Task Completion", 0.7),
    ("refund_step_efficiency", "Refund Step Efficiency", 0.8),
    ("tool_correctness", "Tool Correctness", 1.0),
    ("geval", "G-Eval", 0.8),
    ("knowledge_retention", "Knowledge Retention", 0.8),
)


def _usage(source: str = "target") -> dict[str, object]:
    return {
        "source": source,
        "capture_status": "unavailable",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "note": "服务端未返回 usage，保持显式不可用状态",
    }


def _case_metric(metric_id: str, label: str, threshold: float) -> dict[str, object]:
    return {
        "id": metric_id,
        "label": label,
        "score": 1.0,
        "max_score": 1,
        "threshold": threshold,
        "passed": True,
        "evaluation_model": "deterministic-code",
        "reason": "达到阈值。",
        "error": None,
        "evidence_scope": "case",
        "turn_components": [],
    }


def _turn(index: int, user_input: str) -> dict[str, object]:
    return {
        "turn_index": index,
        "user_input": user_input,
        "agent_status": "completed",
        "raw_model_outputs": [],
        "agent_final_output": "退款处理完成。",
        "model_turns": 1,
        "duration_ms": 120.5,
        "usage": _usage(),
        "trace": [],
    }


def _payload(*, include_multi_turn: bool = False) -> dict[str, object]:
    catalog = [
        {
            "id": metric_id,
            "label": label,
            "max_score": 1,
            "threshold": threshold,
            "applies_to": ["multi_turn"]
            if metric_id == "knowledge_retention"
            else ["single_turn", "multi_turn"],
            "evaluation_type": "llm_judge"
            if metric_id in {"task_completion", "refund_step_efficiency", "geval", "knowledge_retention"}
            else "deterministic",
        }
        for metric_id, label, threshold in METRICS
    ]
    base_metrics = [_case_metric(*metric) for metric in METRICS[:-1]]
    cases: list[dict[str, object]] = [
        {
            "id": "refund_success",
            "name": "正常全额退款",
            "kind": "single_turn",
            "task": "订单正常，可以退款",
            "user_id": "user-001",
            "status": "passed",
            "passed": True,
            "duration_ms": 120.5,
            "usage": _usage(),
            "error": None,
            "metrics": deepcopy(base_metrics),
            "turns": [_turn(1, "请退款订单 A1001")],
        }
    ]
    if include_multi_turn:
        cases.append(
            {
                "id": "context_001",
                "name": "多轮上下文退款",
                "kind": "multi_turn",
                "task": "保持订单上下文",
                "user_id": "user-002",
                "status": "passed",
                "passed": True,
                "duration_ms": 241.0,
                "usage": _usage(),
                "error": None,
                "metrics": [
                    *deepcopy(base_metrics),
                    _case_metric(*METRICS[-1]),
                ],
                "turns": [_turn(1, "我要退款 A1002"), _turn(2, "继续")],
            }
        )

    total = len(cases)
    summary_metrics = []
    for metric_id, label, threshold in METRICS:
        applicable = total if metric_id != "knowledge_retention" else int(include_multi_turn)
        summary_metrics.append(
            {
                "id": metric_id,
                "label": label,
                "score": float(applicable),
                "max_score": applicable,
                "threshold": threshold,
                "applicable_cases": applicable,
                "passed_cases": applicable,
                "failed_cases": 0,
                "available": applicable > 0,
            }
        )
    max_points = total * 5 + int(include_multi_turn)
    return {
        "schema_version": "1.1.0",
        "run": {
            "status": "completed",
            "started_at": "2026-08-25T10:00:00+08:00",
            "finished_at": "2026-08-25T10:00:01+08:00",
            "duration_ms": 361.5 if include_multi_turn else 120.5,
            "target_model": "deepseek-v4-flash",
            "judge_model": "deepseek-v4-pro",
            "dataset_sources": ["single_turn_golden_cases.json"],
            "tokens": {
                "capture_status": "unavailable",
                "target": None,
                "judge": None,
                "total": None,
                "note": "未补零",
            },
        },
        "metric_catalog": catalog,
        "summary": {
            "case_counts": {
                "total": total,
                "passed": total,
                "failed": 0,
                "error": 0,
                "single_turn": 1,
                "multi_turn": int(include_multi_turn),
            },
            "composite": {
                "earned_points": max_points,
                "max_points": max_points,
                "percentage": 100.0,
                "gap": 0.0,
                "method": "累计所有适用 Case 的原始得分；综合得分为六个维度累计得分之和",
            },
            "metrics": summary_metrics,
        },
        "cases": cases,
    }


def test_writes_readable_json_and_safe_offline_html(tmp_path: Path) -> None:
    payload = _payload(include_multi_turn=True)
    payload["cases"][0]["turns"][0]["user_input"] = "恶意文本 </script><script>alert('x')</script> 中文"

    paths = write_evaluation_report(
        payload,
        report_dir=tmp_path,
        generated_at=datetime(2026, 8, 25, 12, 34, 56, tzinfo=timezone.utc),
    )

    assert paths.report_id == "2026-08-25-001"
    assert paths.json_path.name == "2026-08-25-001.json"
    assert paths.html_path.name == "2026-08-25-001.html"
    json_text = paths.json_path.read_text(encoding="utf-8")
    document = json.loads(json_text)
    assert document["report_id"] == paths.report_id
    assert document["cases"][0]["turns"][0]["user_input"].endswith("中文")
    assert "\\u4e2d" not in json_text
    assert '\n  "schema_version"' in json_text
    assert json_text.endswith("\n")

    html = paths.html_path.read_text(encoding="utf-8")
    assert "__REPORT_JSON__" not in html
    assert "UNUSED_PROTOTYPE_REPORT" not in html
    assert "refund_success" in html
    assert "</script><script>alert" not in html
    assert "\\u003c/script\\u003e" in html
    assert "../report-assets/fonts/" in html
    assert '<script src=' not in html
    assert 'href="http' not in html
    assert 'src="http' not in html
    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".html", ".json"]


def test_report_template_keeps_text_clear_of_visual_boundaries_and_closes_dialog_outside(
    tmp_path: Path,
) -> None:
    paths = write_evaluation_report(
        _payload(include_multi_turn=True),
        report_dir=tmp_path,
        generated_at=datetime(2026, 8, 25, 12, 34, 56, tzinfo=timezone.utc),
    )
    html = paths.html_path.read_text(encoding="utf-8")

    assert "累计阈值" not in html
    assert "pytest 的单一结论" not in html
    assert "Agent 整体表现" in html
    assert "AGENT评测报告" in html
    assert "DeepEval架构" in html
    assert "brand-sep" in html and "brand-sub" in html
    assert "Agent详细表现" in html
    assert "选择指标可查看当前分、满分、阈值、是否通过和完整评分理由。" not in html
    assert ".case-overview-card{background:var(--panel);border-radius:24px" in html
    assert "CaseId · ${c.id}" in html and "任务意图：${c.task}" in html
    assert "transform:translate(-6px,-2px);text-align:left" in html
    assert ".turns{margin-top:0;background:var(--panel);border-radius:24px" in html
    assert ".case-detail{min-width:0;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr)" in html
    assert "单轮和多轮共用一套轮次数据结构" not in html
    assert "max-height:900px;overflow-y:auto" in html
    assert 'data-filter="failed"]{color:var(--txt)}' in html
    assert "background:var(--panel);border-radius:24px" in html
    assert "for(let i=0;i<=20;i++)" in html
    assert ".case-list::-webkit-scrollbar-button" in html and "display:none!important" in html
    assert "本报告由本次评估保存的运行数据生成，可离线查看。" not in html
    assert "数据来源：运行评估程序生成的评测数据" in html
    assert "模型调用明细" not in html
    assert "function renderRuntimeEvents(turn)" in html
    assert "运行事件时间线" not in html
    assert "运行事件 · ${events.length} 条" in html
    assert "模型原始输出" in html
    assert "width:1600px;max-width:100%" in html
    assert 'id="report-summary"' in html
    assert ">成功/未通过<" in html
    assert "深绿 = 成功　黄 = 失败" in html
    assert "总Token消耗" in html
    assert "#D4A017" in html
    assert "grid-template-columns:minmax(0,1fr) 430px" in html
    assert "max-height:328px;overflow-y:auto" in html
    assert html.index('id="case-donut"') < html.index('class="dist-actions"') < html.index('class="intro"') < html.index('id="failed-links"')
    assert html.index('id="summary-metrics"') < html.index('id="run-kpis"')
    assert 'class="primary-column"' in html

    score_zones = re.search(
        r"x0=(\d+),axisEnd=(\d+),scoreColumnStart=(\d+),scoreX=(\d+)",
        html,
    )
    assert score_zones is not None
    _, axis_end, score_start, score_end = map(int, score_zones.groups())
    assert axis_end < score_start < score_end
    assert "scoreColumnWidth/Math.max(1,scoreText.length*.62)" in html

    safe_space = re.search(
        r"--kpi-safe-inline:(\d+)px;--kpi-safe-block:(\d+)px",
        html,
    )
    assert safe_space is not None
    inline_space, block_space = map(int, safe_space.groups())
    assert inline_space >= 24
    assert block_space >= 28
    assert "padding:var(--kpi-safe-block) var(--kpi-safe-inline)" in html
    assert ".kpi .v{" in html and "white-space:nowrap" in html

    assert "document.addEventListener('pointerdown'" in html
    assert "dialog.open&&!dialog.contains(e.target)" in html


def test_report_id_uses_highest_existing_sequence_and_resets_each_day(
    tmp_path: Path,
) -> None:
    (tmp_path / "2026-08-25-001.json").write_text("{}", encoding="utf-8")
    (tmp_path / "2026-08-25-003.html").write_text("", encoding="utf-8")

    fourth = write_evaluation_report(
        _payload(),
        report_dir=tmp_path,
        generated_at=datetime(2026, 8, 25, 1, tzinfo=timezone.utc),
    )
    fifth = write_evaluation_report(
        _payload(),
        report_dir=tmp_path,
        generated_at=datetime(2026, 8, 25, 2, tzinfo=timezone.utc),
    )
    next_day = write_evaluation_report(
        _payload(),
        report_dir=tmp_path,
        generated_at=datetime(2026, 8, 26, 1, tzinfo=timezone.utc),
    )

    assert fourth.report_id == "2026-08-25-004"
    assert fifth.report_id == "2026-08-25-005"
    assert next_day.report_id == "2026-08-26-001"


def test_rejects_multi_turn_case_without_knowledge_retention(tmp_path: Path) -> None:
    payload = _payload(include_multi_turn=True)
    payload["cases"][1]["metrics"] = payload["cases"][1]["metrics"][:-1]

    with pytest.raises(ReportValidationError, match="六个指标"):
        write_evaluation_report(
            payload,
            report_dir=tmp_path,
            generated_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
        )

    assert list(tmp_path.iterdir()) == []


def test_case_pass_requires_every_applicable_metric_to_reach_threshold(
    tmp_path: Path,
) -> None:
    payload = _payload()
    metric = payload["cases"][0]["metrics"][1]
    metric["score"] = 0.69
    metric["passed"] = False

    with pytest.raises(ReportValidationError, match="所有适用指标"):
        write_evaluation_report(
            payload,
            report_dir=tmp_path,
            generated_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
        )


def test_threshold_boundary_is_a_pass(tmp_path: Path) -> None:
    payload = _payload()
    metric = payload["cases"][0]["metrics"][1]
    metric["score"] = metric["threshold"]
    summary_metric = payload["summary"]["metrics"][1]
    summary_metric["score"] = metric["threshold"]
    payload["summary"]["composite"].update(
        earned_points=4.7,
        percentage=94.0,
        gap=0.3,
    )

    paths = write_evaluation_report(
        payload,
        report_dir=tmp_path,
        generated_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
    )

    assert paths.json_path.exists()
    assert paths.html_path.exists()


def test_generated_at_must_be_timezone_aware(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="时区"):
        write_evaluation_report(
            _payload(),
            report_dir=tmp_path,
            generated_at=datetime(2026, 8, 25),
        )
