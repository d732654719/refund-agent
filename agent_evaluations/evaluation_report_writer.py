from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

from deepeval_report_template import ReportPaths, render_report


REPORT_PATH = Path(__file__).with_name("latest_llm_evaluation_report.txt")
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[1] / "report"
_LOCK = Lock()
_FAILED_BLOCKS: list[str] = []


def write_evaluation_report(
    payload: dict[str, Any],
    *,
    report_dir: Path | None = None,
    generated_at: datetime | None = None,
) -> ReportPaths:
    """把评测数据交给共享模板包；本分支只负责数据采集和文本日志。"""
    data = dict(payload)
    # 评测层仍使用退款指标名，由源头包统一迁移到 Report Schema 2.0。
    data.setdefault("schema_version", "1.1.0")
    return render_report(
        data,
        output_dir=report_dir if report_dir is not None else DEFAULT_REPORT_DIR,
        generated_at=generated_at,
    )


def reset_full_report() -> None:
    """Start one UTF-8 report for the current pytest evaluation session."""
    header = (
        "Refund Agent DeepEval 完整报告\n"
        f"生成时间：{datetime.now().astimezone().isoformat()}\n"
        "说明：本文件不截断 metric reason。\n"
        + "=" * 80
        + "\n"
    )
    with _LOCK:
        _FAILED_BLOCKS.clear()
        REPORT_PATH.write_text(header, encoding="utf-8")


def record_case_metrics(
    *,
    source: str,
    case_name: str,
    metrics: Iterable[Any],
    target_output: str,
) -> None:
    """Persist complete metric reasons and retain failures for terminal output."""
    lines = [
        f"测试集：{source}",
        f"用例：{case_name}",
        f"Target 输出：{target_output}",
    ]
    failed_lines: list[str] = []

    for metric in metrics:
        error = getattr(metric, "error", None)
        success = getattr(metric, "success", None)
        if error is not None:
            status = "ERROR"
        elif success is True:
            status = "PASSED"
        elif success is False:
            status = "FAILED"
        else:
            status = "UNKNOWN"

        metric_lines = [
            f"Metric：{metric.__name__}",
            f"Status：{status}",
            f"score：{getattr(metric, 'score', None)}",
            f"threshold：{getattr(metric, 'threshold', None)}",
            f"evaluation model：{getattr(metric, 'evaluation_model', None)}",
            f"reason：{getattr(metric, 'reason', None)}",
            f"error：{error}",
        ]
        lines.extend(metric_lines)
        lines.append("-" * 80)
        if status in {"FAILED", "ERROR"}:
            failed_lines.extend(metric_lines)
            failed_lines.append("-" * 80)

    block = "\n".join(lines) + "\n"
    with _LOCK:
        with REPORT_PATH.open("a", encoding="utf-8") as report:
            report.write(block)
        if failed_lines:
            _FAILED_BLOCKS.append(
                "\n".join(
                    [
                        f"测试集：{source}",
                        f"用例：{case_name}",
                        *failed_lines,
                    ]
                )
            )


def failed_metric_blocks() -> list[str]:
    with _LOCK:
        return list(_FAILED_BLOCKS)
