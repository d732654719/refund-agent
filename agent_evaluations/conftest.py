from __future__ import annotations

import pytest

from agent_evaluations.evaluation_report_writer import (
    REPORT_PATH,
    failed_metric_blocks,
    reset_full_report,
)
from agent_evaluations.evaluation_report_session import (
    finalize_evaluation_session,
    latest_report_paths,
    start_evaluation_session,
)


_REPORT_STARTED = pytest.StashKey[bool]()


def _report_started(config: pytest.Config) -> bool:
    return config.stash.get(_REPORT_STARTED, False)


def pytest_collection_finish(session: pytest.Session) -> None:
    # 此时 -m 的筛选已完成；仅为实际选中的 LLM 测试准备报告。
    if session.config.option.collectonly:
        return
    if any(item.get_closest_marker("llm_eval") is not None for item in session.items):
        reset_full_report()
        start_evaluation_session()
        session.config.stash[_REPORT_STARTED] = True


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _report_started(session.config):
        finalize_evaluation_session(exit_status=exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    del exitstatus
    if not _report_started(config):
        return

    failures = failed_metric_blocks()
    if failures:
        terminalreporter.write_sep("=", "完整失败原因（不截断）")
        for block in failures:
            terminalreporter.write_line(block)

    terminalreporter.write_sep("=", "完整评估报告")
    paths = latest_report_paths()
    if paths is not None:
        terminalreporter.write_line(f"JSON：{paths.json_path.resolve()}")
        terminalreporter.write_line(f"HTML：{paths.html_path.resolve()}")
    terminalreporter.write_line(f"兼容文本：{REPORT_PATH.resolve()}")
