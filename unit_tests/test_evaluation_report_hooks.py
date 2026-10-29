from pathlib import Path
import os

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    "arguments, starts_report",
    [
        (["-m", "llm_eval"], True),
        (["-m", "no_llm_eval"], False),
        (["-m", "not llm_eval"], False),
        (["test_sample.py::test_llm"], True),
        (["--collect-only", "-m", "llm_eval"], False),
    ],
)
def test_report_follows_selected_tests(pytester, monkeypatch, arguments, starts_report):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", str(project) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    source = (project / "agent_evaluations" / "conftest.py").read_text(encoding="utf-8")
    # 隔离文件输出，真实执行项目的收集和收尾钩子，不运行模型。
    source += """
from pathlib import Path
_calls = Path(__file__).with_name("report_calls.txt")
def reset_full_report():
    _calls.write_text("reset\\n", encoding="utf-8")
def start_evaluation_session():
    with _calls.open("a", encoding="utf-8") as out:
        out.write("start\\n")
def finalize_evaluation_session(*, exit_status):
    with _calls.open("a", encoding="utf-8") as out:
        out.write("finish\\n")
"""
    pytester.makeconftest(source)
    pytester.makeini("[pytest]\nmarkers =\n    llm_eval: fake model test\n    no_llm_eval: offline test")
    pytester.makepyfile(test_sample="""
import pytest
@pytest.mark.llm_eval
def test_llm():
    pass
@pytest.mark.no_llm_eval
def test_offline():
    pass
""")
    calls = pytester.path / "report_calls.txt"
    calls.write_text("previous report", encoding="utf-8")
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-q", *arguments)
    assert result.ret == 0
    expected = "reset\nstart\nfinish\n" if starts_report else "previous report"
    assert calls.read_text(encoding="utf-8") == expected
