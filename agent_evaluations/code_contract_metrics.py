from __future__ import annotations

from deepeval.metrics import BaseConversationalMetric, BaseMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase

from refund_agent.refund_domain_models import AgentResult
from refund_agent.agent_status_catalog import STATUS_DEFINITIONS


def _describe_status(status: str) -> str:
    definition = STATUS_DEFINITIONS.get(status)
    if definition is None:
        return f"{status}（未在状态目录登记）"
    return (
        f"{status}（含义：{definition.meaning}"
        f"产生条件：{definition.generated_when}"
        f"来源：{definition.source}）"
    )


class StatusCorrectnessMetric(BaseMetric):
    """Deterministically compare the Agent status and record it in DeepEval."""

    def __init__(self, *, actual_status: str, expected_status: str) -> None:
        self.actual_status = actual_status
        self.expected_status = expected_status
        self.threshold = 1.0
        self.strict_mode = True
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_model = "deterministic-code"
        self.evaluation_cost = 0

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        del test_case, args, kwargs
        matches = self.actual_status == self.expected_status
        self.score = 1.0 if matches else 0.0
        self.reason = (
            f"状态符合预期：{_describe_status(self.actual_status)}"
            if matches
            else (
                "状态不符合预期。\n"
                f"expected={_describe_status(self.expected_status)}\n"
                f"actual={_describe_status(self.actual_status)}"
            )
        )
        self.success = self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    @property
    def __name__(self) -> str:
        return "Status Correctness"


class ToolPathCorrectnessMetric(BaseMetric):
    """Compare tool count, names, and ordering without an LLM Judge."""

    def __init__(self) -> None:
        self.threshold = 1.0
        self.strict_mode = True
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_model = "deterministic-code"
        self.evaluation_cost = 0

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        del args, kwargs
        actual_tools = [tool.name for tool in test_case.tools_called or []]
        expected_tools = [tool.name for tool in test_case.expected_tools or []]
        matches = actual_tools == expected_tools
        self.score = 1.0 if matches else 0.0
        self.reason = (
            f"工具路径符合预期：{actual_tools}。"
            if matches
            else (
                "工具路径不符合预期："
                f"expected={expected_tools}, actual={actual_tools}。"
            )
        )
        self.success = self.is_successful()
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    @property
    def __name__(self) -> str:
        return "Tool Correctness"


class AgentExecutionErrorMetric(BaseMetric):
    """Record a Target Agent exception as a failed DeepEval test case."""

    def __init__(self, error: BaseException) -> None:
        self.error_detail = f"{type(error).__name__}: {error}"
        self.threshold = 1.0
        self.strict_mode = True
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_model = "deterministic-code"
        self.evaluation_cost = 0

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        del test_case, args, kwargs
        self.score = 0.0
        self.reason = f"Target Agent 执行异常：{self.error_detail}"
        self.success = False
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    @property
    def __name__(self) -> str:
        return "Agent Execution"


class ConversationExecutionErrorMetric(BaseConversationalMetric):
    """Record a Target Agent exception during a conversation."""

    def __init__(self, error: BaseException) -> None:
        self.error_detail = f"{type(error).__name__}: {error}"
        self.threshold = 1.0
        self.strict_mode = True
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_model = "deterministic-code"
        self.evaluation_cost = 0

    def measure(
        self, test_case: ConversationalTestCase, *args, **kwargs
    ) -> float:
        del test_case, args, kwargs
        self.score = 0.0
        self.reason = f"Target Agent 执行异常：{self.error_detail}"
        self.success = False
        return self.score

    async def a_measure(
        self, test_case: ConversationalTestCase, *args, **kwargs
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    @property
    def __name__(self) -> str:
        return "Agent Execution"


class TaskTimeoutMetric(BaseMetric):
    """Record a single-turn Target or Judge timeout as an error metric."""

    def __init__(self, error: BaseException) -> None:
        self.error_detail = str(error)
        self.threshold = 1.0
        self.strict_mode = True
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_model = "evaluation-timeout-boundary"
        self.evaluation_cost = 0
        self.error = "任务超时"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        del test_case, args, kwargs
        self.score = 0.0
        self.reason = self.error_detail
        self.success = False
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    @property
    def __name__(self) -> str:
        return "Task Timeout"


class ConversationTaskTimeoutMetric(BaseConversationalMetric):
    """Record a multi-turn Target or Judge timeout as an error metric."""

    def __init__(self, error: BaseException) -> None:
        self.error_detail = str(error)
        self.threshold = 1.0
        self.strict_mode = True
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_model = "evaluation-timeout-boundary"
        self.evaluation_cost = 0
        self.error = "任务超时"

    def measure(
        self, test_case: ConversationalTestCase, *args, **kwargs
    ) -> float:
        del test_case, args, kwargs
        self.score = 0.0
        self.reason = self.error_detail
        self.success = False
        return self.score

    async def a_measure(
        self, test_case: ConversationalTestCase, *args, **kwargs
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    @property
    def __name__(self) -> str:
        return "Task Timeout"


class ConversationDataIntegrityMetric(BaseConversationalMetric):
    """Expose an invalid multi-turn Golden without calling either LLM."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        self.threshold = 1.0
        self.strict_mode = True
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_model = "deterministic-code"
        self.evaluation_cost = 0

    def measure(
        self, test_case: ConversationalTestCase, *args, **kwargs
    ) -> float:
        del test_case, args, kwargs
        self.score = 0.0
        self.reason = "多轮 Golden 数据不完整：\n" + "\n".join(
            f"- {error}" for error in self.errors
        )
        self.success = False
        return self.score

    async def a_measure(
        self, test_case: ConversationalTestCase, *args, **kwargs
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    @property
    def __name__(self) -> str:
        return "Evaluation Data Integrity"


class ConversationStatusCorrectnessMetric(BaseConversationalMetric):
    """Compare every observed status with its explicit per-turn contract."""

    def __init__(
        self,
        *,
        actual_results: list[AgentResult],
        expected_turns: list[dict],
    ) -> None:
        self.actual_results = actual_results
        self.expected_turns = expected_turns
        self.threshold = 1.0
        self.strict_mode = True
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_model = "deterministic-code"
        self.evaluation_cost = 0

    def measure(
        self,
        test_case: ConversationalTestCase,
        *args,
        **kwargs,
    ) -> float:
        del test_case, args, kwargs
        mismatches: list[str] = []
        if len(self.actual_results) != len(self.expected_turns):
            mismatches.append(
                "轮数不符合预期："
                f"expected={len(self.expected_turns)}, "
                f"actual={len(self.actual_results)}。"
            )

        for index, (actual, expected) in enumerate(
            zip(self.actual_results, self.expected_turns),
            start=1,
        ):
            expected_status = expected["status"]
            if actual.status != expected_status:
                mismatches.append(
                    f"第 {index} 轮 status 不符合预期："
                    f"expected={_describe_status(expected_status)}, "
                    f"actual={_describe_status(actual.status)}。"
                )

        self.score = 0.0 if mismatches else 1.0
        self.reason = (
            "每轮 status 均符合逐轮 Golden 契约。"
            if not mismatches
            else "\n".join(mismatches)
        )
        self.success = self.is_successful()
        return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        *args,
        **kwargs,
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    @property
    def __name__(self) -> str:
        return "Status Correctness"


class ConversationToolCorrectnessMetric(BaseConversationalMetric):
    """Compare every observed tool path with its explicit per-turn contract."""

    def __init__(
        self,
        *,
        actual_results: list[AgentResult],
        expected_turns: list[dict],
    ) -> None:
        self.actual_results = actual_results
        self.expected_turns = expected_turns
        self.threshold = 1.0
        self.strict_mode = True
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_model = "deterministic-code"
        self.evaluation_cost = 0

    def measure(
        self,
        test_case: ConversationalTestCase,
        *args,
        **kwargs,
    ) -> float:
        del test_case, args, kwargs
        mismatches: list[str] = []
        if len(self.actual_results) != len(self.expected_turns):
            mismatches.append(
                "轮数不符合预期："
                f"expected={len(self.expected_turns)}, "
                f"actual={len(self.actual_results)}。"
            )

        for index, (actual, expected) in enumerate(
            zip(self.actual_results, self.expected_turns),
            start=1,
        ):
            expected_tools = expected["tools"]
            actual_tools = [step.tool_name for step in actual.trace]
            if actual_tools != expected_tools:
                mismatches.append(
                    f"第 {index} 轮工具路径不符合预期："
                    f"expected={expected_tools}, actual={actual_tools}。"
                )

        self.score = 0.0 if mismatches else 1.0
        self.reason = (
            "每轮工具名称、数量和顺序均符合逐轮 Golden 契约。"
            if not mismatches
            else "\n".join(mismatches)
        )
        self.success = self.is_successful()
        return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        *args,
        **kwargs,
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    @property
    def __name__(self) -> str:
        return "Tool Correctness"


class ConversationTurnContractMetric(BaseConversationalMetric):
    """Deterministically validate per-turn status, tools and response contract."""

    def __init__(
        self,
        *,
        actual_results: list[AgentResult],
        expected_turns: list[dict],
    ) -> None:
        self.actual_results = actual_results
        self.expected_turns = expected_turns
        self.threshold = 1.0
        self.strict_mode = True
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_model = "deterministic-code"
        self.evaluation_cost = 0

    def measure(
        self,
        test_case: ConversationalTestCase,
        *args,
        **kwargs,
    ) -> float:
        del test_case, args, kwargs
        mismatches: list[str] = []
        if len(self.actual_results) != len(self.expected_turns):
            mismatches.append(
                "轮数不符合预期："
                f"expected={len(self.expected_turns)}, "
                f"actual={len(self.actual_results)}。"
            )

        for index, (actual, expected) in enumerate(
            zip(self.actual_results, self.expected_turns),
            start=1,
        ):
            expected_status = expected.get("status")
            if expected_status is not None and actual.status != expected_status:
                mismatches.append(
                    f"第 {index} 轮 status 不符合预期："
                    f"expected={expected_status}, actual={actual.status}。"
                )

            expected_tools = expected.get("tools")
            actual_tools = [step.tool_name for step in actual.trace]
            if expected_tools is not None and actual_tools != expected_tools:
                mismatches.append(
                    f"第 {index} 轮工具路径不符合预期："
                    f"expected={expected_tools}, actual={actual_tools}。"
                )

            for phrase in expected.get("response_contains", []):
                if phrase not in actual.response:
                    mismatches.append(
                        f"第 {index} 轮回复缺少预期内容：{phrase!r}。"
                    )

        self.score = 0.0 if mismatches else 1.0
        self.reason = (
            "每轮 status、工具路径和回复契约均符合预期。"
            if not mismatches
            else "\n".join(mismatches)
        )
        self.success = self.is_successful()
        return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        *args,
        **kwargs,
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    @property
    def __name__(self) -> str:
        return "Conversation Turn Contract"
