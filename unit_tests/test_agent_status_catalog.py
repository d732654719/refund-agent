from __future__ import annotations

import json
import unittest
from pathlib import Path

from refund_agent.refund_domain_models import AgentResult
from refund_agent.agent_status_catalog import (
    ALL_AGENT_STATUSES,
    COMPLETED,
    NO_TOOL_FINAL_STATUSES,
    PROMPT_INJECTION_BLOCKED,
    NEEDS_INPUT,
    RESPONDED,
    STATUS_DEFINITIONS,
    TASK_TIMEOUT,
    agent_status_from_tool_code,
)


class AgentStatusContractTests(unittest.TestCase):
    def test_task_timeout_is_registered_as_runtime_failure(self) -> None:
        self.assertEqual(
            STATUS_DEFINITIONS[TASK_TIMEOUT].category,
            "runtime_failure",
        )

    def test_every_golden_expected_status_is_registered(self) -> None:
        goldens_path = (
            Path(__file__).parents[1]
            / "agent_evaluations"
            / "single_turn_golden_cases.json"
        )
        goldens = json.loads(goldens_path.read_text(encoding="utf-8"))

        unknown = {
            case["expected_status"]
            for case in goldens
            if case["expected_status"] not in ALL_AGENT_STATUSES
        }

        self.assertEqual(unknown, set())

    def test_every_registered_status_has_human_description(self) -> None:
        self.assertEqual(set(STATUS_DEFINITIONS), set(ALL_AGENT_STATUSES))
        for definition in STATUS_DEFINITIONS.values():
            self.assertTrue(definition.meaning)
            self.assertTrue(definition.generated_when)
            self.assertTrue(definition.source)

    def test_tool_codes_map_to_public_agent_statuses(self) -> None:
        self.assertEqual(agent_status_from_tool_code("REFUND_CREATED"), COMPLETED)
        self.assertEqual(
            agent_status_from_tool_code("REFUND_WINDOW_EXPIRED"),
            "REFUND_WINDOW_EXPIRED",
        )
        self.assertEqual(agent_status_from_tool_code("ORDER_FOUND"), RESPONDED)

    def test_prompt_injection_is_a_registered_no_tool_security_status(self) -> None:
        self.assertIn(PROMPT_INJECTION_BLOCKED, ALL_AGENT_STATUSES)
        self.assertIn(PROMPT_INJECTION_BLOCKED, NO_TOOL_FINAL_STATUSES)
        self.assertEqual(
            STATUS_DEFINITIONS[PROMPT_INJECTION_BLOCKED].category,
            "security_rejection",
        )
        self.assertNotIn(NEEDS_INPUT, NO_TOOL_FINAL_STATUSES)

    def test_agent_result_rejects_unregistered_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "未登记的 Agent status"):
            AgentResult(
                input="test",
                user_id="user-001",
                response="test",
                status="new_status_without_registration",
                trace=[],
            )


if __name__ == "__main__":
    unittest.main()
