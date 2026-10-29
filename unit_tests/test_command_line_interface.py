from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from refund_agent.command_line_interface import print_result
from refund_agent.refund_domain_models import AgentResult


class CliOutputTests(unittest.TestCase):
    def test_final_model_response_is_not_printed_twice(self) -> None:
        response = "退款已成功创建。"
        result = AgentResult(
            input="退款 A1001",
            user_id="user-001",
            response=response,
            status="completed",
            trace=[],
            model_turns=2,
            conversation_turn=1,
            events=[
                {
                    "event": "model_response",
                    "turn": 2,
                    "content": response,
                    "tool_calls": [],
                }
            ],
        )
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            print_result(result, show_trace=True)

        output = stdout.getvalue()
        self.assertEqual(output.count(response), 1)
        self.assertIn("[与上方 Agent 回答相同]", output)


if __name__ == "__main__":
    unittest.main()
