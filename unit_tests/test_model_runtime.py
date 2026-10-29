from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from refund_agent.model_runtime import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_TARGET_MODEL,
    MODEL_REQUEST_TIMEOUT_SECONDS,
    create_deepseek_client,
)


class LLMConfigurationTests(unittest.TestCase):
    def test_target_model_defaults_to_flash(self) -> None:
        self.assertEqual(DEFAULT_TARGET_MODEL, "deepseek-v4-flash")

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "environment-key"}, clear=True)
    def test_client_prefers_environment_key(self) -> None:
        with patch("refund_agent.model_runtime.OpenAI") as openai:
            create_deepseek_client()

        self.assertEqual(openai.call_args.kwargs["api_key"], "environment-key")

    @patch("refund_agent.model_runtime.OpenAI")
    @patch("refund_agent.model_runtime.load_deepseek_api_key", return_value="test-key")
    def test_client_uses_configured_key_and_default_base_url(
        self, load_api_key, openai
    ) -> None:
        create_deepseek_client()

        load_api_key.assert_called_once_with()
        openai.assert_called_once_with(
            api_key="test-key",
            base_url=DEFAULT_DEEPSEEK_BASE_URL,
            timeout=MODEL_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )

    def test_model_request_timeout_is_fixed_at_sixty_seconds(self) -> None:
        self.assertEqual(MODEL_REQUEST_TIMEOUT_SECONDS, 60)
