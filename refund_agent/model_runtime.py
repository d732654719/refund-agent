"""DeepSeek LLM connection settings and client construction."""

from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI


DEFAULT_TARGET_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_REQUEST_TIMEOUT_SECONDS = 60


def load_deepseek_api_key() -> str:
    """Load the API key from the environment or an explicitly configured file."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        return api_key

    configured_path = os.getenv("REFUND_AGENT_AUTH_FILE")
    auth_path = (
        Path(configured_path)
        if configured_path
        else Path(__file__).resolve().parents[2] / "auth.json"
    )
    if auth_path.exists():
        data = json.loads(auth_path.read_text(encoding="utf-8"))
        api_key = data.get("deepseek", {}).get("api_key")
        if api_key:
            return str(api_key)

    raise RuntimeError(
        "DeepSeek API Key is missing. Set DEEPSEEK_API_KEY or configure "
        "deepseek.api_key in the workspace auth.json file."
    )


def create_deepseek_client() -> OpenAI:
    """Build an OpenAI-compatible client using the configured DeepSeek key."""
    return OpenAI(
        api_key=load_deepseek_api_key(),
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
        timeout=MODEL_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )
