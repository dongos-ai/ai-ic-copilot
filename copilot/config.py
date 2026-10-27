"""Settings read from environment variables (a local .env file is loaded if present)."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

APP_NAME = "AI Investment Committee Copilot"

# SEC asks every automated client to identify itself with a name and contact email.
# https://www.sec.gov/os/accessing-edgar-data
SEC_USER_AGENT_ENV = "SEC_USER_AGENT"

# LLM providers: API key env vars (first one found wins) and default models.
API_KEY_ENVS = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
}
DEFAULT_MODELS = {
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-6.1-sol"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
}
ANTHROPIC_EFFORT = os.getenv("ANTHROPIC_EFFORT", "medium")  # low | medium | high | xhigh | max

# Analysis knobs.
FISCAL_YEARS = 5           # years of XBRL history in the snapshot
CHUNK_WORDS = 180          # target words per retrieval chunk


def get_sec_user_agent() -> str:
    """User-Agent for SEC requests, e.g. 'Jane Doe jane@example.com'."""
    return os.getenv(SEC_USER_AGENT_ENV, "").strip()


def get_api_key(provider: str) -> str | None:
    """Return the API key for a provider, or None if no env var is set."""
    for name in API_KEY_ENVS.get(provider, []):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None
