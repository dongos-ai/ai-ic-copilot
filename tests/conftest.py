"""Shared pytest setup: put the project root on sys.path and locate the optional real-EDGAR fixtures."""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Callable

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from copilot import edgar  # noqa: E402

# Optional real-EDGAR fixtures (one folder per ticker with 10k.html, companyfacts.json, ...), downloaded by
# scripts/fetch_fixtures.py. Tests that need them skip when the folder is missing.
FIXTURE_DIR = Path(os.getenv("FIXTURE_DIR", ROOT / "tests" / "fixtures" / "sec_cache"))


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    if not FIXTURE_DIR.is_dir():
        pytest.skip("real EDGAR fixtures not available (run scripts/fetch_fixtures.py or set FIXTURE_DIR)")
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def tenk_text(fixture_dir: Path) -> Callable[[str], str]:
    """tenk_text("AAPL") -> edgar.html_to_text of that fixture's 10-K, parsed once per test session."""
    @functools.lru_cache(maxsize=None)
    def parse(ticker: str) -> str:
        return edgar.html_to_text((fixture_dir / ticker / "10k.html").read_bytes())
    return parse
