"""Offline tests for copilot.pipeline (end to end on the real EDGAR fixtures) and a headless run of app.py.

No network and no API keys: SEC requests are answered from the fixture folders, Yahoo Finance is stubbed,
and the LLM is a fake client that returns schema-valid JSON (plus deliberately bad claims).
"""
from __future__ import annotations

import functools
import importlib.util
import json
import re
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from copilot import edgar, pipeline
from copilot.edgar import CompanyInfo, Filing, Section, TenK
from copilot.financials import FinancialData
from copilot.llm import LLMError
from copilot.market import MarketData
from copilot.pipeline import Settings, build_documents, run_analysis
from copilot.retrieval import MEMO_QUERIES
from copilot.transcript import MAX_UPLOAD_MB

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
SAMPLE_TRANSCRIPT = ROOT / "tests" / "fixtures" / "sample_transcript.txt"
SETTINGS = Settings(user_agent="AI-IC-Copilot test@example.com")
LLM_SETTINGS = Settings(user_agent="AI-IC-Copilot test@example.com", provider="anthropic")
TAB_NAMES = ["Memo", "Financials", "Red flags", "Evidence", "Committee debate", "Diagnostics"]
STEPS = ["Ticker lookup", "10-K", "Financials", "Market data", "Retrieval", "Red flags", "Evidence pack", "Memo",
         "Total"]

COMMITTEE_HEADINGS = [
    "## 1. Investment thesis", "## 2. Business overview", "## 3. Financial snapshot", "## 4. Bull case",
    "## 5. Bear case", "## 6. Top risks", "## 7. What management said vs. what the numbers show",
    "## 8. Red flags", "## 9. Questions for the committee", "## 10. Vote",
    "## Appendix A: Evidence cited", "## Appendix B: Citation audit",
]
DATA_ONLY_HEADINGS = [
    "## Financial snapshot", "## Red flags (automated)", "## Evidence by memo section",
    "## Questions for the committee", "## Vote", "## Appendix A: Evidence cited",
]

BOGUS_CLAIM = "This claim cites evidence that does not exist."
FAKE_QUOTE = "words that appear in no filing at all"
# Claims with real citations that the citation check must still remove:
FABRICATED = "Revenue grew 45% to $600.0B, the fastest growth in a decade."   # numbers not in the cited item
ONE_SIDED = "Management promised record margins."                            # said-vs-shows row citing only F
UNCITED_QUESTION = "Why did the CFO resign after the SEC opened a probe?"     # plain string, no citation
REAL_SEC_GET = edgar.sec_get


def assert_in_order(text: str, headings: list[str]) -> None:
    positions = [text.find(heading) for heading in headings]
    assert -1 not in positions, [h for h, p in zip(headings, positions) if p == -1]
    assert positions == sorted(positions)


# ---------------------------------------------------------------------------------------------
# Offline stand-ins for SEC, Yahoo Finance and the LLM
# ---------------------------------------------------------------------------------------------

def serve_fixtures(mp: pytest.MonkeyPatch, fixture_dir: Path) -> None:
    """Answer every SEC request from the fixture folders and stub market data (no network)."""
    folders = {json.loads(meta.read_text(encoding="utf-8"))["cik"]: meta.parent
               for meta in fixture_dir.glob("*/meta.json")}

    def fake_sec_get(url: str, user_agent: str, *, timeout: float = 30) -> bytes:
        if url == edgar.TICKERS_URL:
            return (fixture_dir / "company_tickers.json").read_bytes()
        for cik, folder in folders.items():
            if f"CIK{cik:010d}" in url:
                return (folder / ("companyfacts.json" if "companyfacts" in url else "submissions.json")).read_bytes()
            if f"/edgar/data/{cik}/" in url:
                return (folder / "10k.html").read_bytes()
        raise edgar.EdgarError(f"Not found: {url}")

    def fake_market_data(ticker: str, shares_fallback: float | None = None) -> MarketData:
        return MarketData(ticker=ticker, price=250.0, market_cap=3.7e12, currency="USD", as_of="2026-10-16")

    mp.setattr(edgar, "sec_get", fake_sec_get)
    mp.setattr(edgar, "html_to_text", functools.lru_cache(maxsize=None)(edgar.html_to_text))  # parse each 10-K once
    mp.setattr(pipeline, "get_market_data", fake_market_data)


class FakeLLMClient:
    """Answers each committee role with schema-valid JSON that cites real ids from the prompt, plus claims the
    citation check must remove: invalid ids, a made-up quote, a fabricated number, an inline fake tag, a
    one-sided said-vs-shows row and an uncited question."""

    def __init__(self, provider: str, model: str | None = None):
        self.label = f"{provider} / fake-model"
        self.roles: list[str] = []

    def complete_json(self, system: str, user: str, *, max_tokens: int = 16000) -> dict:
        ids = re.findall(r"^\[([EFR]\d+)\]", user, re.MULTILINE)
        e = [i for i in ids if i.startswith("E")]
        f = [i for i in ids if i.startswith("F")]
        r = [i for i in ids if i.startswith("R")]
        first_passage = re.search(r"^\[E1\][^\n]*\n([^\n]+)", user, re.MULTILINE).group(1)
        quote = " ".join(first_passage.split()[:8])

        def claim(text: str, cites: list[str], quote: str = "") -> dict:
            return {"text": text, "citations": cites, "quote": quote}

        bogus = claim(BOGUS_CLAIM, ["E999", "Z1"])
        if "You are the Committee Chair" in system:
            self.roles.append("chair")
            return {
                "thesis": [claim("Thesis one.", [e[0], f[0]], quote), claim("Thesis two.", [f[0]]),
                           claim("Thesis three [E998 · Call Q&A].", [e[1]])],
                "business_overview": [claim("What the company sells.", [e[0]])],
                "bull_case": [claim("Bull point.", [f[0]]), claim(FABRICATED, [f[0]])],
                "bear_case": [claim("Bear point.", [e[1]]), bogus],
                "top_risks": [{**claim("Top risk.", [e[2]]), "severity": "High"}],
                "said_vs_shows": [{"management_said": "Demand is strong.", "numbers_show": "Revenue grew.",
                                   "verdict": "Consistent", "citations": [e[0], f[0]], "quote": FAKE_QUOTE},
                                  {"management_said": ONE_SIDED, "numbers_show": "Margins fell.",
                                   "verdict": "Contradicted", "citations": [f[1]]}],
                "red_flags": [claim("The flag is benign.", [r[0]])] if r else [],
                "questions": [claim("What is the outlook for margins?", [e[0]]),
                              claim("How will capital be allocated?", [f[0]]), UNCITED_QUESTION],
                "vote": {"decision": "Buy", "conviction": "Medium", "rationale": [claim("Cash generation.", [f[0]])],
                         "what_would_change": [claim("Gross margin below 40%.", [f[0]])]},
            }
        if "You are the Risk officer" in system:
            self.roles.append("risk")
            return {"risks": [{**claim("Supply chain risk.", [e[2]]), "severity": "Medium"}, bogus],
                    "red_flag_review": [claim("Benign.", [r[0]])] if r else [],
                    "said_vs_shows": []}
        self.roles.append("analyst")
        return {"points": [claim("Strong cash flow.", [f[0], e[0]], quote), bogus]}


class FailingLLMClient(FakeLLMClient):
    def complete_json(self, system: str, user: str, *, max_tokens: int = 16000) -> dict:
        raise LLMError("rate limit or quota exceeded (HTTP 429).")


class UncitedChairLLMClient(FakeLLMClient):
    """The analysts are fine, but every Chair claim cites evidence that does not exist."""

    def complete_json(self, system: str, user: str, *, max_tokens: int = 16000) -> dict:
        if "You are the Committee Chair" not in system:
            return super().complete_json(system, user, max_tokens=max_tokens)
        bogus = [{"text": BOGUS_CLAIM, "citations": ["E999"]}]
        return {"thesis": bogus, "bull_case": bogus, "vote": {"decision": "Buy", "conviction": "High"}}


@pytest.fixture(scope="module")
def offline(fixture_dir):
    """SEC and market data served from fixtures for every test in this module that asks for it."""
    with pytest.MonkeyPatch.context() as mp:
        serve_fixtures(mp, fixture_dir)
        yield


@pytest.fixture(scope="module")
def aapl(offline):
    return run_analysis("AAPL", SETTINGS)


@pytest.fixture(scope="module")
def aapl_llm(offline):
    clients: list[FakeLLMClient] = []

    def make_client(provider: str, model: str | None = None) -> FakeLLMClient:
        clients.append(FakeLLMClient(provider, model))
        return clients[-1]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline, "LLMClient", make_client)
        result = run_analysis("aapl", LLM_SETTINGS)
    return result, clients


# ---------------------------------------------------------------------------------------------
# build_documents (synthetic, no fixtures)
# ---------------------------------------------------------------------------------------------

def make_tenk(ok: set[str]) -> TenK:
    text = "Cover page.\n" + "\n".join(f"Section {key} body text." for key in ("item1", "item1a", "item7"))
    sections = {}
    for key in ("item1", "item1a", "item7"):
        body = f"Section {key} body text."
        start = text.index(body)
        if key in ok:
            sections[key] = Section(key, edgar.SECTION_TITLES[key], body, start, start + len(body))
        else:
            sections[key] = Section(key, edgar.SECTION_TITLES[key], None, None, None)
    filing = Filing(cik=1, form="10-K", accession="0000000001-25-000001", filing_date="2025-10-31",
                    report_date="2025-09-27", primary_document="acme-20250927.htm")
    return TenK(CompanyInfo("ACME", 1, "Acme Corp"), filing, text, sections)


def test_build_documents_all_sections_parsed():
    docs = build_documents(make_tenk({"item1", "item1a", "item7"}), None, None)
    assert [d.kind for d in docs] == ["business", "risk", "mdna"]
    assert docs[0].source == "10-K FYE 2025-09-27 (filed 2025-10-31)"   # no financials: the period-end date
    assert docs[1].section == "Item 1A. Risk Factors"


@pytest.mark.parametrize("newest_end, label", [("2025-09-27", "10-K FY2025 (filed 2025-10-31)"),
                                                ("2024-09-28", "10-K FYE 2025-09-27 (filed 2025-10-31)")])
def test_10k_source_label_uses_xbrl_fiscal_year_only_for_the_same_period(newest_end, label):
    table = pd.DataFrame({"period_end": ["2023-09-30", newest_end]}, index=["FY2024", "FY2025"])
    docs = build_documents(make_tenk({"item1"}), None, None, FinancialData(table=table))
    assert {d.source for d in docs} == {label}


def test_build_documents_adds_text_outside_parsed_sections_when_one_fails():
    docs = build_documents(make_tenk({"item1", "item1a"}), None, None)
    assert [d.kind for d in docs] == ["business", "risk", "tenk_other"]
    other = docs[-1]
    assert other.section == "10-K text (outside parsed sections)"
    assert "Section item7 body text." in other.text and "Section item1 body" not in other.text


def test_build_documents_uses_full_text_when_nothing_parsed():
    tenk = make_tenk(set())
    docs = build_documents(tenk, None, None)
    assert [(d.kind, d.section) for d in docs] == [("tenk_other", "10-K full text")]
    assert docs[0].text == tenk.text


def test_build_documents_splits_transcript():
    text = SAMPLE_TRANSCRIPT.read_text(encoding="utf-8")
    docs = build_documents(None, "call.txt", text)
    assert [(d.kind, d.section) for d in docs] == [("call_prepared", "Prepared remarks"), ("call_qa", "Q&A")]
    assert docs[0].source == "Earnings call: call.txt"

    short = build_documents(None, "short.txt", "Operator: welcome. Our first question comes from Jane.")
    assert [d.kind for d in short] == ["call_full"]
    assert build_documents(None, None, None) == []


def test_missing_user_agent_is_a_clear_error_not_an_exception(monkeypatch):
    # The real sec_get validates the User-Agent before any request, so this stays offline.
    monkeypatch.setattr(edgar, "sec_get", REAL_SEC_GET)
    result = run_analysis("AAPL", Settings(user_agent=""))
    assert result.company is None
    assert "SEC_USER_AGENT" in result.errors[0]
    assert "The analysis could not be completed" in result.memo_markdown


# ---------------------------------------------------------------------------------------------
# End to end on real fixtures
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("providers, expected", [([], "No LLM API key was found"),
                                                  (["anthropic"], "No LLM provider was selected")])
def test_data_only_reason_says_whether_a_key_exists(monkeypatch, providers, expected):
    monkeypatch.setattr(pipeline, "available_providers", lambda: providers)
    result = pipeline.AnalysisResult(ticker="AAPL", generated_at="2026-10-19")
    assert pipeline._make_llm(SETTINGS, result) is None
    assert result.no_committee_reason.startswith(expected)


def test_data_only_run_for_apple(aapl):
    assert aapl.company.cik == 320193 and aapl.ticker == "AAPL"
    assert aapl.errors == [] and aapl.warnings == []
    assert aapl.tenk.parsed_ok == ["item1", "item1a", "item7"]
    assert aapl.documents[0].source == "10-K FY2025 (filed 2025-10-31)"
    assert "did not run (data-only mode)" in aapl.no_committee_reason
    assert len(aapl.fin.years) == 5
    assert aapl.valuation["pe"] is not None and aapl.valuation["basis_year"] == aapl.fin.years[-1]
    assert [d.kind for d in aapl.documents] == ["business", "risk", "mdna"]
    assert aapl.chunk_count > 50
    assert set(aapl.retrieved) == set(MEMO_QUERIES)
    assert all(aapl.retrieved[key] for key in ("business", "risks", "financial_performance"))
    assert {item.kind for item in aapl.pack.items} >= {"financial", "passage"}
    assert aapl.committee is None and aapl.vote is None and aapl.llm_label is None
    assert all(step in aapl.timings for step in STEPS)

    memo = aapl.memo_markdown
    assert memo.startswith("# Investment Committee Memo: Apple Inc. (AAPL)")
    assert_in_order(memo, DATA_ONLY_HEADINGS)
    assert "Not issued — requires the LLM committee." in memo
    assert "earnings call transcript not provided" in memo
    assert aapl.no_committee_reason in memo and "no LLM key" not in memo


def test_passages_per_section_overrides_k(offline):
    result = run_analysis("AAPL", Settings(user_agent=SETTINGS.user_agent, passages_per_section=2))
    assert all(len(hits) <= 2 for hits in result.retrieved.values())
    assert len(result.retrieved["risks"]) == 2


def test_llm_committee_run_with_fake_client(aapl_llm):
    result, clients = aapl_llm
    assert len(clients) == 1 and sorted(clients[0].roles) == ["analyst", "analyst", "chair", "risk"]
    assert result.llm_label == "anthropic / fake-model"
    assert result.vote == ("Buy", "Medium") and result.no_committee_reason == ""
    assert result.committee.errors == [] and result.errors == [] and result.warnings == []
    assert "Committee" in result.timings

    memo = result.memo_markdown
    assert "**Committee vote: BUY** · Conviction: Medium" in memo
    assert_in_order(memo, COMMITTEE_HEADINGS)
    assert "Thesis one." in memo and "- Thesis three. [E" in memo        # inline fake tag removed from the text
    assert re.search(r"^1\. What is the outlook for margins\? \[E\d+ · ", memo, re.MULTILINE)
    assert re.search(r"^- Gross margin below 40%\. \[F\d+ · ", memo, re.MULTILINE)

    # No citation, no claim: rejected claims are counted in Appendix B but their text appears nowhere.
    audit = result.committee.audit
    assert f"{BOGUS_CLAIM} (no valid citation)" in audit.dropped and {"E999", "Z1", "E998"} <= set(audit.invalid_ids)
    assert any(FABRICATED in d and "numbers not in the cited evidence" in d for d in audit.dropped)
    assert any(ONE_SIDED in d and "needs both an E and an F citation" in d for d in audit.dropped)
    for text in (BOGUS_CLAIM, FABRICATED.replace("$", "\\$"), ONE_SIDED, UNCITED_QUESTION, FAKE_QUOTE, "E998",
                 "E999"):
        assert text not in memo, text
    # The real quote survives; the invented one is blanked.
    assert audit.quotes_verified >= 1 and audit.quotes_removed >= 1
    # Every citation shown in the memo points at an item in the evidence pack.
    cited = set(re.findall(r"\[([EFR]\d+) ·", memo))
    assert cited and cited <= result.pack.ids


def test_analysts_failing_falls_back_to_data_only_memo(offline, monkeypatch):
    monkeypatch.setattr(pipeline, "LLMClient", FailingLLMClient)
    result = run_analysis("AAPL", LLM_SETTINGS)
    assert result.committee is not None and result.committee.memo is None
    assert result.vote is None and "all three analysts failed" in result.no_committee_reason
    assert any(w.startswith("All three analysts failed: rate limit") for w in result.warnings)
    assert any("data-only memo is shown" in w for w in result.warnings)
    assert "**Data-only mode**" in result.memo_markdown
    assert "All three analysts failed: rate limit" in result.memo_markdown


def test_chair_memo_without_valid_citations_falls_back_to_data_only(offline, monkeypatch):
    monkeypatch.setattr(pipeline, "LLMClient", UncitedChairLLMClient)
    result = run_analysis("AAPL", LLM_SETTINGS)
    assert result.committee.memo is not None and result.committee.audit.claims_kept == 0
    assert result.vote is None and "citation check" in result.no_committee_reason
    assert "**Data-only mode**" in result.memo_markdown and "Committee vote" not in result.memo_markdown
    assert BOGUS_CLAIM not in result.memo_markdown


def test_missing_api_key_runs_data_only(offline, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = run_analysis("AAPL", LLM_SETTINGS)
    assert result.committee is None
    assert any("data-only mode" in w and "ANTHROPIC_API_KEY" in w for w in result.warnings)
    assert "ANTHROPIC_API_KEY" in result.no_committee_reason
    assert_in_order(result.memo_markdown, DATA_ONLY_HEADINGS)


def test_transcript_upload_is_indexed(offline):
    data = SAMPLE_TRANSCRIPT.read_bytes().replace(b"Northwind Robotics", b"Apple")  # make it name the company
    result = run_analysis("AAPL", SETTINGS, transcript=("apple_call.txt", data))
    assert result.errors == [] and result.transcript_chars > 5000
    assert {"call_prepared", "call_qa"} <= {d.kind for d in result.documents}
    outlook = result.retrieved["management_outlook"]
    assert outlook and all(chunk.kind.startswith("call_") for chunk, _ in outlook)
    assert "earnings call transcript `apple_call.txt`;" in result.memo_markdown
    assert "Transcript" in result.timings


def test_transcript_of_another_company_is_not_used(offline):
    # The sample is a call of a fictional company, so it must not become Apple's "management said".
    result = run_analysis("AAPL", SETTINGS, transcript=("sample_transcript.txt", SAMPLE_TRANSCRIPT.read_bytes()))
    assert any("never mentions Apple or AAPL, so it was not used" in e for e in result.errors)
    assert result.transcript_chars == 0
    assert not any(d.kind.startswith("call_") for d in result.documents)
    assert "Northwind" not in result.memo_markdown and "Northwind" not in result.pack.to_prompt()
    assert "earnings call transcript `sample_transcript.txt` (not used)" in result.memo_markdown


@pytest.mark.parametrize("ticker, name, text, expected", [
    ("COST", "COSTCO WHOLESALE CORP", "How much did tariffs cost you this quarter?", False),  # ticker as a word
    ("GE", "GENERAL ELECTRIC CO", "General and administrative expenses rose.", False),        # generic first word
    ("GM", "GENERAL MOTORS CO", "Welcome to the General Motors earnings call.", True),
    ("COST", "COSTCO WHOLESALE CORP", "Welcome to the Costco call.", True),
    ("T", "AT&T INC.", "Thank you for joining AT&T's call.", True),
    ("GE", "GE AEROSPACE", "GE delivered record orders.", True),
])
def test_mentions_company(ticker, name, text, expected):
    assert pipeline._mentions_company(text, CompanyInfo(ticker, 1, name)) is expected


def test_sample_transcript_is_refused_for_common_word_tickers():
    sample = SAMPLE_TRANSCRIPT.read_text(encoding="utf-8")
    for ticker, name in [("COST", "COSTCO WHOLESALE CORP"), ("GE", "GENERAL ELECTRIC CO"), ("NOW", "ServiceNow, Inc."),
                         ("LOW", "LOWES COMPANIES INC"), ("ALL", "ALLSTATE CORP"), ("T", "AT&T INC.")]:
        assert not pipeline._mentions_company(sample, CompanyInfo(ticker, 1, name)), ticker


def test_unreadable_transcript_is_an_error_not_an_exception(offline):
    result = run_analysis("AAPL", SETTINGS, transcript=("call.docx", b"not a transcript"))
    assert result.transcript_chars == 0
    assert any("Unsupported file type" in e for e in result.errors)
    assert "(not used)" in result.memo_markdown
    assert result.fin is not None and result.tenk is not None  # the rest of the run carried on


def test_failed_10k_sections_fall_back_to_full_text(offline):
    result = run_analysis("GE", SETTINGS)
    assert result.tenk is not None and result.tenk.failed
    assert any(d.kind == "tenk_other" for d in result.documents)
    assert result.retrieved["risks"], "risk queries should fall back to the full 10-K text"


@pytest.mark.parametrize("ticker, problem", [("TSM", "Form 20-F"), ("XOM", "No 10-K found")])
def test_companies_without_a_10k_still_get_a_memo(offline, monkeypatch, ticker, problem):
    monkeypatch.setattr(pipeline, "LLMClient", FakeLLMClient)
    result = run_analysis(ticker, LLM_SETTINGS)  # provider set, but there is no evidence to argue from
    assert result.company is not None
    assert result.tenk is None and result.fin is None
    assert any(problem in e for e in result.errors)
    assert result.memo_markdown.startswith("# Investment Committee Memo:")
    assert "**Data notes**" in result.memo_markdown
    assert result.committee is None and "skipped" in result.no_committee_reason


def test_bank_gets_bank_appropriate_checks_and_multiples(offline):
    result = run_analysis("JPM", SETTINGS)   # SIC 6021 from the submissions feed
    assert result.tenk.filing.sic == 6021
    assert any("not meaningful for banks and insurers, SIC 6021" in c for c in result.checks_not_run)
    assert not any(flag.name.startswith("Weak cash conversion") for flag in result.flags)
    assert result.valuation["pe"] is not None
    assert result.valuation["ev_sales"] is None and result.valuation["fcf_yield"] is None


def test_unknown_ticker_returns_minimal_memo(offline):
    result = run_analysis("ZZZZQ", LLM_SETTINGS)
    assert result.company is None
    assert "not found" in result.errors[0]
    assert result.memo_markdown.startswith("# Investment Committee Memo: ZZZZQ")
    assert "The analysis could not be completed" in result.memo_markdown
    assert list(result.timings) == ["Ticker lookup", "Total"]


def test_unexpected_step_failure_is_reported_and_run_continues(offline, monkeypatch):
    def broken(*args, **kwargs):
        raise KeyError("boom")

    monkeypatch.setattr(pipeline, "detect_red_flags", broken)
    result = run_analysis("AAPL", SETTINGS)
    assert any(e.startswith("Could not run the red-flag checks (KeyError") for e in result.errors)
    assert result.pack is not None and result.memo_markdown


# ---------------------------------------------------------------------------------------------
# scripts/smoke_test.py
# ---------------------------------------------------------------------------------------------

def load_smoke_test():
    spec = importlib.util.spec_from_file_location("smoke_test", ROOT / "scripts" / "smoke_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_test_cli_writes_memo(offline, tmp_path, capsys):
    smoke_test = load_smoke_test()
    out = tmp_path / "memo.md"
    code = smoke_test.main(["AAPL", "--transcript", str(SAMPLE_TRANSCRIPT), "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 0
    assert out.read_text(encoding="utf-8").startswith("# Investment Committee Memo: Apple Inc. (AAPL)")
    for expected in ("Step timings", "Item 1A. Risk Factors", "Free cash flow", "Red flags (", "Errors (1)"):
        assert expected in printed


def test_smoke_test_cli_unknown_ticker_exits_1(offline, tmp_path, capsys):
    out = tmp_path / "memo.md"
    assert load_smoke_test().main(["ZZZZQ", "--out", str(out)]) == 1
    assert "The analysis could not be completed" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Streamlit app, headless
# ---------------------------------------------------------------------------------------------

def run_app(monkeypatch, result=None, user_agent="Test test@example.com"):
    """Start app.py headless; with a result, submit AAPL and return the app showing that result.

    The run_analysis stand-in records the Settings of each call in `at.calls`.
    """
    from streamlit.testing.v1 import AppTest

    calls = []

    def fake_run_analysis(ticker, settings, *args, **kwargs):
        calls.append(settings)
        return result

    monkeypatch.setattr(pipeline, "run_analysis", fake_run_analysis)
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.calls = calls
    assert not at.exception
    if result is not None:
        at.text_input(key="user_agent").input(user_agent)
        at.text_input(key="ticker").input("AAPL")
        at.form_submit_button[0].click()
        at.run()
        assert not at.exception
    return at


def column_rows(block) -> list[list[int]]:
    """Metrics per column for every row of st.columns under `block`, e.g. [[1, 1, 1], [1, 1, 1]]."""
    children = list(getattr(block, "children", {}).values())
    columns = [child for child in children if child.type == "column"]
    rows = [[len(column.get("metric")) for column in columns]] if columns else []
    return rows + [row for child in children for row in column_rows(child)]


def test_app_starts_without_a_result(monkeypatch):
    at = run_app(monkeypatch)
    assert at.title[0].value == "AI Investment Committee Copilot"
    assert len(at.tabs) == 0


def test_app_settings_live_in_the_form(monkeypatch):
    at = run_app(monkeypatch)
    form = at.get("form")[0]
    assert {w.key for w in form.text_input} == {"ticker", "user_agent", "model"}
    assert [w.key for w in form.selectbox] == ["provider"]
    assert len(at.slider) == 0 and len(at.sidebar.children) == 0
    assert at.text_input(key="model").value == ""             # blank = the provider's default model
    # Changing a setting without submitting does not start (or cancel) a run.
    at.text_input(key="user_agent").input("Someone else s@example.com")
    at.selectbox(key="provider").select("data-only")
    at.run()
    assert at.calls == [] and not at.exception


def test_app_asks_for_a_user_agent_before_running(monkeypatch, aapl):
    at = run_app(monkeypatch, aapl, user_agent="John Smith")
    assert at.calls == []
    assert any("SEC User-Agent" in w.value for w in at.warning)


def test_app_renders_data_only_result(monkeypatch, aapl):
    at = run_app(monkeypatch, aapl)
    assert at.calls[0].passages_per_section is None and at.calls[0].model is None
    assert [tab.label for tab in at.tabs] == TAB_NAMES
    assert any(m.value.startswith("# Investment Committee Memo: Apple Inc.") for m in at.markdown)
    vote = next(metric for metric in at.metric if metric.label == "Committee vote")
    assert vote.value == "Not issued" and vote.help == aapl.no_committee_reason
    assert column_rows(at.main)[-2:] == [[1, 1, 1], [1, 1, 1]]  # two rows of three metrics
    assert any(info.value == aapl.no_committee_reason for info in at.info)  # debate tab
    assert any("| P/E (FY2025 net income) |" in m.value for m in at.markdown)  # Financials tab = memo snapshot
    assert at.get("status")[0].state == "complete"
    assert len(at.dataframe) >= 4


def test_app_status_shows_problems(monkeypatch, offline):
    result = run_analysis("AAPL", SETTINGS, transcript=("call.docx", b"not a transcript"))
    at = run_app(monkeypatch, result)
    status = at.get("status")[0]
    assert status.state == "error" and "(1 problem, see below)" in status.label


def test_app_debate_tab_explains_a_skipped_committee(monkeypatch, offline):
    monkeypatch.setattr(pipeline, "LLMClient", FakeLLMClient)
    at = run_app(monkeypatch, run_analysis("TSM", LLM_SETTINGS))
    assert any(info.value.startswith("The committee was skipped") for info in at.info)
    assert not any("Set an API key" in info.value for info in at.info)


def test_streamlit_upload_limit_matches_the_app():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["maxUploadSize"] == MAX_UPLOAD_MB


def test_app_shows_only_the_error_for_an_unknown_ticker(monkeypatch, offline):
    at = run_app(monkeypatch, run_analysis("ZZZZQ", SETTINGS))
    assert "not found in SEC's company list" in at.error[0].value
    assert len(at.tabs) == 0 and len(at.metric) == 0


def test_app_renders_committee_result(monkeypatch, aapl_llm):
    result, _ = aapl_llm
    at = run_app(monkeypatch, result)
    labels = [tab.label for tab in at.tabs]  # depth-first: the debate's role tabs follow "Committee debate"
    assert [label for label in labels if label in TAB_NAMES] == TAB_NAMES
    assert {"Bull analyst", "Bear analyst", "Risk officer"} <= set(labels)
    assert any(metric.value == "Buy" for metric in at.metric)
    assert any("Committee vote: BUY" in m.value for m in at.markdown)
    memo = next(m.value for m in at.markdown if m.value.startswith("# Investment Committee Memo"))
    shown = " ".join(m.value for m in at.markdown)
    assert "Strong cash flow." in shown                       # debate tab: the analysts' cited claims
    assert BOGUS_CLAIM not in memo and BOGUS_CLAIM in shown   # dropped claims: Diagnostics tab only
