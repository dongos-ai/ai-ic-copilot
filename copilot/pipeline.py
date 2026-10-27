"""Run the whole analysis for one ticker and return everything the UI and memo need.

run_analysis never raises for data, network or LLM problems: each step is timed, and a failure is
recorded in `errors` (problems with the inputs, also listed in the memo) or `warnings` (app notes,
e.g. data-only mode), and the remaining steps carry on with whatever data is available.
"""
from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterator

from .committee import CommitteeResult, EvidencePack, build_evidence_pack, run_committee
from .config import CHUNK_WORDS
from .edgar import CompanyInfo, EdgarError, TenK, load_10k, lookup_company, normalize_ticker, text_outside_sections
from .financials import FinancialData, load_financials
from .llm import LLMClient, LLMError, available_providers
from .market import MarketData, compute_valuation, get_market_data
from .memo import render_memo
from .red_flags import RedFlag, detect_red_flags
from .retrieval import BM25Index, Chunk, Document, chunk_documents, retrieve_for_memo
from .transcript import read_transcript, split_transcript

SECTION_KINDS = {"item1": "business", "item1a": "risk", "item7": "mdna"}
TRANSCRIPT_KINDS = {"Prepared remarks": "call_prepared", "Q&A": "call_qa", "Full transcript": "call_full"}


@dataclass
class Settings:
    user_agent: str
    provider: str | None = None          # None -> data-only mode
    model: str | None = None
    passages_per_section: int | None = None


@dataclass
class AnalysisResult:
    ticker: str
    generated_at: str                    # YYYY-MM-DD
    company: CompanyInfo | None = None
    tenk: TenK | None = None
    fin: FinancialData | None = None
    market: MarketData | None = None
    valuation: dict | None = None
    flags: list[RedFlag] = field(default_factory=list)
    checks_not_run: list[str] = field(default_factory=list)
    transcript_name: str | None = None
    transcript_chars: int = 0
    documents: list[Document] = field(default_factory=list)
    chunk_count: int = 0
    retrieved: dict[str, list[tuple[Chunk, float]]] = field(default_factory=dict)
    pack: EvidencePack | None = None
    committee: CommitteeResult | None = None
    llm_label: str | None = None
    memo_markdown: str = ""
    no_committee_reason: str = ""        # why there is no usable committee memo ("" when there is one)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def committee_memo(self) -> dict | None:
        """The Chair's memo, or None when the committee did not produce one with at least one cited claim."""
        committee = self.committee
        if committee is None or not committee.memo or committee.audit.claims_kept == 0:
            return None
        return committee.memo

    @property
    def vote(self) -> tuple[str, str] | None:
        memo = self.committee_memo
        if memo is None:
            return None
        return memo["vote"]["decision"], memo["vote"]["conviction"]


def build_documents(tenk: TenK | None, transcript_name: str | None, transcript_text: str | None,
                    fin: FinancialData | None = None) -> list[Document]:
    """The documents to search: parsed 10-K sections, a full-text fallback, and the call transcript.

    If any 10-K section failed to parse, the text outside the parsed sections (the whole filing when
    nothing parsed) is added as kind "tenk_other", so retrieval still finds risk / MD&A passages.
    """
    docs: list[Document] = []
    if tenk is not None:
        source = tenk_source_label(tenk, fin)
        for key, kind in SECTION_KINDS.items():
            section = tenk.sections.get(key)
            if section is not None and section.ok:
                docs.append(Document(source, section.title, kind, section.text))
        if tenk.failed:
            label = "10-K text (outside parsed sections)" if tenk.parsed_ok else "10-K full text"
            rest = text_outside_sections(tenk.text, tenk.sections)
            if rest:
                docs.append(Document(source, label, "tenk_other", rest))
    if transcript_text and transcript_text.strip():
        source = f"Earnings call: {transcript_name or 'transcript'}"
        for part, text in split_transcript(transcript_text):
            docs.append(Document(source, part, TRANSCRIPT_KINDS.get(part, "call_full"), text))
    return docs


def tenk_source_label(tenk: TenK, fin: FinancialData | None) -> str:
    """'10-K FY2025 (filed 2025-10-31)': the XBRL fiscal-year label when the financials end on the filing's
    period end (filers name fiscal years differently), else the period-end date ('10-K FYE 2025-09-27 ...')."""
    filing = tenk.filing
    year = filing.fiscal_year_label
    if fin is not None and not fin.table.empty and fin.table["period_end"].iloc[-1] == filing.report_date:
        year = fin.years[-1]
    return f"10-K {year} (filed {filing.filing_date})"


def run_analysis(ticker: str, settings: Settings, transcript: tuple[str, bytes] | None = None,
                 progress: Callable[[str], None] | None = None) -> AnalysisResult:
    """Ticker (+ optional transcript upload as (filename, bytes)) -> AnalysisResult with the memo.

    `progress` receives short status lines as the steps run (for st.status or a console).
    """
    def say(message: str) -> None:
        if progress:
            progress(message)

    started = time.perf_counter()
    result = AnalysisResult(ticker=normalize_ticker(ticker), generated_at=date.today().isoformat())

    # 1. Ticker -> company. Nothing else can run without it.
    say(f"Looking up {result.ticker or 'the ticker'} in SEC's company list...")
    with _step(result, "Ticker lookup", "Could not look up the ticker"):
        result.company = lookup_company(ticker, settings.user_agent)
    if result.company is None:
        result.memo_markdown = _minimal_memo(result)
        result.timings["Total"] = _seconds_since(started)
        return result
    company = result.company
    result.ticker = company.ticker

    # 2. Latest 10-K text and sections.
    say(f"Found {company.name} (CIK {company.cik}). Downloading the latest 10-K...")
    with _step(result, "10-K", "Could not load the 10-K"):
        result.tenk = load_10k(company, settings.user_agent)
    if result.tenk:
        say(_section_status(result.tenk))

    # 3. Five years of XBRL financials.
    say("Pulling five years of financials from SEC XBRL company facts...")
    with _step(result, "Financials", "Could not build the financial table"):
        result.fin = load_financials(company.cik, settings.user_agent)

    # 4. Price, market cap and valuation multiples (get_market_data never raises).
    sic = result.tenk.filing.sic if result.tenk else None   # banks and insurers skip cash-flow screens and EV
    say("Fetching price and market cap from Yahoo Finance...")
    with _step(result, "Market data", "Could not compute the valuation"):
        shares = result.fin.shares_outstanding if result.fin else None
        result.market = get_market_data(company.ticker, shares_fallback=shares)
        if result.market.error:
            result.warnings.append(result.market.error)
        result.valuation = compute_valuation(result.fin, result.market, sic=sic)

    # 5. Earnings-call transcript (optional upload). A call that never names the company is not used.
    transcript_text = None
    if transcript:
        name, data = transcript
        result.transcript_name = name
        say(f"Reading the transcript {name}...")
        with _step(result, "Transcript", f"Could not read the transcript '{name}'"):
            text = read_transcript(name, data)
            if _mentions_company(text, company):
                transcript_text, result.transcript_chars = text, len(text)
            else:
                result.errors.append(f"The transcript '{name}' never mentions {_short_name(company)} or "
                                     f"{company.ticker}, so it was not used; upload this company's call.")

    # 6. Chunk the documents and retrieve evidence for each memo section (BM25).
    with _step(result, "Retrieval", "Could not retrieve evidence passages"):
        result.documents = build_documents(result.tenk, result.transcript_name, transcript_text, result.fin)
        if result.documents:
            chunks = chunk_documents(result.documents, CHUNK_WORDS)
            result.chunk_count = len(chunks)
            say(f"Indexed {len(chunks)} passages; retrieving evidence for each memo section...")
            result.retrieved = retrieve_for_memo(BM25Index(chunks), k_override=settings.passages_per_section)

    # 7. Rule-based red flags (numbers + 10-K text).
    say("Running red-flag checks...")
    with _step(result, "Red flags", "Could not run the red-flag checks"):
        filing_text = result.tenk.text if result.tenk else None
        result.flags, result.checks_not_run = detect_red_flags(result.fin, filing_text, sic=sic)

    # 8. Number every fact, flag and passage the committee may cite.
    with _step(result, "Evidence pack", "Could not build the evidence pack"):
        result.pack = build_evidence_pack(result.retrieved, result.fin, result.valuation, result.flags)

    # 9. The LLM committee (skipped in data-only mode).
    llm = _make_llm(settings, result)
    if llm is None:
        say("Data-only mode: skipping the LLM committee.")
    elif result.pack is None or not (result.documents or result.fin):
        result.no_committee_reason = ("The committee was skipped: there is no 10-K text, financial data or "
                                      "transcript to argue from.")
        result.warnings.append(result.no_committee_reason)
    else:
        result.llm_label = llm.label
        say(f"Convening the committee ({llm.label})...")
        with _step(result, "Committee", "The committee failed"):
            result.committee = run_committee(llm, company.name, company.ticker, _filing_label(result.tenk),
                                             result.pack, on_progress=say)
            result.warnings += result.committee.errors
        result.no_committee_reason = _no_memo_reason(result)
        if result.no_committee_reason:
            result.warnings.append("The committee produced no usable memo, so the data-only memo is shown.")

    # 10. The Markdown memo.
    say("Writing the memo...")
    with _step(result, "Memo", "Could not render the memo"):
        result.memo_markdown = render_memo(result)
    if not result.memo_markdown:
        result.memo_markdown = _minimal_memo(result)
    result.timings["Total"] = _seconds_since(started)
    return result


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------

@contextmanager
def _step(result: AnalysisResult, name: str, failure: str) -> Iterator[None]:
    """Time one step and turn any exception into a user-facing error, so the run continues.

    Code after the failing line inside the `with` block is skipped, like with contextlib.suppress.
    """
    started = time.perf_counter()
    try:
        yield
    except EdgarError as e:          # message already written for users
        result.errors.append(str(e))
    except ValueError as e:          # e.g. an unreadable transcript upload
        result.errors.append(f"{failure}: {e}")
    except Exception as e:  # an unexpected failure must not break the demo: report it and move on
        result.errors.append(f"{failure} ({type(e).__name__}: {e}).")
    finally:
        result.timings[name] = _seconds_since(started)


def _seconds_since(started: float) -> float:
    return round(time.perf_counter() - started, 2)


def _make_llm(settings: Settings, result: AnalysisResult) -> LLMClient | None:
    """The LLM client, or None for data-only mode (no provider chosen, missing key or SDK)."""
    if not settings.provider:
        if available_providers():
            result.no_committee_reason = "No LLM provider was selected, so the committee did not run (data-only mode)."
        else:
            result.no_committee_reason = ("No LLM API key was found (ANTHROPIC_API_KEY, OPENAI_API_KEY or "
                                          "GEMINI_API_KEY), so the committee did not run (data-only mode).")
        return None
    try:
        return LLMClient(settings.provider, settings.model)
    except Exception as e:  # LLMError for a missing key / SDK; anything else from an SDK constructor
        reason = str(e) if isinstance(e, LLMError) else f"{type(e).__name__}: {e}"
        result.no_committee_reason = f"The LLM committee is unavailable: {reason}"
        result.warnings.append(f"LLM committee unavailable, running in data-only mode. {reason}")
        return None


def _no_memo_reason(result: AnalysisResult) -> str:
    """Why a convened committee left no usable memo ("" when there is one)."""
    if result.committee_memo is not None:
        return ""
    committee = result.committee
    if committee is None:
        return "The committee failed with an unexpected error."
    if committee.memo is None:
        return committee.errors[-1] if committee.errors else "The committee produced no memo."
    return "None of the Chair's claims survived the citation check, so its memo is not shown."


def _section_status(tenk: TenK) -> str:
    """'10-K FYE 2025-09-27 (filed 2025-10-31): found Item 1, Item 1A, Item 7.' plus any fallback note."""
    titles = {key: section.title.split(".")[0] for key, section in tenk.sections.items()}
    notes = []
    if tenk.parsed_ok:
        notes.append("found " + ", ".join(titles[key] for key in tenk.parsed_ok) + ".")
    if tenk.failed:
        notes.append("could not isolate " + ", ".join(titles[key] for key in tenk.failed) + ", so the full "
                     "text is searched instead.")
    return f"10-K {tenk.filing.fiscal_year_label} (filed {tenk.filing.filing_date}): " + " ".join(notes)


def _filing_label(tenk: TenK | None) -> str:
    if tenk is None:
        return "No 10-K available (financial data from SEC XBRL company facts only)"
    filing = tenk.filing
    return f"Form 10-K for the fiscal year ended {filing.report_date or 'n/a'} (filed {filing.filing_date})"


# First words too common to identify a company on their own ("General" Electric, "American" Express).
GENERIC_NAME_WORDS = {"american", "bank", "capital", "first", "general", "global", "international", "national",
                      "new", "southern", "the", "united", "us"}


def _short_name(company: CompanyInfo) -> str:
    """Distinctive start of the SEC name: 'Apple Inc.' -> 'Apple', 'GENERAL MOTORS CO' -> 'GENERAL MOTORS'."""
    words = [word for word in re.split(r"[\s,./]+", company.name.strip()) if word]
    if not words:
        return company.ticker
    if words[0].lower() in GENERIC_NAME_WORDS and len(words) > 1:
        return " ".join(words[:2])
    return words[0]


def _mentions_company(text: str, company: CompanyInfo) -> bool:
    """True if the text names the company (any case) or its ticker as written (e.g. 'COST', not 'cost').

    A cheap check against uploading another company's call.
    """
    if re.search(rf"\b{re.escape(_short_name(company))}\b", text, re.IGNORECASE):
        return True
    return len(company.ticker) >= 2 and re.search(rf"\b{re.escape(company.ticker)}\b", text) is not None


def _minimal_memo(result: AnalysisResult) -> str:
    """Memo for a run that could not finish (e.g. unknown ticker), so there is always something to show."""
    problems = "\n".join(f"- {error}" for error in result.errors) or "- Unknown error."
    return (f"# Investment Committee Memo: {result.ticker or 'unknown ticker'}\n\n"
            f"*Generated {result.generated_at} · Draft for discussion — not investment advice.*\n\n"
            "> **The analysis could not be completed.** Fix the problem below and run it again.\n\n"
            f"{problems}\n")
