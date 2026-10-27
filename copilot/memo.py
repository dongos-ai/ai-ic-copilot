"""Render the investment committee memo as Markdown.

Two layouts:
- Committee mode: the Chair's memo wrapped around deterministic tables built from SEC and market data.
  The memo arrives already normalized by committee.enforce_citations: every claim is a
  {"text", "citations", "quote"} dict with valid ids, and the vote fields are canonical.
- Data-only mode: no usable committee memo (data-only mode chosen, no key, the LLM failed, or no claim
  survived the citation check). Shows the financial snapshot, rule-based red flags, and the retrieved
  passages, so the demo always produces a memo.

The financial snapshot is always computed from SEC XBRL / yfinance numbers, never from LLM output.
"""
from __future__ import annotations

import re
from datetime import date
from typing import TYPE_CHECKING

from .committee import SEVERITIES, CommitteeResult, EvidenceItem, EvidencePack
from .financials import FinancialData, format_money, format_multiple, format_pct, format_price, snapshot_table
from .red_flags import RedFlag
from .retrieval import MEMO_QUERIES

if TYPE_CHECKING:
    from .pipeline import AnalysisResult

DISCLAIMER = "Draft for discussion — not investment advice."
DATA_ONLY_BANNER = ("> **Data-only mode** — no committee debate or vote. Everything below is computed directly "
                    "from filings and market data.")
NO_VOTE_TEXT = "Not issued — requires the LLM committee."
NO_CLAIMS = "_No claims in this section survived the citation check._"
NO_FLAGS_TO_REVIEW = "_No automated red flags to review._"
THESIS_CLAIMS = 3

APPENDIX_EXCERPT_WORDS = 30   # passage excerpt length in Appendix A
EVIDENCE_EXCERPT_WORDS = 60   # passage excerpt length in the data-only evidence list
MAX_FLAG_QUESTIONS = 5        # default questions derived from red flags (plus the standard ones)

# (keywords in a red-flag name, question). The first matching entry wins, so specific phrases come first
# (e.g. "Stock comp large relative to free cash flow" must match stock comp, not free cash flow).
FLAG_QUESTIONS = [
    (("going concern", "going-concern"),
     "What financing is in place to resolve the going-concern doubt, and on what terms?"),
    (("material weakness",),
     "What caused the material weakness in internal controls, and when will remediation be complete?"),
    (("restate",),
     "What led to the restatement, and could other reported figures be affected?"),
    (("customer",),
     "How dependent is revenue on the largest customers, and what are the contract terms and renewal risks?"),
    (("receivable",),
     "Why are receivables growing faster than revenue? Have payment terms, customer mix, or collections changed?"),
    (("inventor",),
     "Why is inventory building faster than sales, and is any of it at risk of a write-down?"),
    (("gross margin",),
     "What is driving the gross-margin decline (pricing, mix, input costs), and is it structural?"),
    (("stock-based", "stock comp", "share-based", "sbc"),
     "How much of reported profit and free cash flow is effectively paid in stock, and how will dilution be limited?"),
    (("debt", "leverage"),
     "How will the company service and refinance its debt, and what are the maturities and covenants?"),
    (("margin",),
     "What is compressing operating margin, and what is management's plan to restore it?"),
    (("cash conversion", "ahead of cash", "earnings quality", "net income", "accrual"),
     "Why is operating cash flow lagging net income, and which accruals explain the gap?"),
    (("free cash flow", "runway", "cash burn"),
     "How long can the company fund negative free cash flow, and when should it turn positive?"),
    (("revenue decline", "declining revenue", "revenue fell", "revenue down"),
     "What caused the revenue decline, and what evidence shows it has bottomed?"),
    (("net loss", "unprofitab"),
     "What is the path to sustained profitability, and which costs are truly variable?"),
    (("dilution", "share count"),
     "Why is the share count rising, and how much value per share is created after dilution?"),
]
STANDARD_QUESTIONS = [
    "What growth and margins does today's valuation imply, and how does that compare with our base case?",
    "How is management allocating capital (buybacks, dividends, M&A, capex), and what returns has it earned?",
    "Which single risk would most damage the thesis, and what early indicators would tell us it is playing out?",
]


# ----------------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------------

def render_memo(result: AnalysisResult) -> str:
    """Markdown memo: the committee layout when the Chair's memo kept a cited claim, else the data-only layout."""
    memo = result.committee_memo
    if memo is None:
        return _render_data_only_memo(result)
    try:
        return _render_committee_memo(result, memo)
    except Exception as exc:  # a malformed memo must never break the demo
        return _render_data_only_memo(result, note=f"The committee memo could not be rendered ({exc}).")


def render_citations(ids: list[str], pack: EvidencePack | None) -> str:
    """'[E3 · 10-K Item 1A][F1 · XBRL Revenue]'. IDs missing from the pack render bare, e.g. '[E9]'."""
    parts = []
    for id_ in dict.fromkeys(ids or []):
        item = pack.get(id_) if pack else None
        parts.append(f"[{id_} · {escape_md(item.short_label)}]" if item and item.short_label else f"[{id_}]")
    return "".join(parts)


def render_claim(claim: dict, pack: EvidencePack | None) -> str:
    """'- text [E3 · 10-K Item 1A] — “verbatim quote”' (quote only when present); '' without text or citation."""
    body = _claim_body(claim, pack)
    return f"- {body}" if body else ""


def financial_snapshot_markdown(fin: FinancialData | None, valuation: dict | None) -> str:
    """Five-year table (SEC XBRL), valuation table (yfinance + latest fiscal year), and footnotes."""
    blocks: list[str] = []
    notes: list[str] = []
    currency = (fin.currency if fin else None) or "USD"

    if fin is None or fin.table is None or fin.table.empty:
        blocks.append("_No annual financial data was available from SEC XBRL company facts._")
    else:
        blocks.append(_history_table(fin))
        notes.append("Free cash flow = operating cash flow − capex. Cash & equivalents excludes short-term "
                     "investments (shown separately).")
        revenue_tags = (fin.tags_used or {}).get("revenue") or []
        if revenue_tags:
            notes.append("Revenue XBRL tag(s): " + ", ".join(f"`{tag}`" for tag in revenue_tags) + ".")

    if not valuation:
        blocks.append("_Valuation not available (no market data)._")
    else:
        blocks.append(_valuation_table(valuation, valuation.get("currency") or currency))
        if valuation.get("enterprise_value") is not None:
            notes.append("EV = market cap + total debt − cash − short-term investments.")
        notes.extend(valuation.get("notes") or [])

    if fin is not None:
        notes.extend(fin.notes)
    notes = [n for n in dict.fromkeys(_clean(n) for n in notes) if n]
    if notes:
        blocks.append("**Notes**\n\n" + "\n".join(f"- {escape_md(n)}" for n in notes))
    return "\n\n".join(blocks)


def default_questions(flags: list[RedFlag]) -> list[str]:
    """Deterministic committee questions: one per red-flag theme (High severity first, at most
    MAX_FLAG_QUESTIONS), followed by three standard questions."""
    by_question: dict[str, list[str]] = {}
    ranked = sorted(flags or [], key=lambda f: _rank(f.severity, SEVERITIES))
    for flag in ranked:
        name = (flag.name or "").lower()
        question = next((q for keys, q in FLAG_QUESTIONS if any(k in name for k in keys)), None)
        if question is None:
            question = f"What explains the “{flag.name}” flag, and is it temporary or structural?"
        by_question.setdefault(question, []).append(flag.id)

    questions = [f"{q} (see {', '.join(ids)})" for q, ids in by_question.items()]
    return questions[:MAX_FLAG_QUESTIONS] + STANDARD_QUESTIONS


def memo_filename(result: AnalysisResult) -> str:
    """File name for the Markdown download, e.g. 'AAPL_IC_memo_2026-10-19.md'."""
    ticker = re.sub(r"[^A-Z0-9-]", "-", (result.ticker or "").strip().upper()).strip("-") or "memo"
    day = re.sub(r"[^0-9-]", "", result.generated_at or "") or date.today().isoformat()
    return f"{ticker}_IC_memo_{day}.md"


def escape_md(text) -> str:
    """One line of plain text made safe for Markdown: '$' would start LaTeX in Streamlit, '~' strikes
    through, and '](' would make a live link or image."""
    text = _clean(text).replace("\\$", "$")
    return text.replace("$", "\\$").replace("~", "\\~").replace("](", "]\\(")


# ----------------------------------------------------------------------------------------------
# Layouts
# ----------------------------------------------------------------------------------------------

def _render_committee_memo(result: AnalysisResult, memo: dict) -> str:
    pack = result.pack
    committee = result.committee
    memo = {**memo, "thesis": memo["thesis"][:THESIS_CLAIMS]}   # the memo has a 3-bullet thesis
    vote = memo["vote"]
    decision, conviction = vote["decision"].upper(), vote["conviction"]
    model = committee.model_label or result.llm_label or "LLM committee"

    blocks = [
        _title(result),
        f"*Generated {_day(result)} · {escape_md(model)} · {DISCLAIMER}*",
        f"> **Committee vote: {decision}** · Conviction: {conviction}",
        _sources_line(result),
        _data_notes(result),
        "## 1. Investment thesis",
        _thesis(memo["thesis"], pack),
        "## 2. Business overview",
        _claim_list(memo["business_overview"], pack),
        "## 3. Financial snapshot",
        financial_snapshot_markdown(result.fin, result.valuation),
        "## 4. Bull case",
        _claim_list(memo["bull_case"], pack),
        "## 5. Bear case",
        _claim_list(memo["bear_case"], pack),
        "## 6. Top risks",
        _risk_list(memo["top_risks"], pack),
        "## 7. What management said vs. what the numbers show",
        _said_vs_shows(memo["said_vs_shows"], pack, has_transcript=bool(result.transcript_chars)),
        "## 8. Red flags",
        "### Automated checks",
        _flags_block(result),
        "### Committee view",
        # With no R items an empty list is the expected answer, not a sign that claims were dropped.
        _claim_list(memo["red_flags"], pack, empty=NO_CLAIMS if result.flags else NO_FLAGS_TO_REVIEW),
        "## 9. Questions for the committee",
        _questions(memo["questions"], result.flags, pack),
        "## 10. Vote",
        f"**Decision: {decision}** · Conviction: {conviction}",
        "**Rationale**",
        _claim_list(vote["rationale"], pack),
        "**What would change the call**",
        _claim_list(vote["what_would_change"], pack),
        "## Appendix A: Evidence cited",
        _appendix_a(_cited_ids(memo), pack),
        "## Appendix B: Citation audit",
        _audit_block(committee),
    ]
    return "\n\n".join(b for b in blocks if b) + "\n"


def _render_data_only_memo(result: AnalysisResult, note: str = "") -> str:
    reasons = [note, result.no_committee_reason] + (result.committee.errors if result.committee else [])
    reasons = [r for r in dict.fromkeys(_clean(r) for r in reasons) if r]
    banner = DATA_ONLY_BANNER + ("\n>\n> " + escape_md(" ".join(reasons)) if reasons else "")
    shown_ids, evidence = _evidence_by_section(result)

    blocks = [
        _title(result),
        f"*Generated {_day(result)} · Data-only mode (no LLM) · {DISCLAIMER}*",
        banner,
        _sources_line(result),
        _data_notes(result),
        "## Financial snapshot",
        financial_snapshot_markdown(result.fin, result.valuation),
        "## Red flags (automated)",
        _flags_block(result),
        "## Evidence by memo section",
        evidence,
        "## Questions for the committee",
        _numbered(default_questions(result.flags)),
        "## Vote",
        NO_VOTE_TEXT,
        "## Appendix A: Evidence cited",
        _appendix_sources(shown_ids, result.pack),
    ]
    return "\n\n".join(b for b in blocks if b) + "\n"


# ----------------------------------------------------------------------------------------------
# Header pieces
# ----------------------------------------------------------------------------------------------

def _title(result: AnalysisResult) -> str:
    ticker = (result.company.ticker if result.company else result.ticker or "").upper()
    name = (result.company.name if result.company else "") or (result.fin.entity_name if result.fin else "")
    if name and ticker and name.upper() != ticker:
        return f"# Investment Committee Memo: {escape_md(name)} ({ticker})"
    return f"# Investment Committee Memo: {escape_md(name or ticker or 'Unknown company')}"


def _day(result: AnalysisResult) -> str:
    return result.generated_at or date.today().isoformat()


def _sources_line(result: AnalysisResult) -> str:
    parts = []
    if result.tenk:
        filing = result.tenk.filing
        parts.append(f"10-K for fiscal year ended {filing.report_date or 'n/a'} "
                     f"(filed {filing.filing_date or 'n/a'}, [EDGAR]({filing.url}))")
    else:
        parts.append("10-K not available")

    if result.transcript_name:
        name = result.transcript_name.replace("`", "'")
        unused = "" if result.transcript_chars else " (not used)"
        parts.append(f"earnings call transcript `{name}`{unused}")
    else:
        parts.append("earnings call transcript not provided")

    parts.append("SEC XBRL company facts" if result.fin else "SEC XBRL company facts not available")

    market = result.market
    if market and (market.price is not None or market.market_cap is not None):
        parts.append(f"market data from {escape_md(market.source)} as of {market.as_of}")
    elif market and market.error:
        parts.append(f"market data unavailable ({escape_md(_excerpt(market.error, 25))})")
    else:
        parts.append("market data not available")
    return "**Sources:** " + "; ".join(parts) + "."


def _data_notes(result: AnalysisResult) -> str:
    """Short notes about missing or degraded inputs ('' when everything loaded)."""
    notes = []
    tenk = result.tenk
    if tenk and tenk.failed:
        titles = ", ".join(tenk.sections[key].title for key in tenk.failed)
        notes.append(f"Could not isolate {titles} in the 10-K, so the rest of the filing text is searched instead.")
    notes += result.errors
    notes = [n for n in dict.fromkeys(_clean(n) for n in notes) if n]
    if not notes:
        return ""
    return "**Data notes**\n\n" + "\n".join(f"- {escape_md(n)}" for n in notes)


# ----------------------------------------------------------------------------------------------
# Committee sections (claims are already citation-checked dicts)
# ----------------------------------------------------------------------------------------------

def _claim_body(claim: dict, pack: EvidencePack | None) -> str:
    """'text [E3 · label] — “quote”', or '' when the claim has no text or no citations."""
    text, ids, quote = claim.get("text"), claim.get("citations"), claim.get("quote")
    if not text or not ids:
        return ""
    body = f"{escape_md(text)} {render_citations(ids, pack)}"
    return f"{body} — “{escape_md(quote)}”" if quote else body


def _claim_list(claims: list[dict], pack: EvidencePack | None, empty: str = NO_CLAIMS) -> str:
    lines = [render_claim(claim, pack) for claim in claims]
    return "\n".join(line for line in lines if line) or empty


def _thesis(claims: list[dict], pack: EvidencePack | None) -> str:
    """The thesis claims, with a note when fewer than THESIS_CLAIMS survived the citation check."""
    text = _claim_list(claims, pack)
    if 0 < len(claims) < THESIS_CLAIMS:
        noun = "claim" if len(claims) == 1 else "claims"
        text += f"\n\n_Only {len(claims)} thesis {noun} survived the citation check._"
    return text


def _risk_list(risks: list[dict], pack: EvidencePack | None) -> str:
    """'- **High** — claim', High risks first (the Chair's order is kept within a severity)."""
    rows = [(_rank(risk["severity"], SEVERITIES), f"- **{risk['severity']}** — {_claim_body(risk, pack)}")
            for risk in risks]
    rows.sort(key=lambda row: row[0])
    return "\n".join(line for _, line in rows) or NO_CLAIMS


def _said_vs_shows(rows: list[dict], pack: EvidencePack | None, has_transcript: bool) -> str:
    """Table: Management said | Numbers show | Verdict | Sources."""
    lines = []
    for row in rows:
        said = row["management_said"]
        if row["quote"]:
            said += f" — “{row['quote']}”"
        sources = render_citations(row["citations"], pack).replace("|", "\\|")
        lines.append(f"| {_cell(said)} | {_cell(row['numbers_show'])} | {row['verdict']} | {sources} |")

    note = "" if has_transcript else ("_No earnings call transcript was used, so “management said” "
                                      "draws on the 10-K MD&A only._\n\n")
    if not lines:
        return note + "_No management-vs-numbers comparisons survived the citation check._"
    header = "| Management said | Numbers show | Verdict | Sources |\n|---|---|---|---|\n"
    return note + header + "\n".join(lines)


def _questions(claims: list[dict], flags: list[RedFlag], pack: EvidencePack | None) -> str:
    """The Chair's cited questions, numbered; the standard questions when none survived."""
    lines = [body for body in (_claim_body(claim, pack) for claim in claims) if body]
    if lines:
        return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, start=1))
    return ("_None of the Chair's questions survived the citation check; these are the standard questions._\n\n"
            + _numbered(default_questions(flags)))


def _audit_block(committee: CommitteeResult) -> str:
    audit = committee.audit
    lines = [
        "Rule: every claim must cite at least one evidence ID from the pack, state only numbers found in the "
        "items it cites, and quote a cited passage verbatim. Claims that fail are dropped; failing quotes are "
        "removed.",
        "\n".join([
            f"- Claims proposed: {audit.claims_total} · kept: {audit.claims_kept} · "
            f"dropped: {audit.claims_total - audit.claims_kept}",
            f"- Invalid evidence IDs removed: {len(set(audit.invalid_ids))}",
            f"- Quotes verified verbatim: {audit.quotes_verified} · quotes removed: {audit.quotes_removed}",
        ]),
        "_Dropped claims are not part of this memo; they are listed in the app's Diagnostics tab._",
    ]
    if committee.errors:
        lines.append("**Committee errors**\n\n" + "\n".join(f"- {escape_md(e)}" for e in committee.errors))
    return "\n\n".join(lines)


def _appendix_a(ids: list[str], pack: EvidencePack | None) -> str:
    """Committee mode, one line per cited item.

    Passages: '- **[E3]** source — section: “first ~30 words…”'. Facts and flags: '- **[F1]** source — text'.
    """
    if pack is None:
        return "_No evidence pack was built._"
    lines = []
    for id_ in sorted(set(ids), key=_id_sort_key):
        item = pack.get(id_)
        if item is None:
            continue
        if item.kind == "passage":
            excerpt = escape_md(_excerpt(item.text, APPENDIX_EXCERPT_WORDS))
            lines.append(f"- **[{id_}]** {_provenance(item)}: “{excerpt}”")
        else:  # financial facts and red flags are self-describing one-liners: show them in full
            lines.append(f"- **[{id_}]** {escape_md(item.source)} — {escape_md(item.text)}")
    return "\n".join(lines) or "_No evidence was cited._"


def _appendix_sources(ids: list[str], pack: EvidencePack | None) -> str:
    """Data-only mode: which document section each passage shown above came from."""
    if pack is None:
        return "_No evidence pack was built._"
    groups: dict[str, list[str]] = {}
    for id_ in sorted(set(ids), key=_id_sort_key):
        item = pack.get(id_)
        if item is not None:
            groups.setdefault(_provenance(item), []).append(id_)
    lines = [f"- **{where}:** {', '.join(group)}" for where, group in groups.items()]
    return "\n".join(lines) or "_No evidence was cited._"


def _provenance(item: EvidenceItem) -> str:
    """'10-K FY2025 (filed 2025-10-31) — Item 1A. Risk Factors'."""
    return " — ".join(escape_md(part) for part in (item.source, item.section) if _clean(part))


def _cited_ids(obj) -> list[str]:
    """Every evidence ID under a 'citations' key anywhere in the memo dict."""
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found += list(value) if key == "citations" else _cited_ids(value)
    elif isinstance(obj, list):
        for value in obj:
            found += _cited_ids(value)
    return found


# ----------------------------------------------------------------------------------------------
# Deterministic sections (both layouts)
# ----------------------------------------------------------------------------------------------

def _history_table(fin: FinancialData) -> str:
    """Markdown table of financials.snapshot_table (rows missing in XBRL show n/a; fin.notes say why)."""
    table = snapshot_table(fin)
    years = [str(y) for y in table.columns]
    rows = ["| " + " | ".join([_cell(label)] + [_cell(v) for v in values]) + " |" for label, values in table.iterrows()]

    ends = [e for e in (fin.table["period_end"] if "period_end" in fin.table else []) if isinstance(e, str) and e]
    caption = "*Annual figures from SEC XBRL company facts (10-K filings)"
    caption += f"; fiscal years ended {ends[0]} to {ends[-1]}.*" if ends else ".*"
    header = "| Fiscal year | " + " | ".join(_cell(y) for y in years) + " |"
    align = "|---|" + "---:|" * len(years)
    return "\n".join([caption, "", header, align, *rows])


def _valuation_table(valuation: dict, currency: str) -> str:
    """Caption (as-of date and basis year) plus a two-column table of price, size, and multiples."""
    basis = valuation.get("basis_year") or "latest FY"
    rows = [
        ("Share price", format_price(valuation.get("price"), currency)),
        ("Market cap", format_money(valuation.get("market_cap"), currency)),
        ("Enterprise value", format_money(valuation.get("enterprise_value"), currency)),
        (f"P/E ({basis} net income)", valuation.get("pe_note") or format_multiple(valuation.get("pe"))),
        (f"EV/Sales ({basis} revenue)", format_multiple(valuation.get("ev_sales"))),
        (f"FCF yield ({basis} FCF)", format_pct(valuation.get("fcf_yield"))),
    ]
    caption = (f"**Valuation** — market data as of {valuation.get('as_of') or 'n/a'}; multiples use {basis} "
               "reported figures, not trailing twelve months.")
    lines = [caption, "", "| Metric | Value |", "|---|---:|"]
    lines += [f"| {_cell(label)} | {_cell(value)} |" for label, value in rows]
    return "\n".join(lines)


def _flags_block(result: AnalysisResult) -> str:
    if result.flags:
        parts = ["\n".join(_flag_line(flag) for flag in result.flags)]
    elif result.fin is None and result.tenk is None:
        parts = ["_Red-flag checks could not run: no SEC financial data or 10-K text was available._"]
    else:
        parts = ["_No automated red flags triggered._"]
    if result.checks_not_run:
        parts.append("_Checks not run: " + escape_md("; ".join(result.checks_not_run)) + "._")
    return "\n\n".join(parts)


def _flag_line(flag: RedFlag) -> str:
    """'- **[R1] High: name** — detail “quote” *Source: ...*'."""
    rest = [escape_md(flag.detail)]
    if _clean(flag.quote):
        rest.append(f"“{escape_md(flag.quote)}”")
    if _clean(flag.source):
        rest.append(f"*Source: {escape_md(flag.source)}*")
    rest = [r for r in rest if r]
    head = f"- **[{flag.id}] {flag.severity}: {escape_md(flag.name)}**"
    return f"{head} — {' '.join(rest)}" if rest else head


def _evidence_by_section(result: AnalysisResult) -> tuple[list[str], str]:
    """Data-only mode: top passages per memo section. Returns (ids shown, markdown).

    A passage retrieved for several sections is quoted once and referenced afterwards.
    """
    by_section = result.pack.section_ids if result.pack else result.retrieved
    keys = list(MEMO_QUERIES) + [k for k in by_section if k not in MEMO_QUERIES]
    first_seen: dict[str, str] = {}  # passage id -> title of the section that quoted it
    blocks = []
    for key in keys:
        title = _section_title(key)
        lines = []
        for uid, tag, text in _section_passages(result, key):
            if uid in first_seen:
                lines.append(f"- {tag} — same passage as under *{first_seen[uid]}*")
            else:
                first_seen[uid] = title
                lines.append(f"- {tag} “{escape_md(_excerpt(text, EVIDENCE_EXCERPT_WORDS))}”")
        blocks.append(f"### {title}\n\n" + ("\n".join(lines) or "_No relevant passages found._"))

    if not first_seen:
        return [], "_No document passages were retrieved (no 10-K text or transcript available)._"
    if not result.transcript_chars:
        blocks.insert(0, "_No earnings call transcript was used, so all passages come from the 10-K._")
    return list(first_seen), "\n\n".join(blocks)


def _section_passages(result: AnalysisResult, key: str) -> list[tuple[str, str, str]]:
    """(id, inline tag, text) for each passage retrieved for one memo section."""
    pack = result.pack
    if pack:
        items = [pack.get(id_) for id_ in pack.section_ids.get(key, [])]
        return [(item.id, render_citations([item.id], pack), item.text) for item in items if item]
    # No evidence pack (an earlier step failed): fall back to the raw retrieval results.
    chunks = [chunk for chunk, _score in result.retrieved.get(key, [])]
    return [(chunk.id, f"[{escape_md(chunk.short_label)}]", chunk.text) for chunk in chunks]


def _section_title(key: str) -> str:
    """retrieval.MEMO_QUERIES title, else the key humanized ('management_outlook' -> 'Management outlook')."""
    query = MEMO_QUERIES.get(key)
    title = query.get("title") if isinstance(query, dict) else None
    return title or key.replace("_", " ").capitalize()


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {escape_md(text)}" for i, text in enumerate(items, start=1))


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------

def _rank(value: str, order: tuple[str, ...]) -> int:
    return order.index(value) if value in order else len(order)


def _id_sort_key(id_: str) -> tuple[str, int]:
    """Natural sort: E2 before E10."""
    match = re.fullmatch(r"([A-Z]+)(\d+)", id_)
    return (match.group(1), int(match.group(2))) if match else (id_, 0)


def _clean(text) -> str:
    """One line of plain text: None -> '', all whitespace (including newlines) collapsed."""
    return " ".join(str(text).split()) if text is not None else ""


def _cell(text) -> str:
    """Text for a Markdown table cell: one line, every pipe escaped exactly once, never empty."""
    return re.sub(r"\\*\|", r"\\|", escape_md(text)) or "—"


def _excerpt(text, max_words: int) -> str:
    """First max_words words, with '…' when cut."""
    words = _clean(text).split(" ")
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(".,;:") + "…"
