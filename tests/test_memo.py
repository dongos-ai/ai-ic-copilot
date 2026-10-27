"""Offline tests for copilot.memo: section order, citations, escaping, both layouts, and missing data.

Committee memos are built by hand in the shape committee.enforce_citations returns (every claim a
{"text", "citations", "quote"} dict with valid ids, canonical vote fields)."""
from __future__ import annotations

import json
import re
from datetime import date

import pandas as pd

from copilot import memo as memo_mod
from copilot.committee import CitationAudit, CommitteeResult, EvidenceItem, EvidencePack
from copilot.edgar import CompanyInfo, Filing, Section, TenK
from copilot.financials import FinancialData, format_price
from copilot.market import MarketData
from copilot.memo import (
    default_questions, escape_md, financial_snapshot_markdown, memo_filename, render_citations, render_claim,
    render_memo,
)
from copilot.pipeline import AnalysisResult
from copilot.red_flags import RedFlag
from copilot.retrieval import Chunk

TODAY = "2026-10-19"

LLM_HEADINGS = [
    "# Investment Committee Memo: Acme Corp (ACME)",
    "## 1. Investment thesis",
    "## 2. Business overview",
    "## 3. Financial snapshot",
    "## 4. Bull case",
    "## 5. Bear case",
    "## 6. Top risks",
    "## 7. What management said vs. what the numbers show",
    "## 8. Red flags",
    "### Automated checks",
    "### Committee view",
    "## 9. Questions for the committee",
    "## 10. Vote",
    "## Appendix A: Evidence cited",
    "## Appendix B: Citation audit",
]
DATA_ONLY_HEADINGS = [
    "# Investment Committee Memo: Acme Corp (ACME)",
    "## Financial snapshot",
    "## Red flags (automated)",
    "## Evidence by memo section",
    "## Questions for the committee",
    "## Vote",
    "## Appendix A: Evidence cited",
]


# ------------------------------------------------------------------------------------------------
# Builders (inputs constructed by hand; no other module's logic is exercised)
# ------------------------------------------------------------------------------------------------

def make_fin() -> FinancialData:
    table = pd.DataFrame(
        {
            "period_end": ["2023-12-31", "2024-12-31", "2025-12-31"],
            "revenue": [100e6, 110e6, 121e6],
            "gross_profit": [float("nan")] * 3,
            "operating_income": [20e6, 21e6, 18e6],
            "net_income": [15e6, 16e6, 12e6],
            "operating_cash_flow": [18e6, 17e6, 9e6],
            "capex": [3e6, 4e6, 5e6],
            "fcf": [15e6, 13e6, 4e6],
            "cash": [30e6, 25e6, 20e6],
            "debt": [10e6, 12e6, 15e6],
            "sbc": [2e6, 3e6, 4e6],
            "receivables": [10e6, 13e6, 18e6],
            "inventory": [float("nan")] * 3,
            "diluted_shares": [50e6, 51e6, 52e6],
            "gross_margin": [float("nan")] * 3,
            "operating_margin": [0.20, 0.191, 0.149],
            "net_margin": [0.15, 0.145, 0.099],
            "fcf_margin": [0.15, 0.118, 0.033],
            "revenue_growth": [float("nan"), 0.10, 0.10],
            "sbc_pct_revenue": [0.02, 0.027, 0.033],
        },
        index=["FY2023", "FY2024", "FY2025"],
    )
    return FinancialData(
        table=table,
        tags_used={"revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]},
        currency="USD",
        entity_name="Acme Corp",
        notes=["Total debt uses LongTermDebt + CommercialPaper."],
    )


def make_valuation() -> dict:
    return {
        "price": 12.345, "market_cap": 640e6, "enterprise_value": 635e6, "pe": 53.333, "ev_sales": 5.25,
        "fcf_yield": 0.00625, "basis_year": "FY2025", "as_of": TODAY,
        "notes": ["Short-term investments not tagged for FY2025 (they were in earlier years)."],
    }


def make_flags() -> list[RedFlag]:
    return [
        RedFlag("R1", "Receivables growing faster than revenue", "Medium",
                "Receivables +38.5% vs revenue +10.0% in FY2025 (DSO 43 → 54 days).",
                "SEC XBRL company facts: AccountsReceivableNetCurrent, Revenues (FY2024–FY2025)"),
        RedFlag("R2", "Material weakness in internal controls", "High",
                "The 10-K reports a material weakness.", "10-K text: Item 9A",
                quote="we identified a material weakness in our internal control over financial reporting"),
    ]


def make_pack() -> EvidencePack:
    items = [
        EvidenceItem("E1", "passage", "10-K FY2025 (filed 2026-02-20)", "Item 1A. Risk Factors", "10-K Item 1A",
                     "Our largest customer accounted for 18% of revenue and may reduce orders at any time. " * 5),
        EvidenceItem("E2", "passage", "Earnings call: acme_q4.txt", "Q&A", "Call Q&A",
                     "We expect margins to expand next year as pricing actions take hold."),
        EvidenceItem("E3", "passage", "10-K FY2025 (filed 2026-02-20)", "Item 1. Business", "10-K Item 1",
                     "Acme makes industrial widgets for factories worldwide."),
        EvidenceItem("F1", "financial", "SEC XBRL company facts", "Revenue", "XBRL Revenue",
                     "Revenue (USD): FY2023 $100.0M; FY2024 $110.0M; FY2025 $121.0M [tag: Revenues]"),
        EvidenceItem("F2", "financial", "SEC XBRL company facts", "Operating margin", "XBRL Operating margin",
                     "Operating margin: FY2023 20.0%; FY2024 19.1%; FY2025 14.9%"),
        EvidenceItem("R1", "red_flag", "Rule-based check", "Receivables growing faster than revenue",
                     "Red-flag check", "Medium: Receivables growing faster than revenue — Receivables +38.5%"),
        EvidenceItem("R2", "red_flag", "Rule-based check", "Material weakness in internal controls",
                     "Red-flag check", "High: Material weakness in internal controls"),
    ]
    section_ids = {"business": ["E3"], "risks": ["E1"], "management_outlook": ["E2"], "growth": []}
    return EvidencePack(items=items, section_ids=section_ids)


def claim(text: str, *ids: str, quote: str = "") -> dict:
    return {"text": text, "citations": list(ids), "quote": quote}


def svs(said: str, shows: str, verdict: str, *ids: str, quote: str = "") -> dict:
    return {"management_said": said, "numbers_show": shows, "verdict": verdict, "citations": list(ids),
            "quote": quote}


def make_committee_memo() -> dict:
    return {
        "thesis": [claim("Revenue compounded at 10% a year.", "F1"),
                   claim("Margins are under pressure.", "F2"),
                   claim("Customer concentration is the key debate.", "E1",
                         quote="Our largest customer accounted for 18% of revenue")],
        "business_overview": [claim("Acme sells industrial widgets.", "E3")],
        "bull_case": [claim("Pricing actions should lift margins.", "E2", quote="We expect margins to expand next year")],
        "bear_case": [claim("Operating margin fell to 14.9% in FY2025.", "F2")],
        "top_risks": [{**claim("Lower orders from the largest customer.", "E1"), "severity": "Medium"},
                      {**claim("Control weakness could delay filings.", "R2"), "severity": "High"}],
        "said_vs_shows": [svs("Margins will expand", "Operating margin fell 4 pts", "Contradicted", "E2", "F2")],
        "red_flags": [claim("Receivables growth looks real, not seasonal.", "R1")],
        "questions": [claim("Why did DSO rise?", "R1"), claim("What is the customer contract term?", "E1")],
        "vote": {"decision": "Hold", "conviction": "Medium",
                 "rationale": [claim("Growth is solid but margins are falling.", "F1", "F2")],
                 "what_would_change": [claim("Two quarters of margin expansion.", "F2")]},
    }


def make_tenk(failed: tuple[str, ...] = ()) -> TenK:
    company = CompanyInfo("ACME", 12345, "Acme Corp")
    filing = Filing(12345, "10-K", "0000012345-26-000010", "2026-02-20", "2025-12-31", "acme-20251231.htm")
    sections = {}
    for key, title in (("item1", "Item 1. Business"), ("item1a", "Item 1A. Risk Factors"),
                       ("item7", "Item 7. Management's Discussion and Analysis")):
        ok = key not in failed
        sections[key] = Section(key, title, "text" if ok else None, 0 if ok else None, 4 if ok else None)
    return TenK(company=company, filing=filing, text="full text", sections=sections)


def make_result(memo: dict | None = None, **overrides) -> AnalysisResult:
    committee = None
    if memo is not None:
        audit = CitationAudit(claims_total=12, claims_kept=11, dropped=["Revenue will triple."],
                              invalid_ids=["E99"], quotes_verified=2, quotes_removed=1)
        committee = CommitteeResult(bull={}, bear={}, risk={}, memo=memo, audit=audit, errors=[], raw={},
                                    model_label="anthropic / claude-opus-5-5")
    values = dict(
        ticker="ACME", generated_at=TODAY, company=CompanyInfo("ACME", 12345, "Acme Corp"), tenk=make_tenk(),
        fin=make_fin(), market=MarketData("ACME", 12.345, 640e6, "USD", TODAY), valuation=make_valuation(),
        flags=make_flags(), checks_not_run=["Inventory vs revenue (no inventory data)"],
        transcript_name="acme_q4.txt", transcript_chars=5000, pack=make_pack(), committee=committee,
        llm_label="anthropic / claude-opus-5-5",
    )
    values.update(overrides)
    return AnalysisResult(**values)


def heading_lines(markdown: str) -> list[str]:
    return [line for line in markdown.splitlines() if line.startswith("#")]


def section(markdown: str, heading: str) -> str:
    """Text between a heading and the next heading of the same or higher level."""
    level = heading.split(" ")[0]
    start = markdown.index(heading + "\n") + len(heading)
    rest = markdown[start:]
    match = re.search(rf"^#{{1,{len(level)}}} ", rest, flags=re.M)
    return rest[: match.start()] if match else rest


def table_column_counts(markdown: str) -> list[int]:
    """Column count of every Markdown table line (escaped pipes are not separators)."""
    return [len(re.split(r"(?<!\\)\|", line)) for line in markdown.splitlines() if line.startswith("|")]


# ------------------------------------------------------------------------------------------------
# Citations and claims
# ------------------------------------------------------------------------------------------------

def test_render_citations_uses_short_labels():
    pack = make_pack()
    assert render_citations(["E1", "F1"], pack) == "[E1 · 10-K Item 1A][F1 · XBRL Revenue]"
    assert render_citations(["E1", "E1"], pack) == "[E1 · 10-K Item 1A]"         # deduped
    assert render_citations(["E99"], pack) == "[E99]"                           # unknown id stays visible
    assert render_citations(["E1"], None) == "[E1]"
    assert render_citations([], pack) == ""
    assert render_citations(None, pack) == ""


def test_render_claim_formats_text_citations_and_quote():
    pack = make_pack()
    line = render_claim(claim("Customers are concentrated.", "E1", quote="Our largest customer accounted for 18%"), pack)
    assert line == "- Customers are concentrated. [E1 · 10-K Item 1A] — “Our largest customer accounted for 18%”"
    assert render_claim(claim("Revenue grew.", "F1"), pack) == "- Revenue grew. [F1 · XBRL Revenue]"


def test_render_claim_drops_claims_without_text_or_citations():
    pack = make_pack()
    assert render_claim(claim("No source for this."), pack) == ""
    assert render_claim(claim("", "E1"), pack) == ""
    assert render_claim({"citations": ["E1"]}, pack) == ""


def test_render_claim_keeps_markdown_clean():
    line = render_claim(claim("Revenue was $121.0M\n- up ~10%", "F1"), make_pack())
    assert line == "- Revenue was \\$121.0M - up \\~10% [F1 · XBRL Revenue]"


def test_links_and_images_in_model_text_are_not_rendered():
    line = render_claim(claim("Gross margin 46.9% ![x](https://attacker.example/p.png) [filing](https://evil.example)",
                              "F1"), make_pack())
    assert "](http" not in line and "attacker.example" in line           # text stays readable, not a link
    assert escape_md("a](b") == "a]\\(b"


def test_table_cells_escape_every_pipe_once():
    assert memo_mod._cell("a|b") == "a\\|b"
    assert memo_mod._cell("a\\|b") == "a\\|b"                              # an escaped pipe is not unescaped
    memo = make_committee_memo()
    memo["said_vs_shows"] = [svs("Demand is strong \\| Consistent \\| see filing", "Revenue grew", "Mixed",
                                 "E2", "F1")]
    table = section(render_memo(make_result(memo)), "## 7. What management said vs. what the numbers show")
    rows = [line for line in table.splitlines() if line.startswith("|")]
    assert set(table_column_counts("\n".join(rows))) == {6}             # 4 columns, no shifted cells


# ------------------------------------------------------------------------------------------------
# Committee (LLM) layout
# ------------------------------------------------------------------------------------------------

def test_llm_layout_headings_in_order():
    markdown = render_memo(make_result(make_committee_memo()))
    assert heading_lines(markdown) == LLM_HEADINGS


def test_llm_layout_header_vote_and_sources():
    markdown = render_memo(make_result(make_committee_memo()))
    lines = markdown.splitlines()
    assert lines[2] == (f"*Generated {TODAY} · anthropic / claude-opus-5-5 · "
                        "Draft for discussion — not investment advice.*")
    assert lines[4] == "> **Committee vote: HOLD** · Conviction: Medium"
    sources = lines[6]
    edgar_url = "https://www.sec.gov/Archives/edgar/data/12345/000001234526000010/acme-20251231.htm"
    assert sources.startswith(f"**Sources:** 10-K for fiscal year ended 2025-12-31 (filed 2026-02-20, [EDGAR]({edgar_url}))")
    assert "earnings call transcript `acme_q4.txt`" in sources
    assert "SEC XBRL company facts" in sources
    assert f"market data from Yahoo Finance via yfinance as of {TODAY}" in sources


def test_thesis_shows_at_most_three_claims():
    memo = make_committee_memo()
    memo["thesis"] = [claim(f"Thesis point {word}.", "F1") for word in ("one", "two", "three")]
    memo["thesis"] += [claim("Thesis point four.", "E3"), claim("Thesis point five.", "E3")]
    memo["business_overview"] = []                             # E3 is now cited only by the hidden claims
    markdown = render_memo(make_result(memo))
    thesis = section(markdown, "## 1. Investment thesis").strip().splitlines()
    assert [line for line in thesis if line.startswith("- ")] == [
        "- Thesis point one. [F1 · XBRL Revenue]", "- Thesis point two. [F1 · XBRL Revenue]",
        "- Thesis point three. [F1 · XBRL Revenue]"]
    assert "[E3]" not in section(markdown, "## Appendix A: Evidence cited")

    memo["thesis"] = memo["thesis"][:2]
    thesis = section(render_memo(make_result(memo)), "## 1. Investment thesis")
    assert thesis.count("\n- ") == 2
    assert "_Only 2 thesis claims survived the citation check._" in thesis


def test_llm_layout_claim_sections():
    markdown = render_memo(make_result(make_committee_memo()))
    thesis = section(markdown, "## 1. Investment thesis").strip().splitlines()
    assert thesis == [
        "- Revenue compounded at 10% a year. [F1 · XBRL Revenue]",
        "- Margins are under pressure. [F2 · XBRL Operating margin]",
        "- Customer concentration is the key debate. [E1 · 10-K Item 1A] — "
        "“Our largest customer accounted for 18% of revenue”",
    ]
    # every bullet in the claim sections carries at least one citation
    for heading in ("## 1. Investment thesis", "## 2. Business overview", "## 4. Bull case", "## 5. Bear case",
                    "## 6. Top risks", "### Committee view"):
        bullets = [l for l in section(markdown, heading).splitlines() if l.startswith("- ")]
        assert bullets and all(re.search(r"\[[EFR]\d+ · ", b) for b in bullets), heading


def test_top_risks_sorted_by_severity():
    markdown = render_memo(make_result(make_committee_memo()))
    risks = section(markdown, "## 6. Top risks").strip().splitlines()
    assert risks[0] == "- **High** — Control weakness could delay filings. [R2 · Red-flag check]"
    assert risks[1].startswith("- **Medium** — Lower orders from the largest customer.")


def test_said_vs_shows_table_and_escaping():
    memo = make_committee_memo()
    memo["said_vs_shows"].append(svs("Demand is strong | backlog\nis full", "", "Mixed", "E2", "F1",
                                     quote="We expect margins"))
    markdown = render_memo(make_result(memo))
    table = section(markdown, "## 7. What management said vs. what the numbers show").strip().splitlines()
    assert table[0] == "| Management said | Numbers show | Verdict | Sources |"
    assert table[2] == ("| Margins will expand | Operating margin fell 4 pts | Contradicted | "
                        "[E2 · Call Q&A][F2 · XBRL Operating margin] |")
    assert table[3] == ("| Demand is strong \\| backlog is full — “We expect margins” | — | Mixed | "
                        "[E2 · Call Q&A][F1 · XBRL Revenue] |")
    assert len(table) == 4
    assert len(set(table_column_counts("\n".join(table)))) == 1


def test_said_vs_shows_notes_missing_transcript():
    markdown = render_memo(make_result(make_committee_memo(), transcript_name=None, transcript_chars=0))
    body = section(markdown, "## 7. What management said vs. what the numbers show")
    assert "No earnings call transcript was used" in body
    assert "earnings call transcript not provided" in markdown


def test_red_flags_section_lists_checks_and_committee_view():
    markdown = render_memo(make_result(make_committee_memo()))
    automated = section(markdown, "### Automated checks")
    assert ("- **[R1] Medium: Receivables growing faster than revenue** — Receivables +38.5% vs revenue +10.0% "
            "in FY2025 (DSO 43 → 54 days). *Source: SEC XBRL company facts: AccountsReceivableNetCurrent, "
            "Revenues (FY2024–FY2025)*") in automated
    assert "“we identified a material weakness in our internal control over financial reporting”" in automated
    assert "_Checks not run: Inventory vs revenue (no inventory data)._" in automated
    committee_view = section(markdown, "### Committee view")
    assert "- Receivables growth looks real, not seasonal. [R1 · Red-flag check]" in committee_view


def test_committee_view_without_red_flags_is_neutral():
    memo = make_committee_memo()
    memo["red_flags"] = []
    no_flags = section(render_memo(make_result(memo, flags=[])), "### Committee view").strip()
    assert no_flags == "_No automated red flags to review._"         # nothing was there to drop
    with_flags = section(render_memo(make_result(memo)), "### Committee view").strip()
    assert with_flags == "_No claims in this section survived the citation check._"


def test_questions_and_vote_section_are_cited_claims():
    markdown = render_memo(make_result(make_committee_memo()))
    assert section(markdown, "## 9. Questions for the committee").strip().splitlines() == [
        "1. Why did DSO rise? [R1 · Red-flag check]", "2. What is the customer contract term? [E1 · 10-K Item 1A]"]
    vote = section(markdown, "## 10. Vote")
    assert "**Decision: HOLD** · Conviction: Medium" in vote
    assert "- Growth is solid but margins are falling. [F1 · XBRL Revenue][F2 · XBRL Operating margin]" in vote
    assert "**What would change the call**\n\n- Two quarters of margin expansion. [F2 · XBRL Operating margin]" in vote


def test_questions_fall_back_to_defaults():
    memo = make_committee_memo()
    memo["questions"] = []
    memo["vote"]["what_would_change"] = []
    markdown = render_memo(make_result(memo))
    questions = section(markdown, "## 9. Questions for the committee").strip().splitlines()
    assert questions[0] == ("_None of the Chair's questions survived the citation check; "
                            "these are the standard questions._")
    numbered = [q for q in questions if re.match(r"\d+\. ", q)]
    assert len(numbered) == len(default_questions(make_flags()))
    assert "**What would change the call**\n\n_No claims in this section survived" in section(markdown, "## 10. Vote")


def test_appendix_a_lists_only_cited_ids():
    markdown = render_memo(make_result(make_committee_memo()))
    appendix = section(markdown, "## Appendix A: Evidence cited").strip().splitlines()
    ids = [re.match(r"- \*\*\[(\w+)\]\*\*", line).group(1) for line in appendix]
    assert ids == ["E1", "E2", "E3", "F1", "F2", "R1", "R2"]   # natural sort by type, then number
    pack = make_pack()
    pack.items.append(EvidenceItem("E10", "passage", "10-K", "Item 7", "10-K Item 7", "never cited"))
    markdown = render_memo(make_result(make_committee_memo(), pack=pack))
    assert "[E10]" not in markdown
    first = appendix[0]
    assert first.startswith("- **[E1]** 10-K FY2025 (filed 2026-02-20) — Item 1A. Risk Factors: “Our largest customer")
    assert first.endswith("…”") and len(first.split("“")[1].split()) == 30
    assert "- **[F1]** SEC XBRL company facts — Revenue (USD): FY2023 \\$100.0M" in markdown


def test_appendix_a_skips_uncited_pack_items():
    memo = make_committee_memo()
    memo["business_overview"] = []                             # E3 is now cited nowhere
    markdown = render_memo(make_result(memo))
    appendix = section(markdown, "## Appendix A: Evidence cited")
    assert "[E3]" not in appendix and "[E1]" in appendix
    assert "_No claims in this section survived the citation check._" in section(markdown, "## 2. Business overview")


def test_appendix_b_citation_audit_shows_counts_only():
    result = make_result(make_committee_memo())
    result.committee.errors.append("Bear analyst failed: timeout")
    markdown = render_memo(result)
    audit = section(markdown, "## Appendix B: Citation audit")
    assert "- Claims proposed: 12 · kept: 11 · dropped: 1" in audit
    assert "- Invalid evidence IDs removed: 1" in audit
    assert "- Quotes verified verbatim: 2 · quotes removed: 1" in audit
    assert "listed in the app's Diagnostics tab" in audit
    assert "Revenue will triple." not in markdown and "E99" not in markdown   # dropped content never shown
    assert "- Bear analyst failed: timeout" in audit


def test_llm_memo_has_no_unescaped_dollar_signs_and_consistent_tables():
    markdown = render_memo(make_result(make_committee_memo()))
    assert not re.search(r"(?<!\\)\$", markdown)               # bare "$" would render as LaTeX in Streamlit
    for block in markdown.split("\n\n"):
        counts = table_column_counts(block)
        assert len(set(counts)) <= 1, block


def test_malformed_memo_falls_back_to_data_only():
    odd = {"thesis": "A bare string thesis", "questions": [None, "Plain question?"], "vote": "Buy"}
    markdown = render_memo(make_result(odd))
    assert "**Data-only mode**" in markdown and "The committee memo could not be rendered" in markdown
    assert "A bare string thesis" not in markdown and "Plain question?" not in markdown


def test_memo_without_kept_claims_uses_data_only_layout():
    result = make_result(make_committee_memo(),
                         no_committee_reason="None of the Chair's claims survived the citation check.")
    result.committee.audit = CitationAudit(claims_total=4, claims_kept=0)
    result.committee.errors = ["Vote withheld: none of the Chair's rationale claims survived the citation check."]
    markdown = render_memo(result)
    assert "**Data-only mode**" in markdown and "Committee vote" not in markdown
    assert ("> None of the Chair's claims survived the citation check. Vote withheld: none of the Chair's "
            "rationale claims survived the citation check.") in markdown
    assert "Revenue compounded" not in markdown and result.vote is None


def test_committee_render_failure_falls_back_to_data_only(monkeypatch):
    def boom(result, memo):
        raise ValueError("bad shape")

    monkeypatch.setattr(memo_mod, "_render_committee_memo", boom)
    markdown = render_memo(make_result(make_committee_memo()))
    assert "**Data-only mode**" in markdown
    assert "The committee memo could not be rendered (bad shape)." in markdown


# ------------------------------------------------------------------------------------------------
# Data-only layout
# ------------------------------------------------------------------------------------------------

def test_data_only_layout(monkeypatch):
    monkeypatch.setattr(memo_mod, "MEMO_QUERIES", {
        "business": {"title": "Business overview"}, "growth": {"title": "Growth drivers"},
        "risks": {"title": "Key risks"}, "management_outlook": {"title": "Management outlook & guidance"},
    })
    markdown = render_memo(make_result(None))
    headings = heading_lines(markdown)
    assert [h for h in headings if not h.startswith("### ")] == DATA_ONLY_HEADINGS
    assert [h for h in headings if h.startswith("### ")] == [
        "### Business overview", "### Growth drivers", "### Key risks", "### Management outlook & guidance"]
    assert markdown.splitlines()[2].startswith(f"*Generated {TODAY} · Data-only mode (no LLM)")
    assert memo_mod.DATA_ONLY_BANNER in markdown
    assert "## Vote\n\nNot issued — requires the LLM committee." in markdown
    assert "Committee vote" not in markdown
    evidence = section(markdown, "## Evidence by memo section")
    assert "- [E3 · 10-K Item 1] “Acme makes industrial widgets for factories worldwide.”" in evidence
    assert "### Growth drivers\n\n_No relevant passages found._" in evidence
    long_line = next(l for l in evidence.splitlines() if l.startswith("- [E1"))
    assert long_line.endswith("…”") and len(long_line.split("“")[1].split()) == 60
    # Appendix A: where each passage shown came from, grouped by document section
    appendix = section(markdown, "## Appendix A: Evidence cited").strip().splitlines()
    assert appendix == ["- **10-K FY2025 (filed 2026-02-20) — Item 1A. Risk Factors:** E1",
                        "- **Earnings call: acme_q4.txt — Q&A:** E2",
                        "- **10-K FY2025 (filed 2026-02-20) — Item 1. Business:** E3"]
    assert "No earnings call transcript was used" not in evidence
    questions = section(markdown, "## Questions for the committee").strip().splitlines()
    assert questions[-1].startswith(f"{len(questions)}. Which single risk")


def test_data_only_quotes_repeated_passages_once(monkeypatch):
    monkeypatch.setattr(memo_mod, "MEMO_QUERIES", {"risks": {"title": "Key risks"}, "headwinds": {"title": "Headwinds"}})
    pack = make_pack()
    pack.section_ids = {"risks": ["E1"], "headwinds": ["E1", "E3"]}
    markdown = render_memo(make_result(None, pack=pack, transcript_name=None, transcript_chars=0))
    evidence = section(markdown, "## Evidence by memo section")
    assert evidence.count("- [E1 · 10-K Item 1A] “Our largest customer") == 1
    assert "- [E1 · 10-K Item 1A] — same passage as under *Key risks*" in evidence
    assert evidence.strip().startswith("_No earnings call transcript was used, so all passages come from the 10-K._")


def test_data_only_humanizes_section_keys_without_memo_queries(monkeypatch):
    monkeypatch.setattr(memo_mod, "MEMO_QUERIES", {})
    markdown = render_memo(make_result(None))
    assert "### Management outlook" in markdown
    assert "### Business" in markdown


def test_data_only_banner_states_the_reason():
    markdown = render_memo(make_result(None, no_committee_reason="No LLM provider was selected."))
    assert memo_mod.DATA_ONLY_BANNER + "\n>\n> No LLM provider was selected." in markdown
    assert "no LLM key" not in markdown


def test_data_only_when_committee_failed_shows_reason():
    result = make_result(make_committee_memo())
    result.committee.memo = None
    result.committee.errors = ["Chair failed: rate limited"]
    markdown = render_memo(result)
    assert "**Data-only mode**" in markdown
    assert "> Chair failed: rate limited" in markdown
    assert "## Vote\n\nNot issued — requires the LLM committee." in markdown


def test_data_only_without_pack_uses_retrieved_chunks(monkeypatch):
    monkeypatch.setattr(memo_mod, "MEMO_QUERIES", {"risks": {"title": "Key risks"}})
    chunk = Chunk("C7", "10-K FY2025", "Item 1A. Risk Factors", "risk", "Supply chain disruption could hurt results.")
    markdown = render_memo(make_result(None, pack=None, retrieved={"risks": [(chunk, 3.2)]}))
    assert "- [10-K Item 1A] “Supply chain disruption could hurt results.”" in markdown
    assert "_No evidence pack was built._" in markdown


def test_minimal_result_never_crashes():
    result = AnalysisResult(ticker="XOM", generated_at=TODAY, errors=["No 10-K found for XOM."])
    markdown = render_memo(result)
    assert markdown.startswith("# Investment Committee Memo: XOM\n")
    for text in ("10-K not available", "earnings call transcript not provided", "SEC XBRL company facts not available",
                 "market data not available", "- No 10-K found for XOM.",
                 "_No annual financial data was available from SEC XBRL company facts._",
                 "_Valuation not available (no market data)._",
                 "_Red-flag checks could not run: no SEC financial data or 10-K text was available._",
                 "_No document passages were retrieved", "## Vote\n\nNot issued", "_No evidence pack was built._"):
        assert text in markdown, text


def test_missing_pieces_render_as_short_notes():
    result = make_result(
        None, tenk=make_tenk(failed=("item7",)), transcript_chars=0,
        market=MarketData("ACME", None, None, None, TODAY, error="Yahoo rate limit | try later"), valuation=None,
    )
    markdown = render_memo(result)
    assert ("- Could not isolate Item 7. Management's Discussion and Analysis in the 10-K, so the rest of the "
            "filing text is searched instead.") in markdown
    assert "earnings call transcript `acme_q4.txt` (not used)" in markdown
    assert "market data unavailable (Yahoo rate limit | try later)" in markdown


# ------------------------------------------------------------------------------------------------
# Financial snapshot
# ------------------------------------------------------------------------------------------------

def test_financial_snapshot_table():
    markdown = financial_snapshot_markdown(make_fin(), make_valuation())
    lines = markdown.splitlines()
    assert lines[0] == ("*Annual figures from SEC XBRL company facts (10-K filings); "
                        "fiscal years ended 2023-12-31 to 2025-12-31.*")
    assert "| Fiscal year | FY2023 | FY2024 | FY2025 |" in lines
    assert "|---|---:|---:|---:|" in lines
    assert "| Revenue | \\$100.0M | \\$110.0M | \\$121.0M |" in lines
    assert "| Revenue growth | n/a | 10.0% | 10.0% |" in lines
    assert "| Operating margin | 20.0% | 19.1% | 14.9% |" in lines
    assert "| Gross margin | n/a | n/a | n/a |" in lines                # every row is shown; fin.notes say why
    assert "| Short-term investments | n/a | n/a | n/a |" in lines
    assert "Not reported in XBRL for these years" not in markdown     # financials owns the missing-data note
    assert len(set(table_column_counts(markdown.split("**Valuation**")[0]))) == 1


def test_financial_snapshot_valuation_and_footnotes():
    markdown = financial_snapshot_markdown(make_fin(), make_valuation())
    caption = ("**Valuation** — market data as of 2026-10-19; multiples use FY2025 reported figures, "
               "not trailing twelve months.")
    for line in (caption, "| Share price | \\$12.35 |", "| Market cap | \\$640.0M |",
                 "| Enterprise value | \\$635.0M |", "| P/E (FY2025 net income) | 53.3x |",
                 "| EV/Sales (FY2025 revenue) | 5.2x |", "| FCF yield (FY2025 FCF) | 0.6% |"):
        assert line in markdown, line
    notes = markdown.split("**Notes**")[1]
    assert "- Free cash flow = operating cash flow − capex. Cash & equivalents excludes short-term" in notes
    assert "- EV = market cap + total debt − cash − short-term investments." in notes
    assert "- Short-term investments not tagged for FY2025 (they were in earlier years)." in notes  # valuation
    assert "- Total debt uses LongTermDebt + CommercialPaper." in notes             # financials note
    assert "Total debt = long-term debt" not in notes                              # one owner per note
    assert "Revenue XBRL tag(s): `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`." in notes


def test_financial_snapshot_is_deterministic_and_ignores_llm():
    first = financial_snapshot_markdown(make_fin(), make_valuation())
    assert first == financial_snapshot_markdown(make_fin(), make_valuation())
    markdown = render_memo(make_result(make_committee_memo()))
    assert first in markdown


def test_financial_snapshot_missing_inputs():
    assert financial_snapshot_markdown(None, None) == (
        "_No annual financial data was available from SEC XBRL company facts._\n\n"
        "_Valuation not available (no market data)._")
    empty = FinancialData(table=pd.DataFrame())
    assert "No annual financial data" in financial_snapshot_markdown(empty, None)
    partial = {"price": None, "market_cap": None, "pe": float("nan"), "ev_sales": "bad",
               "notes": ["Market data unavailable."]}
    markdown = financial_snapshot_markdown(None, partial)
    assert "| Share price | n/a |" in markdown and "| P/E (latest FY net income) | n/a |" in markdown
    assert "- Market data unavailable." in markdown


def test_valuation_shows_loss_pe_and_currency():
    valuation = {**make_valuation(), "pe": None, "pe_note": "n/m (loss)", "currency": "EUR"}
    markdown = financial_snapshot_markdown(make_fin(), valuation)
    assert "| P/E (FY2025 net income) | n/m (loss) |" in markdown
    assert f"| Share price | {format_price(12.345, 'EUR')} |" in markdown and "| Market cap | EUR 640.0M |" in markdown


def test_financial_snapshot_escapes_table_cells():
    fin = make_fin()
    fin.table.index = ["FY2023", "FY2024", "FY2025 | restated"]
    markdown = financial_snapshot_markdown(fin, None)
    assert "| Fiscal year | FY2023 | FY2024 | FY2025 \\| restated |" in markdown
    assert len(set(table_column_counts(markdown))) == 1


# ------------------------------------------------------------------------------------------------
# Default questions and file name
# ------------------------------------------------------------------------------------------------

def test_default_questions_from_flags():
    questions = default_questions(make_flags())
    assert questions[0].startswith("What caused the material weakness") and questions[0].endswith("(see R2)")
    assert questions[1].startswith("Why are receivables growing faster than revenue") and questions[1].endswith("(see R1)")
    assert questions[2:] == memo_mod.STANDARD_QUESTIONS


def test_default_questions_dedupe_cap_and_fallback():
    assert default_questions([]) == memo_mod.STANDARD_QUESTIONS
    flags = [RedFlag("R1", "Operating margin decline", "Medium", "", ""),
             RedFlag("R2", "Sustained margin compression", "Medium", "", ""),
             RedFlag("R3", "Something unusual", "Low", "", "")]
    questions = default_questions(flags)
    assert questions[0].endswith("(see R1, R2)")
    assert questions[1] == "What explains the “Something unusual” flag, and is it temporary or structural? (see R3)"
    many = [RedFlag(f"R{i}", f"Odd flag {i}", "Medium", "", "") for i in range(1, 10)]
    assert len(default_questions(many)) == memo_mod.MAX_FLAG_QUESTIONS + len(memo_mod.STANDARD_QUESTIONS)


def test_memo_filename():
    assert memo_filename(make_result(None)) == "ACME_IC_memo_2026-10-19.md"
    assert memo_filename(AnalysisResult(ticker="brk.b", generated_at=TODAY)) == "BRK-B_IC_memo_2026-10-19.md"
    assert memo_filename(AnalysisResult(ticker="", generated_at="")) == f"memo_IC_memo_{date.today().isoformat()}.md"


# ------------------------------------------------------------------------------------------------
# Real EDGAR fixture: the memo links to the actual filing
# ------------------------------------------------------------------------------------------------

def test_sources_link_points_to_real_filing(fixture_dir):
    meta = json.loads((fixture_dir / "AAPL" / "meta.json").read_text(encoding="utf-8"))
    company = CompanyInfo(meta["ticker"], meta["cik"], meta["name"])
    primary_document = meta["url"].rsplit("/", 1)[1]
    filing = Filing(meta["cik"], "10-K", meta["accession"], meta["filing_date"], meta["report_date"], primary_document)
    tenk = TenK(company=company, filing=filing, text="", sections={})
    markdown = render_memo(AnalysisResult(ticker="AAPL", generated_at=TODAY, company=company, tenk=tenk))
    assert markdown.startswith("# Investment Committee Memo: Apple Inc. (AAPL)\n")
    assert (f"10-K for fiscal year ended {meta['report_date']} (filed {meta['filing_date']}, "
            f"[EDGAR]({meta['url']}))") in markdown
