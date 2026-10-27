"""Streamlit UI for the AI Investment Committee Copilot. Thin by design: all logic lives in copilot/.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from copilot import config
from copilot.committee import ROLE_NAMES, EvidencePack
from copilot.financials import format_money, format_multiple, format_pct, format_price
from copilot.llm import PROVIDER_LABELS, available_providers
from copilot.memo import escape_md, financial_snapshot_markdown, memo_filename, render_claim
from copilot.pipeline import AnalysisResult, Settings, run_analysis
from copilot.retrieval import MEMO_QUERIES
from copilot.transcript import MAX_UPLOAD_MB

DATA_ONLY = "data-only"
TAB_NAMES = ["Memo", "Financials", "Red flags", "Evidence", "Committee debate", "Diagnostics"]
SEVERITY_COLORS = {"High": "red", "Medium": "orange", "Low": "blue"}
UA_MISSING = ("Enter your name and email in Settings → SEC User-Agent before running (SEC requires it), "
              "e.g. 'Jane Doe jane@example.com'.")


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------

def render_form() -> tuple[bool, str, object, Settings]:
    """Ticker, optional transcript and settings. Returns (submitted, ticker, uploaded file or None, settings).

    Everything sits in one form, so changing a setting never reruns (and silently cancels) an analysis.
    """
    providers = available_providers()
    default_ua = config.get_sec_user_agent()
    with st.form("analyze"):
        left, right = st.columns([1, 2])
        ticker = left.text_input("Ticker", placeholder="AAPL", max_chars=12, key="ticker")
        upload = right.file_uploader(
            "Earnings-call transcript (optional)", type=["txt", "pdf"], key="transcript",
            help=f"A .txt or .pdf transcript you have access to (up to {MAX_UPLOAD_MB} MB). Nothing is scraped.")
        if not providers:
            st.caption("No LLM API key found (ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY). "
                       "Data-only mode still shows the financials, red flags and evidence; "
                       "the debate and the vote need a key.")
        with st.expander("Settings", expanded="@" not in default_ua):
            user_agent = st.text_input(
                "SEC User-Agent", value=default_ua, placeholder="Your Name you@email.com", key="user_agent",
                help="SEC EDGAR asks every automated client for a name and contact email. "
                     "Set SEC_USER_AGENT in .env to prefill this.")
            provider = st.selectbox("LLM provider", providers + [DATA_ONLY], key="provider",
                                    format_func=lambda p: PROVIDER_LABELS.get(p, "None — data-only mode"))
            defaults = ", ".join(f"{PROVIDER_LABELS[p]}: {model}" for p, model in config.DEFAULT_MODELS.items())
            model = st.text_input("Model", placeholder="provider default", key="model",
                                  help=f"Leave blank for the provider's default ({defaults}).")
        submitted = st.form_submit_button("Convene the committee", type="primary")
    settings = Settings(user_agent=user_agent.strip(), provider=None if provider == DATA_ONLY else provider,
                        model=model.strip() or None)
    return submitted, ticker, upload, settings


def analyze(ticker: str, upload, settings: Settings) -> AnalysisResult:
    """Run the pipeline inside a status box that streams its progress lines."""
    transcript = (upload.name, upload.getvalue()) if upload is not None else None
    with st.status(f"Analyzing {ticker.strip().upper()}...", expanded=True) as status:
        result = run_analysis(ticker, settings, transcript, progress=lambda message: status.write(escape_md(message)))
        if result.company is None:
            status.update(label="Analysis failed", state="error", expanded=True)
        else:
            problems = len(result.errors)
            label = f"Done in {result.timings.get('Total', 0):.1f} s"
            if problems:
                label += f" ({problems} problem{'s' if problems > 1 else ''}, see below)"
            status.update(label=label, state="error" if problems else "complete", expanded=False)
    return result


# ---------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------

def render_result(result: AnalysisResult) -> None:
    if result.company is None:  # the ticker lookup failed, so the error is all there is to show
        st.error(escape_md(" ".join(result.errors)))
        return
    render_header(result)
    for error in result.errors:
        st.error(escape_md(error))
    for warning in result.warnings:
        st.warning(escape_md(warning))

    memo_tab, fin_tab, flags_tab, evidence_tab, debate_tab, diag_tab = st.tabs(TAB_NAMES)
    with memo_tab:
        render_memo_tab(result)
    with fin_tab:
        render_financials_tab(result)
    with flags_tab:
        render_red_flags_tab(result)
    with evidence_tab:
        render_evidence_tab(result)
    with debate_tab:
        render_debate_tab(result)
    with diag_tab:
        render_diagnostics_tab(result)


def render_header(result: AnalysisResult) -> None:
    """Company name and two rows of metrics: price, market cap, the vote; then the multiples."""
    st.subheader(f"{result.company.name} ({result.company.ticker})")
    valuation = result.valuation or {}
    currency = valuation.get("currency") or "USD"
    basis = f"Uses {valuation.get('basis_year') or 'latest fiscal year'} reported figures, not TTM."
    vote = result.vote

    top, bottom = st.columns(3), st.columns(3)
    top[0].metric("Price", format_price(valuation.get("price"), currency),
                  help=f"As of {valuation.get('as_of') or 'n/a'}")
    top[1].metric("Market cap", format_money(valuation.get("market_cap"), currency))
    if vote and vote[0] != "No vote":
        top[2].metric("Committee vote", vote[0], help=f"Conviction: {vote[1]}")
    else:
        top[2].metric("Committee vote", "Not issued", help=result.no_committee_reason
                      or "The Chair did not cast a cited Buy, Hold or Pass vote (see the warnings).")
    bottom[0].metric("P/E", valuation.get("pe_note") or format_multiple(valuation.get("pe")), help=basis)
    bottom[1].metric("EV/Sales", format_multiple(valuation.get("ev_sales")), help=basis)
    bottom[2].metric("FCF yield", format_pct(valuation.get("fcf_yield")), help=basis)


def render_memo_tab(result: AnalysisResult) -> None:
    st.download_button("Download memo (.md)", data=result.memo_markdown, file_name=memo_filename(result),
                       mime="text/markdown")
    st.markdown(result.memo_markdown)  # already Markdown-escaped by copilot.memo


def render_financials_tab(result: AnalysisResult) -> None:
    st.markdown(financial_snapshot_markdown(result.fin, result.valuation))
    if result.fin is not None and result.fin.tags_used:
        st.markdown("**XBRL tags used**")
        tags = [(metric, ", ".join(names)) for metric, names in result.fin.tags_used.items()]
        st.dataframe(pd.DataFrame(tags, columns=["Metric", "Tags"]), hide_index=True)


def render_red_flags_tab(result: AnalysisResult) -> None:
    if not result.flags:
        if result.fin is None and result.tenk is None:
            st.info("Red-flag checks could not run: no SEC financial data or 10-K text was available.")
        else:
            st.success("No automated red flags triggered.")
    for flag in result.flags:
        with st.container(border=True):
            color = SEVERITY_COLORS.get(flag.severity, "gray")
            st.markdown(f":{color}[**{flag.severity}**] **[{flag.id}] {escape_md(flag.name)}**\n\n"
                        f"{escape_md(flag.detail)}")
            if flag.quote:
                st.markdown(f"> {escape_md(flag.quote)}")
            st.caption(escape_md(f"Source: {flag.source}"))
    if result.checks_not_run:
        st.caption("Checks not run: " + escape_md("; ".join(result.checks_not_run)))


def render_evidence_tab(result: AnalysisResult) -> None:
    if not result.retrieved:
        st.info("No passages were retrieved (no 10-K text or transcript was available).")
        return
    st.caption(f"BM25 top passages per memo section, from {result.chunk_count} chunks of "
               f"{len(result.documents)} documents. IDs match the memo's citations.")
    # Passage text -> evidence ID (chunk texts are unique, so this is a safe key).
    ids = {item.text: item.id for item in result.pack.items if item.kind == "passage"} if result.pack else {}
    for key, hits in result.retrieved.items():
        spec = MEMO_QUERIES.get(key, {})
        with st.expander(f"{spec.get('title', key)} ({len(hits)} passages)"):
            st.caption(escape_md(f"Query: {spec.get('query', '')}"))
            for chunk, score in hits:
                label = f"[{ids.get(chunk.text, chunk.id)}] {chunk.short_label}"
                st.markdown(f"**{escape_md(label)}** · score {score:.1f} · "
                            f"{escape_md(chunk.source)}, {escape_md(chunk.section)}")
                st.text(chunk.text)


def render_debate_tab(result: AnalysisResult) -> None:
    committee = result.committee
    if committee is None:
        st.info(escape_md(result.no_committee_reason or "The committee did not run."))
        return
    st.caption(f"Model: {committee.model_label}. Only claims that cite valid evidence IDs are shown.")
    for error in committee.errors:
        st.warning(escape_md(error))
    chair_ran = committee.memo is not None
    bull_tab, bear_tab, risk_tab = st.tabs([ROLE_NAMES["bull"], ROLE_NAMES["bear"], ROLE_NAMES["risk"]])
    with bull_tab:
        render_role(committee.bull, result.pack, [("points", "Points")], chair_ran)
    with bear_tab:
        render_role(committee.bear, result.pack, [("points", "Points")], chair_ran)
    with risk_tab:
        render_role(committee.risk, result.pack, [("risks", "Risks"), ("red_flag_review", "Red-flag review"),
                                                  ("said_vs_shows", "Management said vs. numbers show")], chair_ran)


def render_role(output: dict | None, pack: EvidencePack | None, parts: list[tuple[str, str]],
                chair_ran: bool) -> None:
    """One analyst's cited claims."""
    if output is None:
        without = ", so the Chair worked without it" if chair_ran else ""
        st.warning(f"This role failed{without} (see the errors above).")
        return
    for key, title in parts:
        lines = [render_claim(_as_claim(claim), pack) for claim in output.get(key) or []]
        st.markdown(f"**{title}**\n\n" + ("\n".join(line for line in lines if line) or "_None._"))


def _as_claim(item: dict) -> dict:
    """Show severity / verdict in the claim text so render_claim can display any role's items."""
    if "management_said" in item:
        text = (f"**{item['verdict']}** — management said: {item['management_said']}; "
                f"numbers show: {item['numbers_show']}")
        return {**item, "text": text}
    if item.get("severity"):
        return {**item, "text": f"**{item['severity']}** — {item['text']}"}
    return item


def render_diagnostics_tab(result: AnalysisResult) -> None:
    st.markdown("**10-K sections**")
    tenk = result.tenk
    if tenk is None:
        st.info("No 10-K was loaded.")
    else:
        filing = tenk.filing
        st.markdown(f"[Form {filing.form}, fiscal year ended {filing.report_date}, filed {filing.filing_date}]"
                    f"({filing.url}) · {len(tenk.text):,} characters of text")
        rows = [(s.title, "✅" if s.ok else "❌ (full-text fallback)", len(s.text) if s.ok else 0)
                for s in tenk.sections.values()]
        st.dataframe(pd.DataFrame(rows, columns=["Section", "Parsed", "Characters"]), hide_index=True)

    st.markdown("**Documents indexed**")
    if result.documents:
        docs = [(d.source, d.section, d.kind, len(d.text)) for d in result.documents]
        st.dataframe(pd.DataFrame(docs, columns=["Source", "Section", "Kind", "Characters"]), hide_index=True)
    transcript = "not provided"
    if result.transcript_name:
        used = f"{result.transcript_chars:,} characters" if result.transcript_chars else "not used"
        transcript = f"{escape_md(result.transcript_name)} ({used})"
    st.caption(f"{result.chunk_count} chunks indexed · transcript: {transcript}")

    st.markdown("**Step timings**")
    st.dataframe(pd.DataFrame(list(result.timings.items()), columns=["Step", "Seconds"]), hide_index=True)

    committee = result.committee
    if committee is None:
        return
    audit = committee.audit
    st.markdown("**Citation audit (Chair memo)**")
    cols = st.columns(4)
    cols[0].metric("Claims kept", f"{audit.claims_kept} / {audit.claims_total}")
    cols[1].metric("Invalid IDs removed", len(set(audit.invalid_ids)))
    cols[2].metric("Quotes verified", audit.quotes_verified)
    cols[3].metric("Quotes removed", audit.quotes_removed)
    if audit.dropped:
        with st.expander(f"Dropped claims ({len(audit.dropped)})"):
            st.markdown("\n".join(f"- {escape_md(text)}" for text in audit.dropped))
    st.markdown("**Raw LLM output**")
    for role, raw in committee.raw.items():
        with st.expander(ROLE_NAMES.get(role, role)):
            st.code(raw or "(no output)", language="json")


# ---------------------------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=config.APP_NAME, page_icon=":material/gavel:", layout="wide")
    st.title(config.APP_NAME)
    st.caption("Reads a company's latest 10-K, five years of SEC XBRL financials and (optionally) an earnings-call "
               "transcript. A Bull analyst, a Bear analyst and a Risk officer argue from cited evidence; the Chair "
               "writes the memo and casts the vote. Draft for discussion, not investment advice.")

    submitted, ticker, upload, settings = render_form()
    if submitted:
        if not ticker.strip():
            st.warning("Enter a ticker symbol, e.g. AAPL.")
        elif "@" not in settings.user_agent:
            st.warning(UA_MISSING)
        else:
            st.session_state["result"] = analyze(ticker, upload, settings)

    result = st.session_state.get("result")  # survives reruns (e.g. the download button)
    if result is not None:
        render_result(result)


if __name__ == "__main__":
    main()
