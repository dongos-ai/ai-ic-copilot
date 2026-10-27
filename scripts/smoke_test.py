"""Run the full pipeline for one ticker from the command line and print what the app would show.

    python scripts/smoke_test.py AAPL
    python scripts/smoke_test.py AAPL --transcript call.pdf --provider anthropic --out memo.md

Reads SEC_USER_AGENT (and any LLM API key) from the environment or a .env file in the project root.
Without --provider it runs in data-only mode (no LLM calls). Exit code 1 if the ticker could not be analyzed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # lets the script run from anywhere without installing the package

from copilot.config import get_sec_user_agent  # noqa: E402
from copilot.financials import format_money, format_multiple, format_pct, format_price, snapshot_table  # noqa: E402
from copilot.llm import PROVIDERS  # noqa: E402
from copilot.memo import memo_filename  # noqa: E402
from copilot.pipeline import AnalysisResult, Settings, run_analysis  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the AI Investment Committee Copilot pipeline.")
    parser.add_argument("ticker", help="ticker symbol, e.g. AAPL or BRK.B")
    parser.add_argument("--transcript", type=Path, help="earnings-call transcript (.txt or .pdf)")
    parser.add_argument("--provider", choices=PROVIDERS, help="LLM provider (default: data-only mode)")
    parser.add_argument("--model", help="model id (default: the provider's default from copilot/config.py)")
    parser.add_argument("--passages", type=int, help="passages per memo section (default: per-section k)")
    parser.add_argument("--out", type=Path, help="where to write the memo (default: <TICKER>_IC_memo_<date>.md)")
    return parser.parse_args(argv)


def print_report(result: AnalysisResult) -> None:
    """Print the step timings, 10-K parse status, financial snapshot, red flags and problems."""
    company = result.company
    title = f"{company.name} ({company.ticker}), CIK {company.cik}" if company else result.ticker
    print(f"\n=== {title} ===")

    print("\nStep timings (s):")
    for step, seconds in result.timings.items():
        print(f"  {step:<14} {seconds:6.2f}")

    print("\n10-K sections:")
    if result.tenk is None:
        print("  (no 10-K)")
    else:
        print(f"  {result.tenk.filing.url}")
        for section in result.tenk.sections.values():
            status = f"OK   {len(section.text):>8,} chars" if section.ok else "FAIL (full-text fallback)"
            print(f"  {section.title:<48} {status}")
    print(f"  Documents: {len(result.documents)} · chunks indexed: {result.chunk_count} · "
          f"transcript chars: {result.transcript_chars}")

    print("\nFinancial snapshot (SEC XBRL):")
    if result.fin is None:
        print("  (not available)")
    else:
        print(snapshot_table(result.fin).to_string())
    if result.valuation:
        v = result.valuation
        currency = v.get("currency") or "USD"
        pe = v.get("pe_note") or format_multiple(v.get("pe"))
        print(f"  Valuation as of {v.get('as_of')} (basis {v.get('basis_year')}): "
              f"price {format_price(v.get('price'), currency)}, "
              f"market cap {format_money(v.get('market_cap'), currency)}, "
              f"EV {format_money(v.get('enterprise_value'), currency)}, P/E {pe}, "
              f"EV/Sales {format_multiple(v.get('ev_sales'))}, FCF yield {format_pct(v.get('fcf_yield'))}")

    print(f"\nRed flags ({len(result.flags)}):")
    for flag in result.flags:
        print(f"  [{flag.id}] {flag.severity}: {flag.name} -- {flag.detail}")
    for check in result.checks_not_run:
        print(f"  not run: {check}")

    if result.committee:
        audit = result.committee.audit
        print(f"\nCommittee ({result.llm_label}): vote {result.vote}; claims kept {audit.claims_kept}/"
              f"{audit.claims_total}; invalid ids {len(audit.invalid_ids)}; quotes removed {audit.quotes_removed}")
    if result.no_committee_reason:
        print(f"\nNo committee memo: {result.no_committee_reason}")

    for label, messages in (("Warnings", result.warnings), ("Errors", result.errors)):
        print(f"\n{label} ({len(messages)}):")
        for message in messages:
            print(f"  - {message}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    args = parse_args(argv)

    transcript = None
    if args.transcript:
        if not args.transcript.is_file():
            print(f"Transcript file not found: {args.transcript}", file=sys.stderr)
            return 2
        transcript = (args.transcript.name, args.transcript.read_bytes())

    settings = Settings(user_agent=get_sec_user_agent(), provider=args.provider, model=args.model,
                        passages_per_section=args.passages)
    result = run_analysis(args.ticker, settings, transcript, progress=lambda message: print(f"> {message}"))
    print_report(result)

    out = args.out or Path(memo_filename(result))
    out.write_text(result.memo_markdown, encoding="utf-8")
    print(f"\nMemo written to {out.resolve()} ({len(result.memo_markdown):,} chars)")
    return 0 if result.company else 1


if __name__ == "__main__":
    sys.exit(main())
