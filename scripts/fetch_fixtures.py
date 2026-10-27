"""Download the real-EDGAR test fixtures (about 140 MB) into tests/fixtures/sec_cache, the tests' default folder.

    python scripts/fetch_fixtures.py                      # every ticker the tests use
    python scripts/fetch_fixtures.py AAPL --out C:/tmp/fx  # one ticker into another folder (then set FIXTURE_DIR)

Requests go one at a time through copilot.edgar.sec_get (User-Agent, throttle, retries). Reads SEC_USER_AGENT
from the environment or .env. Writes <TICKER>/{10k.html, companyfacts.json, submissions.json, meta.json}
plus company_tickers.json. The tests were written against the filings available in October 2026; a newer
10-K can change some expected values.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # lets the script run from anywhere without installing the package

from copilot import edgar  # noqa: E402
from copilot.config import get_sec_user_agent  # noqa: E402
from copilot.financials import COMPANY_FACTS_URL  # noqa: E402

TICKERS = ["AAPL", "AMZN", "BRK-B", "COST", "GE", "INTC", "JPM", "KO", "MSFT", "NFLX", "NVDA", "O", "PFE",
           "PLTR", "SNOW", "TSLA", "TSM", "WMT", "XOM"]


def fetch(ticker: str, out: Path, ua: str) -> None:
    company = edgar.lookup_company(ticker, ua)
    folder = out / company.ticker
    folder.mkdir(parents=True, exist_ok=True)
    for name, url in [("submissions.json", edgar.SUBMISSIONS_URL), ("companyfacts.json", COMPANY_FACTS_URL)]:
        (folder / name).write_bytes(edgar.sec_get(url.format(cik10=company.cik10), ua))
    meta = {"ticker": company.ticker, "cik": company.cik, "name": company.name}
    try:
        filing = edgar.get_latest_10k(company.cik, ua)     # reads the submissions cached above
    except edgar.EdgarError:                               # 20-F filer or no 10-K yet (TSM, XOM)
        meta["no_10k"] = True
    else:
        (folder / "10k.html").write_bytes(edgar.sec_get(filing.url, ua))
        meta.update(accession=filing.accession, filing_date=filing.filing_date,
                    report_date=filing.report_date, url=filing.url)
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{company.ticker}: {meta.get('url', 'no 10-K')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the real-EDGAR test fixtures.")
    parser.add_argument("tickers", nargs="*", default=TICKERS, help="default: every ticker the tests use")
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "fixtures" / "sec_cache")
    args = parser.parse_args(argv)
    ua = get_sec_user_agent()
    try:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "company_tickers.json").write_bytes(edgar.sec_get(edgar.TICKERS_URL, ua))
        for ticker in args.tickers:
            fetch(ticker, args.out, ua)
    except edgar.EdgarError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
