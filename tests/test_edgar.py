"""Offline tests for copilot.edgar (no network: requests.get and sec_get_json are monkeypatched)."""
from __future__ import annotations

import json
import re

import pytest
import requests

from copilot import edgar
from copilot.edgar import (
    CompanyInfo, EdgarError, Filing, Section, extract_sections, get_latest_10k, html_to_text, load_10k,
    lookup_company, normalize_ticker, sec_get, sec_get_json, text_outside_sections,
)

UA = "Test Runner test@example.com"
FILLER = "The company sells products and services to customers around the world. " * 30   # ~2,200 chars


@pytest.fixture(autouse=True)
def clean_http_state(monkeypatch):
    """Empty cache, no throttle waits, and a recorder instead of real sleeps."""
    edgar._cache.clear()
    monkeypatch.setattr(edgar, "MIN_REQUEST_INTERVAL", 0.0)
    sleeps: list[float] = []
    monkeypatch.setattr(edgar.time, "sleep", sleeps.append)
    yield sleeps
    edgar._cache.clear()


# --------------------------------------------------------------------------------------------
# sec_get
# --------------------------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content


def fake_get(outcomes: list):
    """requests.get replacement returning (or raising) the given outcomes in order; records calls."""
    calls: list[dict] = []

    def get(url, headers=None, timeout=None):
        calls.append({"url": url, "headers": headers, "timeout": timeout})
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    get.calls = calls
    return get


@pytest.mark.parametrize("bad_agent", ["", "   ", "MyApp/1.0", None])
def test_sec_get_requires_contact_email_in_user_agent(monkeypatch, bad_agent):
    get = fake_get([FakeResponse(200, b"ok")])
    monkeypatch.setattr(edgar.requests, "get", get)
    with pytest.raises(EdgarError, match=r"SEC User-Agent box or SEC_USER_AGENT in \.env"):
        sec_get("https://www.sec.gov/x", bad_agent)
    assert get.calls == []


def test_sec_get_sends_headers_and_caches(monkeypatch):
    get = fake_get([FakeResponse(200, b"payload")])
    monkeypatch.setattr(edgar.requests, "get", get)
    assert sec_get("https://www.sec.gov/a", UA) == b"payload"
    assert sec_get("https://www.sec.gov/a", UA) == b"payload"    # second call served from cache
    assert len(get.calls) == 1
    assert get.calls[0]["headers"] == {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
    assert get.calls[0]["timeout"] == 20


def test_sec_get_retries_429_5xx_and_network_errors(monkeypatch, clean_http_state):
    get = fake_get([FakeResponse(429), requests.ConnectionError("reset"), FakeResponse(503),
                    FakeResponse(200, b"finally")])
    monkeypatch.setattr(edgar.requests, "get", get)
    assert sec_get("https://www.sec.gov/b", UA) == b"finally"
    assert len(get.calls) == 4
    assert clean_http_state == [1, 2, 4]        # backoff before each retry


@pytest.mark.parametrize("last, message", [
    (FakeResponse(429), r"^SEC EDGAR is rate-limiting requests \(HTTP 429\)\. Wait a minute and try again\."),
    (FakeResponse(500), r"^SEC EDGAR is unavailable \(HTTP 500\)\. Try again in a few minutes\."),
    (requests.ConnectionError("reset"), r"^Could not reach SEC EDGAR\. Check your internet connection"),
])
def test_sec_get_gives_up_after_retries_with_advice_first_and_url_last(monkeypatch, clean_http_state, last, message):
    get = fake_get([requests.ConnectionError("reset"), FakeResponse(503), FakeResponse(500), last])
    monkeypatch.setattr(edgar.requests, "get", get)
    with pytest.raises(EdgarError, match=message) as info:
        sec_get("https://www.sec.gov/c", UA)
    assert str(info.value).endswith(" (https://www.sec.gov/c)")
    assert len(get.calls) == 4
    assert clean_http_state == [1, 2, 4]


@pytest.mark.parametrize("error", [requests.ReadTimeout("stalled"), requests.ConnectTimeout("stalled")])
def test_sec_get_does_not_retry_timeouts(monkeypatch, clean_http_state, error):
    get = fake_get([error, FakeResponse(200, b"too late")])
    monkeypatch.setattr(edgar.requests, "get", get)
    with pytest.raises(EdgarError, match=r"^SEC EDGAR did not respond within 20 s\. Check your internet connection"):
        sec_get("https://www.sec.gov/t", UA)
    assert len(get.calls) == 1 and clean_http_state == []


@pytest.mark.parametrize("status, message", [
    (403, r"HTTP 403\).*SEC User-Agent box"),
    (404, r"^Not found on SEC EDGAR \(HTTP 404\)\. \(https://www\.sec\.gov/d\)$"),
    (400, r"^SEC EDGAR returned HTTP 400\. \(https://www\.sec\.gov/d\)$"),
])
def test_sec_get_does_not_retry_client_errors(monkeypatch, status, message):
    get = fake_get([FakeResponse(status)] * 4)
    monkeypatch.setattr(edgar.requests, "get", get)
    with pytest.raises(EdgarError, match=message):
        sec_get("https://www.sec.gov/d", UA)
    assert len(get.calls) == 1


def test_throttle_waits_between_requests(monkeypatch, clean_http_state):
    monkeypatch.setattr(edgar, "MIN_REQUEST_INTERVAL", 0.15)
    monkeypatch.setattr(edgar, "_last_request_at", 100.0)
    monkeypatch.setattr(edgar.time, "monotonic", lambda: 100.05)
    edgar._wait_for_turn()
    assert clean_http_state == [pytest.approx(0.10)]


def test_sec_get_json(monkeypatch):
    get = fake_get([FakeResponse(200, b'{"a": 1}'), FakeResponse(200, b"<html>")])
    monkeypatch.setattr(edgar.requests, "get", get)
    assert sec_get_json("https://data.sec.gov/ok.json", UA) == {"a": 1}
    with pytest.raises(EdgarError, match="invalid JSON"):
        sec_get_json("https://data.sec.gov/bad.json", UA)


# --------------------------------------------------------------------------------------------
# Ticker lookup and latest 10-K (synthetic JSON)
# --------------------------------------------------------------------------------------------

TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
}


def submissions(*rows: tuple[str, str, str, str, str], name: str = "Example Corp", sic: str = "") -> dict:
    """Build a submissions JSON from (form, accession, filing_date, report_date, primary_document) rows."""
    keys = ["form", "accessionNumber", "filingDate", "reportDate", "primaryDocument"]
    return {"name": name, "sic": sic, "filings": {"recent": {k: [r[i] for r in rows] for i, k in enumerate(keys)}}}


def serve_json(monkeypatch, routes: dict[str, dict]) -> None:
    """Make sec_get_json return routes[url-substring] without any network."""
    def fake(url, user_agent):
        for key, payload in routes.items():
            if key in url:
                return payload
        raise AssertionError(f"unexpected url {url}")
    monkeypatch.setattr(edgar, "sec_get_json", fake)


@pytest.mark.parametrize("raw, expected", [("aapl", "AAPL"), ("  brk.b ", "BRK-B"), ("BRK-B", "BRK-B"), ("", "")])
def test_normalize_ticker(raw, expected):
    assert normalize_ticker(raw) == expected


def test_lookup_company(monkeypatch):
    serve_json(monkeypatch, {"company_tickers.json": TICKERS})
    assert lookup_company("aapl", UA) == CompanyInfo("AAPL", 320193, "Apple Inc.")
    brk = lookup_company("BRK.B", UA)
    assert (brk.ticker, brk.cik, brk.cik10) == ("BRK-B", 1067983, "0001067983")
    with pytest.raises(EdgarError, match="Ticker 'ZZZZ' not found"):
        lookup_company("zzzz", UA)
    with pytest.raises(EdgarError, match="Enter a ticker"):
        lookup_company("  ", UA)


def test_get_latest_10k_skips_amendments(monkeypatch):
    serve_json(monkeypatch, {"CIK0000000042.json": submissions(
        ("8-K", "0000000042-26-000009", "2026-05-01", "", "x8k.htm"),
        ("10-K/A", "0000000042-26-000005", "2026-04-30", "2025-12-31", "amend.htm"),
        ("10-K", "0000000042-26-000001", "2026-02-01", "2025-12-31", "ex-20251231.htm"),
        ("10-K", "0000000042-25-000001", "2025-02-01", "2024-12-31", "ex-20241231.htm"),
    )})
    filing = get_latest_10k(42, UA)
    assert filing == Filing(42, "10-K", "0000000042-26-000001", "2026-02-01", "2025-12-31", "ex-20251231.htm")
    assert filing.url == "https://www.sec.gov/Archives/edgar/data/42/000000004226000001/ex-20251231.htm"
    assert filing.sic is None                              # no SIC code in the feed
    assert filing.fiscal_year_label == "FYE 2025-12-31"
    assert Filing(42, "10-K", "a", "b", "", "c").fiscal_year_label == "FYE ?"


def test_get_latest_10k_reads_sic_code(monkeypatch):
    serve_json(monkeypatch, {"CIK0000000042.json": submissions(
        ("10-K", "0000000042-26-000001", "2026-02-01", "2025-12-31", "bank.htm"), sic="6021")})
    assert get_latest_10k(42, UA).sic == 6021


def test_get_latest_10k_explains_missing_10k(monkeypatch):
    serve_json(monkeypatch, {
        "CIK0000000001.json": submissions(("6-K", "a", "2026-01-01", "", "x.htm"), ("20-F", "b", "2025-04-01", "", "y.htm"),
                                          name="Foreign Co"),
        "CIK0000000002.json": submissions(("10-Q", "a", "2026-08-01", "2026-06-30", "q.htm"), name="New Holdco"),
    })
    with pytest.raises(EdgarError, match="Foreign Co is a foreign private issuer .* Form 20-F"):
        get_latest_10k(1, UA)
    with pytest.raises(EdgarError, match="No 10-K found in the recent EDGAR filings of New Holdco"):
        get_latest_10k(2, UA)


# --------------------------------------------------------------------------------------------
# Ticker lookup and latest 10-K (real EDGAR fixtures)
# --------------------------------------------------------------------------------------------

def serve_fixture(monkeypatch, fixture_dir, ticker: str) -> None:
    def fake(url, user_agent):
        if url == edgar.TICKERS_URL:
            return json.loads((fixture_dir / "company_tickers.json").read_text(encoding="utf-8"))
        assert "submissions" in url
        return json.loads((fixture_dir / ticker / "submissions.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(edgar, "sec_get_json", fake)


@pytest.mark.parametrize("ticker, sic", [("AAPL", 3571), ("BRK-B", 6331), ("JPM", 6021), ("TSLA", 3711),
                                         ("COST", 5331)])
def test_fixture_lookup_and_latest_10k(monkeypatch, fixture_dir, ticker, sic):
    meta = json.loads((fixture_dir / ticker / "meta.json").read_text(encoding="utf-8"))
    serve_fixture(monkeypatch, fixture_dir, ticker)
    company = lookup_company(ticker.replace("-", ".").lower(), UA)
    assert (company.ticker, company.cik) == (ticker, meta["cik"])
    filing = get_latest_10k(company.cik, UA)
    assert filing.form == "10-K"            # TSLA's newer 10-K/A (Part III amendment) is skipped
    assert (filing.accession, filing.filing_date, filing.report_date) == (
        meta["accession"], meta["filing_date"], meta["report_date"])
    assert filing.url == meta["url"]
    assert filing.sic == sic


def test_fixture_tsm_is_a_20f_filer(monkeypatch, fixture_dir):
    serve_fixture(monkeypatch, fixture_dir, "TSM")
    company = lookup_company("TSM", UA)
    with pytest.raises(EdgarError, match="foreign private issuer .* Form 20-F"):
        get_latest_10k(company.cik, UA)


def test_fixture_xom_successor_has_no_10k(monkeypatch, fixture_dir):
    serve_fixture(monkeypatch, fixture_dir, "XOM")
    company = lookup_company("XOM", UA)
    assert company.cik == 2115436
    with pytest.raises(EdgarError, match=r"No 10-K found .*ExxonMobil Holdings Corp \(CIK 2115436\)"):
        get_latest_10k(company.cik, UA)


# --------------------------------------------------------------------------------------------
# html_to_text
# --------------------------------------------------------------------------------------------

def test_html_to_text_keeps_span_split_headings_on_one_line():
    html = ('<div><span style="font-weight:700">Item&#160;</span><span style="font-weight:700">1A.</span>'
            '<span>&#160;&#160;Risk\nFactors</span></div><div>Body <a href="#x">text</a> here.</div>')
    assert html_to_text(html) == "Item 1A. Risk Factors\nBody text here."


def test_html_to_text_drops_hidden_and_non_content():
    html = """<?xml version='1.0' encoding='ASCII'?>
    <html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><head><title>t</title><style>p {}</style></head>
    <body><div style="display:none"><ix:header><ix:hidden>0000320193 2025 FY false</ix:hidden></ix:header></div>
    <ix:header><ix:references>dei-junk</ix:references></ix:header>
    <script>var x = 1;</script><!-- a comment -->
    <p style="DISPLAY: NONE">hidden paragraph</p>
    <p>Visible&#8203;text&#8201;here</p></body></html>"""
    assert html_to_text(html) == "Visible text here"


def test_html_to_text_drops_superscript_footnote_markers_only():
    html = ('<p><span style="vertical-align:super;font-size:7pt">2</span><span>Geographic data reflects</span> '
            'the iPhone<sup>®</sup> line<sup>(1)</sup> and our 1<sup>st</sup> quarter<sup>*</sup>.</p>')
    assert html_to_text(html) == "Geographic data reflects the iPhone® line and our 1st quarter."


def test_html_to_text_table_rows_become_pipe_lines():
    html = """<table>
      <tr><td>Fiscal year</td><td></td><td>2025</td><td>&#160;</td><td>2024</td></tr>
      <tr><td>Net <span>sales</span></td><td>$</td><td>416,161</td><td></td><td>$</td><td>391,035</td></tr>
      <tr><td>Other income</td><td>(</td><td>321</td><td>)</td><td>12.5</td><td>%</td></tr>
      <tr><td style="display:none">secret</td><td>Gross<br/>margin</td><td>46.9</td></tr>
    </table><p>After the table.</p>"""
    assert html_to_text(html).splitlines() == [
        "Fiscal year | 2025 | 2024",
        "Net sales | $416,161 | $391,035",
        "Other income | (321) | 12.5%",
        "Gross margin | 46.9",
        "After the table.",
    ]


def test_html_to_text_one_cell_layout_row_keeps_paragraphs():
    html = "<table><tr><td></td><td><p>First paragraph.</p><p>Second paragraph.</p></td></tr></table>"
    assert html_to_text(html).splitlines() == ["First paragraph.", "Second paragraph."]


def test_html_to_text_block_boundaries_and_bytes_input():
    html = b"<body>Intro<div>Block one<br>after break</div>tail<ul><li>a</li><li>b</li></ul><pre>x  y\nz</pre></body>"
    assert html_to_text(html).splitlines() == ["Intro", "Block one", "after break", "tail", "a", "b", "x y", "z"]


def test_html_to_text_drops_page_furniture():
    furniture = ["26", "Table of Contents", "TABLE OF CONTENTS | Alphabet Inc.", "8.", "F-3", "K-12",
                 "Apple Inc. | 2025 Form 10-K | 4", "Fiscal 2025 Form 10-K | 31", "2025 Annual Report 12",
                 "47 | JPMorgan Chase & Co./2025 Form 10-K"]
    content = ["MQ-25", "Item 1A | Risk Factors | 12", "2025", "See our 2025 Form 10-K for details."]
    html = "".join(f"<p>{line}</p>" for line in ["a sentence that", *furniture, "continues here.", *content])
    assert html_to_text(html).splitlines() == ["a sentence that", "continues here.", *content]


# --------------------------------------------------------------------------------------------
# Item headings and section extraction
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("line, item", [
    ("Item 1. Business", "1"),
    ("ITEM 1A.RISK FACTORS", "1a"),
    ("Item 1A: Risk Factors", "1a"),
    ("Item 1A - Risk Factors", "1a"),
    ("Item 1A—Risk Factors", "1a"),
    ("Item 7 — Management’s Discussion and Analysis", "7"),
    ("Item 7 -", "7"),
    ("Item 7", "7"),
    ("Item 7A. Quantitative and Qualitative Disclosures", "7a"),
    ("Item 1A. | Risk Factors | 12", "1a"),
    ("Item 1.Business", "1"),
    ("ITEM1. BUSINESS", "1"),
    ("Items 1 and 2. Business and Properties", "1"),
    ("Item 1B, 1C", "1b"),
    ("Item 10. Directors, Executive Officers", "10"),
    ("Item 1Business", None),            # no separator: ambiguous, not a heading
    ("Item 2.02 Results of Operations", None),   # 8-K item numbers are not 10-K headings
    ("See Item 1A of this report.", None),       # mid-line reference
    ("Itemized deductions", None),
])
def test_heading_regex(line, item):
    found = edgar._find_item_headings(line)
    assert [i for _, i in found] == ([item] if item else [])


def make_10k(*, item1: str = FILLER, item1a: str = FILLER, item7: str = FILLER, extra: str = "") -> str:
    """A plain-text 10-K with a table of contents (page numbers) followed by the body."""
    toc = "\n".join([
        "TABLE OF CONTENTS", "Item 1. | Business | 3", "Item 1A. | Risk Factors | 12", "Item 1B. | Unresolved Staff Comments | 20",
        "Item 2. | Properties | 21", "Item 7. | Management’s Discussion and Analysis | 30", "Item 7A. | Market Risk | 45",
        "Item 8. | Financial Statements | 46", "Item 10. | Directors | 90",
    ])
    body = "\n".join([
        "PART I", "Item 1. Business", item1,
        "ITEM 1A.RISK FACTORS", item1a,
        "Item 1B. Unresolved Staff Comments", "None.",
        "Item 2. Properties", "We lease offices.",
        "PART II", "Item 7 — Management’s Discussion and Analysis", item7,
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk", "Rates.",
        "Item 8. Financial Statements", extra or "Balance sheet.",
        "Item 10. Directors", "See proxy.",
    ])
    return toc + "\n" + body


def test_extract_sections_skips_table_of_contents():
    text = make_10k(item1="Our business is widgets. " + FILLER, item1a="Competition is intense. " + FILLER,
                    item7="Revenue grew 12%. " + FILLER)
    sections = extract_sections(text)
    assert list(sections) == ["item1", "item1a", "item7"]
    s1, s1a, s7 = sections["item1"], sections["item1a"], sections["item7"]
    assert s1.ok and s1.text.startswith("Item 1. Business\nOur business is widgets.")
    assert s1.text.endswith(FILLER.strip()) and "Item 1A" not in s1.text   # stops before the next item
    assert s1a.text.startswith("ITEM 1A.RISK FACTORS\nCompetition is intense.")
    assert s7.text.startswith("Item 7 — Management’s Discussion and Analysis\nRevenue grew 12%.")
    assert "Item 7A" not in s7.text
    assert (s1.title, s1a.title) == ("Item 1. Business", "Item 1A. Risk Factors")
    for s in sections.values():   # start/end are offsets into the full text
        assert text[s.start:s.end].strip() == s.text


def test_extract_sections_rejects_short_spans_and_never_raises():
    sections = extract_sections(make_10k(item1a="Not applicable."))
    assert sections["item1"].ok and sections["item7"].ok
    assert sections["item1a"] == Section("item1a", "Item 1A. Risk Factors", None, None, None)
    for junk in ["", "no headings at all", "Item 1A.\nItem 1A.\n", "Item 7\n" * 3 + "x" * 2000]:
        assert not any(s.ok for s in extract_sections(junk).values())


def test_extract_sections_rejects_cross_reference_pointers():
    # JPM, WFC and DE answer Item 7 with a pointer to another document. It must count as a failed section,
    # so the pipeline also searches the text outside the parsed sections.
    pointer = ("Information in response to this Item 7 can be found in the 2025 Annual Report to Shareholders "
               "under \"Financial Review\" on pages 46 through 160. That information is incorporated into "
               "this item by reference. ") * 6
    assert 500 < len(pointer) < edgar.MIN_SECTION_CHARS
    text = make_10k(item7=pointer)
    sections = extract_sections(text)
    assert sections["item1"].ok and sections["item1a"].ok and not sections["item7"].ok
    assert pointer.strip() in text_outside_sections(text, sections)


def test_extract_sections_ignores_stray_item_heading_inside_another_item():
    # Realty Income's 10-K has an "Item 1: Financial Statements" line inside Item 8; it must not win
    # as Item 1 even though it would be the longest span.
    stray = "Item 1: Financial Statements\n" + FILLER * 20
    sections = extract_sections(make_10k(item1="Real business text. " + FILLER, extra=stray))
    assert sections["item1"].text.startswith("Item 1. Business\nReal business text.")


def test_extract_sections_running_page_headers_do_not_end_a_section():
    # Some filers repeat "Item 7" as a page header on every page of the section.
    item7 = "\n".join([FILLER, "48", "PART II", "Item 7", FILLER])
    section = extract_sections(make_10k(item7=item7))["item7"]
    assert section.text.count(FILLER.strip()) == 2


def test_extract_sections_without_1b_or_7a():
    text = "\n".join(["Item 1. Business", FILLER, "Item 1A. Risk Factors", FILLER, "Item 1C. Cybersecurity",
                      "Item 7. MD&A", FILLER, "Item 8. Financial Statements"])
    assert all(s.ok for s in extract_sections(text).values())


def test_text_outside_sections():
    text = make_10k(item1="BUSINESS " + FILLER, item1a="RISKS " + FILLER, item7="MDNA " + FILLER)
    sections = extract_sections(text)
    rest = text_outside_sections(text, sections)
    assert "BUSINESS" not in rest and "RISKS" not in rest and "MDNA" not in rest
    assert rest.startswith("TABLE OF CONTENTS") and "Item 2. Properties\nWe lease offices." in rest
    # Nothing parsed -> the whole text is the fallback document.
    nothing = {k: Section(k, edgar.SECTION_TITLES[k], None, None, None) for k in edgar.SECTION_KEYS}
    assert text_outside_sections(text, nothing) == text


def test_extract_sections_from_span_split_html():
    toc = ("<table><tr><td><span>Item 1.</span></td><td>Business</td><td>3</td></tr>"
           "<tr><td><span>Item 1A.</span></td><td>Risk Factors</td><td>9</td></tr>"
           "<tr><td><span>Item 1B.</span></td><td>Unresolved Staff Comments</td><td>20</td></tr></table>")
    body = (f"<div><span>ITEM</span><span>&#160;1.</span><span>&#160;BUSINESS</span></div><p>{FILLER}</p>"
            f"<div><span>ITEM</span><span>&#160;1A.</span><span>RISK FACTORS</span></div><p>{FILLER}</p>"
            "<div><span>ITEM</span><span>&#160;1B.</span><span>&#160;UNRESOLVED STAFF COMMENTS</span></div><p>None.</p>")
    text = html_to_text(f"<html><body>{toc}{body}</body></html>")
    assert text.splitlines()[:2] == ["Item 1. | Business | 3", "Item 1A. | Risk Factors | 9"]
    sections = extract_sections(text)
    assert sections["item1"].text == "ITEM 1. BUSINESS\n" + FILLER.strip()
    assert sections["item1a"].text == "ITEM 1A.RISK FACTORS\n" + FILLER.strip()
    assert not sections["item7"].ok


COMPANY = CompanyInfo("EX", 42, "Example Corp")
FILING = Filing(42, "10-K", "0000000042-26-000001", "2026-02-01", "2025-12-31", "ex.htm")
FOLDER = "https://www.sec.gov/Archives/edgar/data/42/000000004226000001"
POINTER = "Information in response to this Item can be found in the Annual Report to Shareholders."
ANNUAL_REPORT = b"<p>Exhibit 13</p><p>Risk Factors</p><p>UNIQUE_RISK_TEXT: credit losses could rise.</p>"


def as_html(text: str) -> bytes:
    return "".join(f"<p>{line}</p>" for line in text.splitlines()).encode()


def listing(*files: tuple[str, int]) -> dict:
    """A filing folder's index.json with the given (name, size) entries."""
    return {"directory": {"item": [{"name": name, "type": "text.gif", "size": str(size)} for name, size in files]}}


def serve_filing(monkeypatch, primary_text: str, index) -> list[str]:
    """Serve FILING's primary document, its index.json (`index`, or an exception to raise) and an
    'ex13.htm' annual report. Returns the list of requested URLs."""
    requested: list[str] = []
    documents = {FILING.url: as_html(primary_text), f"{FOLDER}/ex13.htm": ANNUAL_REPORT}

    def fake_get(url, ua):
        requested.append(url)
        return documents[url]

    def fake_get_json(url, ua):
        requested.append(url)
        if isinstance(index, Exception):
            raise index
        return index

    monkeypatch.setattr(edgar, "get_latest_10k", lambda cik, ua: FILING)
    monkeypatch.setattr(edgar, "sec_get", fake_get)
    monkeypatch.setattr(edgar, "sec_get_json", fake_get_json)
    return requested


def test_load_10k_offline(monkeypatch):
    requested = serve_filing(monkeypatch, make_10k(), index=None)
    tenk = load_10k(COMPANY, UA)
    assert tenk.filing is FILING and tenk.parsed_ok == ["item1", "item1a", "item7"] and tenk.failed == []
    assert requested == [FILING.url]          # every section parsed: the filing index is not read

    monkeypatch.setattr(edgar, "sec_get", lambda url, ua: b"<html><body> </body></html>")
    with pytest.raises(EdgarError, match="no readable text"):
        load_10k(COMPANY, UA)


def test_load_10k_appends_companion_annual_report(monkeypatch):
    # Wells Fargo-style 10-K: Items 1A and 7 point to an annual report filed as a separate, larger document.
    primary = make_10k(item1a=POINTER, item7=POINTER)
    requested = serve_filing(monkeypatch, primary, listing(
        ("ex.htm", 9_000), ("ex13.htm", 5_000_000), ("ex21.htm", 2_000),
        ("0000000042-26-000001-index.html", 9_000_000), ("Financial_Report.xlsx", 9_000_000)))
    tenk = load_10k(COMPANY, UA)
    assert requested == [FILING.url, f"{FOLDER}/index.json", f"{FOLDER}/ex13.htm"]
    primary_text = html_to_text(as_html(primary))
    assert tenk.text == primary_text + "\n" + html_to_text(ANNUAL_REPORT)
    assert tenk.sections == extract_sections(primary_text) and tenk.failed == ["item1a", "item7"]
    assert "UNIQUE_RISK_TEXT" in text_outside_sections(tenk.text, tenk.sections)


@pytest.mark.parametrize("index", [
    EdgarError("SEC EDGAR returned invalid JSON."),
    {"unexpected": "shape"},
    listing(("ex.htm", 9_000_000), ("ex21.htm", 1_000)),     # no other document is bigger than the 10-K
])
def test_load_10k_without_usable_companion_keeps_the_10k_text(monkeypatch, index):
    primary = make_10k(item7=POINTER)
    requested = serve_filing(monkeypatch, primary, index)
    tenk = load_10k(COMPANY, UA)
    assert requested == [FILING.url, f"{FOLDER}/index.json"]
    assert tenk.text == html_to_text(as_html(primary)) and tenk.failed == ["item7"]


# --------------------------------------------------------------------------------------------
# Real 10-K fixtures
# --------------------------------------------------------------------------------------------

# Filings that use a "Form 10-K Cross-Reference Index" instead of Item headings (GE, INTC), or whose
# Item 7 only points to the MD&A elsewhere in the document (JPM), fall back to the full text.
EXPECTED_FAILURES = {"GE": {"item1", "item1a", "item7"}, "INTC": {"item1", "item1a", "item7"}, "JPM": {"item7"}}
FIXTURE_TICKERS = ["AAPL", "AMZN", "BRK-B", "COST", "GE", "INTC", "JPM", "KO", "MSFT", "NFLX", "NVDA", "O",
                   "PFE", "PLTR", "SNOW", "TSLA", "WMT"]
NEXT_HEADING = {"item1": r"items? ?1[abc]|items? ?2", "item1a": r"items? ?1[bc]|items? ?[23]", "item7": r"items? ?(7a|8)"}


@pytest.mark.parametrize("ticker", FIXTURE_TICKERS)
def test_fixture_section_extraction(tenk_text, ticker):
    text = tenk_text(ticker)
    assert len(text) > 100_000
    assert not [line for line in text.splitlines()                 # page furniture is dropped
                if line.lower() == "table of contents" or re.fullmatch(r"\d{1,3}", line)]
    sections = extract_sections(text)
    assert {k for k, s in sections.items() if not s.ok} == EXPECTED_FAILURES.get(ticker, set())
    for key, s in sections.items():
        if not s.ok:
            continue
        number = key[len("item"):]
        assert re.match(rf"items? ?{number}(?![0-9a-z])", s.text, re.IGNORECASE), s.text[:80]
        assert re.match(NEXT_HEADING[key], text[s.end:], re.IGNORECASE), text[s.end:s.end + 80]
        assert not re.search(r"\|\s*\d+$", s.text.splitlines()[0])   # not a TOC row ending in a page number
        assert len(s.text) > 5_000
    # The fallback document never repeats a parsed section.
    rest = text_outside_sections(text, sections)
    assert len(rest) == pytest.approx(len(text) - sum(len(s.text) for s in sections.values() if s.ok), rel=0.01)


def test_fixture_apple_text_quality(tenk_text):
    text = tenk_text("AAPL")
    lines = text.splitlines()
    assert lines[:3] == ["UNITED STATES", "SECURITIES AND EXCHANGE COMMISSION", "Washington, D.C. 20549"]
    for junk in ("us-gaap:", "xbrli", "iso4217", "http://fasb.org", "0000320193 "):
        assert junk not in text                     # inline-XBRL header stays hidden
    assert "Total net sales | $416,161 | $391,035 | $383,285" in lines
    assert "Item 1A. | Risk Factors | 5" in lines     # TOC row as a pipe line
    assert all(line == line.strip() and "  " not in line for line in lines)
