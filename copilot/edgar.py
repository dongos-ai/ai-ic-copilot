"""SEC EDGAR access: ticker -> CIK, latest 10-K, HTML -> text, and Item 1 / 1A / 7 extraction."""
from __future__ import annotations

import json
import re
import threading
import time
import warnings
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"


class EdgarError(Exception):
    """A user-facing problem talking to EDGAR (message is safe to show in the UI)."""


@dataclass
class CompanyInfo:
    ticker: str
    cik: int
    name: str

    @property
    def cik10(self) -> str:
        return f"{self.cik:010d}"


@dataclass
class Filing:
    cik: int
    form: str
    accession: str
    filing_date: str
    report_date: str
    primary_document: str
    sic: int | None = None       # SEC industry code from the submissions feed

    @property
    def url(self) -> str:
        return (f"https://www.sec.gov/Archives/edgar/data/{self.cik}/"
                f"{self.accession.replace('-', '')}/{self.primary_document}")

    @property
    def fiscal_year_label(self) -> str:
        """Fallback label for the filing's year (the period-end date; filers name fiscal years differently)."""
        return f"FYE {self.report_date or '?'}"


SECTION_KEYS = ("item1", "item1a", "item7")
SECTION_TITLES = {
    "item1": "Item 1. Business",
    "item1a": "Item 1A. Risk Factors",
    "item7": "Item 7. Management's Discussion and Analysis",
}


@dataclass
class Section:
    key: str
    title: str
    text: str | None
    start: int | None
    end: int | None

    @property
    def ok(self) -> bool:
        return self.text is not None


@dataclass
class TenK:
    company: CompanyInfo
    filing: Filing
    text: str
    sections: dict[str, Section]

    @property
    def parsed_ok(self) -> list[str]:
        return [k for k, s in self.sections.items() if s.ok]

    @property
    def failed(self) -> list[str]:
        return [k for k, s in self.sections.items() if not s.ok]


# --------------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------------

MIN_REQUEST_INTERVAL = 0.15      # seconds between requests; SEC allows at most 10 requests/second
RETRY_DELAYS = (1, 2, 4)         # seconds to wait before each retry
MAX_CACHE_ENTRIES = 100          # the cache is cleared when it grows past this

_cache: dict[str, bytes] = {}
_throttle_lock = threading.Lock()
_last_request_at = 0.0


def _wait_for_turn() -> None:
    """Block until at least MIN_REQUEST_INTERVAL has passed since the previous request (all threads)."""
    global _last_request_at
    with _throttle_lock:
        wait = _last_request_at + MIN_REQUEST_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def sec_get(url: str, user_agent: str, *, timeout: float = 20) -> bytes:
    """GET an SEC URL politely (identified, throttled, retried) and cache the body for this process.

    Rate limits (429), server errors (5xx) and dropped connections are retried with backoff. A timeout
    is not: a stalled network would otherwise freeze the app for minutes. Error messages put the
    user's next step first and the URL last.
    """
    if not user_agent or "@" not in user_agent:
        raise EdgarError("Enter your name and email in the SEC User-Agent box or SEC_USER_AGENT in .env, "
                         "e.g. 'Jane Doe jane@example.com' (the SEC requires a contact email in the "
                         "User-Agent of every automated request).")
    if url in _cache:
        return _cache[url]

    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    problem = ""
    for delay in (0, *RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        _wait_for_turn()
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
        except requests.Timeout as exc:       # before ConnectionError: ConnectTimeout is both
            raise EdgarError(f"SEC EDGAR did not respond within {timeout:.0f} s. Check your internet "
                             f"connection and try again. ({url})") from exc
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError):
            problem = "Could not reach SEC EDGAR. Check your internet connection and try again."
            continue
        except requests.RequestException as exc:
            raise EdgarError(f"Could not request SEC EDGAR ({exc.__class__.__name__}). ({url})") from exc
        if resp.status_code == 200:
            if len(_cache) >= MAX_CACHE_ENTRIES:
                _cache.clear()
            _cache[url] = resp.content
            return resp.content
        if resp.status_code == 403:
            raise EdgarError("SEC rejected the request (HTTP 403). This usually means the User-Agent is "
                             "missing or not descriptive, or too many requests were sent. Enter your name "
                             "and email in the SEC User-Agent box or SEC_USER_AGENT in .env and try again "
                             f"in a few minutes. ({url})")
        if resp.status_code == 404:
            raise EdgarError(f"Not found on SEC EDGAR (HTTP 404). ({url})")
        if resp.status_code == 429:
            problem = "SEC EDGAR is rate-limiting requests (HTTP 429). Wait a minute and try again."
        elif resp.status_code >= 500:
            problem = f"SEC EDGAR is unavailable (HTTP {resp.status_code}). Try again in a few minutes."
        else:
            raise EdgarError(f"SEC EDGAR returned HTTP {resp.status_code}. ({url})")
    raise EdgarError(f"{problem} ({url})")


def sec_get_json(url: str, user_agent: str) -> dict:
    """sec_get + JSON decoding."""
    try:
        return json.loads(sec_get(url, user_agent))
    except ValueError as exc:
        raise EdgarError(f"SEC EDGAR returned invalid JSON. Try again in a few minutes. ({url})") from exc


# --------------------------------------------------------------------------------------------
# Ticker -> company -> latest 10-K
# --------------------------------------------------------------------------------------------

def normalize_ticker(ticker: str) -> str:
    """'  brk.b ' -> 'BRK-B' (SEC and Yahoo both write share classes with a dash)."""
    return (ticker or "").strip().upper().replace(".", "-")


def lookup_company(ticker: str, user_agent: str) -> CompanyInfo:
    """Map a ticker to its CIK and company name using SEC's company_tickers.json."""
    symbol = normalize_ticker(ticker)
    if not symbol:
        raise EdgarError("Enter a ticker symbol, e.g. AAPL.")
    for row in sec_get_json(TICKERS_URL, user_agent).values():
        if str(row.get("ticker", "")).upper() == symbol:
            return CompanyInfo(ticker=symbol, cik=int(row["cik_str"]), name=str(row.get("title", symbol)))
    raise EdgarError(f"Ticker '{symbol}' not found in SEC's company list. Check the symbol; only "
                     "SEC-registered companies are covered (write share classes like BRK-B).")


def get_latest_10k(cik: int, user_agent: str) -> Filing:
    """Return the most recent original 10-K (not a 10-K/A amendment) from the company's submissions feed."""
    data = sec_get_json(SUBMISSIONS_URL.format(cik10=f"{cik:010d}"), user_agent)
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    sic = str(data.get("sic") or "")
    for i, form in enumerate(forms):
        if form == "10-K":
            return Filing(
                cik=cik,
                form=form,
                accession=recent["accessionNumber"][i],
                filing_date=recent["filingDate"][i],
                report_date=recent["reportDate"][i] or "",
                primary_document=recent["primaryDocument"][i],
                sic=int(sic) if sic.isdigit() else None,
            )

    name = data.get("name") or f"CIK {cik}"
    foreign_form = next((f for f in forms if f in ("20-F", "40-F")), None)
    if foreign_form:
        raise EdgarError(f"No 10-K found: {name} is a foreign private issuer that files annual reports on "
                         f"Form {foreign_form}, which this app does not support (10-K filers only).")
    raise EdgarError(f"No 10-K found in the recent EDGAR filings of {name} (CIK {cik}). Newly registered "
                     "companies (e.g. a new holding company) may not have filed one yet, and foreign private "
                     "issuers file 20-F/40-F, which this app does not support.")


# --------------------------------------------------------------------------------------------
# HTML -> text
# --------------------------------------------------------------------------------------------

_SKIP_TAGS = {"script", "style", "head", "title", "noscript", "template", "ix:header"}
_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "body", "br", "caption", "center", "dd", "div", "dl", "dt",
    "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr",
    "html", "li", "main", "nav", "ol", "p", "section", "table", "tbody", "tfoot", "thead", "ul",
}
_HIDDEN_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
# Superscript footnote markers ("segment data<sup>2</sup>", "(1)", "*") would otherwise glue onto the next word.
_SUPERSCRIPT_RE = re.compile(r"vertical-align\s*:\s*super", re.IGNORECASE)
_FOOTNOTE_MARK_RE = re.compile(r"\(?\d{1,2}\)?|[*†‡§]+")
# Whitespace inside a text node (newlines in the HTML source included) is just a space.
_SPACE_RE = re.compile(r"[\s\N{ZERO WIDTH SPACE}]+")
# Financial tables put "$" and ")" in cells of their own: "$" | "1,234" -> "$1,234", "(12" | ")" -> "(12)".
_GLUE_TO_NEXT = {"$", "€", "£", "¥", "(", "($"}
_GLUE_TO_PREVIOUS = {")", "%", ")%", "%)"}
# Whole lines that are page furniture, not content. Dropping them keeps a sentence that runs across a
# page break in one piece, so passages read cleanly and verbatim quotes still match.
_PAGE_FURNITURE_RE = re.compile("|".join([
    r"table of contents(?: \|.{0,40})?",      # "Table of Contents", "Table of Contents | Alphabet Inc."
    r"(?:[a-z]-)?\d{1,3}\.?",                 # page numbers: "47", "8.", "F-3", "K-12"
    # short footers ending in a page number: "Apple Inc. | 2025 Form 10-K | 4", "... Annual Report 12"
    r"(?!item)(?=.{0,60}$).*\b(?:form 10-k|annual report)\b(?: \|)? \d{1,3}",
    # short footers starting with a page number: "47 | JPMorgan Chase & Co./2025 Form 10-K"
    r"\d{1,3} (?:\| )?(?=.{0,60}$).*\bform 10-k",
]), re.IGNORECASE)


def _skip(tag: Tag) -> bool:
    style = tag.get("style") or ""
    if tag.name in _SKIP_TAGS or _HIDDEN_RE.search(style):
        return True
    is_superscript = tag.name == "sup" or bool(_SUPERSCRIPT_RE.search(style))
    return is_superscript and bool(_FOOTNOTE_MARK_RE.fullmatch(tag.get_text(strip=True)))


def _render(root: Tag) -> str:
    """Text of `root` with "\n" at block boundaries only; inline runs (span, font, a, ix:*) stay on one line.

    Walks the tree with an explicit stack (filings can nest deeper than Python's recursion limit);
    None on the stack marks the end of a block element.
    """
    parts: list[str] = []
    stack: list[Tag | NavigableString | None] = [root]
    while stack:
        node = stack.pop()
        if node is None:
            parts.append("\n")
        elif isinstance(node, NavigableString):
            # Plain text only: comments and doctypes are NavigableString subclasses.
            if type(node) is NavigableString:
                parts.append(_SPACE_RE.sub(" ", node))
        elif _skip(node):
            continue
        elif node.name == "tr":
            parts.append("\n" + _row_text(node) + "\n")
        elif node.name == "pre":
            parts.append("\n" + node.get_text() + "\n")
        else:
            if node.name in _BLOCK_TAGS:
                parts.append("\n")
                stack.append(None)
            stack.extend(reversed(node.contents))
    return "".join(parts)


def _row_text(tr: Tag) -> str:
    """One table row -> 'cell | cell | cell' (empty cells skipped)."""
    rendered = [_render(cell) for cell in tr.find_all(["td", "th"], recursive=False) if not _skip(cell)]
    rendered = [text for text in rendered if text.strip()]
    if len(rendered) == 1:
        # A one-cell row is usually a layout wrapper around paragraphs: keep its line breaks.
        return rendered[0]
    cells: list[str] = []
    for text in rendered:
        text = " ".join(text.split())
        if cells and (cells[-1] in _GLUE_TO_NEXT or text in _GLUE_TO_PREVIOUS):
            cells[-1] += text
        else:
            cells.append(text)
    return " | ".join(cells)


def html_to_text(html: bytes | str) -> str:
    """Filing HTML -> plain text with one paragraph, heading or table row ("a | b | c") per line.

    Scripts, styles, the inline-XBRL header, anything styled display:none and page furniture (page
    numbers, "Table of Contents" links, running footers) are dropped.
    """
    with warnings.catch_warnings():
        # Inline XBRL filings start with an <?xml?> declaration; parsing them as HTML is intended.
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    lines = (_SPACE_RE.sub(" ", line).strip() for line in _render(soup).split("\n"))
    return "\n".join(line for line in lines if line and not _PAGE_FURNITURE_RE.fullmatch(line))


# --------------------------------------------------------------------------------------------
# Item 1 / 1A / 7 extraction
# --------------------------------------------------------------------------------------------

# Item headings at the start of a line: "Item 1.", "ITEM 1A.RISK FACTORS", "Item 7 -", "Items 1 and 2.",
# "Item 1A | Risk Factors | 12". The number may not run into another digit or letter ("Item 1" never
# matches "Item 10" or "Item 1A") nor into ".02" (8-K item numbers like "Item 2.02").
_HEADING_RE = re.compile(r"^items?[ \t]*(\d{1,2}[a-c]?)(?![0-9a-z]|\.\d)", re.IGNORECASE | re.MULTILINE)
# Items that may directly follow each target section (filers can omit 1B, 1C and 7A).
_NEXT_ITEMS = {"1": {"1a", "1b", "1c", "2"}, "1a": {"1b", "1c", "2", "3"}, "7": {"7a", "8"}}
# Shorter spans are table-of-contents rows, "Not applicable" or a pointer to another document
# (the longest pointer seen is ~400 chars; the shortest real section ~9,000).
MIN_SECTION_CHARS = 2_000


def _find_item_headings(text: str) -> list[tuple[int, str]]:
    """(offset, item) for every line that starts with a 10-K item heading, e.g. (1234, "1a")."""
    return [(m.start(), m.group(1).lower()) for m in _HEADING_RE.finditer(text)]


def extract_sections(text: str) -> dict[str, Section]:
    """Locate Items 1, 1A and 7. Never raises; a section that can't be found has text=None.

    Every heading of the target item is a candidate start. A candidate ends at the next heading of a
    different item, which must be one that can follow the target (so a stray "Item 1: Financial
    Statements" line inside Item 8 is ignored). The longest valid span wins, which skips the table
    of contents.
    """
    headings = _find_item_headings(text or "")
    sections: dict[str, Section] = {}
    for key in SECTION_KEYS:
        target = key[len("item"):]
        best: tuple[int, int] | None = None
        for i, (start, item) in enumerate(headings):
            if item != target:
                continue
            following = next(((pos, other) for pos, other in headings[i + 1:] if other != target), None)
            if following is None or following[1] not in _NEXT_ITEMS[target]:
                continue
            end = following[0]
            if best is None or end - start > best[1] - best[0]:
                best = (start, end)
        if best and best[1] - best[0] >= MIN_SECTION_CHARS:
            sections[key] = Section(key, SECTION_TITLES[key], text[best[0]:best[1]].strip(), best[0], best[1])
        else:
            sections[key] = Section(key, SECTION_TITLES[key], None, None, None)
    return sections


def text_outside_sections(text: str, sections: dict[str, Section]) -> str:
    """The full text minus the spans of successfully parsed sections (the full-text fallback document)."""
    spans = sorted((s.start, s.end) for s in sections.values()
                   if s.ok and s.start is not None and s.end is not None)
    pieces: list[str] = []
    pos = 0
    for start, end in spans:
        if start > pos:
            pieces.append(text[pos:start].strip())
        pos = max(pos, end)
    pieces.append(text[pos:].strip())
    return "\n".join(p for p in pieces if p)


def _companion_document_url(filing: Filing, user_agent: str, primary_size: int) -> str | None:
    """URL of the filing's largest other HTML document if it is bigger than the primary one, else None.

    Some 10-Ks (e.g. Wells Fargo's) are a short wrapper whose Items 1A and 7 point to an annual report
    filed as a separate, much larger document in the same filing.
    """
    folder = filing.url.rsplit("/", 1)[0]
    items = sec_get_json(f"{folder}/index.json", user_agent)["directory"]["item"]
    sizes = {item["name"]: int(item.get("size") or 0) for item in items
             if item["name"].lower().endswith((".htm", ".html"))
             and "index" not in item["name"] and item["name"] != filing.primary_document}
    largest = max(sizes, key=sizes.get, default=None)
    return f"{folder}/{largest}" if largest and sizes[largest] > primary_size else None


def load_10k(company: CompanyInfo, user_agent: str) -> TenK:
    """Download the latest 10-K, convert it to text and extract the key sections.

    If a section is missing, the text of a larger companion document (see _companion_document_url) is
    appended after the 10-K text, so the full-text fallback can search it; section offsets stay valid.
    """
    filing = get_latest_10k(company.cik, user_agent)
    html = sec_get(filing.url, user_agent)
    text = html_to_text(html)
    if not text:
        raise EdgarError(f"The 10-K document contained no readable text. ({filing.url})")
    sections = extract_sections(text)
    if not all(s.ok for s in sections.values()):
        try:
            url = _companion_document_url(filing, user_agent, len(html))
            if url:
                text += "\n" + html_to_text(sec_get(url, user_agent))
        except (EdgarError, KeyError, TypeError, ValueError):
            pass        # best effort: the 10-K document alone is still usable
    return TenK(company=company, filing=filing, text=text, sections=sections)
