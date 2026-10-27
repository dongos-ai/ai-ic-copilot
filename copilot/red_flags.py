"""Rule-based red flags from the financial table and the 10-K text.

Numeric rules compare the latest fiscal year with the prior one (or a short history) in the SEC XBRL
table from financials.py. Text rules look for a few high-signal disclosures in the 10-K. Both are
deliberately conservative: a missed flag is better than a noisy one, and every flag carries the
numbers, years, and source it came from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .financials import FinancialData, format_money, to_float


@dataclass
class RedFlag:
    id: str          # "R1", "R2", ...
    name: str
    severity: str    # "High" | "Medium" | "Low"
    detail: str
    source: str
    quote: str = ""


# Numeric thresholds. Ratios are fractions: 0.10 means 10% or 10 percentage points.
RECEIVABLES_GAP_MEDIUM = 0.10        # receivables growth minus revenue growth
RECEIVABLES_GAP_HIGH = 0.25
INVENTORY_GAP_MEDIUM = 0.15          # inventory growth minus revenue growth
MIN_BALANCE_SHARE_OF_REVENUE = 0.02  # receivables/inventory below 2% of revenue are too small to matter
GROSS_MARGIN_DROP_MEDIUM = 0.02
GROSS_MARGIN_DROP_HIGH = 0.05
OPERATING_MARGIN_DROP_MEDIUM = 0.03
OPERATING_MARGIN_DROP_HIGH = 0.08
MARGIN_DECLINE_STREAK = 3            # consecutive yearly operating-margin declines
SBC_SHARE_OF_REVENUE_MEDIUM = 0.10
SBC_SHARE_OF_REVENUE_HIGH = 0.20
SBC_SHARE_OF_FCF_MEDIUM = 0.50
MIN_OCF_TO_NET_INCOME = 0.8
ACCRUAL_YEARS = 3                    # window for cumulative OCF vs cumulative net income
ACCRUAL_MIN_RATIO = 0.9              # cumulative OCF below this share of cumulative net income
FCF_LOOKBACK_YEARS = 3
NEGATIVE_FCF_YEARS_HIGH = 2          # negative FCF in this many of the lookback years -> High
MIN_CASH_RUNWAY_YEARS = 2.0
REVENUE_DECLINE_MEDIUM = 0.02        # smaller declines are noise
REVENUE_DECLINE_HIGH = 0.10
DEBT_GROWTH_MEDIUM = 0.25
DEBT_GROWTH_MAX_REVENUE_GROWTH = 0.10
MAX_DEBT_TO_OCF = 4.0
DILUTION_MEDIUM = 0.03
DILUTION_HIGH = 0.08

# Text-scan settings.
MIN_CUSTOMER_SHARE_PCT = 10.0
NEGATION_WINDOW_WORDS = 5            # "no", "not", ... this close before the key phrase cancels a match
HEDGE_WINDOW_WORDS = 20              # "if", "may", "could", ... this close before the key phrase cancels it
HEDGE_AFTER_WORDS = 1                # ... or right after it ("a restatement of ... statements could harm us")
MAX_QUOTE_WORDS = 40

# Derived metrics are cited by the XBRL tags they were computed from.
_SOURCE_METRICS = {"fcf": ["operating_cash_flow", "capex"]}


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------

def _last(fin: FinancialData, column: str, n: int) -> list[float] | None:
    """The last n values of a table column (oldest first), or None if any is missing."""
    table = fin.table
    if column not in table.columns or len(table) < n:
        return None
    values = [to_float(v) for v in table[column].iloc[-n:]]
    return None if any(v is None for v in values) else values


def _growth(old: float, new: float) -> float | None:
    """Year-over-year growth; None when the base is zero or negative."""
    return new / old - 1 if old > 0 else None


def _years(fin: FinancialData, n: int) -> list[str]:
    return [str(y) for y in fin.years[-n:]]


def _span(fin: FinancialData, n: int) -> str:
    """'FY2025' or 'FY2024–FY2025' for the last n fiscal years."""
    years = _years(fin, n)
    return years[0] if len(years) == 1 else f"{years[0]}–{years[-1]}"


def _source(fin: FinancialData, metrics: list[str], n: int) -> str:
    """Citation naming the XBRL tags behind the metrics (metric name if no tag was recorded)."""
    tags: list[str] = []
    for metric in metrics:
        for base in _SOURCE_METRICS.get(metric, [metric]):
            for tag in fin.tags_used.get(base) or [base]:
                if tag not in tags:
                    tags.append(tag)
    return f"SEC XBRL company facts: {', '.join(tags)} ({_span(fin, n)})"


def _flag(fin: FinancialData, name: str, severity: str, detail: str, metrics: list[str], n: int) -> RedFlag:
    return RedFlag(id="", name=name, severity=severity, detail=detail, source=_source(fin, metrics, n))


def _money(fin: FinancialData, x: float) -> str:
    return format_money(x, fin.currency)


def _signed(x: float) -> str:
    return f"{x:+.1%}"


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _shares(x: float) -> str:
    return f"{x / 1e9:.2f}B" if abs(x) >= 1e9 else f"{x / 1e6:.1f}M"


# ---------------------------------------------------------------------------------------------
# Numeric rules. Each returns a RedFlag or None; the columns it reads are guaranteed present
# (see NUMERIC_CHECKS). Values are listed oldest -> newest.
# ---------------------------------------------------------------------------------------------

def _check_receivables(fin: FinancialData) -> RedFlag | None:
    old, new = _last(fin, "receivables", 2)
    rev_old, rev_new = _last(fin, "revenue", 2)
    growth, rev_growth = _growth(old, new), _growth(rev_old, rev_new)
    if growth is None or rev_growth is None or new < MIN_BALANCE_SHARE_OF_REVENUE * rev_new:
        return None
    gap = growth - rev_growth
    if gap <= RECEIVABLES_GAP_MEDIUM:
        return None
    prev_year, year = _years(fin, 2)
    detail = (f"Receivables {_signed(growth)} ({_money(fin, old)} → {_money(fin, new)}) vs revenue "
              f"{_signed(rev_growth)} in {year} vs {prev_year}; days sales outstanding "
              f"{old / rev_old * 365:.0f} → {new / rev_new * 365:.0f}.")
    severity = "High" if gap > RECEIVABLES_GAP_HIGH else "Medium"
    return _flag(fin, "Receivables growing faster than revenue", severity, detail, ["receivables", "revenue"], 2)


def _check_inventory(fin: FinancialData) -> RedFlag | None:
    old, new = _last(fin, "inventory", 2)
    rev_old, rev_new = _last(fin, "revenue", 2)
    growth, rev_growth = _growth(old, new), _growth(rev_old, rev_new)
    if growth is None or rev_growth is None or new < MIN_BALANCE_SHARE_OF_REVENUE * rev_new:
        return None
    if growth - rev_growth <= INVENTORY_GAP_MEDIUM:
        return None
    prev_year, year = _years(fin, 2)
    detail = (f"Inventory {_signed(growth)} ({_money(fin, old)} → {_money(fin, new)}) vs revenue "
              f"{_signed(rev_growth)} in {year} vs {prev_year}; inventory {_pct(old / rev_old)} → "
              f"{_pct(new / rev_new)} of revenue.")
    return _flag(fin, "Inventory growing faster than revenue", "Medium", detail, ["inventory", "revenue"], 2)


def _margins(fin: FinancialData, column: str, n: int) -> list[float] | None:
    """column / revenue for the last n years; None if revenue is not positive."""
    values, revenue = _last(fin, column, n), _last(fin, "revenue", n)
    if any(r <= 0 for r in revenue):
        return None
    return [v / r for v, r in zip(values, revenue)]


def _margin_drop(fin: FinancialData, column: str, label: str, medium: float, high: float) -> RedFlag | None:
    margins = _margins(fin, column, 2)
    if margins is None:
        return None
    drop = margins[0] - margins[1]
    if drop <= medium:
        return None
    prev_year, year = _years(fin, 2)
    detail = (f"{label} fell from {_pct(margins[0])} to {_pct(margins[1])} "
              f"(-{drop * 100:.1f} pts) in {year} vs {prev_year}.")
    severity = "High" if drop > high else "Medium"
    return _flag(fin, f"{label} contraction", severity, detail, [column, "revenue"], 2)


def _check_gross_margin(fin: FinancialData) -> RedFlag | None:
    return _margin_drop(fin, "gross_profit", "Gross margin", GROSS_MARGIN_DROP_MEDIUM, GROSS_MARGIN_DROP_HIGH)


def _check_operating_margin(fin: FinancialData) -> RedFlag | None:
    return _margin_drop(fin, "operating_income", "Operating margin",
                        OPERATING_MARGIN_DROP_MEDIUM, OPERATING_MARGIN_DROP_HIGH)


def _check_margin_streak(fin: FinancialData) -> RedFlag | None:
    n = MARGIN_DECLINE_STREAK + 1
    margins = _margins(fin, "operating_income", n)
    if margins is None or not all(new < old for old, new in zip(margins, margins[1:])):
        return None
    history = ", ".join(f"{y} {_pct(m)}" for y, m in zip(_years(fin, n), margins))
    detail = f"Operating margin declined {MARGIN_DECLINE_STREAK} years in a row: {history}."
    return _flag(fin, "Sustained margin compression", "Medium", detail, ["operating_income", "revenue"], n)


def _check_sbc_vs_revenue(fin: FinancialData) -> RedFlag | None:
    (sbc,), (revenue,) = _last(fin, "sbc", 1), _last(fin, "revenue", 1)
    if revenue <= 0:
        return None
    share = sbc / revenue
    if share <= SBC_SHARE_OF_REVENUE_MEDIUM:
        return None
    detail = (f"Stock-based compensation of {_money(fin, sbc)} was {_pct(share)} of revenue "
              f"({_money(fin, revenue)}) in {_span(fin, 1)}.")
    severity = "High" if share > SBC_SHARE_OF_REVENUE_HIGH else "Medium"
    return _flag(fin, "Heavy stock-based compensation", severity, detail, ["sbc", "revenue"], 1)


def _check_sbc_vs_fcf(fin: FinancialData) -> RedFlag | None:
    (sbc,), (fcf,) = _last(fin, "sbc", 1), _last(fin, "fcf", 1)
    if fcf <= 0 or sbc <= SBC_SHARE_OF_FCF_MEDIUM * fcf:
        return None
    detail = (f"Stock-based compensation of {_money(fin, sbc)} equals {_pct(sbc / fcf)} of free cash flow "
              f"({_money(fin, fcf)}) in {_span(fin, 1)}, so much of the FCF is paid out in stock.")
    return _flag(fin, "Stock comp large relative to free cash flow", "Medium", detail, ["sbc", "fcf"], 1)


def _check_cash_conversion(fin: FinancialData) -> RedFlag | None:
    (ocf,), (net_income,) = _last(fin, "operating_cash_flow", 1), _last(fin, "net_income", 1)
    if net_income <= 0 or ocf >= MIN_OCF_TO_NET_INCOME * net_income:
        return None
    detail = (f"Operating cash flow of {_money(fin, ocf)} was {ocf / net_income:.2f}x net income "
              f"({_money(fin, net_income)}) in {_span(fin, 1)}.")
    return _flag(fin, "Weak cash conversion", "Medium", detail, ["operating_cash_flow", "net_income"], 1)


def _check_accruals(fin: FinancialData) -> RedFlag | None:
    ocf = sum(_last(fin, "operating_cash_flow", ACCRUAL_YEARS))
    net_income = sum(_last(fin, "net_income", ACCRUAL_YEARS))
    if net_income <= 0 or ocf >= ACCRUAL_MIN_RATIO * net_income:
        return None
    detail = (f"Cumulative operating cash flow of {_money(fin, ocf)} trailed cumulative net income of "
              f"{_money(fin, net_income)} over {_span(fin, ACCRUAL_YEARS)}.")
    return _flag(fin, "Earnings running ahead of cash flow", "Medium", detail,
                 ["operating_cash_flow", "net_income"], ACCRUAL_YEARS)


def _check_negative_fcf(fin: FinancialData) -> RedFlag | None:
    years = _years(fin, FCF_LOOKBACK_YEARS)
    values = [to_float(v) for v in fin.table["fcf"].iloc[-FCF_LOOKBACK_YEARS:]]
    history = [(year, v) for year, v in zip(years, values) if v is not None]  # latest is never missing
    negatives = sum(v < 0 for _, v in history)
    if history[-1][1] >= 0 and negatives < NEGATIVE_FCF_YEARS_HIGH:
        return None
    listed = ", ".join(f"{year} {_money(fin, v)}" for year, v in history)
    detail = f"Free cash flow was negative in {negatives} of the last {len(history)} years ({listed})."
    severity = "High" if negatives >= NEGATIVE_FCF_YEARS_HIGH else "Medium"
    return _flag(fin, "Negative free cash flow", severity, detail, ["fcf"], FCF_LOOKBACK_YEARS)


def _check_cash_runway(fin: FinancialData) -> RedFlag | None:
    (fcf,), (cash,) = _last(fin, "fcf", 1), _last(fin, "cash", 1)
    if fcf >= 0:
        return None
    (investments,) = _last(fin, "short_term_investments", 1) or [0.0]   # optional: many companies hold none
    runway = (cash + investments) / -fcf
    if runway >= MIN_CASH_RUNWAY_YEARS:
        return None
    if investments:
        what, metrics = "cash and short-term investments", ["cash", "short_term_investments", "fcf"]
    else:
        what, metrics = "cash", ["cash", "fcf"]
    detail = (f"{_money(fin, cash + investments)} of {what} covers about {runway:.1f} years of the "
              f"{_span(fin, 1)} free-cash-flow burn ({_money(fin, fcf)}).")
    return _flag(fin, "Short cash runway", "High", detail, metrics, 1)


def _check_revenue_decline(fin: FinancialData) -> RedFlag | None:
    old, new = _last(fin, "revenue", 2)
    growth = _growth(old, new)
    if growth is None or -growth < REVENUE_DECLINE_MEDIUM:
        return None
    prev_year, year = _years(fin, 2)
    detail = (f"Revenue fell {-growth:.1%} from {_money(fin, old)} to {_money(fin, new)} "
              f"in {year} vs {prev_year}.")
    severity = "High" if -growth > REVENUE_DECLINE_HIGH else "Medium"
    return _flag(fin, "Revenue decline", severity, detail, ["revenue"], 2)


def _check_net_loss(fin: FinancialData) -> RedFlag | None:
    (net_income,) = _last(fin, "net_income", 1)
    if net_income >= 0:
        return None
    detail = f"Net loss of {_money(fin, -net_income)} in {_span(fin, 1)}"
    revenue = _last(fin, "revenue", 1)
    if revenue and revenue[0] > 0:
        detail += f" (net margin {_pct(net_income / revenue[0])})"
    return _flag(fin, "Net loss", "Medium", detail + ".", ["net_income"], 1)


def _check_debt_growth(fin: FinancialData) -> RedFlag | None:
    old, new = _last(fin, "debt", 2)
    rev_old, rev_new = _last(fin, "revenue", 2)
    growth, rev_growth = _growth(old, new), _growth(rev_old, rev_new)
    if growth is None or rev_growth is None:
        return None
    if growth <= DEBT_GROWTH_MEDIUM or rev_growth >= DEBT_GROWTH_MAX_REVENUE_GROWTH:
        return None
    prev_year, year = _years(fin, 2)
    detail = (f"Total debt {_signed(growth)} ({_money(fin, old)} → {_money(fin, new)}) while revenue grew "
              f"{_signed(rev_growth)} in {year} vs {prev_year}.")
    return _flag(fin, "Debt rising faster than revenue", "Medium", detail, ["debt", "revenue"], 2)


def _check_debt_to_ocf(fin: FinancialData) -> RedFlag | None:
    (debt,), (ocf,) = _last(fin, "debt", 1), _last(fin, "operating_cash_flow", 1)
    if ocf <= 0 or debt <= MAX_DEBT_TO_OCF * ocf:
        return None
    detail = (f"Total debt of {_money(fin, debt)} is {debt / ocf:.1f}x operating cash flow "
              f"({_money(fin, ocf)}) in {_span(fin, 1)}.")
    return _flag(fin, "High debt relative to cash flow", "Medium", detail, ["debt", "operating_cash_flow"], 1)


def _check_dilution(fin: FinancialData) -> RedFlag | None:
    old, new = _last(fin, "diluted_shares", 2)
    growth = _growth(old, new)
    if growth is None or growth <= DILUTION_MEDIUM:
        return None
    prev_year, year = _years(fin, 2)
    detail = (f"Diluted share count {_signed(growth)} ({_shares(old)} → {_shares(new)} shares) "
              f"in {year} vs {prev_year}.")
    severity = "High" if growth > DILUTION_HIGH else "Medium"
    return _flag(fin, "Shareholder dilution", severity, detail, ["diluted_shares"], 2)


# (name shown in checks_not_run, table columns needed, years of history needed, rule)
NUMERIC_CHECKS = [
    ("Receivables vs revenue", ["receivables", "revenue"], 2, _check_receivables),
    ("Inventory vs revenue", ["inventory", "revenue"], 2, _check_inventory),
    ("Gross margin trend", ["gross_profit", "revenue"], 2, _check_gross_margin),
    ("Operating margin trend", ["operating_income", "revenue"], 2, _check_operating_margin),
    ("Sustained margin compression", ["operating_income", "revenue"], MARGIN_DECLINE_STREAK + 1,
     _check_margin_streak),
    ("Stock comp vs revenue", ["sbc", "revenue"], 1, _check_sbc_vs_revenue),
    ("Stock comp vs free cash flow", ["sbc", "fcf"], 1, _check_sbc_vs_fcf),
    ("Cash conversion (OCF / net income)", ["operating_cash_flow", "net_income"], 1, _check_cash_conversion),
    ("Cumulative OCF vs net income", ["operating_cash_flow", "net_income"], ACCRUAL_YEARS, _check_accruals),
    ("Negative free cash flow", ["fcf"], 1, _check_negative_fcf),
    ("Cash runway", ["fcf", "cash"], 1, _check_cash_runway),
    ("Revenue trend", ["revenue"], 2, _check_revenue_decline),
    ("Net loss", ["net_income"], 1, _check_net_loss),
    ("Debt growth vs revenue", ["debt", "revenue"], 2, _check_debt_growth),
    ("Debt vs operating cash flow", ["debt", "operating_cash_flow"], 1, _check_debt_to_ocf),
    ("Share dilution", ["diluted_shares"], 2, _check_dilution),
]

# Cash-flow and leverage screens that misfire for banks and insurers: deposits, loans and insurance float run
# through operating cash flow, and their borrowings fund the business rather than lever it.
FINANCIAL_SKIP_CHECKS = {"Cash conversion (OCF / net income)", "Cumulative OCF vs net income",
                         "Negative free cash flow", "Cash runway", "Debt growth vs revenue",
                         "Debt vs operating cash flow", "Stock comp vs free cash flow"}


def is_financial_sic(sic: int | None) -> bool:
    """Banks, brokers and insurers (SIC 6000-6399) except health plans (6324). REITs (6798) are not included."""
    return sic is not None and 6000 <= sic <= 6399 and sic != 6324


def _why_not_runnable(fin: FinancialData, columns: list[str], n: int) -> str | None:
    """Reason a check cannot run (too little history or missing values), else None."""
    if len(fin.table) < n:
        return f"needs {n} years of data, have {len(fin.table)}"
    missing = [c.replace("_", " ") for c in columns if _last(fin, c, n) is None]
    if missing:
        return f"no {', '.join(missing)} data for {_span(fin, n)}"
    return None


# ---------------------------------------------------------------------------------------------
# 10-K text scan
# ---------------------------------------------------------------------------------------------

_GOING_CONCERN = re.compile(
    r"\bsubstantial\s+doubt\b[^.]{0,120}?\bability\s+to\s+continue\s+as\s+a\s+going\s+concern\b",
    re.IGNORECASE)
_MATERIAL_WEAKNESS = re.compile(
    r"\bidentified\s+(?:a\s+)?material\s+weakness(?:es)?\b"
    r"|\bmaterial\s+weakness(?:es)?\s+(?:was|were|has\s+been|have\s+been)\s+identified\b",
    re.IGNORECASE)
# "restated our previously issued financial statements", "restatement of the consolidated financial statements"
_RESTATEMENT = re.compile(
    r"\brestate(?:d|ment)?\s+(?:of\s+)?(?:our\s+|its\s+|the\s+)?"
    r"(?:previously\s+issued\s+(?:consolidated\s+)?|consolidated\s+)financial\s+statements\b",
    re.IGNORECASE)
# "sales to one direct customer represented 22% of total revenue",
# "Customer A accounted for approximately 15%, 14% and 13% of net sales" (the first figure is captured).
_CUSTOMER = re.compile(
    r"\b(?:one\s+(?:\w+\s+)?customer|a\s+single\s+customer|largest\s+customer|customer\s+(?-i:[A-Z]))\b"
    r"[^.]{0,80}?\b(?:accounted\s+for|represented)\s+"
    r"(?:approximately\s+|about\s+|nearly\s+|over\s+|more\s+than\s+)?"
    r"(?P<pct>\d{1,3}(?:\.\d+)?)\s?(?:%|percent)"
    r"(?:,?\s+(?:and\s+)?\d{1,3}(?:\.\d+)?\s?(?:%|percent))*"
    r"(?:\s+or\s+more)?"
    r"\s+of\s+(?:(?:our|its|the|total|consolidated|net|worldwide)\s+){0,3}(?:revenues?|net\s+sales|sales)\b",
    re.IGNORECASE)

# (flag name, severity, pattern, detail; {pct} is filled from the match when the pattern captures it)
TEXT_RULES = [
    ("Going-concern doubt", "High", _GOING_CONCERN,
     "The 10-K discloses substantial doubt about the company's ability to continue as a going concern."),
    ("Material weakness in internal control", "High", _MATERIAL_WEAKNESS,
     "The 10-K reports an identified material weakness in internal control over financial reporting."),
    ("Restatement of financial statements", "High", _RESTATEMENT,
     "The 10-K refers to restating previously issued financial statements."),
    ("Customer concentration", "Medium", _CUSTOMER,
     "One customer accounted for {pct}% of revenue, per the 10-K (see quote)."),
]

_NEGATIONS = {"no", "not", "never", "none", "nor", "without", "cannot"}
# Conditional / hypothetical wording typical of risk-factor boilerplate: "if we identify ...",
# "we may be required to restate ...", "a restatement of ... financial statements could ...".
_HEDGES = {"if", "whether", "unless", "event", "could", "may", "might", "would", "should", "can", "any",
           "potential", "potentially", "possible"}
# Benign contexts: a resolved going-concern question, or financial statements recast for a split / accounting change.
_BENIGN_CONTEXT = re.compile(r"alleviat|stock split|retrospective|discontinued operations", re.IGNORECASE)
_SENTENCE_ENDS = (". ", "? ", "! ", "\n")
_ITEM_HEADING = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\b", re.IGNORECASE | re.MULTILINE)


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Start/end offsets of the sentence (or line) containing text[start:end]."""
    left = max(text.rfind(sep, 0, start) for sep in _SENTENCE_ENDS)
    rights = [i for i in (text.find(sep, end) for sep in _SENTENCE_ENDS) if i >= 0]
    return (left + 1 if left >= 0 else 0), (min(rights) + 1 if rights else len(text))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:['’][a-z]+)?", text.lower())


def _is_hedged(before: str, after: str) -> bool:
    """True when the sentence around a match negates it, makes it hypothetical, or is benign.

    before / after are the parts of the sentence preceding and following the matched phrase.
    """
    words_before, words_after = _words(before), _words(after)
    if any(w in _NEGATIONS or w.endswith(("n't", "n’t")) for w in words_before[-NEGATION_WINDOW_WORDS:]):
        return True
    if any(w in _HEDGES for w in words_before[-HEDGE_WINDOW_WORDS:] + words_after[:HEDGE_AFTER_WORDS]):
        return True
    return bool(_BENIGN_CONTEXT.search(before + " " + after))


def _quote(sentence: str, words_before: int) -> str:
    """The sentence, or a window of MAX_QUOTE_WORDS words around the match with … marking cuts."""
    words = sentence.split()
    if len(words) <= MAX_QUOTE_WORDS:
        return " ".join(words)
    first = max(0, min(words_before - 10, len(words) - MAX_QUOTE_WORDS))
    quote = " ".join(words[first:first + MAX_QUOTE_WORDS])
    if first > 0:
        quote = "…" + quote
    if first + MAX_QUOTE_WORDS < len(words):
        quote += "…"
    return quote


def _item_before(text: str, pos: int) -> str | None:
    """Label of the last 'Item N.' heading line before pos, e.g. 'Item 9A'."""
    headings = list(_ITEM_HEADING.finditer(text, 0, pos))
    return f"Item {headings[-1].group(1).upper()}" if headings else None


def _scan_text(text: str) -> list[RedFlag]:
    """At most one flag per text rule: the first match that is not negated or hypothetical."""
    flags = []
    for name, severity, pattern, detail in TEXT_RULES:
        for match in pattern.finditer(text):
            pct = match.groupdict().get("pct")
            if pct is not None and float(pct) < MIN_CUSTOMER_SHARE_PCT:
                continue
            start, end = _sentence_bounds(text, match.start(), match.end())
            before = text[start:match.start()]
            if _is_hedged(before, text[match.end():end]):
                continue
            item = _item_before(text, match.start())
            flags.append(RedFlag(
                id="", name=name, severity=severity, detail=detail.format(pct=pct),
                source=f"10-K text: {item}" if item else "10-K text",
                quote=_quote(text[start:end], len(before.split())),
            ))
            break
    return flags


# ---------------------------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------------------------

def detect_red_flags(fin: FinancialData | None, filing_text: str | None = None,
                     sic: int | None = None) -> tuple[list[RedFlag], list[str]]:
    """Run every numeric and text rule (banks and insurers by `sic` skip FINANCIAL_SKIP_CHECKS).

    Returns (flags, checks_not_run). Flags are numbered R1, R2, ... in rule order; checks_not_run
    names each check skipped and why, e.g. "Inventory vs revenue (no inventory data for FY2025)".
    """
    flags: list[RedFlag] = []
    not_run: list[str] = []

    if fin is None or fin.table.empty:
        not_run.append("Financial checks (no XBRL financial data)")
    else:
        financial = is_financial_sic(sic)
        for name, columns, years, rule in NUMERIC_CHECKS:
            if financial and name in FINANCIAL_SKIP_CHECKS:
                reason = f"not meaningful for banks and insurers, SIC {sic}"
            else:
                reason = _why_not_runnable(fin, columns, years)
            if reason:
                not_run.append(f"{name} ({reason})")
                continue
            flag = rule(fin)
            if flag:
                flags.append(flag)

    if filing_text and filing_text.strip():
        flags.extend(_scan_text(filing_text))
    else:
        not_run.append("10-K text scan (no 10-K text)")

    for i, flag in enumerate(flags, start=1):
        flag.id = f"R{i}"
    return flags, not_run
