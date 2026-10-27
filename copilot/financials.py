"""Five years of fundamentals from SEC XBRL company facts.

How annual values are picked (build_financials):
- Annual facts only: forms 10-K, 10-K/A, 10-KT. Flow items (revenue, cash flow, ...) must span 330-400 days
  (52/53-week years included); balance-sheet items are point-in-time.
- A period is identified by its end date. When several filings report the same period, the latest filing wins,
  so restatements replace the original numbers. The `fy` field only names the year (see _year_labels).
- Fiscal years = the newest period ends of the revenue and net-income series, at least 300 days apart.
- Revenue: companies switch tags over time and some report a subset next to the total (e.g. REITs), so each
  year takes the largest value across REVENUE_TAGS.
- Other metrics: the first tag in priority order with a value that year. Tags that report the newest year are
  tried first, so a series does not mix a retired tag with its replacement.
- Balance-sheet values match the fiscal-year end exactly, else the nearest date within 10 days.
- Total debt: see _total_debt (operating leases excluded). Short-term investments: see _short_term_investments.
  FCF = operating cash flow - capex.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from .config import FISCAL_YEARS
from .edgar import EdgarError, sec_get_json

COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"

ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A"}
YEAR_DAYS = (330, 400)          # length of a fiscal year in days, 52/53-week years included
WEEK53_DAYS = 370               # a 53-week fiscal year spans 370 days (a 52-week one 363-364)
MIN_DAYS_BETWEEN_YEARS = 300    # period ends closer than this are the same fiscal year reported twice
MATCH_TOLERANCE_DAYS = 10       # a balance-sheet date may differ slightly from the fiscal-year end

# Revenue: per year, the MAX across these tags.
REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
    "RevenuesNetOfInterestExpense",              # banks and brokers
    "RegulatedAndUnregulatedOperatingRevenue",   # utilities
    "OperatingLeaseLeaseIncome",                 # landlords and REITs
    "RealEstateRevenueNet",
]

# Other metrics: per year, the first tag (in priority order) that reports a value.
DURATION_TAGS = {
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsForCapitalImprovements", "PaymentsToAcquireOtherPropertyPlantAndEquipment"],  # last: LLY
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
    "diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding",          # unit "shares"
                       "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
                       "WeightedAverageNumberOfSharesOutstandingBasic"],
}
INSTANT_TAGS = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash", "CashAndDueFromBanks"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent",
                    "AccountsNotesAndLoansReceivableNetCurrent"],
    "inventory": ["InventoryNet", "RetailRelatedInventoryMerchandise", "InventoryFinishedGoodsNetOfReserves",
                  "InventoryFinishedGoods"],
}

# Short-term investments (balance-sheet tags); see _short_term_investments.
CASH_AND_STI_TAG = "CashCashEquivalentsAndShortTermInvestments"
STI_TAGS = ["ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"]

# Total-debt building blocks (balance-sheet tags); see _total_debt.
DEBT_TOTAL_TAGS = ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities"]
DEBT_NONCURRENT_TAGS = ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermNotesPayable"]
DEBT_CURRENT_TAGS = ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent", "NotesPayableCurrent"]
SHORT_TERM_DEBT_TAGS = ["ShortTermBorrowings", "CommercialPaper", "OtherShortTermBorrowings"]
OTHER_DEBT_TAGS = ["NotesPayable", "LoansPayable", "SecuredDebt", "ConvertibleDebtNoncurrent"]
DEBT_TAGS = (DEBT_TOTAL_TAGS + DEBT_NONCURRENT_TAGS + DEBT_CURRENT_TAGS + SHORT_TERM_DEBT_TAGS
             + ["DebtCurrent"] + OTHER_DEBT_TAGS)

# Columns of FinancialData.table (besides period_end), in display order.
METRIC_COLUMNS = [
    "revenue", "gross_profit", "operating_income", "net_income", "operating_cash_flow", "capex", "fcf",
    "cash", "short_term_investments", "debt", "sbc", "receivables", "inventory", "diluted_shares",
    "gross_margin", "operating_margin", "net_margin", "fcf_margin", "revenue_growth", "sbc_pct_revenue",
]
# Metrics named in a note when missing in every year (not short-term investments: many companies hold none).
EXPECTED_METRICS = ["revenue", "gross_profit", "operating_income", "net_income", "operating_cash_flow", "capex",
                    "cash", "sbc", "receivables", "inventory", "diluted_shares"]

# (display label, column, kind) rows for the memo's financial snapshot table.
SNAPSHOT_ROWS = [
    ("Revenue", "revenue", "money"),
    ("Revenue growth", "revenue_growth", "pct"),
    ("Gross margin", "gross_margin", "pct"),
    ("Operating margin", "operating_margin", "pct"),
    ("Net income", "net_income", "money"),
    ("Net margin", "net_margin", "pct"),
    ("Operating cash flow", "operating_cash_flow", "money"),
    ("Capex", "capex", "money"),
    ("Free cash flow", "fcf", "money"),
    ("FCF margin", "fcf_margin", "pct"),
    ("Cash & equivalents", "cash", "money"),
    ("Short-term investments", "short_term_investments", "money"),
    ("Total debt", "debt", "money"),
    ("Stock-based comp", "sbc", "money"),
    ("SBC % revenue", "sbc_pct_revenue", "pct"),
]


@dataclass
class FinancialData:
    table: pd.DataFrame                      # index: "FY2021".."FY2025" oldest -> newest; columns: period_end + METRIC_COLUMNS
    tags_used: dict[str, list[str]] = field(default_factory=dict)
    currency: str = "USD"
    entity_name: str = ""
    shares_outstanding: float | None = None  # dei:EntityCommonStockSharesOutstanding (cover page)
    notes: list[str] = field(default_factory=list)

    @property
    def latest(self) -> pd.Series:
        return self.table.iloc[-1]

    @property
    def years(self) -> list[str]:
        return list(self.table.index)


def to_float(x) -> float | None:
    """A finite float, or None for None / NaN / inf / non-numbers (bools included)."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):   # numbers.Real covers numpy scalars
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def format_price(x, currency: str = "USD") -> str:
    """342.9 -> '$342.90' (other currencies: 'EUR 342.90'); None/NaN -> 'n/a'."""
    x = to_float(x)
    if x is None:
        return "n/a"
    return f"${x:,.2f}" if currency == "USD" else f"{currency} {x:,.2f}"


def format_multiple(x) -> str:
    """44.71 -> '44.7x'; None/NaN -> 'n/a'."""
    x = to_float(x)
    return "n/a" if x is None else f"{x:,.1f}x"


def format_money(x, currency: str = "USD") -> str:
    """391_035_000_000 -> '$391.0B'; None/NaN -> 'n/a'."""
    x = to_float(x)
    if x is None:
        return "n/a"
    symbol = "$" if currency == "USD" else f"{currency} "
    sign = "-" if x < 0 else ""
    x = abs(x)
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= div:
            return f"{sign}{symbol}{x / div:,.1f}{suffix}"
    return f"{sign}{symbol}{x:,.0f}"


def format_pct(x) -> str:
    """0.4523 -> '45.2%'; None/NaN -> 'n/a'."""
    x = to_float(x)
    return "n/a" if x is None else f"{x * 100:.1f}%"


def snapshot_table(fin: FinancialData) -> pd.DataFrame:
    """Display table: one row per metric, one column per fiscal year (oldest -> newest), formatted strings."""
    data = {}
    for year in fin.table.index:
        row = fin.table.loc[year]
        data[year] = [
            format_money(row.get(col), fin.currency) if kind == "money" else format_pct(row.get(col))
            for _, col, kind in SNAPSHOT_ROWS
        ]
    return pd.DataFrame(data, index=[label for label, _, _ in SNAPSHOT_ROWS])


def get_company_facts(cik: int, user_agent: str) -> dict:
    """Download every XBRL fact SEC holds for a company (one JSON document, typically a few MB)."""
    try:
        return sec_get_json(COMPANY_FACTS_URL.format(cik10=f"{cik:010d}"), user_agent)
    except EdgarError as exc:
        if str(exc).startswith("Not found"):   # HTTP 404: the company never filed XBRL financial statements
            raise EdgarError("SEC has no XBRL financial data (company facts) for this company.") from exc
        raise


def load_financials(cik: int, user_agent: str) -> FinancialData:
    """Fetch company facts from EDGAR and build the five-year table."""
    return build_financials(get_company_facts(cik, user_agent))


def build_financials(facts: dict, years: int = FISCAL_YEARS) -> FinancialData:
    """Turn a company-facts JSON into up to `years` fiscal years of fundamentals (pure function).

    Raises EdgarError (user-facing message) when there is no annual US-GAAP data at all, e.g. a 20-F filer
    reporting under IFRS or a new registrant that has not filed a 10-K yet. Partial data returns normally,
    with gaps as NaN and an explanation in `notes`.
    """
    name = facts.get("entityName") or "this company"
    gaap = _us_gaap(facts, name)
    currency = _reporting_currency(gaap)
    revenue = _revenue_series(gaap, currency)
    net_income_ends = {end for tag in DURATION_TAGS["net_income"]
                       for end in _annual_series(gaap, tag, currency, duration=True)}
    ends = _fiscal_year_ends(set(revenue) | net_income_ends, years)
    if not ends:
        raise EdgarError(f"SEC XBRL data has no annual (10-K) revenue or net income for {name}; "
                         "it may not have filed a 10-K yet.")
    labels = _year_labels(gaap, currency, ends)

    picked = {"revenue": [_lookup(revenue, end) for end in ends]}
    for metric, tags in DURATION_TAGS.items():
        unit = "shares" if metric == "diluted_shares" else currency
        picked[metric] = _pick(gaap, tags, unit, ends, duration=True)
    for metric, tags in INSTANT_TAGS.items():
        picked[metric] = _pick(gaap, tags, currency, ends, duration=False)
    data = {metric: _values(series) for metric, series in picked.items()}
    tags_used = {metric: _tags(series) for metric, series in picked.items() if _tags(series)}

    notes = _derive_columns(data, tags_used)
    data["short_term_investments"], sti_tags = _short_term_investments(gaap, currency, ends, data["cash"])
    data["debt"], debt_tags, latest_debt_tags = _debt_series(gaap, currency, ends)
    if sti_tags:
        tags_used["short_term_investments"] = sti_tags
    if debt_tags:
        tags_used["debt"] = debt_tags
    notes += _debt_notes(data["debt"], labels, latest_debt_tags)

    data["diluted_shares"], split_years = _split_adjusted_shares(gaap, picked["diluted_shares"], labels)
    if split_years:
        notes.append(f"Diluted shares for {', '.join(split_years)} multiplied for later stock splits "
                     "(older 10-Ks are not restated).")

    table = _make_table(data, ends, labels)
    restated = _restated_growth_years(gaap, currency, picked["revenue"], labels)
    if restated:
        table.loc[restated, "revenue_growth"] = math.nan
        notes.append(f"Revenue growth for {', '.join(restated)} not shown: a later 10-K restated revenue (e.g. a "
                     "spin-off or discontinued operations), so the prior year is on a different basis.")
    notes += _coverage_notes(data, picked["revenue"], labels, years, currency)

    return FinancialData(
        table=table,
        tags_used=tags_used,
        currency=currency,
        entity_name=facts.get("entityName") or "",
        shares_outstanding=_cover_page_shares(facts["facts"].get("dei") or {}, ends[-1]),
        notes=notes,
    )


def _us_gaap(facts: dict, name: str) -> dict:
    """The us-gaap facts, or a clear EdgarError for IFRS filers and companies without XBRL data."""
    taxonomies = facts.get("facts") or {}
    gaap = taxonomies.get("us-gaap") or {}
    if not gaap and taxonomies.get("ifrs-full"):
        raise EdgarError(f"{name} reports under IFRS (a 20-F/40-F filer), not US GAAP. "
                         "This app reads US-GAAP financials from 10-K filings only.")
    if not gaap:
        raise EdgarError(f"SEC has no US-GAAP XBRL financial data for {name}.")
    return gaap


def _derive_columns(data: dict[str, list[float | None]], tags_used: dict[str, list[str]]) -> list[str]:
    """Normalize capex, fill gross profit from cost of revenue and compute FCF (in place); returns notes."""
    notes = []
    # Capex is a payment (positive in XBRL); a few filers sign it negative.
    data["capex"] = [None if v is None else abs(v) for v in data["capex"]]

    computed_gross = False
    for i, (rev, cost) in enumerate(zip(data["revenue"], data["cost_of_revenue"])):
        if data["gross_profit"][i] is None and rev is not None and cost is not None:
            data["gross_profit"][i] = rev - cost
            computed_gross = True
    if computed_gross:
        cost_tags = ", ".join(tags_used["cost_of_revenue"])
        tags_used.setdefault("gross_profit", []).append(f"revenue - {cost_tags}")
        notes.append(f"Gross profit = revenue - cost of revenue ({cost_tags}) where GrossProfit is not tagged. "
                     "Revenue here includes all revenue (e.g. membership or service fees), so this margin can "
                     "differ from the gross margin the company reports.")

    data["fcf"] = [ocf - capex if ocf is not None and capex is not None else None
                   for ocf, capex in zip(data["operating_cash_flow"], data["capex"])]
    fcf_tags = tags_used.get("operating_cash_flow", []) + tags_used.get("capex", [])
    if fcf_tags:
        tags_used["fcf"] = fcf_tags
    return notes


def _debt_notes(debt: list[float | None], labels: list[str], latest_tags: list[str]) -> list[str]:
    """The notes on total debt; market.compute_valuation follows the same rules for EV."""
    missing = [label for label, value in zip(labels, debt) if value is None]
    if len(missing) == len(debt):
        return ["Total debt not tagged in XBRL in any year (the company may use its own tags), "
                "so no enterprise value is computed."]
    notes = []
    if debt[-1] is not None:
        notes.append(f"Total debt ({labels[-1]}) = {' + '.join(latest_tags)}; excludes operating leases.")
    if missing:
        ev = "; EV treats it as zero" if debt[-1] is None else ""
        notes.append(f"Total debt not tagged in XBRL for {', '.join(missing)} (shown as n/a{ev}).")
    return notes


def _coverage_notes(data: dict[str, list[float | None]], revenue: list[dict | None], labels: list[str],
                    years: int, currency: str) -> list[str]:
    """Notes on metrics missing in every year, a short history, 53-week years and a non-USD currency."""
    notes = []
    missing = [m.replace("_", " ") for m in EXPECTED_METRICS if all(v is None for v in data[m])]
    if missing:
        notes.append("Not reported in XBRL (shown as n/a): " + ", ".join(missing) + ".")
    if len(labels) < years:
        notes.append(f"Only {len(labels)} fiscal year(s) of 10-K data available.")
    long_years = [label for label, fact in zip(labels, revenue) if fact and _days(fact) >= WEEK53_DAYS]
    if long_years:
        notes.append(f"{', '.join(long_years)} had 53 weeks, which adds about 2 points to that year's revenue "
                     "growth and takes about 2 points off the next year's.")
    if currency != "USD":
        notes.append(f"Figures are in {currency}, the currency of the XBRL filings.")
    return notes


# ---------------------------------------------------------------------------------------------------
# Fact selection
# ---------------------------------------------------------------------------------------------------

def _days(fact: dict) -> int:
    return (date.fromisoformat(fact["end"]) - date.fromisoformat(fact["start"])).days


def _is_annual(fact: dict, duration: bool) -> bool:
    """True for a 10-K fact covering one fiscal year (duration) or one balance-sheet date (instant)."""
    if fact.get("form") not in ANNUAL_FORMS or fact.get("val") is None or not fact.get("end"):
        return False
    if not duration:
        return "start" not in fact
    return "start" in fact and YEAR_DAYS[0] <= _days(fact) <= YEAR_DAYS[1]


def _annual_series(gaap: dict, tag: str, unit: str, duration: bool) -> dict[date, dict]:
    """Annual facts for one tag keyed by period end; the latest filing wins (restatements)."""
    series: dict[date, dict] = {}
    for fact in gaap.get(tag, {}).get("units", {}).get(unit, []):
        if not _is_annual(fact, duration):
            continue
        end = date.fromisoformat(fact["end"])
        if end not in series or fact.get("filed", "") > series[end].get("filed", ""):
            series[end] = fact
    return series


def _revenue_series(gaap: dict, unit: str) -> dict[date, dict]:
    """Largest revenue fact per period end across REVENUE_TAGS (each fact gains a "tag" key)."""
    best: dict[date, dict] = {}
    for tag in REVENUE_TAGS:
        for end, fact in _annual_series(gaap, tag, unit, duration=True).items():
            if end not in best or fact["val"] > best[end]["val"]:
                best[end] = {**fact, "tag": tag}
    return best


def _pick(gaap: dict, tags: list[str], unit: str, ends: list[date], duration: bool) -> list[dict | None]:
    """Per fiscal-year end, the fact from the first tag that reports it (each fact gains a "tag" key)."""
    series = {tag: _annual_series(gaap, tag, unit, duration) for tag in tags}
    # Tags still used in the newest year go first (stable sort keeps the priority order otherwise).
    tags = sorted(tags, key=lambda tag: _lookup(series[tag], ends[-1]) is None)
    picked: list[dict | None] = []
    for end in ends:
        hit = None
        for tag in tags:
            fact = _lookup(series[tag], end)
            if fact is not None:
                hit = {**fact, "tag": tag}
                break
        picked.append(hit)
    return picked


def _lookup(series: dict[date, dict], end: date) -> dict | None:
    """The fact for a fiscal-year end: exact date, else the nearest within MATCH_TOLERANCE_DAYS."""
    if end in series:
        return series[end]
    near = [d for d in series if abs((d - end).days) <= MATCH_TOLERANCE_DAYS]
    return series[min(near, key=lambda d: abs((d - end).days))] if near else None


def _values(facts: list[dict | None]) -> list[float | None]:
    return [None if fact is None else float(fact["val"]) for fact in facts]


def _tags(facts: list[dict | None]) -> list[str]:
    """Distinct tags behind a picked series, oldest year first."""
    used: list[str] = []
    for fact in facts:
        if fact is not None and fact["tag"] not in used:
            used.append(fact["tag"])
    return used


def _reporting_currency(gaap: dict) -> str:
    """'USD' unless revenue and net income are reported only in another currency."""
    units = [unit for tag in REVENUE_TAGS + DURATION_TAGS["net_income"]
             for unit in gaap.get(tag, {}).get("units", {})]
    return "USD" if "USD" in units or not units else units[0]


def _fiscal_year_ends(candidates: set[date], years: int) -> list[date]:
    """The newest `years` period ends, at least MIN_DAYS_BETWEEN_YEARS apart, oldest first."""
    chosen: list[date] = []
    for end in sorted(candidates, reverse=True):
        if chosen and (chosen[-1] - end).days < MIN_DAYS_BETWEEN_YEARS:
            continue
        chosen.append(end)
        if len(chosen) == years:
            break
    return chosen[::-1]


def _year_labels(gaap: dict, unit: str, ends: list[date]) -> list[str]:
    """'FY2025' in the filer's own numbering; 'FY2020 (2020-01-31)' when two periods get the same number.

    Retailers often name a year by its start (Home Depot's fiscal 2025 ends 2026-02-01), and 52/53-week years
    end on either side of 1 January (Snap-on), so the calendar year of the end is not enough. Each period takes
    the `fy` of the 10-K that reported it as its own (newest) year. Periods only seen as comparatives (years
    before a company's first 10-K) shift by the newest year's offset.
    """
    own_fy = _own_year_fy(gaap, unit)
    offset = own_fy.get(ends[-1], ends[-1].year) - ends[-1].year
    years = [own_fy.get(end, end.year + offset) for end in ends]
    return [f"FY{y}" if years.count(y) == 1 else f"FY{y} ({end.isoformat()})" for y, end in zip(years, ends)]


def _own_year_fy(gaap: dict, unit: str) -> dict[date, int]:
    """Period end -> `fy` of a 10-K whose newest annual revenue / net-income period ends then.

    A 10-K tags its comparatives with its own fy too, so only each filing's newest period is used. An fy more
    than a year before the end's calendar year (or after it) is treated as a tagging error and ignored.
    """
    facts = [fact for tag in REVENUE_TAGS + DURATION_TAGS["net_income"]
             for fact in gaap.get(tag, {}).get("units", {}).get(unit, []) if _is_annual(fact, duration=True)]
    newest: dict[str, str] = {}
    for fact in facts:
        newest[fact.get("accn")] = max(newest.get(fact.get("accn"), ""), fact["end"])
    own_fy: dict[date, int] = {}
    for fact in facts:
        end = date.fromisoformat(fact["end"])
        if fact["end"] == newest[fact.get("accn")] and fact.get("fy") in (end.year - 1, end.year):
            own_fy[end] = fact["fy"]
    return own_fy


# ---------------------------------------------------------------------------------------------------
# Short-term investments, debt, share counts, table
# ---------------------------------------------------------------------------------------------------

def _short_term_investments(gaap: dict, unit: str, ends: list[date],
                            cash: list[float | None]) -> tuple[list[float | None], list[str]]:
    """Short-term investments per fiscal-year end, plus the tags used.

    When the balance sheet tags one 'cash, cash equivalents and short-term investments' total, the value is
    that total minus cash, so nothing already in cash is counted twice (Target's short-term investments are
    cash equivalents). Otherwise the first of STI_TAGS that reports the year.
    """
    totals = _values(_pick(gaap, [CASH_AND_STI_TAG], unit, ends, duration=False))
    picked = _pick(gaap, STI_TAGS, unit, ends, duration=False)
    values: list[float | None] = []
    tags: list[str] = []
    for total, cash_value, fact in zip(totals, cash, picked):
        if total is not None and cash_value is not None:
            value, tag = max(total - cash_value, 0.0), f"{CASH_AND_STI_TAG} - cash"
        elif fact is not None:
            value, tag = float(fact["val"]), fact["tag"]
        else:
            value, tag = None, None
        values.append(value)
        if tag and tag not in tags:
            tags.append(tag)
    return values, tags


def _total_debt(values: dict[str, float]) -> tuple[float | None, list[str]]:
    """Total debt at one balance-sheet date from {tag: value}, plus the tags summed.

    1. Long-term debt including current maturities (LongTermDebt), else noncurrent + current portion.
       A zero LongTermDebt next to positive noncurrent/current tags is a stray fact and is ignored (ORCL).
    2. Plus short-term borrowings: ShortTermBorrowings, else the larger of CommercialPaper and
       OtherShortTermBorrowings (BAC tags its whole short-term line as the latter; KO only a sliver).
       When neither a current portion nor short-term borrowings is tagged, DebtCurrent (which covers both)
       is used instead. This also applies next to LongTermDebt, which some filers (TSLA) use for the
       noncurrent part only.
    3. Filers with no long-term debt tag at all (common for REITs) -> sum of OTHER_DEBT_TAGS + short-term.
    """
    def first(tags: list[str]) -> str | None:
        return next((tag for tag in tags if tag in values), None)

    short = "ShortTermBorrowings" if "ShortTermBorrowings" in values else max(
        (tag for tag in SHORT_TERM_DEBT_TAGS[1:] if tag in values), key=values.get, default=None)
    current = first(DEBT_CURRENT_TAGS)
    noncurrent = first(DEBT_NONCURRENT_TAGS)
    total = first(DEBT_TOTAL_TAGS)
    if total and values[total] == 0 and any(values.get(tag, 0) > 0 for tag in (noncurrent, current) if tag):
        total = None
    if current is None and short is None and total in (None, "LongTermDebt"):
        short = first(["DebtCurrent"])
    if total:
        parts = [total, short]
    elif noncurrent or current:
        parts = [noncurrent, current, short]
    else:
        parts = [tag for tag in OTHER_DEBT_TAGS if tag in values] + [short]
    parts = [tag for tag in parts if tag]
    if not parts:
        return None, []
    return float(sum(values[tag] for tag in parts)), parts


def _debt_series(gaap: dict, unit: str, ends: list[date]) -> tuple[list[float | None], list[str], list[str]]:
    """Total debt per fiscal-year end, all tags used, and the tags behind the newest year."""
    series = {tag: _annual_series(gaap, tag, unit, duration=False) for tag in DEBT_TAGS}
    totals: list[float | None] = []
    all_tags: list[str] = []
    parts: list[str] = []
    for end in ends:
        at_end = {}
        for tag, by_end in series.items():
            fact = _lookup(by_end, end)
            if fact is not None:
                at_end[tag] = float(fact["val"])
        total, parts = _total_debt(at_end)
        totals.append(total)
        all_tags += [tag for tag in parts if tag not in all_tags]
    return totals, all_tags, parts


def _split_adjusted_shares(gaap: dict, picked: list[dict | None],
                           labels: list[str]) -> tuple[list[float | None], list[str]]:
    """Share counts on the latest post-split basis, and the labels of the years that needed adjusting."""
    values: list[float | None] = []
    adjusted: list[str] = []
    for label, fact in zip(labels, picked):
        if fact is None:
            values.append(None)
            continue
        factor = _split_factor(gaap, fact["tag"], fact.get("filed", ""))
        if factor != 1:
            adjusted.append(label)
        values.append(float(fact["val"]) * factor)
    return values, adjusted


def _split_factor(gaap: dict, tag: str, filed: str) -> float:
    """Multiplier that puts a share count filed on `filed` on the latest post-split basis.

    After a stock split, the next 10-K restates prior-year share counts. So when a later 10-K reports the same
    period at a whole-number ratio (2x to 50x, or the inverse for a reverse split), a split happened between
    the two filings, and counts filed before the restating 10-K must be multiplied by that ratio.
    """
    reports: dict[str, list[dict]] = {}
    for fact in gaap.get(tag, {}).get("units", {}).get("shares", []):
        if _is_annual(fact, duration=True):
            reports.setdefault(fact["end"], []).append(fact)
    splits: dict[str, float] = {}  # filing date of the first restated 10-K -> split ratio
    for facts in reports.values():
        facts.sort(key=lambda f: f.get("filed", ""))
        for old, new in zip(facts, facts[1:]):
            if old["val"] > 0 and new["val"] > 0 and old.get("filed") != new.get("filed"):
                ratio = new["val"] / old["val"]
                big = max(ratio, 1 / ratio)
                if 2 <= round(big) <= 50 and abs(big - round(big)) <= 0.02 * big:
                    splits[new.get("filed", "")] = ratio
    factor = 1.0
    for split_filed, ratio in splits.items():
        if filed < split_filed:
            factor *= ratio
    return factor


def _restated_growth_years(gaap: dict, unit: str, revenue: list[dict | None], labels: list[str]) -> list[str]:
    """Labels of years whose revenue growth would compare two different reporting bases.

    Each year shows its latest (restated) figure. If the filing behind the prior year's figure reported this
    year more than 10% differently, this year was restated afterwards (e.g. a spin-off moved to discontinued
    operations) while the prior year was not.
    """
    restated = []
    for i in range(1, len(revenue)):
        current, prior = revenue[i], revenue[i - 1]
        if current is None or prior is None:
            continue
        in_prior_filing = [f["val"] for f in gaap.get(current["tag"], {}).get("units", {}).get(unit, [])
                           if f.get("accn") == prior.get("accn") and f["end"] == current["end"]
                           and _is_annual(f, duration=True)]
        if in_prior_filing and abs(in_prior_filing[0] - current["val"]) > 0.10 * abs(current["val"]):
            restated.append(labels[i])
    return restated


def _cover_page_shares(dei: dict, newest_end: date) -> float | None:
    """Latest dei:EntityCommonStockSharesOutstanding, if current and for a single share class."""
    facts = dei.get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    if not facts:
        return None
    latest = max(facts, key=lambda f: (f.get("end", ""), f.get("filed", "")))
    same_date = {f["val"] for f in facts if f.get("end") == latest.get("end") and f.get("accn") == latest.get("accn")}
    if len(same_date) > 1:   # one count per share class: no single number to multiply by one price
        return None
    if date.fromisoformat(latest["end"]) < newest_end - timedelta(days=365):   # stale (stopped reporting it)
        return None
    return float(latest["val"])


def _make_table(data: dict[str, list[float | None]], ends: list[date], labels: list[str]) -> pd.DataFrame:
    """FinancialData.table: raw metrics plus ratios, oldest year first."""
    raw = [col for col in METRIC_COLUMNS if col in data]
    table = pd.DataFrame({col: data[col] for col in raw}, index=labels, dtype="float64")
    revenue = table["revenue"].where(table["revenue"] > 0)
    table["gross_margin"] = table["gross_profit"] / revenue
    table["operating_margin"] = table["operating_income"] / revenue
    table["net_margin"] = table["net_income"] / revenue
    table["fcf_margin"] = table["fcf"] / revenue
    table["sbc_pct_revenue"] = table["sbc"] / revenue
    # Growth only between consecutive fiscal years (not across a gap in the data).
    previous = revenue.shift(1)
    gap_days = pd.Series([math.nan] + [(b - a).days for a, b in zip(ends, ends[1:])], index=labels)
    table["revenue_growth"] = (table["revenue"] / previous - 1).where(gap_days <= YEAR_DAYS[1])
    table.insert(0, "period_end", [end.isoformat() for end in ends])
    return table[["period_end"] + METRIC_COLUMNS]
