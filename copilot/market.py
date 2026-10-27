"""Price and market cap from yfinance, plus valuation multiples."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import yfinance as yf

from .edgar import normalize_ticker
from .financials import FinancialData, format_money, to_float
from .red_flags import is_financial_sic


@dataclass
class MarketData:
    ticker: str
    price: float | None
    market_cap: float | None
    currency: str | None
    as_of: str                      # YYYY-MM-DD
    source: str = "Yahoo Finance via yfinance"
    error: str | None = None


def get_market_data(ticker: str, shares_fallback: float | None = None) -> MarketData:
    """Latest price, market cap and currency from Yahoo Finance. Never raises; problems go to `error`.

    `shares_fallback` (e.g. the SEC cover-page share count) gives market cap = price x shares when Yahoo
    has a price but no market cap.
    """
    symbol = normalize_ticker(ticker)          # Yahoo writes share classes like EDGAR: BRK-B
    mkt = MarketData(ticker=symbol, price=None, market_cap=None, currency=None, as_of=date.today().isoformat())
    try:
        stock = yf.Ticker(symbol)
        info = stock.fast_info              # lazy: each attribute may trigger a request or raise KeyError
        mkt.price = _positive(lambda: info.last_price)
        mkt.market_cap = _positive(lambda: info.market_cap)
        mkt.currency = _text(lambda: info.currency)
        if mkt.price is None:
            closes = stock.history(period="5d")["Close"].dropna()
            if not closes.empty:
                mkt.price = float(closes.iloc[-1])
                mkt.as_of = closes.index[-1].date().isoformat()
    except Exception as exc:  # network errors, rate limits, Yahoo format changes
        mkt.error = f"Yahoo Finance request failed ({type(exc).__name__}: {str(exc)[:200]})"

    if mkt.price is None:
        mkt.market_cap = None
        mkt.error = mkt.error or f"No price data on Yahoo Finance for '{symbol}' (invalid or delisted ticker?)."
    elif mkt.market_cap is None and shares_fallback and shares_fallback > 0:
        mkt.market_cap = mkt.price * shares_fallback
        mkt.source += " (market cap = price x SEC cover-page shares outstanding)"
    return mkt


def compute_valuation(fin: FinancialData | None, mkt: MarketData | None, sic: int | None = None) -> dict:
    """P/E, EV/Sales and FCF yield from the market cap and the LATEST fiscal year (not trailing twelve months).

    EV: see _enterprise_value. Banks and insurers (`sic`, see red_flags.is_financial_sic) get P/E only. A
    multiple whose inputs are missing, or whose currencies differ, is None with a plain-English note; when P/E
    is None because the latest year was a loss, `pe_note` is "n/m (loss)" for display.
    """
    out = {"price": None, "market_cap": None, "currency": None, "enterprise_value": None, "pe": None,
           "pe_note": None, "ev_sales": None, "fcf_yield": None, "basis_year": None, "as_of": None, "notes": []}
    notes = out["notes"]
    if mkt is not None:
        out.update(price=mkt.price, market_cap=mkt.market_cap, currency=mkt.currency, as_of=mkt.as_of)
        if mkt.error:
            notes.append(f"Market data unavailable: {mkt.error}")
    else:
        notes.append("No market data.")
    if fin is None or fin.table.empty:
        notes.append("No XBRL financials, so no valuation multiples.")
        return out

    row = fin.latest
    basis = fin.years[-1]
    out["basis_year"] = basis
    out["currency"] = out["currency"] or fin.currency
    market_cap = to_float(out["market_cap"])
    if market_cap is None:
        notes.append("Market cap unavailable, so no valuation multiples.")
        return out
    market_currency = mkt.currency if mkt else None
    if market_currency and market_currency != fin.currency:
        notes.append(f"Currency mismatch: market data in {market_currency}, filings in {fin.currency}; "
                     "multiples not computed.")
        return out

    revenue, net_income, fcf = to_float(row["revenue"]), to_float(row["net_income"]), to_float(row["fcf"])
    if net_income is None:
        notes.append(f"P/E not computed: no {basis} net income.")
    elif net_income <= 0:
        out["pe_note"] = "n/m (loss)"
        notes.append(f"P/E n/m (loss): {basis} net income was {format_money(net_income, fin.currency)}.")
    else:
        out["pe"] = market_cap / net_income

    if is_financial_sic(sic):
        notes.append("EV, EV/Sales and FCF yield are not meaningful for banks and insurers "
                     "(deposits and insurance float are not ordinary cash or debt); use P/E.")
        return out

    ev = out["enterprise_value"] = _enterprise_value(fin, market_cap, notes)
    if ev is not None and revenue is not None and revenue > 0:
        out["ev_sales"] = ev / revenue
    elif ev is not None:
        notes.append(f"EV/Sales not computed: no {basis} revenue.")

    if fcf is None:
        notes.append(f"FCF yield not computed: no {basis} free cash flow (operating cash flow or capex missing).")
    else:
        out["fcf_yield"] = fcf / market_cap
    return out


def _enterprise_value(fin: FinancialData, market_cap: float, notes: list[str]) -> float | None:
    """Market cap + total debt - cash - short-term investments for the latest fiscal year.

    Missing debt counts as zero when debt was tagged in another year (e.g. loans repaid); with no debt tags in
    any year (Ford and Berkshire use their own) there is no EV. financials' notes already explain debt gaps.
    Missing short-term investments count as zero, with a note when they were tagged in earlier years.
    """
    row, basis = fin.latest, fin.years[-1]
    debt, cash = to_float(row["debt"]), to_float(row["cash"])
    sti = to_float(row.get("short_term_investments"))
    if cash is None:
        notes.append(f"Cash not reported for {basis}, so no enterprise value.")
        return None
    if debt is None and not _ever_reported(fin, "debt"):
        return None
    if sti is None and _ever_reported(fin, "short_term_investments"):
        notes.append(f"Short-term investments not tagged for {basis} (they were in earlier years), "
                     "so EV subtracts cash only and may be overstated.")
    return market_cap + (debt or 0.0) - cash - (sti or 0.0)


def _ever_reported(fin: FinancialData, column: str) -> bool:
    values = fin.table.get(column)
    return values is not None and bool(values.notna().any())


def _positive(get) -> float | None:
    """Call a lazy yfinance getter; None if it raises or returns a missing / non-positive number."""
    try:
        value = float(get())
    except Exception:
        return None
    return value if math.isfinite(value) and value > 0 else None


def _text(get) -> str | None:
    try:
        value = get()
    except Exception:
        return None
    return value if isinstance(value, str) and value else None
