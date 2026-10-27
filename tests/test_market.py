"""Offline tests for copilot.market: valuation math and get_market_data with a fake yfinance."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from copilot import market
from copilot.financials import FinancialData, build_financials
from copilot.market import MarketData, compute_valuation, get_market_data
from copilot.red_flags import detect_red_flags

VALUATION_KEYS = {"price", "market_cap", "currency", "enterprise_value", "pe", "pe_note", "ev_sales",
                  "fcf_yield", "basis_year", "as_of", "notes"}
NAN = float("nan")


def make_fin(currency: str = "USD", prior: dict | None = None, **latest) -> FinancialData:
    """FinancialData whose newest row (FY2025) has the given values; FY2024 copies it, updated with `prior`."""
    row = {"period_end": "2025-12-31", "revenue": 1000.0, "net_income": 100.0, "fcf": 50.0,
           "debt": 200.0, "cash": 100.0}
    row.update(latest)
    table = pd.DataFrame([{**row, "period_end": "2024-12-31", **(prior or {})}, row], index=["FY2024", "FY2025"])
    return FinancialData(table=table, currency=currency, entity_name="Test Co")


def make_mkt(market_cap: float | None = 2000.0, currency: str | None = "USD", error: str | None = None) -> MarketData:
    return MarketData(ticker="TEST", price=20.0 if market_cap else None, market_cap=market_cap,
                      currency=currency, as_of="2026-10-19", error=error)


# ---------------------------------------------------------------------------------------------------
# compute_valuation
# ---------------------------------------------------------------------------------------------------

def test_valuation_multiples():
    val = compute_valuation(make_fin(), make_mkt())
    assert set(val) == VALUATION_KEYS
    assert val["price"] == 20.0 and val["market_cap"] == 2000.0 and val["currency"] == "USD"
    assert val["enterprise_value"] == 2100.0               # 2000 + 200 debt - 100 cash
    assert val["pe"] == pytest.approx(20.0)
    assert val["ev_sales"] == pytest.approx(2.1)
    assert val["fcf_yield"] == pytest.approx(0.025)
    assert val["basis_year"] == "FY2025" and val["as_of"] == "2026-10-19"
    assert val["notes"] == []                               # the not-TTM caveat is the renderers' caption


def test_short_term_investments_reduce_enterprise_value():
    val = compute_valuation(make_fin(short_term_investments=300.0), make_mkt())
    assert val["enterprise_value"] == 1800.0               # 2000 + 200 debt - 100 cash - 300 investments
    assert val["notes"] == []


def test_short_term_investments_untagged_in_latest_year_is_noted():
    val = compute_valuation(make_fin(short_term_investments=NAN, prior={"short_term_investments": 300.0}),
                            make_mkt())
    assert val["enterprise_value"] == 2100.0
    assert any("Short-term investments not tagged for FY2025" in n for n in val["notes"])


def test_loss_making_company_has_no_pe():
    val = compute_valuation(make_fin(net_income=-50e6, fcf=-10.0), make_mkt())
    assert val["pe"] is None and val["pe_note"] == "n/m (loss)"
    assert any("n/m (loss)" in n and "-$50.0M" in n for n in val["notes"])
    assert val["ev_sales"] == pytest.approx(2.1)            # other multiples still work
    assert val["fcf_yield"] == pytest.approx(-0.005)


def test_debt_missing_in_latest_year_counts_as_zero_in_ev():
    val = compute_valuation(make_fin(debt=NAN, prior={"debt": 0.0}), make_mkt())
    assert val["enterprise_value"] == 1900.0
    assert not any("debt" in n.lower() for n in val["notes"])   # financials' note explains the gap


def test_no_debt_tags_in_any_year_means_no_enterprise_value():
    val = compute_valuation(make_fin(debt=NAN), make_mkt())
    assert val["enterprise_value"] is None and val["ev_sales"] is None
    assert val["pe"] == pytest.approx(20.0) and val["fcf_yield"] == pytest.approx(0.025)


@pytest.mark.parametrize("sic, financial", [(6021, True), (6331, True), (6798, False), (6324, False), (None, False)])
def test_banks_and_insurers_get_pe_only(sic, financial):
    val = compute_valuation(make_fin(), make_mkt(), sic=sic)
    assert val["pe"] == pytest.approx(20.0)
    if financial:
        assert val["enterprise_value"] is None and val["ev_sales"] is None and val["fcf_yield"] is None
        assert any("not meaningful for banks and insurers" in n for n in val["notes"])
    else:
        assert val["ev_sales"] == pytest.approx(2.1) and val["fcf_yield"] == pytest.approx(0.025)


def test_missing_cash_means_no_enterprise_value():
    val = compute_valuation(make_fin(cash=float("nan")), make_mkt())
    assert val["enterprise_value"] is None and val["ev_sales"] is None
    assert val["pe"] == pytest.approx(20.0)
    assert any("Cash not reported" in n for n in val["notes"])


def test_missing_revenue_and_fcf():
    val = compute_valuation(make_fin(revenue=0.0, fcf=None), make_mkt())
    assert val["ev_sales"] is None and val["fcf_yield"] is None
    assert any("EV/Sales not computed" in n for n in val["notes"])
    assert any("FCF yield not computed" in n for n in val["notes"])


def test_currency_mismatch_blocks_multiples():
    val = compute_valuation(make_fin(currency="USD"), make_mkt(currency="EUR"))
    assert val["market_cap"] == 2000.0 and val["currency"] == "EUR"   # market data itself is kept
    assert val["enterprise_value"] is None and val["pe"] is None
    assert val["ev_sales"] is None and val["fcf_yield"] is None
    assert any("Currency mismatch" in n for n in val["notes"])


def test_none_inputs_never_crash():
    val = compute_valuation(None, None)
    assert set(val) == VALUATION_KEYS
    assert all(val[k] is None for k in VALUATION_KEYS - {"notes"})
    assert val["notes"]

    only_fin = compute_valuation(make_fin(), None)
    assert only_fin["basis_year"] == "FY2025" and only_fin["pe"] is None
    assert any("No market data" in n for n in only_fin["notes"])

    only_mkt = compute_valuation(None, make_mkt())
    assert only_mkt["market_cap"] == 2000.0 and only_mkt["pe"] is None

    empty = FinancialData(table=pd.DataFrame(columns=["period_end", "revenue"]))
    assert compute_valuation(empty, make_mkt())["pe"] is None


def annual_facts(val: float, duration: bool = True) -> dict:
    """One companyfacts tag with the same value for calendar years 2024 and 2025."""
    facts = []
    for year in (2024, 2025):
        fact = {"end": f"{year}-12-31", "val": val, "accn": f"a{year}", "fy": year, "form": "10-K",
                "filed": f"{year + 1}-02-15"}
        if duration:
            fact["start"] = f"{year}-01-01"
        facts.append(fact)
    return {"units": {"USD": facts}}


def test_short_term_investments_count_as_cash_end_to_end():
    # Cash $100M plus $900M of current available-for-sale securities, burning $200M a year.
    gaap = {
        "Revenues": annual_facts(500e6),
        "NetIncomeLoss": annual_facts(-250e6),
        "NetCashProvidedByUsedInOperatingActivities": annual_facts(-200e6),
        "PaymentsToAcquirePropertyPlantAndEquipment": annual_facts(0.0),
        "CashAndCashEquivalentsAtCarryingValue": annual_facts(100e6, duration=False),
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent": annual_facts(900e6, duration=False),
        "LongTermDebt": annual_facts(50e6, duration=False),
    }
    fin = build_financials({"entityName": "Burn Co", "facts": {"us-gaap": gaap}})
    flags, _ = detect_red_flags(fin)
    assert "Short cash runway" not in [f.name for f in flags]          # 1.0B covers 5 years of burn
    val = compute_valuation(fin, MarketData(ticker="BURN", price=5.0, market_cap=5e9, currency="USD",
                                            as_of="2026-10-19"))
    assert val["enterprise_value"] == pytest.approx(5e9 - 1e9 + 50e6)


def test_fixture_without_debt_tags_has_no_ev_sales(fixture_dir):
    path = fixture_dir / "BRK-B" / "companyfacts.json"
    if not path.exists():
        pytest.skip("no BRK-B companyfacts fixture")
    fin = build_financials(json.loads(path.read_text(encoding="utf-8")))
    val = compute_valuation(fin, make_mkt(market_cap=1e12))
    assert val["enterprise_value"] is None and val["ev_sales"] is None and val["pe"] is not None


def test_market_data_error_is_reported():
    val = compute_valuation(make_fin(), make_mkt(market_cap=None, error="rate limited"))
    assert val["pe"] is None and val["enterprise_value"] is None
    assert any("rate limited" in n for n in val["notes"])
    assert any("Market cap unavailable" in n for n in val["notes"])


# ---------------------------------------------------------------------------------------------------
# get_market_data with a fake yfinance (no network)
# ---------------------------------------------------------------------------------------------------

class FakeFastInfo:
    """Mimics yfinance's lazy FastInfo: attributes may raise (e.g. KeyError for unknown symbols)."""

    def __init__(self, **values):
        self._values = values

    def __getattr__(self, name):
        value = self._values.get(name)
        if isinstance(value, Exception):
            raise value
        return value


def fake_yfinance(monkeypatch, info: FakeFastInfo | None = None, closes: list[float] | None = None,
                  error: Exception | None = None) -> list[str]:
    """Install a fake `yf` module in copilot.market; returns the list of requested symbols."""
    requested: list[str] = []

    def history(period: str) -> pd.DataFrame:
        values = closes or []
        index = pd.date_range(end="2026-10-16", periods=len(values), freq="D")   # last close on 2026-10-16
        return pd.DataFrame({"Close": values}, index=index, dtype="float64")

    def ticker(symbol: str):
        requested.append(symbol)
        if error:
            raise error
        return SimpleNamespace(fast_info=info, history=history)

    monkeypatch.setattr(market, "yf", SimpleNamespace(Ticker=ticker))
    return requested


def test_get_market_data_from_fast_info(monkeypatch):
    requested = fake_yfinance(monkeypatch, FakeFastInfo(last_price=532.2, market_cap=1.1e12, currency="USD"))
    mkt = get_market_data(" brk.b ")
    assert requested == ["BRK-B"]
    assert (mkt.ticker, mkt.price, mkt.market_cap, mkt.currency, mkt.error) == ("BRK-B", 532.2, 1.1e12, "USD", None)
    assert len(mkt.as_of) == 10 and mkt.source == "Yahoo Finance via yfinance"


def test_price_falls_back_to_history(monkeypatch):
    info = FakeFastInfo(last_price=KeyError("currentTradingPeriod"), market_cap=None, currency="USD")
    fake_yfinance(monkeypatch, info, closes=[9.5, 10.0])
    mkt = get_market_data("TEST")
    assert mkt.price == 10.0 and mkt.as_of == "2026-10-16"
    assert mkt.market_cap is None and mkt.error is None


def test_market_cap_falls_back_to_sec_shares(monkeypatch):
    fake_yfinance(monkeypatch, FakeFastInfo(last_price=10.0, market_cap=float("nan"), currency="USD"))
    mkt = get_market_data("TEST", shares_fallback=50.0)
    assert mkt.market_cap == 500.0
    assert "SEC cover-page shares" in mkt.source


def test_invalid_ticker_returns_error(monkeypatch):
    info = FakeFastInfo(last_price=KeyError("currentTradingPeriod"), market_cap=None, currency=KeyError("currency"))
    fake_yfinance(monkeypatch, info, closes=[])
    mkt = get_market_data("ZZZZNOTREAL", shares_fallback=100.0)
    assert mkt.price is None and mkt.market_cap is None and mkt.currency is None
    assert "No price data" in mkt.error and "ZZZZNOTREAL" in mkt.error


def test_network_failure_never_raises(monkeypatch):
    fake_yfinance(monkeypatch, error=ConnectionError("Too Many Requests"))
    mkt = get_market_data("AAPL")
    assert mkt.price is None and mkt.market_cap is None
    assert "Yahoo Finance request failed" in mkt.error and "Too Many Requests" in mkt.error
    assert len(mkt.as_of) == 10
