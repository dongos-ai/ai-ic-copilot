"""Tests for copilot.red_flags (offline; FinancialData objects are built by hand)."""
from __future__ import annotations

import functools
import json
import re

import pandas as pd
import pytest

from copilot.edgar import html_to_text
from copilot.financials import FinancialData, build_financials
from copilot.red_flags import FINANCIAL_SKIP_CHECKS, MAX_QUOTE_WORDS, RedFlag, detect_red_flags, is_financial_sic

M = 1e6  # table values below are in $ millions


def make_fin(tags: dict | None = None, **columns: list) -> FinancialData:
    """FinancialData whose table has one row per value (oldest first), labelled up to FY2025."""
    n = len(next(iter(columns.values())))
    years = [f"FY{2025 - n + 1 + i}" for i in range(n)]
    scaled = {}
    for name, values in columns.items():
        factor = 1 if name == "diluted_shares" else M
        scaled[name] = [float("nan") if v is None else v * factor for v in values]
    return FinancialData(table=pd.DataFrame(scaled, index=years), tags_used=tags or {}, entity_name="TestCo")


def healthy(**overrides: list) -> FinancialData:
    """Four steady years in which no rule fires; overrides replace whole columns."""
    columns = {
        "revenue": [1000, 1100, 1200, 1300],
        "gross_profit": [500, 550, 600, 650],          # 50% gross margin
        "operating_income": [200, 220, 240, 260],      # 20% operating margin
        "net_income": [150, 165, 180, 195],
        "operating_cash_flow": [200, 220, 240, 260],
        "capex": [50, 55, 60, 65],
        "fcf": [150, 165, 180, 195],
        "cash": [500, 500, 500, 500],
        "debt": [200, 220, 240, 260],
        "sbc": [20, 22, 24, 26],
        "receivables": [100, 110, 120, 130],
        "inventory": [100, 110, 120, 130],
        "diluted_shares": [100e6, 100e6, 100e6, 100e6],
    }
    columns.update(overrides)
    return make_fin(**columns)


def named(flags: list[RedFlag], name: str) -> list[RedFlag]:
    return [f for f in flags if f.name == name]


def only(fin: FinancialData, name: str) -> RedFlag | None:
    """The single flag with this name, or None if it did not fire."""
    matches = named(detect_red_flags(fin)[0], name)
    assert len(matches) <= 1
    return matches[0] if matches else None


# --- baseline and plumbing -----------------------------------------------------------------------

def test_healthy_company_has_no_flags_and_runs_every_check():
    flags, not_run = detect_red_flags(healthy())
    assert flags == []
    assert not_run == ["10-K text scan (no 10-K text)"]


def test_fin_none():
    flags, not_run = detect_red_flags(None)
    assert flags == []
    assert not_run == ["Financial checks (no XBRL financial data)", "10-K text scan (no 10-K text)"]


def test_fin_none_still_scans_text():
    flags, not_run = detect_red_flags(None, "A material weakness was identified in our revenue controls.")
    assert [f.id for f in flags] == ["R1"]
    assert not_run == ["Financial checks (no XBRL financial data)"]


def test_empty_table_is_treated_as_no_data():
    fin = FinancialData(table=pd.DataFrame())
    assert detect_red_flags(fin)[1][0] == "Financial checks (no XBRL financial data)"


def test_ids_are_sequential_in_rule_order():
    fin = healthy(revenue=[1000, 1100, 1200, 1000], net_income=[150, 165, 180, -50])
    flags, _ = detect_red_flags(fin, "A material weakness was identified in our revenue controls.")
    assert [f.id for f in flags] == [f"R{i}" for i in range(1, len(flags) + 1)]
    names = [f.name for f in flags]
    assert names.index("Revenue decline") < names.index("Net loss") < names.index(
        "Material weakness in internal control")


def test_source_names_xbrl_tags_and_years():
    fin = make_fin(tags={"receivables": ["AccountsReceivableNetCurrent"], "revenue": ["Revenues"]},
                   revenue=[1000, 1100], receivables=[100, 140])
    flag = only(fin, "Receivables growing faster than revenue")
    assert flag.source == "SEC XBRL company facts: AccountsReceivableNetCurrent, Revenues (FY2024–FY2025)"


def test_source_falls_back_to_metric_names_and_expands_fcf():
    fin = make_fin(tags={"operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"]},
                   sbc=[60], fcf=[100])
    flag = only(fin, "Stock comp large relative to free cash flow")
    assert flag.source == "SEC XBRL company facts: sbc, NetCashProvidedByUsedInOperatingActivities, capex (FY2025)"


# --- missing data ----------------------------------------------------------------------------------

def test_nan_value_skips_check_and_reports_it():
    fin = healthy(receivables=[100, 110, 120, None])
    flags, not_run = detect_red_flags(fin)
    assert flags == []
    assert "Receivables vs revenue (no receivables data for FY2024–FY2025)" in not_run


def test_missing_columns_do_not_crash():
    fin = make_fin(revenue=[1000, 900])
    flags, not_run = detect_red_flags(fin)
    assert [f.name for f in flags] == ["Revenue decline"]
    assert "Inventory vs revenue (no inventory data for FY2024–FY2025)" in not_run
    assert "Share dilution (no diluted shares data for FY2024–FY2025)" in not_run
    assert "Cash conversion (OCF / net income) (no operating cash flow, net income data for FY2025)" in not_run


def test_single_year_runs_one_year_checks_only():
    fin = make_fin(revenue=[1000], net_income=[-50], operating_cash_flow=[20], fcf=[-100], cash=[150],
                   sbc=[250], debt=[100], receivables=[100], diluted_shares=[1e8])
    flags, not_run = detect_red_flags(fin)
    assert {f.name for f in flags} == {"Net loss", "Negative free cash flow", "Short cash runway",
                                       "Heavy stock-based compensation", "High debt relative to cash flow"}
    assert "Receivables vs revenue (needs 2 years of data, have 1)" in not_run
    assert "Sustained margin compression (needs 4 years of data, have 1)" in not_run
    assert "Revenue trend (needs 2 years of data, have 1)" in not_run


def test_none_values_in_object_columns():
    years = ["FY2024", "FY2025"]
    table = pd.DataFrame({"period_end": ["2024-12-31", "2025-12-31"],
                          "revenue": pd.Series([1000 * M, None], index=years, dtype=object),
                          "net_income": pd.Series([None, -5 * M], index=years, dtype=object)}, index=years)
    flags, not_run = detect_red_flags(FinancialData(table=table))
    assert [f.name for f in flags] == ["Net loss"]
    assert "Revenue trend (no revenue data for FY2024–FY2025)" in not_run


def test_zero_or_negative_bases_are_skipped():
    fin = make_fin(revenue=[0, 100], receivables=[0, 50], debt=[0, 100], diluted_shares=[0, 1e8])
    flags, _ = detect_red_flags(fin)
    assert named(flags, "Receivables growing faster than revenue") == []
    assert named(flags, "Debt rising faster than revenue") == []
    assert named(flags, "Shareholder dilution") == []


# --- numeric rules: fire above, silent below --------------------------------------------------------

def test_receivables_vs_revenue():
    medium = only(make_fin(revenue=[1000, 1100], receivables=[120, 146]), "Receivables growing faster than revenue")
    assert medium.severity == "Medium"
    assert medium.detail == ("Receivables +21.7% ($120.0M → $146.0M) vs revenue +10.0% in FY2025 vs FY2024; "
                             "days sales outstanding 44 → 48.")
    high = only(make_fin(revenue=[1000, 1100], receivables=[100, 140]), "Receivables growing faster than revenue")
    assert high.severity == "High"
    assert only(make_fin(revenue=[1000, 1100], receivables=[100, 119]), "Receivables growing faster than revenue") is None


def test_receivables_too_small_to_matter():
    fin = make_fin(revenue=[1000, 1100], receivables=[5, 15])   # +200%, but only 1.4% of revenue
    assert only(fin, "Receivables growing faster than revenue") is None


def test_inventory_vs_revenue():
    flag = only(make_fin(revenue=[1000, 1100], inventory=[100, 127]), "Inventory growing faster than revenue")
    assert flag.severity == "Medium"
    assert "+27.0%" in flag.detail and "+10.0%" in flag.detail and "10.0% → 11.5% of revenue" in flag.detail
    assert only(make_fin(revenue=[1000, 1100], inventory=[100, 123]), "Inventory growing faster than revenue") is None


def test_gross_margin_contraction():
    medium = only(make_fin(revenue=[1000, 1000], gross_profit=[500, 470]), "Gross margin contraction")
    assert medium.severity == "Medium"
    assert medium.detail == "Gross margin fell from 50.0% to 47.0% (-3.0 pts) in FY2025 vs FY2024."
    assert only(make_fin(revenue=[1000, 1000], gross_profit=[500, 440]), "Gross margin contraction").severity == "High"
    assert only(make_fin(revenue=[1000, 1000], gross_profit=[500, 485]), "Gross margin contraction") is None


def test_operating_margin_contraction():
    medium = only(make_fin(revenue=[1000, 1000], operating_income=[200, 160]), "Operating margin contraction")
    assert medium.severity == "Medium"
    assert "20.0% to 16.0% (-4.0 pts)" in medium.detail
    high = only(make_fin(revenue=[1000, 1000], operating_income=[200, 110]), "Operating margin contraction")
    assert high.severity == "High"
    assert only(make_fin(revenue=[1000, 1000], operating_income=[200, 175]), "Operating margin contraction") is None


def test_sustained_margin_compression():
    fin = make_fin(revenue=[1000] * 4, operating_income=[300, 290, 285, 280])
    flags, _ = detect_red_flags(fin)
    assert [f.name for f in flags] == ["Sustained margin compression"]   # each step is < 3 pts
    assert flags[0].detail == ("Operating margin declined 3 years in a row: "
                               "FY2022 30.0%, FY2023 29.0%, FY2024 28.5%, FY2025 28.0%.")
    assert "(FY2022–FY2025)" in flags[0].source
    assert only(make_fin(revenue=[1000] * 4, operating_income=[300, 290, 295, 280]),
                "Sustained margin compression") is None


def test_sustained_margin_compression_needs_four_years():
    _, not_run = detect_red_flags(make_fin(revenue=[1000] * 3, operating_income=[300, 290, 280]))
    assert "Sustained margin compression (needs 4 years of data, have 3)" in not_run


def test_heavy_stock_comp():
    medium = only(make_fin(revenue=[1000], sbc=[110]), "Heavy stock-based compensation")
    assert medium.severity == "Medium"
    assert medium.detail == "Stock-based compensation of $110.0M was 11.0% of revenue ($1.0B) in FY2025."
    assert only(make_fin(revenue=[1000], sbc=[210]), "Heavy stock-based compensation").severity == "High"
    assert only(make_fin(revenue=[1000], sbc=[90]), "Heavy stock-based compensation") is None


def test_stock_comp_vs_fcf():
    flag = only(make_fin(sbc=[60], fcf=[100]), "Stock comp large relative to free cash flow")
    assert flag.severity == "Medium" and "60.0% of free cash flow ($100.0M)" in flag.detail
    assert only(make_fin(sbc=[40], fcf=[100]), "Stock comp large relative to free cash flow") is None
    assert only(make_fin(sbc=[40], fcf=[-100]), "Stock comp large relative to free cash flow") is None


def test_weak_cash_conversion():
    flag = only(make_fin(net_income=[100], operating_cash_flow=[70]), "Weak cash conversion")
    assert flag.detail == "Operating cash flow of $70.0M was 0.70x net income ($100.0M) in FY2025."
    assert only(make_fin(net_income=[100], operating_cash_flow=[90]), "Weak cash conversion") is None
    assert only(make_fin(net_income=[-100], operating_cash_flow=[-200]), "Weak cash conversion") is None


def test_cumulative_ocf_below_net_income():
    fin = make_fin(net_income=[100, 100, 100], operating_cash_flow=[80, 85, 90])     # 0.85x
    flag = only(fin, "Earnings running ahead of cash flow")
    assert flag.detail == ("Cumulative operating cash flow of $255.0M trailed cumulative net income of "
                           "$300.0M over FY2023–FY2025.")
    assert only(make_fin(net_income=[100, 100, 100], operating_cash_flow=[90, 95, 100]),   # 0.95x: noise
                "Earnings running ahead of cash flow") is None
    _, not_run = detect_red_flags(make_fin(net_income=[100, 100], operating_cash_flow=[90, 95]))
    assert "Cumulative OCF vs net income (needs 3 years of data, have 2)" in not_run


def test_negative_fcf():
    medium = only(make_fin(fcf=[10, 20, -5]), "Negative free cash flow")
    assert medium.severity == "Medium"
    assert medium.detail == ("Free cash flow was negative in 1 of the last 3 years "
                             "(FY2023 $10.0M, FY2024 $20.0M, FY2025 -$5.0M).")
    assert only(make_fin(fcf=[10, -20, -5]), "Negative free cash flow").severity == "High"
    assert only(make_fin(fcf=[-10, -20, 5]), "Negative free cash flow").severity == "High"
    assert only(make_fin(fcf=[10, 20, 30]), "Negative free cash flow") is None
    assert only(make_fin(fcf=[-10, 20, 30]), "Negative free cash flow") is None


def test_negative_fcf_skips_missing_years():
    flag = only(make_fin(fcf=[-10, None, -5]), "Negative free cash flow")
    assert flag.severity == "High" and "negative in 2 of the last 2 years" in flag.detail


def test_short_cash_runway():
    flag = only(make_fin(fcf=[-100], cash=[150]), "Short cash runway")
    assert flag.severity == "High"
    assert flag.detail == "$150.0M of cash covers about 1.5 years of the FY2025 free-cash-flow burn (-$100.0M)."
    assert only(make_fin(fcf=[-100], cash=[300]), "Short cash runway") is None
    assert only(make_fin(fcf=[100], cash=[10]), "Short cash runway") is None


def test_short_term_investments_count_toward_runway():
    assert only(make_fin(fcf=[-100], cash=[50], short_term_investments=[250]), "Short cash runway") is None
    flag = only(make_fin(tags={"short_term_investments": ["MarketableSecuritiesCurrent"]},
                         fcf=[-100], cash=[50], short_term_investments=[100]), "Short cash runway")
    assert flag.detail.startswith("$150.0M of cash and short-term investments covers about 1.5 years")
    assert "MarketableSecuritiesCurrent" in flag.source


def test_revenue_decline():
    medium = only(make_fin(revenue=[1000, 950]), "Revenue decline")
    assert medium.severity == "Medium"
    assert medium.detail == "Revenue fell 5.0% from $1.0B to $950.0M in FY2025 vs FY2024."
    assert only(make_fin(revenue=[1000, 975]), "Revenue decline").severity == "Medium"
    assert only(make_fin(revenue=[1000, 880]), "Revenue decline").severity == "High"
    assert only(make_fin(revenue=[1000, 985]), "Revenue decline") is None          # -1.5%: noise
    assert only(make_fin(revenue=[1000, 1010]), "Revenue decline") is None


def test_net_loss():
    flag = only(make_fin(revenue=[1000], net_income=[-50]), "Net loss")
    assert flag.severity == "Medium"
    assert flag.detail == "Net loss of $50.0M in FY2025 (net margin -5.0%)."
    assert only(make_fin(net_income=[-50]), "Net loss").detail == "Net loss of $50.0M in FY2025."
    assert only(make_fin(net_income=[50]), "Net loss") is None


def test_debt_rising_faster_than_revenue():
    flag = only(make_fin(debt=[100, 130], revenue=[1000, 1050]), "Debt rising faster than revenue")
    assert flag.detail == "Total debt +30.0% ($100.0M → $130.0M) while revenue grew +5.0% in FY2025 vs FY2024."
    assert only(make_fin(debt=[100, 130], revenue=[1000, 1150]), "Debt rising faster than revenue") is None
    assert only(make_fin(debt=[100, 120], revenue=[1000, 1050]), "Debt rising faster than revenue") is None


def test_debt_to_operating_cash_flow():
    flag = only(make_fin(debt=[500], operating_cash_flow=[100]), "High debt relative to cash flow")
    assert flag.detail == "Total debt of $500.0M is 5.0x operating cash flow ($100.0M) in FY2025."
    assert only(make_fin(debt=[300], operating_cash_flow=[100]), "High debt relative to cash flow") is None
    assert only(make_fin(debt=[300], operating_cash_flow=[-10]), "High debt relative to cash flow") is None


def test_dilution():
    medium = only(make_fin(diluted_shares=[100e6, 104e6]), "Shareholder dilution")
    assert medium.severity == "Medium"
    assert medium.detail == "Diluted share count +4.0% (100.0M → 104.0M shares) in FY2025 vs FY2024."
    assert only(make_fin(diluted_shares=[2e9, 2.2e9]), "Shareholder dilution").severity == "High"
    assert only(make_fin(diluted_shares=[100e6, 102e6]), "Shareholder dilution") is None


# --- banks and insurers ----------------------------------------------------------------------------------

def test_is_financial_sic():
    assert [is_financial_sic(s) for s in (6021, 6331, 6211, 6798, 6324, 3571, None)] == [
        True, True, True, False, False, False, False]


def test_banks_skip_cash_flow_and_leverage_checks():
    bank = healthy(operating_cash_flow=[200, 220, -500, -400], debt=[200, 220, 2400, 2600])
    assert {"Weak cash conversion", "Earnings running ahead of cash flow"} <= {
        f.name for f in detect_red_flags(bank)[0]}
    flags, not_run = detect_red_flags(bank, sic=6021)
    assert flags == []
    skipped = [n for n in not_run if n.endswith("(not meaningful for banks and insurers, SIC 6021)")]
    assert len(skipped) == len(FINANCIAL_SKIP_CHECKS) == 7
    assert detect_red_flags(bank, sic=6798) == detect_red_flags(bank)            # REITs keep every check


# --- realistic scenarios ------------------------------------------------------------------------------

def test_high_sbc_software_company():
    fin = make_fin(
        revenue=[1200, 1650, 2070, 2800, 3600],
        gross_profit=[780, 1100, 1400, 1890, 2420],
        operating_income=[-500, -720, -840, -830, -900],
        net_income=[-540, -680, -800, -840, -1290],
        operating_cash_flow=[110, 80, 545, 850, 960],
        fcf=[80, 60, 500, 780, 910],
        cash=[3000, 3900, 3800, 2600, 2000],
        sbc=[300, 600, 860, 1170, 1480],
        receivables=[300, 545, 715, 925, 1100],
        diluted_shares=[290e6, 305e6, 316e6, 324e6, 335e6],
    )
    flags, _ = detect_red_flags(fin)
    by_name = {f.name: f for f in flags}
    assert by_name["Heavy stock-based compensation"].severity == "High"
    assert "41.1% of revenue" in by_name["Heavy stock-based compensation"].detail
    assert "Stock comp large relative to free cash flow" in by_name
    assert "Net loss" in by_name and "Shareholder dilution" in by_name
    assert "Negative free cash flow" not in by_name


def test_declining_margin_chipmaker():
    fin = make_fin(
        revenue=[79000, 63000, 54200, 53100, 52800],
        gross_profit=[43800, 26900, 21700, 17300, 15600],
        operating_income=[19500, 2300, 100, -11700, -2500],
        net_income=[19900, 8000, 1700, -18800, -300],
        operating_cash_flow=[29900, 15400, 11500, 8300, 9700],
        fcf=[11000, -9600, -14300, -15700, -8100],
        cash=[4800, 11100, 7100, 8200, 9500],
        debt=[38100, 42100, 49300, 50000, 46600],
        inventory=[11100, 13200, 11100, 12200, 11400],
        diluted_shares=[4090e6, 4123e6, 4212e6, 4280e6, 4360e6],
    )
    flags, _ = detect_red_flags(fin)
    by_name = {f.name: f for f in flags}
    assert by_name["Negative free cash flow"].severity == "High"
    assert by_name["Short cash runway"].severity == "High"
    assert by_name["Gross margin contraction"].severity == "Medium"
    assert "High debt relative to cash flow" in by_name
    assert "Operating margin contraction" not in by_name   # margin recovered in the latest year


# --- 10-K text scan ------------------------------------------------------------------------------------

BOILERPLATE = [
    "If we identify a material weakness in our internal control over financial reporting, investors could lose "
    "confidence.",
    "We have not identified any material weaknesses in our internal control over financial reporting.",
    "Management has not identified a material weakness in our internal control over financial reporting.",
    "Although no material weaknesses were identified, we cannot assure you that none will arise.",
    "If a material weakness were identified in the future, our stock price could decline.",
    "Our audit included assessing the risk that a material weakness exists.",
    "Any failure to maintain effective controls could result in a restatement of our financial statements.",
    "If we fail to remediate the material weakness, we may be required to restate our previously issued "
    "financial statements.",
    "A restatement of our previously issued financial statements could cause investors to lose confidence.",
    "In the event of a restatement of our previously issued financial statements, the Committee will seek "
    "recovery of incentive compensation.",
    "All share and per share amounts have been retroactively restated in the consolidated financial statements.",
    "We restated our consolidated financial statements to reflect the stock split effected in June 2024.",
    "Indicate by check mark whether any of those error corrections are restatements that required a recovery "
    "analysis of incentive-based compensation.",
    "No customer accounted for more than 10% of revenue in fiscal 2025, 2024 or 2023.",
    "No one customer accounted for 10% or more of our net sales.",
    "No single customer represented more than 10% of our consolidated revenues in 2025.",
    "Our three largest customers accounted for 43% of our net revenue in 2025.",
    "One customer accounted for 12% of our accounts receivable as of December 31, 2025.",
    "One customer accounted for 8% of total revenue in fiscal 2025.",
    "Sales to one customer represented less than 10% of revenue.",
    "Customer I represented 25 % and 26 % of total accounts receivable as of December 31, 2025 and 2024.",
    "There is no substantial doubt about our ability to continue as a going concern.",
    "These conditions could raise substantial doubt about our ability to continue as a going concern.",
    "Management evaluates whether there are conditions or events, considered in the aggregate, that raise "
    "substantial doubt about the Company's ability to continue as a going concern within one year.",
    "Management concluded that its plans alleviate the substantial doubt about the Company's ability to "
    "continue as a going concern.",
    "If we are unable to raise additional capital, there would be substantial doubt about our ability to "
    "continue as a going concern.",
    "Substantial doubt about our ability to continue as a going concern may arise if financing is unavailable.",
]


@pytest.mark.parametrize("sentence", BOILERPLATE)
def test_boilerplate_does_not_fire(sentence):
    flags, _ = detect_red_flags(None, f"Item 1A. Risk Factors\n{sentence}")
    assert flags == []


DISCLOSURES = [
    ("In connection with the audit, we identified a material weakness in our internal control over financial "
     "reporting related to revenue recognition.", "Material weakness in internal control", "High"),
    ("A material weakness was identified in the design of controls over inventory costing.",
     "Material weakness in internal control", "High"),
    ("Management identified material weaknesses in our internal control over financial reporting as of "
     "December 31, 2025.", "Material weakness in internal control", "High"),
    ("As a result, we restated our previously issued consolidated financial statements for the years ended "
     "December 31, 2024 and 2023.", "Restatement of financial statements", "High"),
    ("Note 2. Restatement of Previously Issued Consolidated Financial Statements.",
     "Restatement of financial statements", "High"),
    ("Our recurring losses from operations raise substantial doubt about our ability to continue as a going "
     "concern.", "Going-concern doubt", "High"),
    ("The Company has suffered recurring losses from operations and has a net capital deficiency that raise "
     "substantial doubt about its ability to continue as a going concern.", "Going-concern doubt", "High"),
    ("These factors raise substantial doubt about our ability to continue as a going concern, which may "
     "adversely affect our stock price.", "Going-concern doubt", "High"),
    ("Management identified a material weakness that could result in a misstatement of revenue.",
     "Material weakness in internal control", "High"),
    ("For fiscal year 2026, sales to one direct customer represented 22% of total revenue.",
     "Customer concentration", "Medium"),
    ("Customer A accounted for approximately 15%, 14% and 13% of net sales in fiscal 2025, 2024 and 2023, "
     "respectively.", "Customer concentration", "Medium"),
    ("Our largest customer accounted for 31% of our total revenues in 2025.", "Customer concentration", "Medium"),
]


@pytest.mark.parametrize("sentence, name, severity", DISCLOSURES)
def test_disclosures_fire(sentence, name, severity):
    flags, _ = detect_red_flags(None, f"Some earlier sentence. {sentence} A later sentence.")
    assert [(f.name, f.severity) for f in flags] == [(name, severity)]
    assert flags[0].quote and flags[0].quote in sentence
    assert flags[0].source == "10-K text"


def test_customer_detail_has_the_percentage():
    flags, _ = detect_red_flags(None, "Sales to one direct customer represented 22% of total revenue.")
    assert flags[0].detail == "One customer accounted for 22% of revenue, per the 10-K (see quote)."


def test_customer_percentage_below_threshold_then_above():
    text = ("One customer accounted for 9% of revenue in 2025. "
            "In 2024, one customer accounted for 11% of revenue.")
    flags, _ = detect_red_flags(None, text)
    assert flags[0].detail.startswith("One customer accounted for 11%")


def test_at_most_one_flag_per_text_rule_and_item_source():
    text = ("Item 1A. Risk Factors\nIf we identify a material weakness, we could suffer.\n"
            "Item 9A. Controls and Procedures\nWe identified a material weakness in revenue controls. "
            "A second material weakness was identified in inventory controls.\n")
    flags, _ = detect_red_flags(None, text)
    assert len(flags) == 1
    assert flags[0].source == "10-K text: Item 9A"
    assert flags[0].quote == "We identified a material weakness in revenue controls."


def test_long_sentence_quote_is_trimmed_around_the_match():
    filler = " ".join(f"word{i}" for i in range(60))
    text = f"{filler} and we identified a material weakness in controls over revenue {filler}."
    flags, _ = detect_red_flags(None, text)
    quote = flags[0].quote
    assert len(quote.split()) <= MAX_QUOTE_WORDS
    assert "identified a material weakness" in quote
    assert quote.startswith("…") and quote.endswith("…")


def test_blank_text_is_reported_as_not_run():
    assert detect_red_flags(None, "   ")[1][-1] == "10-K text scan (no 10-K text)"


# --- real 10-K fixtures ----------------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _parsed_10k(path) -> str:
    return html_to_text(path.read_bytes())       # the same text the app scans


def _fixture_text(fixture_dir, ticker: str) -> str:
    path = fixture_dir / ticker / "10k.html"
    if not path.exists():
        pytest.skip(f"no 10-K fixture for {ticker}")
    return _parsed_10k(path)


def _fixture_flags(fixture_dir, ticker: str, sic: int | None = None) -> tuple[set[str], list[str]]:
    """Names of the numeric flags for a fixture's companyfacts, and the checks not run."""
    path = fixture_dir / ticker / "companyfacts.json"
    if not path.exists():
        pytest.skip(f"no companyfacts fixture for {ticker}")
    flags, not_run = detect_red_flags(build_financials(json.loads(path.read_text(encoding="utf-8"))), sic=sic)
    return {f.name for f in flags}, not_run


@pytest.mark.parametrize("ticker", ["AAPL", "KO", "PLTR", "SNOW", "INTC"])
def test_fixture_boilerplate_does_not_fire(fixture_dir, ticker):
    flags, _ = detect_red_flags(None, _fixture_text(fixture_dir, ticker))
    assert flags == []


def test_fixture_nvda_customer_concentration(fixture_dir):
    flags, _ = detect_red_flags(None, _fixture_text(fixture_dir, "NVDA"))
    assert [f.name for f in flags] == ["Customer concentration"]
    assert re.search(r"one direct customer represented 22\s?% of total revenue", flags[0].quote)
    assert flags[0].source == "10-K text: Item 1A"


def test_fixture_bank_cash_flow_flags_are_skipped(fixture_dir):
    misfires = {"Weak cash conversion", "Earnings running ahead of cash flow"}
    assert misfires <= _fixture_flags(fixture_dir, "JPM")[0]
    names, not_run = _fixture_flags(fixture_dir, "JPM", sic=6021)
    assert not misfires & names
    assert any("banks and insurers" in n for n in not_run)


def test_fixture_small_gaps_are_not_flagged(fixture_dir):
    # NFLX: 3-year OCF is 0.99x net income; INTC and PFE revenue fell by less than 2%.
    assert "Earnings running ahead of cash flow" not in _fixture_flags(fixture_dir, "NFLX")[0]
    assert "Revenue decline" not in _fixture_flags(fixture_dir, "INTC")[0] | _fixture_flags(fixture_dir, "PFE")[0]
