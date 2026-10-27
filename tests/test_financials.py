"""Offline tests for copilot.financials: synthetic company-facts dicts plus (optionally) real EDGAR fixtures."""
from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

import numpy as np

from copilot import financials
from copilot.edgar import EdgarError
from copilot.financials import (
    METRIC_COLUMNS, FinancialData, _total_debt, build_financials, format_money, format_pct, get_company_facts,
    snapshot_table, to_float,
)


# ---------------------------------------------------------------------------------------------------
# Synthetic company-facts builders
# ---------------------------------------------------------------------------------------------------

def annual(end: str, val: float, *, filed: str | None = None, days: int = 364, form: str = "10-K",
           fy: int | None = None) -> dict:
    """A duration fact as it appears in companyfacts JSON. `fy` is the filing's fiscal year (default: the year
    of `end`, as for a 10-K's own year); the accession number is derived from the filing date."""
    filed = filed or (date.fromisoformat(end) + timedelta(days=60)).isoformat()
    start = (date.fromisoformat(end) - timedelta(days=days)).isoformat()
    return {"start": start, "end": end, "val": val, "accn": f"accn-{filed}", "fy": fy or int(end[:4]), "fp": "FY",
            "form": form, "filed": filed}


def instant(end: str, val: float, *, filed: str | None = None, form: str = "10-K") -> dict:
    filed = filed or (date.fromisoformat(end) + timedelta(days=60)).isoformat()
    return {"end": end, "val": val, "accn": f"accn-{filed}", "fy": int(end[:4]), "fp": "FY", "form": form,
            "filed": filed}


def company(usd: dict[str, list[dict]], shares: dict[str, list[dict]] | None = None,
            dei: dict | None = None, name: str = "Test Co") -> dict:
    gaap = {tag: {"units": {"USD": facts}} for tag, facts in usd.items()}
    for tag, facts in (shares or {}).items():
        gaap[tag] = {"units": {"shares": facts}}
    facts = {"us-gaap": gaap}
    if dei is not None:
        facts["dei"] = dei
    return {"cik": 1, "entityName": name, "facts": facts}


ENDS = ["2021-12-31", "2022-12-31", "2023-12-31"]


def simple_company() -> dict:
    return company({
        "Revenues": [annual(e, v) for e, v in zip(ENDS, [100.0, 120.0, 150.0])],
        "GrossProfit": [annual(e, v) for e, v in zip(ENDS, [40.0, 50.0, 60.0])],
        "OperatingIncomeLoss": [annual(e, v) for e, v in zip(ENDS, [10.0, 12.0, 30.0])],
        "NetIncomeLoss": [annual(e, v) for e, v in zip(ENDS, [8.0, -6.0, 20.0])],
        "NetCashProvidedByUsedInOperatingActivities": [annual(e, v) for e, v in zip(ENDS, [15.0, 18.0, 25.0])],
        "PaymentsToAcquirePropertyPlantAndEquipment": [annual(e, v) for e, v in zip(ENDS, [5.0, 6.0, 7.0])],
        "ShareBasedCompensation": [annual(e, v) for e, v in zip(ENDS, [3.0, 4.0, 30.0])],
        "CashAndCashEquivalentsAtCarryingValue": [instant(e, v) for e, v in zip(ENDS, [50.0, 40.0, 30.0])],
        "LongTermDebt": [instant(e, v) for e, v in zip(ENDS, [80.0, 70.0, 60.0])],
        "CommercialPaper": [instant(e, v) for e, v in zip(ENDS, [5.0, 5.0, 5.0])],
        "AccountsReceivableNetCurrent": [instant(e, v) for e, v in zip(ENDS, [10.0, 11.0, 12.0])],
    }, shares={"WeightedAverageNumberOfDilutedSharesOutstanding": [annual(e, 1000.0) for e in ENDS]})


def nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


# ---------------------------------------------------------------------------------------------------
# Table shape, ratios, FCF, growth
# ---------------------------------------------------------------------------------------------------

def test_table_shape_and_ratios():
    fin = build_financials(simple_company())
    assert isinstance(fin, FinancialData)
    assert fin.years == ["FY2021", "FY2022", "FY2023"]                     # oldest -> newest
    assert list(fin.table.columns) == ["period_end"] + METRIC_COLUMNS
    assert list(fin.table["period_end"]) == ENDS
    latest = fin.latest
    assert latest["revenue"] == 150.0
    assert latest["gross_margin"] == pytest.approx(0.4)
    assert latest["operating_margin"] == pytest.approx(0.2)
    assert latest["net_margin"] == pytest.approx(20 / 150)
    assert latest["fcf"] == 18.0                                           # 25 - 7
    assert latest["fcf_margin"] == pytest.approx(18 / 150)
    assert latest["sbc_pct_revenue"] == pytest.approx(0.2)
    assert latest["debt"] == 65.0                                          # LongTermDebt + CommercialPaper
    assert latest["cash"] == 30.0 and latest["receivables"] == 12.0
    assert nan(latest["inventory"])
    growth = list(fin.table["revenue_growth"])
    assert nan(growth[0]) and growth[1] == pytest.approx(0.2) and growth[2] == pytest.approx(0.25)
    assert fin.table.loc["FY2022", "net_income"] == -6.0
    assert fin.entity_name == "Test Co" and fin.currency == "USD"
    assert fin.tags_used["revenue"] == ["Revenues"]
    assert fin.tags_used["fcf"] == ["NetCashProvidedByUsedInOperatingActivities",
                                    "PaymentsToAcquirePropertyPlantAndEquipment"]
    assert any("Total debt (FY2023) = LongTermDebt + CommercialPaper" in n for n in fin.notes)
    assert any("inventory" in n for n in fin.notes)                       # missing metric is explained


def test_duration_filter_ignores_quarters_and_10q():
    data = simple_company()
    revenue = data["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    revenue.append(annual("2024-03-31", 40.0, days=90))                   # quarterly fact inside a 10-K
    revenue.append(annual("2024-06-30", 999.0, form="10-Q"))              # annual-length fact in a 10-Q
    fin = build_financials(data)
    assert fin.years == ["FY2021", "FY2022", "FY2023"]
    assert fin.latest["revenue"] == 150.0


def test_latest_filing_wins_and_comparative_fy_ignored():
    data = simple_company()
    revenue = data["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    # A later 10-K restates FY2022 as a comparative tagged with its own fy (2024); the label stays FY2022.
    revenue += [annual("2022-12-31", 125.0, filed="2025-02-01", fy=2024),
                annual("2024-12-31", 160.0, filed="2025-02-01", fy=2024)]
    fin = build_financials(data)
    assert fin.table.loc["FY2022", "revenue"] == 125.0
    assert fin.years == ["FY2021", "FY2022", "FY2023", "FY2024"]


def test_revenue_tag_switch_over_years():
    ends = ["2016-12-31", "2017-12-31", "2018-12-31", "2019-12-31", "2020-12-31"]
    data = company({
        "SalesRevenueNet": [annual(e, v) for e, v in zip(ends[:2], [10.0, 11.0])],
        "RevenueFromContractWithCustomerExcludingAssessedTax": [annual(e, v)
                                                                for e, v in zip(ends[2:], [12.0, 13.0, 14.0])],
        "NetIncomeLoss": [annual(e, 1.0) for e in ends],
    })
    fin = build_financials(data)
    assert list(fin.table["revenue"]) == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert fin.tags_used["revenue"] == ["SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"]
    assert fin.table.loc["FY2018", "revenue_growth"] == pytest.approx(12 / 11 - 1)


def test_revenue_max_rule_prefers_total_over_subset():
    # REIT-style: "Revenues" is the total, RevenueFromContract... is only part of it.
    data = company({
        "Revenues": [annual(e, v) for e, v in zip(ENDS, [100.0, 110.0, 120.0])],
        "RevenueFromContractWithCustomerExcludingAssessedTax": [annual(e, v)
                                                                for e, v in zip(ENDS, [20.0, 22.0, 130.0])],
    })
    fin = build_financials(data)
    assert list(fin.table["revenue"]) == [100.0, 110.0, 130.0]
    assert fin.tags_used["revenue"] == ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]


def test_instant_matching_within_ten_days():
    data = simple_company()
    gaap = data["facts"]["us-gaap"]
    gaap["CashAndCashEquivalentsAtCarryingValue"]["units"]["USD"] = [
        instant("2022-01-07", 51.0),     # 7 days after the FY2021 end -> matched
        instant("2023-01-20", 41.0),     # 20 days after the FY2022 end -> too far
        instant("2023-12-31", 30.0),
    ]
    fin = build_financials(data)
    cash = list(fin.table["cash"])
    assert cash[0] == 51.0 and nan(cash[1]) and cash[2] == 30.0


def test_keeps_newest_years_and_skips_ends_too_close_together():
    ends = [f"{y}-12-31" for y in range(2015, 2022)]                       # 7 fiscal years
    facts = [annual(e, float(i + 1)) for i, e in enumerate(ends)]
    facts.append(annual("2021-12-01", 99.0))                               # 30 days before the newest end
    fin = build_financials(company({"Revenues": facts}))
    assert fin.years == ["FY2017", "FY2018", "FY2019", "FY2020", "FY2021"]
    assert fin.latest["revenue"] == 7.0
    assert build_financials(company({"Revenues": facts}), years=3).years == ["FY2019", "FY2020", "FY2021"]


def test_labels_for_two_periods_ending_in_the_same_year():
    # Fiscal-year change: January year-end, then December year-end (both ~12-month periods).
    data = company({"Revenues": [annual("2019-01-31", 10.0), annual("2020-01-31", 11.0),
                                 annual("2020-12-31", 12.0, days=334), annual("2021-12-31", 13.0)]})
    fin = build_financials(data)
    assert fin.years == ["FY2019", "FY2020 (2020-01-31)", "FY2020 (2020-12-31)", "FY2021"]


def test_labels_follow_the_filers_fiscal_year():
    # Home Depot style: the year is named by its start. Each 10-K's own (newest) year carries the fy; the
    # restated comparative in the second 10-K is tagged fy 2025 but must not rename the earlier year.
    data = company({"Revenues": [
        annual("2025-02-01", 10.0, filed="2025-03-12", fy=2024),
        annual("2025-02-01", 10.5, filed="2026-03-11", fy=2025),
        annual("2026-01-31", 11.0, filed="2026-03-11", fy=2025),
    ]})
    assert build_financials(data).years == ["FY2024", "FY2025"]


def test_labels_for_52_53_week_years_around_new_year():
    # Snap-on style: fiscal 2021 ends 2022-01-01 and fiscal 2025 ends 2026-01-03.
    ends = {"2022-01-01": 2021, "2022-12-31": 2022, "2023-12-30": 2023, "2024-12-28": 2024, "2026-01-03": 2025}
    data = company({"Revenues": [annual(end, 1.0, fy=fy) for end, fy in ends.items()]})
    assert build_financials(data).years == ["FY2021", "FY2022", "FY2023", "FY2024", "FY2025"]


@pytest.mark.parametrize("ends, fy, expected", [
    (["2022-12-31", "2023-12-31", "2024-12-31"], 2024, ["FY2022", "FY2023", "FY2024"]),
    (["2023-01-28", "2024-02-03", "2025-02-01"], 2024, ["FY2022", "FY2023", "FY2024"]),   # named by start year
])
def test_labels_for_years_only_seen_as_comparatives(ends, fy, expected):
    # A company's first 10-K tags all three years with its own fy.
    data = company({"Revenues": [annual(end, 1.0, filed="2025-03-01", fy=fy) for end in ends]})
    assert build_financials(data).years == expected


def test_growth_not_computed_across_a_missing_year():
    data = company({"Revenues": [annual("2019-12-31", 10.0), annual("2021-12-31", 20.0), annual("2022-12-31", 22.0)]})
    growth = list(build_financials(data).table["revenue_growth"])
    assert nan(growth[0]) and nan(growth[1]) and growth[2] == pytest.approx(0.1)


def test_growth_blank_when_later_10k_restated_the_year():
    # Filing A (2023) reports FY2022=100, FY2021=90. Filing B (2024) restates FY2022 to 45 after a spin-off.
    data = company({"Revenues": [
        annual("2021-12-31", 90.0, filed="2023-02-01"), annual("2022-12-31", 100.0, filed="2023-02-01"),
        annual("2022-12-31", 45.0, filed="2024-02-01"), annual("2023-12-31", 50.0, filed="2024-02-01"),
    ]})
    fin = build_financials(data)
    assert list(fin.table["revenue"]) == [90.0, 45.0, 50.0]
    growth = list(fin.table["revenue_growth"])
    assert nan(growth[1])                                                  # 90 -> 45 mixes two bases
    assert growth[2] == pytest.approx(50 / 45 - 1)
    assert any("Revenue growth for FY2022 not shown" in n for n in fin.notes)


def test_gross_profit_computed_from_cost_of_revenue():
    data = company({
        "Revenues": [annual(e, 100.0) for e in ENDS],
        "CostOfGoodsAndServicesSold": [annual(e, 70.0) for e in ENDS],
    })
    fin = build_financials(data)
    assert fin.latest["gross_profit"] == 30.0
    assert fin.latest["gross_margin"] == pytest.approx(0.3)
    assert fin.tags_used["gross_profit"] == ["revenue - CostOfGoodsAndServicesSold"]
    assert any("Gross profit = revenue - cost of revenue" in n for n in fin.notes)


def test_bank_without_cost_of_revenue_has_no_gross_margin():
    data = company({"RevenuesNetOfInterestExpense": [annual(e, 100.0) for e in ENDS],
                    "NetIncomeLoss": [annual(e, 30.0) for e in ENDS]})
    fin = build_financials(data)
    assert fin.latest["revenue"] == 100.0
    assert nan(fin.latest["gross_profit"]) and nan(fin.latest["gross_margin"])
    assert any("gross profit" in n and "operating income" in n for n in fin.notes)


def test_fcf_missing_without_capex_and_capex_sign_normalized():
    data = simple_company()
    gaap = data["facts"]["us-gaap"]
    gaap["PaymentsToAcquirePropertyPlantAndEquipment"]["units"]["USD"] = [
        annual("2022-12-31", -6.0),       # wrongly signed by the filer
        annual("2023-12-31", 7.0),
    ]
    fin = build_financials(data)
    assert nan(fin.table.loc["FY2021", "fcf"])                             # no capex reported that year
    assert fin.table.loc["FY2022", "capex"] == 6.0
    assert fin.table.loc["FY2022", "fcf"] == 12.0                          # 18 - 6


def test_capex_from_other_property_plant_and_equipment_tag():
    data = company({
        "Revenues": [annual(e, 100.0) for e in ENDS],
        "NetCashProvidedByUsedInOperatingActivities": [annual(e, 30.0) for e in ENDS],
        "PaymentsToAcquireOtherPropertyPlantAndEquipment": [annual(e, 8.0) for e in ENDS],   # LLY's capex tag
    })
    fin = build_financials(data)
    assert fin.latest["capex"] == 8.0 and fin.latest["fcf"] == 22.0
    assert fin.tags_used["capex"] == ["PaymentsToAcquireOtherPropertyPlantAndEquipment"]


def test_53_week_year_is_noted():
    data = simple_company()
    data["facts"]["us-gaap"]["Revenues"]["units"]["USD"][1] = annual("2022-12-31", 120.0, days=371)
    notes = [n for n in build_financials(data).notes if "53 weeks" in n]
    assert len(notes) == 1 and notes[0].startswith("FY2022 had 53 weeks")
    assert not any("53 weeks" in n for n in build_financials(simple_company()).notes)


# ---------------------------------------------------------------------------------------------------
# Short-term investments
# ---------------------------------------------------------------------------------------------------

def test_short_term_investments_from_first_tag():
    data = simple_company()
    data["facts"]["us-gaap"]["AvailableForSaleSecuritiesDebtSecuritiesCurrent"] = {
        "units": {"USD": [instant(e, v) for e, v in zip(ENDS, [5.0, 6.0, 7.0])]}}
    fin = build_financials(data)
    assert list(fin.table["short_term_investments"]) == [5.0, 6.0, 7.0]
    assert fin.tags_used["short_term_investments"] == ["AvailableForSaleSecuritiesDebtSecuritiesCurrent"]
    assert nan(build_financials(simple_company()).latest["short_term_investments"])


@pytest.mark.parametrize("sti_tag_value, total, expected", [
    (None, 42.0, 12.0),     # KO style: only the cash + short-term investments total is tagged
    (25.0, 30.0, 0.0),      # Target style: the tagged short-term investments are cash equivalents (in cash)
])
def test_short_term_investments_from_cash_and_investments_total(sti_tag_value, total, expected):
    data = simple_company()                                               # cash FY2023 = 30
    gaap = data["facts"]["us-gaap"]
    gaap["CashCashEquivalentsAndShortTermInvestments"] = {"units": {"USD": [instant("2023-12-31", total)]}}
    if sti_tag_value is not None:
        gaap["ShortTermInvestments"] = {"units": {"USD": [instant("2023-12-31", sti_tag_value)]}}
    fin = build_financials(data)
    assert fin.latest["short_term_investments"] == expected
    assert fin.tags_used["short_term_investments"] == ["CashCashEquivalentsAndShortTermInvestments - cash"]


def test_prefers_tag_still_used_in_newest_year():
    # The retired tag (PPE) has stale FY2021 values; the current tag covers all years.
    data = simple_company()
    gaap = data["facts"]["us-gaap"]
    gaap["PaymentsToAcquirePropertyPlantAndEquipment"]["units"]["USD"] = [annual("2021-12-31", 99.0)]
    gaap["PaymentsToAcquireProductiveAssets"] = {"units": {"USD": [annual(e, 5.0) for e in ENDS]}}
    fin = build_financials(data)
    assert list(fin.table["capex"]) == [5.0, 5.0, 5.0]
    assert fin.tags_used["capex"] == ["PaymentsToAcquireProductiveAssets"]


# ---------------------------------------------------------------------------------------------------
# Total debt rule
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("values, expected, tags", [
    ({"LongTermDebt": 100.0, "LongTermDebtCurrent": 10.0, "CommercialPaper": 5.0},
     105.0, ["LongTermDebt", "CommercialPaper"]),                       # total already includes current portion
    ({"LongTermDebt": 20.0, "ShortTermBorrowings": 7.0, "CommercialPaper": 4.0},
     27.0, ["LongTermDebt", "ShortTermBorrowings"]),                    # ShortTermBorrowings preferred over CP
    ({"LongTermDebtNoncurrent": 90.0, "LongTermDebtCurrent": 10.0, "ShortTermBorrowings": 3.0},
     103.0, ["LongTermDebtNoncurrent", "LongTermDebtCurrent", "ShortTermBorrowings"]),
    ({"LongTermDebtAndCapitalLeaseObligations": 90.0, "DebtCurrent": 12.0},
     102.0, ["LongTermDebtAndCapitalLeaseObligations", "DebtCurrent"]),  # DebtCurrent covers both current pieces
    ({"LongTermDebtNoncurrent": 90.0, "LongTermDebtCurrent": 10.0, "DebtCurrent": 12.0},
     100.0, ["LongTermDebtNoncurrent", "LongTermDebtCurrent"]),          # no double counting
    ({"LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities": 40.0, "CommercialPaper": 1.5},
     41.5, ["LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities", "CommercialPaper"]),
    ({"NotesPayable": 25.0, "LoansPayable": 1.7, "SecuredDebt": 0.1, "CommercialPaper": 0.5},
     27.3, ["NotesPayable", "LoansPayable", "SecuredDebt", "CommercialPaper"]),  # REIT-style
    ({"LongTermDebt": 6584.0, "DebtCurrent": 1569.0},
     8153.0, ["LongTermDebt", "DebtCurrent"]),                           # TSLA: LongTermDebt is noncurrent only
    ({"LongTermDebt": 20469.0, "DebtCurrent": 1686.0, "ShortTermBorrowings": 25.0},
     20494.0, ["LongTermDebt", "ShortTermBorrowings"]),                  # GE
    ({"LongTermDebt": 8468.0, "LongTermDebtCurrent": 999.0, "DebtCurrent": 999.0},
     8468.0, ["LongTermDebt"]),                                          # NVDA: current portion already included
    ({"LongTermNotesPayable": 122342.0, "NotesPayableCurrent": 7199.0, "DebtCurrent": 7199.0},
     129541.0, ["LongTermNotesPayable", "NotesPayableCurrent"]),          # ORCL: notes payable tags
    ({"LongTermDebt": 0.0, "LongTermNotesPayable": 75000.0, "NotesPayableCurrent": 3000.0},
     78000.0, ["LongTermNotesPayable", "NotesPayableCurrent"]),           # ORCL: stray zero LongTermDebt ignored
    ({"LongTermDebt": 317816.0, "CommercialPaper": 20400.0, "OtherShortTermBorrowings": 48088.0},
     365904.0, ["LongTermDebt", "OtherShortTermBorrowings"]),             # BAC: the larger short-term line
    ({"LongTermDebt": 42000.0, "CommercialPaper": 3500.0, "OtherShortTermBorrowings": 56.0},
     45500.0, ["LongTermDebt", "CommercialPaper"]),                       # KO: CP is the short-term line
    ({"LongTermDebt": 0.0, "DebtCurrent": 0.0}, 0.0, ["LongTermDebt", "DebtCurrent"]),  # a real zero stays zero
    ({"OperatingLeaseLiability": 50.0}, None, []),                      # leases are not debt
    ({}, None, []),
])
def test_total_debt_rule(values, expected, tags):
    total, used = _total_debt(values)
    assert used == tags
    if expected is None:
        assert total is None
    else:
        assert total == pytest.approx(expected)


def test_no_debt_tags_note():
    data = simple_company()
    del data["facts"]["us-gaap"]["LongTermDebt"], data["facts"]["us-gaap"]["CommercialPaper"]
    fin = build_financials(data)
    assert fin.table["debt"].isna().all()
    assert "debt" not in fin.tags_used
    assert [n for n in fin.notes if "debt" in n.lower()] == [
        "Total debt not tagged in XBRL in any year (the company may use its own tags), "
        "so no enterprise value is computed."]


def test_debt_missing_in_some_years_has_one_note():
    data = simple_company()
    gaap = data["facts"]["us-gaap"]
    gaap["LongTermDebt"]["units"]["USD"] = [instant("2021-12-31", 0.0)]      # PLTR style: $0, then untagged
    del gaap["CommercialPaper"]
    debt_notes = [n for n in build_financials(data).notes if "debt" in n.lower()]
    assert debt_notes == ["Total debt not tagged in XBRL for FY2022, FY2023 (shown as n/a; EV treats it as zero)."]


# ---------------------------------------------------------------------------------------------------
# Share counts
# ---------------------------------------------------------------------------------------------------

def test_stock_split_restatement_adjusts_older_share_counts():
    shares = [
        annual("2020-12-31", 990.0, filed="2022-02-01"), annual("2021-12-31", 1000.0, filed="2022-02-01"),
        annual("2021-12-31", 1000.0, filed="2023-02-01"), annual("2022-12-31", 1010.0, filed="2023-02-01"),
        # 10-for-1 split in 2023: the next 10-K restates FY2022.
        annual("2022-12-31", 10100.0, filed="2024-02-01"), annual("2023-12-31", 10200.0, filed="2024-02-01"),
    ]
    data = company({"Revenues": [annual(e, 1.0) for e in ["2020-12-31"] + ENDS]},
                   shares={"WeightedAverageNumberOfDilutedSharesOutstanding": shares})
    fin = build_financials(data)
    assert list(fin.table["diluted_shares"]) == [9900.0, 10000.0, 10100.0, 10200.0]
    assert any("FY2020, FY2021" in n and "stock splits" in n for n in fin.notes)


def test_real_dilution_is_not_mistaken_for_a_split():
    shares = [annual("2021-12-31", 1000.0), annual("2022-12-31", 2000.0), annual("2023-12-31", 2050.0)]
    data = company({"Revenues": [annual(e, 1.0) for e in ENDS]},
                   shares={"WeightedAverageNumberOfDilutedSharesOutstanding": shares})
    assert list(build_financials(data).table["diluted_shares"]) == [1000.0, 2000.0, 2050.0]


def test_cover_page_shares_outstanding():
    def dei(*facts):
        return {"EntityCommonStockSharesOutstanding": {"units": {"shares": list(facts)}}}

    base = simple_company()["facts"]["us-gaap"]
    latest = {"end": "2024-04-20", "val": 950.0, "accn": "q1", "form": "10-Q", "filed": "2024-04-30"}
    older = {"end": "2024-01-31", "val": 960.0, "accn": "k", "form": "10-K", "filed": "2024-02-15"}
    data = {"entityName": "X", "facts": {"us-gaap": base, "dei": dei(older, latest)}}
    assert build_financials(data).shares_outstanding == 950.0

    stale = {"end": "2011-02-17", "val": 943242.0, "accn": "old", "form": "10-K", "filed": "2011-02-28"}
    stale_only = {"entityName": "X", "facts": {"us-gaap": base, "dei": dei(stale)}}
    assert build_financials(stale_only).shares_outstanding is None

    class_a = {"end": "2024-04-20", "val": 100.0, "accn": "q1", "form": "10-Q", "filed": "2024-04-30"}
    class_b = {**class_a, "val": 200.0}
    two_classes = {"entityName": "X", "facts": {"us-gaap": base, "dei": dei(class_a, class_b)}}
    assert build_financials(two_classes).shares_outstanding is None


# ---------------------------------------------------------------------------------------------------
# Graceful failures and currency
# ---------------------------------------------------------------------------------------------------

def test_ifrs_filer_raises_clear_error():
    data = {"entityName": "Foreign Co", "facts": {"dei": {}, "ifrs-full": {"Revenue": {"units": {"TWD": []}}}}}
    with pytest.raises(EdgarError, match="IFRS"):
        build_financials(data)


def test_no_annual_data_raises_clear_error():
    data = company({"Revenues": [annual("2026-06-30", 10.0, form="10-Q")]}, name="New Holdco")
    with pytest.raises(EdgarError, match="New Holdco.*10-K"):
        build_financials(data)
    with pytest.raises(EdgarError):
        build_financials({})


def test_partial_history_is_noted():
    data = company({"Revenues": [annual(e, 10.0) for e in ENDS[-2:]]})
    fin = build_financials(data)
    assert fin.years == ["FY2022", "FY2023"]
    assert any("Only 2 fiscal year(s)" in n for n in fin.notes)


def test_non_usd_reporting_currency():
    data = {"entityName": "Euro Co", "facts": {"us-gaap": {
        "Revenues": {"units": {"EUR": [annual(e, 100.0) for e in ENDS]}},
        "NetIncomeLoss": {"units": {"EUR": [annual(e, 10.0) for e in ENDS]}},
    }}}
    fin = build_financials(data)
    assert fin.currency == "EUR"
    assert fin.latest["revenue"] == 100.0
    assert any("EUR" in n for n in fin.notes)


# ---------------------------------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------------------------------

def test_format_helpers_and_snapshot_table():
    assert to_float(np.int64(5)) == 5.0 and to_float(np.float64("nan")) is None and to_float(True) is None
    assert format_money(391_035_000_000) == "$391.0B"
    assert format_money(950e6) == "$950.0M"
    assert format_money(-1.2e9) == "-$1.2B"
    assert format_money(float("nan")) == "n/a" and format_money(None) == "n/a"
    assert format_pct(0.4523) == "45.2%" and format_pct(float("nan")) == "n/a"
    snap = snapshot_table(build_financials(simple_company()))
    assert list(snap.columns) == ["FY2021", "FY2022", "FY2023"]
    assert snap.loc["Revenue", "FY2023"] == "$150"
    assert snap.loc["Revenue growth", "FY2021"] == "n/a"
    assert snap.loc["Total debt", "FY2023"] == "$65"
    assert snap.loc["Cash & equivalents", "FY2023"] == "$30" and snap.loc["Short-term investments", "FY2023"] == "n/a"


def test_missing_company_facts_gives_a_plain_message(monkeypatch):
    def not_found(url, user_agent):
        raise EdgarError(f"Not found on SEC EDGAR (HTTP 404). ({url})")

    monkeypatch.setattr(financials, "sec_get_json", not_found)
    with pytest.raises(EdgarError) as info:
        get_company_facts(320193, "Test test@example.com")
    assert str(info.value) == "SEC has no XBRL financial data (company facts) for this company."


# ---------------------------------------------------------------------------------------------------
# Real EDGAR fixtures (skipped when FIXTURE_DIR is not available)
# ---------------------------------------------------------------------------------------------------

def load_fixture(fixture_dir, ticker: str) -> dict:
    path = fixture_dir / ticker / "companyfacts.json"
    if not path.exists():
        pytest.skip(f"no companyfacts fixture for {ticker}")
    return json.loads(path.read_text(encoding="utf-8"))


# Latest fiscal year: label, period end, revenue and net income (USD millions) as printed in each 10-K's
# income statement (net income attributable to the company).
FILING_CHECKS = {
    "AAPL": ("FY2025", "2025-09-27", 416_161, 112_010),
    "MSFT": ("FY2026", "2026-06-30", 331_839, 133_749),
    "KO": ("FY2025", "2025-12-31", 47_941, 13_107),
    "NVDA": ("FY2026", "2026-01-25", 215_938, 120_067),
    "WMT": ("FY2026", "2026-01-31", 713_163, 21_893),
    "JPM": ("FY2025", "2025-12-31", 182_447, 57_048),
    "O": ("FY2025", "2025-12-31", 5_749.377, 1_058.590),
    "BRK-B": ("FY2025", "2025-12-31", 371_444, 66_968),
}


@pytest.mark.parametrize("ticker", sorted(FILING_CHECKS))
def test_fixture_latest_year_matches_10k(fixture_dir, ticker):
    label, period_end, revenue_m, net_income_m = FILING_CHECKS[ticker]
    fin = build_financials(load_fixture(fixture_dir, ticker))
    assert fin.years[-1] == label
    assert fin.latest["period_end"] == period_end
    assert fin.latest["revenue"] / 1e6 == pytest.approx(revenue_m, abs=0.5)
    assert fin.latest["net_income"] / 1e6 == pytest.approx(net_income_m, abs=0.5)


MATURE = ["AAPL", "AMZN", "BRK-B", "COST", "GE", "INTC", "JPM", "KO", "MSFT", "NFLX", "NVDA", "O", "PFE",
          "PLTR", "SNOW", "TSLA", "WMT"]


@pytest.mark.parametrize("ticker", MATURE)
def test_fixture_sanity(fixture_dir, ticker):
    fin = build_financials(load_fixture(fixture_dir, ticker))
    table = fin.table
    assert len(table) == 5 and table.index.is_unique
    ends = [date.fromisoformat(e) for e in table["period_end"]]
    assert all(330 <= (b - a).days <= 400 for a, b in zip(ends, ends[1:]))   # consecutive fiscal years
    assert (table["revenue"] > 0).all()
    for col in ("gross_margin", "operating_margin", "net_margin"):
        assert table[col].dropna().between(-3, 1).all(), col
    shares = table["diluted_shares"].dropna()
    assert ((shares / shares.shift(1)).dropna().between(0.5, 2.0)).all()    # no unadjusted stock splits


def test_fixture_traps(fixture_dir):
    jpm = build_financials(load_fixture(fixture_dir, "JPM"))                   # bank: no COGS / gross profit
    assert jpm.table["gross_profit"].isna().all()
    assert jpm.latest["debt"] > 0
    o = build_financials(load_fixture(fixture_dir, "O"))                       # REIT: total, not subset
    assert o.tags_used["revenue"] == ["Revenues"]
    assert o.latest["debt"] / 1e9 == pytest.approx(27.3, abs=0.1)
    brk = build_financials(load_fixture(fixture_dir, "BRK-B"))
    assert brk.table["debt"].isna().all()
    assert any("not tagged in XBRL in any year" in n for n in brk.notes)
    wmt = build_financials(load_fixture(fixture_dir, "WMT"))                   # late-January fiscal year
    assert wmt.years == ["FY2022", "FY2023", "FY2024", "FY2025", "FY2026"]
    cost = build_financials(load_fixture(fixture_dir, "COST"))                 # 52/53-week years
    assert list(cost.table["period_end"])[1] == "2023-09-03"
    ge = build_financials(load_fixture(fixture_dir, "GE"))                     # restated after spin-offs
    assert math.isnan(ge.table.loc["FY2022", "revenue_growth"])


# Fiscal-year labels must match each filer's own numbering (January year-ends named by the end year here).
LABELS = {
    "AAPL": ["FY2021", "FY2022", "FY2023", "FY2024", "FY2025"],
    "COST": ["FY2022", "FY2023", "FY2024", "FY2025", "FY2026"],
    "INTC": ["FY2021", "FY2022", "FY2023", "FY2024", "FY2025"],
    "MSFT": ["FY2022", "FY2023", "FY2024", "FY2025", "FY2026"],
    "NVDA": ["FY2022", "FY2023", "FY2024", "FY2025", "FY2026"],
    "SNOW": ["FY2022", "FY2023", "FY2024", "FY2025", "FY2026"],
    "WMT": ["FY2022", "FY2023", "FY2024", "FY2025", "FY2026"],
}


@pytest.mark.parametrize("ticker", sorted(LABELS))
def test_fixture_year_labels(fixture_dir, ticker):
    assert build_financials(load_fixture(fixture_dir, ticker)).years == LABELS[ticker]


def test_fixture_cash_debt_and_week_counts(fixture_dir):
    msft = build_financials(load_fixture(fixture_dir, "MSFT")).latest       # 10-K: cash + ST investments
    assert (msft["cash"] + msft["short_term_investments"]) / 1e6 == pytest.approx(76_843)
    tsla = build_financials(load_fixture(fixture_dir, "TSLA"))              # 10-K "Total debt" incl. current
    assert list(tsla.table["debt"].iloc[-2:] / 1e6) == pytest.approx([7_878, 8_153])
    cost = build_financials(load_fixture(fixture_dir, "COST"))
    assert [n for n in cost.notes if "53 weeks" in n][0].startswith("FY2023 had 53 weeks")


@pytest.mark.parametrize("ticker, message", [("TSM", "IFRS"), ("XOM", "10-K")])
def test_fixture_graceful_failures(fixture_dir, ticker, message):
    with pytest.raises(EdgarError, match=message):
        build_financials(load_fixture(fixture_dir, ticker))
