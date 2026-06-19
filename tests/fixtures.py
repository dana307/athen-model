"""
A yfinance-shaped fixture.

Mirrors the exact structure yfinance returns (statement DataFrames indexed by
line-item label, columns = period-end Timestamps) so we can test the loader's
normalize() logic deterministically, without network access. Numbers are
RELIANCE-like (in absolute INR) but illustrative, not official.
"""
from __future__ import annotations

import pandas as pd

_COLS = [pd.Timestamp(f"{y}-03-31") for y in (2024, 2023, 2022, 2021)]


def _df(rows: dict[str, list]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=_COLS).T  # rows become the index


INCOME = _df({
    "Total Revenue":      [9_00_000e7, 8_00_000e7, 7_00_000e7, 5_00_000e7],
    "EBITDA":             [1_60_000e7, 1_45_000e7, 1_25_000e7, 90_000e7],
    "EBIT":               [1_10_000e7, 1_00_000e7, 85_000e7, 60_000e7],
    "Net Income":         [70_000e7, 66_000e7, 60_000e7, 49_000e7],
})

BALANCE = _df({
    "Cash And Cash Equivalents": [2_00_000e7, 1_90_000e7, 1_75_000e7, 1_50_000e7],
    "Total Debt":                [3_20_000e7, 3_00_000e7, 2_80_000e7, 2_50_000e7],
    "Stockholders Equity":       [7_50_000e7, 6_90_000e7, 6_30_000e7, 5_60_000e7],
    # receivables & inventory roughly track sales -> stable ratios (low risk)
    "Accounts Receivable":       [90_000e7, 80_000e7, 70_000e7, 50_000e7],
    "Inventory":                 [1_00_000e7, 90_000e7, 78_000e7, 56_000e7],
    "Current Assets":            [2_50_000e7, 2_30_000e7, 2_10_000e7, 1_80_000e7],
    "Current Liabilities":       [3_00_000e7, 2_80_000e7, 2_60_000e7, 2_20_000e7],
    "Ordinary Shares Number":    [676e7, 676e7, 633e7, 633e7],
})

CASHFLOW = _df({
    # Yahoo reports capex as a negative outflow — loader must abs() it.
    "Capital Expenditure":          [-1_20_000e7, -1_00_000e7, -90_000e7, -80_000e7],
    "Depreciation And Amortization": [50_000e7, 45_000e7, 40_000e7, 30_000e7],
})

INFO = {"sharesOutstanding": 676e7, "longName": "Reliance Industries Limited"}


def raw_bundle() -> dict:
    """The dict shape YFinanceLoader.fetch_raw() returns."""
    return {
        "income": INCOME.copy(),
        "balance": BALANCE.copy(),
        "cashflow": CASHFLOW.copy(),
        "info": dict(INFO),
    }


def stressed_bundle() -> dict:
    """A deliberately distressed company: high/rising leverage, collapsing
    margins, negative FCF, ballooning receivables & inventory. Exercises the
    HIGH-risk branches of the Phase 9 engine."""
    income = _df({
        "Total Revenue": [5_00_000e7, 5_20_000e7, 5_00_000e7, 4_50_000e7],
        "EBITDA":        [50_000e7, 70_000e7, 80_000e7, 75_000e7],   # margin collapses
        "EBIT":          [20_000e7, 45_000e7, 55_000e7, 50_000e7],
        "Net Income":    [5_000e7, 30_000e7, 40_000e7, 38_000e7],
    })
    balance = _df({
        "Cash And Cash Equivalents": [10_000e7, 30_000e7, 40_000e7, 50_000e7],
        "Total Debt":                [6_00_000e7, 4_00_000e7, 3_50_000e7, 3_20_000e7],  # +50% YoY
        "Stockholders Equity":       [2_00_000e7, 2_50_000e7, 2_60_000e7, 2_55_000e7],
        "Accounts Receivable":       [1_50_000e7, 1_10_000e7, 90_000e7, 70_000e7],  # rising % sales
        "Inventory":                 [1_40_000e7, 1_00_000e7, 85_000e7, 70_000e7],
        "Current Assets":            [2_00_000e7, 1_80_000e7, 1_70_000e7, 1_50_000e7],
        "Current Liabilities":       [3_00_000e7, 2_40_000e7, 2_20_000e7, 2_00_000e7],
        "Ordinary Shares Number":    [500e7, 500e7, 500e7, 500e7],
    })
    cashflow = _df({
        "Capital Expenditure":           [-1_50_000e7, -1_00_000e7, -90_000e7, -80_000e7],
        "Depreciation And Amortization": [40_000e7, 38_000e7, 35_000e7, 30_000e7],
    })
    return {"income": income, "balance": balance, "cashflow": cashflow,
            "info": dict(INFO)}


def scaled_bundle(scale: float, margin_boost: float = 0.0) -> dict:
    """A peer-like bundle: same shape, scaled magnitudes + optional margin tweak.

    Lets tests build a small peer set with genuinely different multiples.
    """
    b = raw_bundle()
    for key in ("income", "balance", "cashflow"):
        b[key] = b[key] * scale
    if margin_boost:
        b["income"].loc["EBITDA"] *= (1 + margin_boost)
        b["income"].loc["Net Income"] *= (1 + margin_boost)
    b["info"] = dict(INFO)
    return b
