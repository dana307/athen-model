"""
Bundled offline sample dataset for the public demo.

The deployed dashboard tries live data first (yfinance), but shared cloud IPs
can get rate-limited. This sample lets the demo *always* show a fully working
valuation with no network — illustrative RELIANCE-like figures in absolute INR.
"""
from __future__ import annotations

import pandas as pd

from utils.schema import CANONICAL_FIELDS

# years (fiscal), most recent first
_YEARS = [2024, 2023, 2022, 2021, 2020]

# all values in absolute INR; shares in absolute count
_DATA = {
    "revenue":            [9.00e12, 8.00e12, 7.00e12, 5.50e12, 5.00e12],
    "ebitda":             [1.60e12, 1.42e12, 1.25e12, 0.95e12, 0.90e12],
    "ebit":               [1.10e12, 0.97e12, 0.85e12, 0.63e12, 0.60e12],
    "pat":                [0.70e12, 0.63e12, 0.55e12, 0.42e12, 0.40e12],
    "depreciation":       [0.50e12, 0.45e12, 0.40e12, 0.32e12, 0.30e12],
    "capex":              [1.10e12, 1.00e12, 0.90e12, 0.80e12, 0.75e12],
    "working_capital":    [-0.50e12, -0.48e12, -0.45e12, -0.40e12, -0.38e12],
    "receivables":        [0.46e12, 0.41e12, 0.36e12, 0.28e12, 0.26e12],
    "inventory":          [0.55e12, 0.50e12, 0.44e12, 0.34e12, 0.31e12],
    "cash":               [2.00e12, 1.90e12, 1.75e12, 1.50e12, 1.40e12],
    "debt":               [3.20e12, 3.00e12, 2.80e12, 2.55e12, 2.50e12],
    "total_equity":       [7.50e12, 6.90e12, 6.30e12, 5.60e12, 5.20e12],
    "shares_outstanding": [6.76e9, 6.76e9, 6.76e9, 6.33e9, 6.33e9],
}


def get_sample(ticker: str = "DEMO"):
    """Return (canonical fundamentals DataFrame, market-data dict)."""
    df = pd.DataFrame(_DATA, index=pd.Index(_YEARS, name="fiscal_year"))
    df = df[CANONICAL_FIELDS].sort_index(ascending=False)
    md = {"price": 2900.0, "beta": 1.05,
          "name": f"{ticker.upper()} (demo data)"}
    return df, md
