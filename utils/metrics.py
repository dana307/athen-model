"""
Phase 2 — normalization & derived metrics.

The loader already maps every source onto the canonical schema. What's left for
"normalization" is the analytical layer: turning the 10 raw canonical fields
into the ratios, growth rates, and cash-flow building blocks that the valuation
engine actually consumes.

Two public entry points:
    add_derived_metrics(df)  -> wide frame + derived columns (per fiscal year)
    historical_drivers(df)   -> DriverProfile: the summary ratios Phase 3 forecasts

All inputs/outputs use the canonical convention: index = fiscal year, and the
caller may pass it in either order — we sort internally where order matters.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from config import settings
from utils.schema import CANONICAL_FIELDS

# Columns added by add_derived_metrics (documented for downstream consumers).
DERIVED_FIELDS = [
    "ebitda_margin", "ebit_margin", "net_margin",
    "revenue_growth",
    "net_debt",
    "capex_pct_sales", "dep_pct_sales", "wc_pct_sales",
    "receivables_pct_sales", "inventory_pct_sales",
    "implied_tax_rate",
    "nopat", "change_in_wc", "fcff",
]


def _safe_div(a, b):
    """Element-wise divide that yields NaN instead of inf/0-division noise."""
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.replace(0, np.nan)


def add_derived_metrics(
    df: pd.DataFrame, tax_rate: float | None = None
) -> pd.DataFrame:
    """Return a copy of canonical fundamentals with analytical columns added.

    Rows are sorted ascending by year so that year-over-year deltas (growth,
    change in working capital) are well defined.
    """
    tax_rate = settings.DEFAULT_TAX_RATE if tax_rate is None else tax_rate

    out = df.copy()
    out = out.sort_index(ascending=True)  # oldest -> newest for deltas

    rev = out["revenue"]

    # --- profitability margins ---------------------------------------------
    out["ebitda_margin"] = _safe_div(out["ebitda"], rev)
    out["ebit_margin"] = _safe_div(out["ebit"], rev)
    out["net_margin"] = _safe_div(out["pat"], rev)

    # --- growth -------------------------------------------------------------
    out["revenue_growth"] = rev.pct_change()

    # --- balance-sheet derived ---------------------------------------------
    out["net_debt"] = out["debt"] - out["cash"]

    # --- intensity ratios (drivers for forecasting) ------------------------
    out["capex_pct_sales"] = _safe_div(out["capex"], rev)
    out["dep_pct_sales"] = _safe_div(out["depreciation"], rev)
    out["wc_pct_sales"] = _safe_div(out["working_capital"], rev)
    out["receivables_pct_sales"] = _safe_div(out["receivables"], rev)
    out["inventory_pct_sales"] = _safe_div(out["inventory"], rev)

    # --- tax (approximate: ignores interest/other income) ------------------
    # Implied from EBIT->PAT, clamped to a sane band; informational only.
    implied = 1 - _safe_div(out["pat"], out["ebit"])
    out["implied_tax_rate"] = implied.clip(lower=0.0, upper=0.5)

    # --- cash-flow building blocks (historical FCFF) -----------------------
    out["change_in_wc"] = out["working_capital"].diff()
    out["nopat"] = out["ebit"] * (1 - tax_rate)
    # FCFF = NOPAT + D&A - Capex - increase in working capital
    out["fcff"] = (
        out["nopat"]
        + out["depreciation"].fillna(0)
        - out["capex"].fillna(0)
        - out["change_in_wc"].fillna(0)
    )

    return out.sort_index(ascending=False)  # back to canonical (newest first)


@dataclass
class DriverProfile:
    """Summary of historical drivers — the seed for Phase 3 forecasting."""
    latest_year: int
    latest_revenue: float
    n_years: int
    revenue_cagr: float            # geometric, over the available window
    avg_revenue_growth: float      # arithmetic mean YoY
    avg_ebitda_margin: float
    latest_ebitda_margin: float
    avg_capex_pct_sales: float
    avg_dep_pct_sales: float
    avg_wc_pct_sales: float
    latest_net_debt: float
    avg_implied_tax_rate: float

    def as_dict(self) -> dict:
        return asdict(self)


def historical_drivers(df: pd.DataFrame) -> DriverProfile:
    """Collapse the historical metrics into forecastable driver assumptions."""
    m = add_derived_metrics(df).sort_index(ascending=True)
    rev = m["revenue"].dropna()
    if len(rev) < 2:
        raise ValueError("Need at least 2 years of revenue to profile drivers.")

    first, last = rev.iloc[0], rev.iloc[-1]
    span = len(rev) - 1
    cagr = (last / first) ** (1 / span) - 1 if first > 0 else np.nan

    return DriverProfile(
        latest_year=int(m.index.max()),
        latest_revenue=float(rev.iloc[-1]),
        n_years=int(len(rev)),
        revenue_cagr=float(cagr),
        avg_revenue_growth=float(m["revenue_growth"].mean(skipna=True)),
        avg_ebitda_margin=float(m["ebitda_margin"].mean(skipna=True)),
        latest_ebitda_margin=float(m["ebitda_margin"].dropna().iloc[-1]),
        avg_capex_pct_sales=float(m["capex_pct_sales"].mean(skipna=True)),
        avg_dep_pct_sales=float(m["dep_pct_sales"].mean(skipna=True)),
        avg_wc_pct_sales=float(m["wc_pct_sales"].mean(skipna=True)),
        latest_net_debt=float(m["net_debt"].dropna().iloc[-1]),
        avg_implied_tax_rate=float(m["implied_tax_rate"].mean(skipna=True)),
    )
