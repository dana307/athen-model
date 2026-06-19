"""
Phase 6 — Comparable Company Analysis.

Two halves:

1. compute_multiples(): the trading multiples & returns ratios for one company
   from its fundamentals + market price:
       EV/Sales, EV/EBITDA, P/E, PEG, ROE, ROCE, Debt/Equity

2. peer_comparison(): take a target + a peer set, compute the *median* peer
   multiple for each valuation multiple, apply it to the target's metrics, and
   back out an implied per-share value (per multiple, plus a blended figure).

This is the market-based cross-check on the intrinsic DCF: the DCF asks "what
are the cash flows worth?"; comps ask "what does the market pay for similar
businesses right now?".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

from utils.logging import get_logger

log = get_logger("valuation.multiples")

# valuation multiples used to imply a target price from peers
VALUATION_MULTIPLES = ["ev_sales", "ev_ebitda", "pe"]
# quality/returns ratios shown for context (not used to imply price)
QUALITY_RATIOS = ["peg", "roe", "roce", "debt_to_equity"]


def _latest(df: pd.DataFrame, field_: str) -> float:
    s = df.sort_index(ascending=False)[field_].dropna()
    return float(s.iloc[0]) if not s.empty else np.nan


def _cagr(series: pd.Series) -> float:
    s = series.dropna().sort_index()
    if len(s) < 2 or s.iloc[0] <= 0:
        return np.nan
    return (s.iloc[-1] / s.iloc[0]) ** (1 / (len(s) - 1)) - 1


@dataclass
class CompanyMultiples:
    ticker: str
    # raw inputs (latest year, absolute INR)
    market_cap: float
    enterprise_value: float
    revenue: float
    ebitda: float
    pat: float
    net_debt: float
    total_equity: float
    shares: float
    earnings_growth: float
    # multiples
    ev_sales: float
    ev_ebitda: float
    pe: float
    peg: float
    roe: float
    roce: float
    debt_to_equity: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_multiples(
    ticker: str,
    fundamentals: pd.DataFrame,
    market_price: float,
    earnings_growth: float | None = None,
) -> CompanyMultiples:
    """Trading multiples + returns ratios for a single company."""
    rev = _latest(fundamentals, "revenue")
    ebitda = _latest(fundamentals, "ebitda")
    ebit = _latest(fundamentals, "ebit")
    pat = _latest(fundamentals, "pat")
    cash = _latest(fundamentals, "cash")
    debt = _latest(fundamentals, "debt")
    equity = _latest(fundamentals, "total_equity")
    shares = _latest(fundamentals, "shares_outstanding")

    net_debt = debt - cash
    market_cap = market_price * shares
    ev = market_cap + net_debt

    # earnings growth for PEG: explicit, else historical PAT CAGR
    g = earnings_growth if earnings_growth is not None else _cagr(fundamentals["pat"])

    def safe(n, d):
        return n / d if (d and not np.isnan(d) and d != 0) else np.nan

    pe = safe(market_cap, pat)
    cm = CompanyMultiples(
        ticker=ticker.upper(),
        market_cap=market_cap, enterprise_value=ev,
        revenue=rev, ebitda=ebitda, pat=pat,
        net_debt=net_debt, total_equity=equity, shares=shares,
        earnings_growth=g,
        ev_sales=safe(ev, rev),
        ev_ebitda=safe(ev, ebitda),
        pe=pe,
        peg=safe(pe, g * 100) if (g and not np.isnan(g)) else np.nan,
        roe=safe(pat, equity),
        roce=safe(ebit, equity + debt),
        debt_to_equity=safe(debt, equity),
    )
    return cm


@dataclass
class ComparablesResult:
    target: CompanyMultiples
    peers: list[CompanyMultiples]
    table: pd.DataFrame                 # all companies x metrics
    peer_medians: dict                  # median peer multiple per valuation multiple
    implied_prices: dict                # valuation multiple -> implied price/share
    implied_price: float                # blended (mean of available implied)
    current_price: float
    upside: float = field(default=np.nan)

    def summary(self) -> str:
        parts = [f"{k}=₹{v:,.0f}" for k, v in self.implied_prices.items()
                 if not np.isnan(v)]
        return (f"Implied (blended) ₹{self.implied_price:,.0f}/sh "
                f"[{', '.join(parts)}] vs mkt ₹{self.current_price:,.0f} "
                f"({self.upside:+.1%})")


def peer_comparison(
    target_fundamentals: pd.DataFrame,
    target_ticker: str,
    target_price: float,
    peers: dict[str, tuple[pd.DataFrame, float]],
    multiples: list[str] | None = None,
) -> ComparablesResult:
    """Value the target off median peer multiples.

    peers maps ticker -> (fundamentals_df, market_price).
    """
    multiples = multiples or VALUATION_MULTIPLES
    target = compute_multiples(target_ticker, target_fundamentals, target_price)
    peer_cms = [compute_multiples(t, f, p) for t, (f, p) in peers.items()]
    if not peer_cms:
        raise ValueError("need at least one peer for a comparable analysis")

    # table for display
    all_cms = [target] + peer_cms
    cols = ["ev_sales", "ev_ebitda", "pe"] + QUALITY_RATIOS
    table = pd.DataFrame(
        {cm.ticker: {c: getattr(cm, c) for c in cols} for cm in all_cms}
    ).T
    table.index.name = "ticker"

    # median peer multiple (peers only — exclude the target)
    peer_df = pd.DataFrame(
        {cm.ticker: {m: getattr(cm, m) for m in multiples} for cm in peer_cms}
    ).T
    medians = {m: float(np.nanmedian(peer_df[m].values)) for m in multiples}

    # imply target price from each multiple
    implied = {}
    for m in multiples:
        med = medians[m]
        if np.isnan(med):
            implied[m] = np.nan
            continue
        if m == "ev_sales":
            ev = med * target.revenue
            eq = ev - target.net_debt
        elif m == "ev_ebitda":
            ev = med * target.ebitda
            eq = ev - target.net_debt
        elif m == "pe":
            eq = med * target.pat
        else:
            implied[m] = np.nan
            continue
        implied[m] = eq / target.shares if target.shares else np.nan

    vals = [v for v in implied.values() if not np.isnan(v)]
    blended = float(np.mean(vals)) if vals else np.nan
    upside = (blended / target_price - 1) if target_price else np.nan

    res = ComparablesResult(
        target=target, peers=peer_cms, table=table,
        peer_medians=medians, implied_prices=implied,
        implied_price=blended, current_price=target_price, upside=upside,
    )
    log.info("Comps | %s", res.summary())
    return res
