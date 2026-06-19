"""
Phase 7 — Monte Carlo simulation.

The deterministic DCF answers "what's it worth given *these* assumptions?".
Monte Carlo answers "given the *uncertainty* in those assumptions, what's the
distribution of value, and how likely is the stock undervalued?".

We replace point assumptions with distributions:

    revenue growth ~ Normal(μ_g, σ_g)
    EBITDA margin  ~ Normal(μ_m, σ_m)
    WACC           ~ Normal(μ_w, σ_w)
    terminal growth~ Normal(μ_tg, σ_tg)

and run N simulations. The whole thing is vectorized over simulations with
numpy (no Python loop), so 10,000 paths cost milliseconds. The formulas mirror
forecasting.engine + valuation.dcf exactly; a zero-variance run reproduces the
deterministic DCF (asserted in the tests).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import settings
from utils.metrics import historical_drivers
from utils.logging import get_logger

log = get_logger("valuation.montecarlo")


@dataclass
class MonteCarloConfig:
    n_sims: int = 10_000
    years: int = settings.FORECAST_YEARS
    tax_rate: float = settings.DEFAULT_TAX_RATE
    seed: int | None = 42

    # (mean, std). mean=None -> filled from the company's history.
    revenue_growth: tuple = (None, 0.02)
    ebitda_margin: tuple = (None, 0.01)
    wacc: tuple = (0.11, 0.01)
    terminal_growth: tuple = (settings.TERMINAL_GROWTH, 0.005)


@dataclass
class MonteCarloResult:
    prices: np.ndarray
    n_sims: int
    mean: float
    median: float
    std: float
    p5: float
    p95: float
    prob_undervalued: float
    current_price: float | None = None
    assumptions: dict = field(default_factory=dict)

    def histogram(self, bins: int = 40):
        counts, edges = np.histogram(self.prices, bins=bins)
        return counts, edges

    def summary(self) -> str:
        s = (f"intrinsic ₹{self.mean:,.0f} mean / ₹{self.median:,.0f} median "
             f"(σ ₹{self.std:,.0f}) | 90% CI ₹{self.p5:,.0f}–₹{self.p95:,.0f}")
        if self.current_price is not None:
            s += f" | P(undervalued) {self.prob_undervalued:.0%}"
        return s


def simulate(
    fundamentals: pd.DataFrame,
    config: MonteCarloConfig | None = None,
    *,
    current_price: float | None = None,
) -> MonteCarloResult:
    cfg = config or MonteCarloConfig()
    p = historical_drivers(fundamentals)
    rng = np.random.default_rng(cfg.seed)

    # --- base values & fixed ratios (held at historical averages) ----------
    R0 = p.latest_revenue
    last_wc = float(fundamentals.sort_index(ascending=False)["working_capital"]
                    .dropna().iloc[0])
    shares = float(fundamentals.sort_index(ascending=False)["shares_outstanding"]
                   .dropna().iloc[0])
    net_debt = p.latest_net_debt
    capex_r, dep_r, wc_r = (p.avg_capex_pct_sales, p.avg_dep_pct_sales,
                            p.avg_wc_pct_sales)

    # --- resolve distribution means ----------------------------------------
    g_mu = p.revenue_cagr if cfg.revenue_growth[0] is None else cfg.revenue_growth[0]
    m_mu = p.avg_ebitda_margin if cfg.ebitda_margin[0] is None else cfg.ebitda_margin[0]
    w_mu, tg_mu = cfg.wacc[0], cfg.terminal_growth[0]

    n, N = cfg.n_sims, cfg.years

    # --- draw assumptions (shape: n) ---------------------------------------
    g = rng.normal(g_mu, cfg.revenue_growth[1], n)
    m = np.clip(rng.normal(m_mu, cfg.ebitda_margin[1], n), 0.01, 0.95)
    w = np.maximum(rng.normal(w_mu, cfg.wacc[1], n), 0.02)
    tg = rng.normal(tg_mu, cfg.terminal_growth[1], n)
    tg = np.minimum(tg, w - 0.005)   # keep Gordon-growth valid (WACC > g)

    # --- build the operating model, vectorized over sims -------------------
    t = np.arange(1, N + 1)                      # (N,)
    revenue = R0 * np.power(1 + g[:, None], t[None, :])   # (n, N)
    ebitda = revenue * m[:, None]
    dep = revenue * dep_r
    ebit = ebitda - dep
    capex = revenue * capex_r
    wc = revenue * wc_r
    nopat = ebit * (1 - cfg.tax_rate)

    wc_prev = np.concatenate(
        [np.full((n, 1), last_wc), wc[:, :-1]], axis=1)
    change_wc = wc - wc_prev
    fcff = nopat + dep - capex - change_wc

    # --- discount ----------------------------------------------------------
    disc = 1.0 / np.power(1 + w[:, None], t[None, :])     # (n, N)
    pv_explicit = (fcff * disc).sum(axis=1)
    tv = fcff[:, -1] * (1 + tg) / (w - tg)
    pv_tv = tv / np.power(1 + w, N)
    ev = pv_explicit + pv_tv
    equity = ev - net_debt
    prices = equity / shares

    # --- stats -------------------------------------------------------------
    prob = float(np.mean(prices > current_price)) if current_price else float("nan")
    res = MonteCarloResult(
        prices=prices, n_sims=n,
        mean=float(np.mean(prices)), median=float(np.median(prices)),
        std=float(np.std(prices)),
        p5=float(np.percentile(prices, 5)), p95=float(np.percentile(prices, 95)),
        prob_undervalued=prob, current_price=current_price,
        assumptions={
            "revenue_growth": (g_mu, cfg.revenue_growth[1]),
            "ebitda_margin": (m_mu, cfg.ebitda_margin[1]),
            "wacc": (w_mu, cfg.wacc[1]),
            "terminal_growth": (tg_mu, cfg.terminal_growth[1]),
        },
    )
    log.info("MonteCarlo (%d sims) | %s", n, res.summary())
    return res
