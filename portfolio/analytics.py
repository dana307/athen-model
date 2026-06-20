"""
Phase 12 — Portfolio Analytics.

Aggregates a set of holdings into portfolio-level measures:

    portfolio intrinsic value · market value · weights · sector allocation ·
    concentration risk (HHI, effective holdings, top-N) ·
    weighted expected return · weighted valuation gap

A Holding pairs a position (quantity) with its market price and the model's
intrinsic price (from the Phase 4-5 DCF). The Portfolio rolls them up. A
build_holding() helper wires a holding straight from fundamentals via the
valuation pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from valuation import value_company
from forecasting import ForecastConfig
from utils.logging import get_logger

log = get_logger("portfolio")


@dataclass
class Holding:
    ticker: str
    quantity: float
    market_price: float
    intrinsic_price: float
    sector: str = "Unknown"
    expected_return: float | None = None   # defaults to the valuation gap

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def intrinsic_value(self) -> float:
        return self.quantity * self.intrinsic_price

    @property
    def valuation_gap(self) -> float:
        """Upside if price converges to intrinsic value."""
        return self.intrinsic_price / self.market_price - 1 if self.market_price else np.nan

    @property
    def exp_return(self) -> float:
        return self.valuation_gap if self.expected_return is None else self.expected_return


@dataclass
class PortfolioAnalytics:
    holdings: list
    total_market_value: float
    total_intrinsic_value: float
    weights: dict
    sector_allocation: dict
    hhi: float
    effective_holdings: float
    top_weight: float
    top3_weight: float
    concentration_level: str
    weighted_expected_return: float
    weighted_valuation_gap: float
    portfolio_upside: float
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        return (f"MV ₹{self.total_market_value/1e7:,.0f}cr → intrinsic "
                f"₹{self.total_intrinsic_value/1e7:,.0f}cr "
                f"({self.portfolio_upside:+.1%}) | "
                f"exp.return {self.weighted_expected_return:+.1%} | "
                f"HHI {self.hhi:.2f} ({self.concentration_level}), "
                f"~{self.effective_holdings:.1f} effective names")


def _concentration_level(hhi: float) -> str:
    if hhi < 0.15:
        return "Diversified"
    if hhi < 0.25:
        return "Moderately concentrated"
    return "Concentrated"


class Portfolio:
    def __init__(self, holdings: list[Holding] | None = None):
        self.holdings: list[Holding] = list(holdings or [])

    def add(self, holding: Holding) -> None:
        self.holdings.append(holding)

    def analyze(self) -> PortfolioAnalytics:
        if not self.holdings:
            raise ValueError("portfolio has no holdings")

        total_mv = sum(h.market_value for h in self.holdings)
        total_iv = sum(h.intrinsic_value for h in self.holdings)
        if total_mv <= 0:
            raise ValueError("total market value must be positive")

        weights = {h.ticker: h.market_value / total_mv for h in self.holdings}

        # sector allocation
        sector_alloc: dict[str, float] = {}
        for h in self.holdings:
            sector_alloc[h.sector] = sector_alloc.get(h.sector, 0.0) + weights[h.ticker]

        # concentration
        w = np.array(list(weights.values()))
        hhi = float((w ** 2).sum())
        effective = 1.0 / hhi
        top_weight = float(w.max())
        top3 = float(np.sort(w)[::-1][:3].sum())

        # weighted measures
        wer = float(sum(weights[h.ticker] * h.exp_return for h in self.holdings))
        wgap = float(sum(weights[h.ticker] * h.valuation_gap for h in self.holdings))
        port_upside = total_iv / total_mv - 1

        table = pd.DataFrame([{
            "ticker": h.ticker, "sector": h.sector, "qty": h.quantity,
            "mkt_price": h.market_price, "intrinsic": h.intrinsic_price,
            "weight": weights[h.ticker], "mkt_value": h.market_value,
            "valuation_gap": h.valuation_gap, "exp_return": h.exp_return,
        } for h in self.holdings]).set_index("ticker")

        res = PortfolioAnalytics(
            holdings=self.holdings,
            total_market_value=total_mv, total_intrinsic_value=total_iv,
            weights=weights, sector_allocation=sector_alloc,
            hhi=hhi, effective_holdings=effective,
            top_weight=top_weight, top3_weight=top3,
            concentration_level=_concentration_level(hhi),
            weighted_expected_return=wer, weighted_valuation_gap=wgap,
            portfolio_upside=port_upside, table=table,
        )
        log.info("Portfolio | %s", res.summary())
        return res


def build_holding(
    ticker: str,
    fundamentals: pd.DataFrame,
    *,
    market_price: float,
    quantity: float,
    sector: str = "Unknown",
    beta: float = 1.0,
    years: int = 5,
    **val_kwargs,
) -> Holding:
    """Value a company via the DCF pipeline and wrap it as a Holding."""
    val = value_company(fundamentals, market_price=market_price, beta=beta,
                        forecast_config=ForecastConfig(years=years), **val_kwargs)
    return Holding(ticker=ticker.upper(), quantity=quantity,
                   market_price=market_price,
                   intrinsic_price=val.dcf.intrinsic_price, sector=sector)
