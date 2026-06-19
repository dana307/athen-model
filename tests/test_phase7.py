"""Phase 7 tests — Monte Carlo simulation (offline)."""
from __future__ import annotations

import numpy as np
import pytest

from loaders.yfinance_loader import YFinanceLoader
from forecasting import forecast, ForecastConfig
from valuation.dcf import run_dcf
from valuation.montecarlo import simulate, MonteCarloConfig
from utils.metrics import historical_drivers
from tests import fixtures


class FakeYFLoader(YFinanceLoader):
    def fetch_raw(self):
        return fixtures.raw_bundle()


@pytest.fixture
def fundamentals():
    return FakeYFLoader("RELIANCE").load()


def test_result_shape_and_stats(fundamentals):
    res = simulate(fundamentals, MonteCarloConfig(n_sims=5000, seed=1),
                   current_price=2900.0)
    assert len(res.prices) == 5000
    assert res.p5 <= res.median <= res.p95
    assert 0.0 <= res.prob_undervalued <= 1.0


def test_reproducible_with_seed(fundamentals):
    a = simulate(fundamentals, MonteCarloConfig(n_sims=2000, seed=7))
    b = simulate(fundamentals, MonteCarloConfig(n_sims=2000, seed=7))
    assert a.mean == pytest.approx(b.mean)
    assert np.allclose(a.prices, b.prices)


def test_zero_variance_matches_deterministic_dcf(fundamentals):
    """The crucial consistency check: with all σ=0, Monte Carlo must reproduce
    the deterministic DCF for the same point assumptions."""
    g, m, w, tg = 0.10, 0.18, 0.12, 0.045

    # deterministic path through the real engine
    cfg = ForecastConfig(
        years=5, tax_rate=0.25,
        revenue_method="constant_growth", revenue_params={"growth": g},
        margin_method="constant", margin_params={"value": m},
    )
    proj = forecast(fundamentals, cfg)
    p = historical_drivers(fundamentals)
    shares = fundamentals.sort_index(ascending=False)["shares_outstanding"].iloc[0]
    dcf = run_dcf(proj, w, net_debt=p.latest_net_debt,
                  shares_outstanding=shares, terminal_growth=tg)

    # Monte Carlo with zero variance on every driver
    mc = simulate(
        fundamentals,
        MonteCarloConfig(
            n_sims=100, seed=0, years=5, tax_rate=0.25,
            revenue_growth=(g, 0.0), ebitda_margin=(m, 0.0),
            wacc=(w, 0.0), terminal_growth=(tg, 0.0),
        ),
    )
    assert mc.mean == pytest.approx(dcf.intrinsic_price, rel=1e-9)
    assert mc.std == pytest.approx(0.0, abs=1e-6)


def test_higher_growth_raises_value(fundamentals):
    lo = simulate(fundamentals, MonteCarloConfig(
        n_sims=3000, seed=3, revenue_growth=(0.05, 0.0)))
    hi = simulate(fundamentals, MonteCarloConfig(
        n_sims=3000, seed=3, revenue_growth=(0.15, 0.0)))
    assert hi.mean > lo.mean
