"""Phase 5 tests — FCFF DCF engine & full valuation pipeline (offline)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from valuation.dcf import run_dcf
from valuation import value_company
from forecasting import ForecastConfig
from loaders.yfinance_loader import YFinanceLoader
from tests import fixtures


def _flat_forecast(fcff=100.0, n=3):
    return pd.DataFrame({"fcff": [fcff] * n},
                        index=[2025 + i for i in range(n)])


def test_dcf_matches_independent_math():
    fcff, n, w, g = 100.0, 3, 0.10, 0.04
    res = run_dcf(_flat_forecast(fcff, n), wacc=w, net_debt=50.0,
                  shares_outstanding=10.0, terminal_growth=g)

    # independent recomputation
    t = np.arange(1, n + 1)
    pv_explicit = (fcff / (1 + w) ** t).sum()
    tv = fcff * (1 + g) / (w - g)
    pv_tv = tv / (1 + w) ** n
    ev = pv_explicit + pv_tv

    assert res.pv_explicit == pytest.approx(pv_explicit)
    assert res.pv_terminal == pytest.approx(pv_tv)
    assert res.enterprise_value == pytest.approx(ev)
    assert res.equity_value == pytest.approx(ev - 50.0)
    assert res.intrinsic_price == pytest.approx((ev - 50.0) / 10.0)


def test_ev_equity_bridge():
    res = run_dcf(_flat_forecast(), wacc=0.10, net_debt=200.0,
                  shares_outstanding=10.0, terminal_growth=0.03)
    # with positive net debt, EV must exceed equity value
    assert res.enterprise_value > res.equity_value
    assert res.equity_value == pytest.approx(res.enterprise_value - 200.0)


def test_mid_year_raises_value():
    eoy = run_dcf(_flat_forecast(), wacc=0.10, net_debt=0.0,
                  shares_outstanding=10.0, terminal_growth=0.04)
    mid = run_dcf(_flat_forecast(), wacc=0.10, net_debt=0.0,
                  shares_outstanding=10.0, terminal_growth=0.04, mid_year=True)
    # cash discounted half a year less -> higher PV
    assert mid.pv_explicit > eoy.pv_explicit


def test_upside_computed():
    res = run_dcf(_flat_forecast(), wacc=0.10, net_debt=0.0,
                  shares_outstanding=10.0, terminal_growth=0.04,
                  current_price=100.0)
    assert res.upside == pytest.approx(res.intrinsic_price / 100.0 - 1)


def test_wacc_must_exceed_growth():
    with pytest.raises(ValueError):
        run_dcf(_flat_forecast(), wacc=0.04, net_debt=0.0,
                shares_outstanding=10.0, terminal_growth=0.05)


def test_shares_must_be_positive():
    with pytest.raises(ValueError):
        run_dcf(_flat_forecast(), wacc=0.10, net_debt=0.0,
                shares_outstanding=0.0, terminal_growth=0.04)


# --- full pipeline on the fixture ------------------------------------------
class FakeYFLoader(YFinanceLoader):
    def fetch_raw(self):
        return fixtures.raw_bundle()


def test_value_company_pipeline():
    df = FakeYFLoader("RELIANCE").load()
    res = value_company(
        df, market_price=2900.0, beta=1.1,
        forecast_config=ForecastConfig(years=5, revenue_method="fade_growth"),
        risk_free_rate=0.07, equity_risk_premium=0.06,
    )
    assert len(res.forecast) == 5
    assert res.wacc.wacc > res.dcf.terminal_growth      # DCF precondition held
    assert res.dcf.intrinsic_price > 0
    assert res.dcf.upside == pytest.approx(
        res.dcf.intrinsic_price / 2900.0 - 1)
