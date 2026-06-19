"""Phase 11 tests — dashboard compute helper + app import (offline)."""
from __future__ import annotations

import importlib

import pytest

from loaders.yfinance_loader import YFinanceLoader
from forecasting import forecast, ForecastConfig
from valuation import intrinsic_price
from valuation.dcf import run_dcf
from utils.metrics import historical_drivers
from tests import fixtures


def _load(bundle):
    class L(YFinanceLoader):
        def fetch_raw(self):
            return bundle
    return L("X").load()


@pytest.fixture
def healthy():
    return _load(fixtures.raw_bundle())


def test_intrinsic_price_matches_manual_dcf(healthy):
    """The slider helper must equal a hand-wired forecast+DCF."""
    g, m, w, tg = 0.10, 0.18, 0.12, 0.045
    live = intrinsic_price(healthy, growth=g, margin=m, wacc=w,
                           terminal_growth=tg, years=5, tax_rate=0.25)

    cfg = ForecastConfig(
        years=5, tax_rate=0.25,
        revenue_method="constant_growth", revenue_params={"growth": g},
        margin_method="constant", margin_params={"value": m})
    proj = forecast(healthy, cfg)
    p = historical_drivers(healthy)
    shares = healthy.sort_index(ascending=False)["shares_outstanding"].iloc[0]
    expected = run_dcf(proj, w, net_debt=p.latest_net_debt,
                       shares_outstanding=shares, terminal_growth=tg).intrinsic_price
    assert live == pytest.approx(expected)


def test_intrinsic_price_responds_to_sliders(healthy):
    base = intrinsic_price(healthy, growth=0.08, margin=0.16, wacc=0.13)
    higher = intrinsic_price(healthy, growth=0.12, margin=0.20, wacc=0.11)
    assert higher > base       # more growth/margin, lower WACC -> higher value


def test_dashboard_module_imports():
    """The Streamlit app must import cleanly (catches syntax/import errors)."""
    mod = importlib.import_module("dashboard.streamlit_app")
    assert hasattr(mod, "main")
    assert hasattr(mod, "load_company")
