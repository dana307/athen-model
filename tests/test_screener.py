"""Tests for the screener.in loader (offline, against an HTML fixture)."""
from __future__ import annotations

import pytest

from loaders.screener import ScreenerLoader
from loaders import get_loader, available_sources
from utils.schema import CANONICAL_FIELDS
from tests.screener_fixture import screener_html


class FakeScreener(ScreenerLoader):
    def fetch_raw(self):
        return screener_html()


@pytest.fixture
def df():
    return FakeScreener("RELIANCE").load()


def test_registered():
    assert "screener" in available_sources()
    assert isinstance(get_loader("screener", "RELIANCE"), ScreenerLoader)


def test_schema_contract(df):
    assert list(df.columns) == CANONICAL_FIELDS
    assert list(df.index) == [2024, 2023, 2022]


def test_revenue_scaled_to_inr(df):
    # 9,00,000 crore -> 9e12 INR
    assert df.loc[2024, "revenue"] == pytest.approx(9_00_000 * 1e7)


def test_operating_profit_is_ebitda(df):
    assert df.loc[2024, "ebitda"] == pytest.approx(1_60_000 * 1e7)


def test_ebit_derived(df):
    # EBIT = Operating Profit - Depreciation = 160000 - 50000 = 110000 cr
    assert df.loc[2024, "ebit"] == pytest.approx(1_10_000 * 1e7)


def test_debt_from_borrowings(df):
    assert df.loc[2024, "debt"] == pytest.approx(3_20_000 * 1e7)


def test_total_equity_capital_plus_reserves(df):
    # 6,760 + 7,43,240 = 7,50,000 cr
    assert df.loc[2024, "total_equity"] == pytest.approx(7_50_000 * 1e7)


def test_shares_from_marketcap_over_price(df):
    # Market Cap 19,60,400 cr / price 2,900 = 6.76e9 shares
    assert df.loc[2024, "shares_outstanding"] == pytest.approx(19_60_400 * 1e7 / 2900,
                                                               rel=1e-4)


def test_unavailable_fields_are_nan(df):
    # screener's condensed BS doesn't break these out
    for col in ("cash", "receivables", "inventory", "capex", "working_capital"):
        assert df[col].isna().all()


def test_bad_html_raises():
    class Empty(ScreenerLoader):
        def fetch_raw(self):
            return "<html><body>no tables here</body></html>"
    with pytest.raises(Exception):
        Empty("X").load()
