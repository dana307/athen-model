"""Phase 6 tests — comparable company analysis (offline)."""
from __future__ import annotations

import numpy as np
import pytest

from loaders.yfinance_loader import YFinanceLoader
from valuation.multiples import compute_multiples, peer_comparison
from tests import fixtures


def _load(bundle):
    class L(YFinanceLoader):
        def fetch_raw(self):
            return bundle
    return L("X").load()


@pytest.fixture
def target():
    return _load(fixtures.raw_bundle())


def test_multiples_math(target):
    price, shares = 2900.0, 676e7
    cm = compute_multiples("RELIANCE", target, price)

    market_cap = price * shares
    net_debt = 3_20_000e7 - 2_00_000e7
    ev = market_cap + net_debt

    assert cm.market_cap == pytest.approx(market_cap)
    assert cm.enterprise_value == pytest.approx(ev)
    assert cm.ev_sales == pytest.approx(ev / 9_00_000e7)
    assert cm.ev_ebitda == pytest.approx(ev / 1_60_000e7)
    assert cm.pe == pytest.approx(market_cap / 70_000e7)
    assert cm.roe == pytest.approx(70_000e7 / 7_50_000e7)
    assert cm.debt_to_equity == pytest.approx(3_20_000e7 / 7_50_000e7)


def test_roce_formula(target):
    cm = compute_multiples("RELIANCE", target, 2900.0)
    # EBIT / (equity + debt) = 1,10,000 / (7,50,000 + 3,20,000)
    assert cm.roce == pytest.approx(1_10_000e7 / (7_50_000e7 + 3_20_000e7))


def test_peer_comparison_implied_price(target):
    peers = {
        "PEER1": (_load(fixtures.scaled_bundle(0.5)), 2900.0),
        "PEER2": (_load(fixtures.scaled_bundle(2.0, margin_boost=0.1)), 3200.0),
    }
    res = peer_comparison(target, "RELIANCE", 2900.0, peers)

    # table includes target + both peers
    assert set(res.table.index) == {"RELIANCE", "PEER1", "PEER2"}
    # medians computed for each valuation multiple
    for m in ("ev_sales", "ev_ebitda", "pe"):
        assert m in res.peer_medians
        assert not np.isnan(res.peer_medians[m])
    # blended implied price is the mean of the per-multiple implied prices
    vals = [v for v in res.implied_prices.values() if not np.isnan(v)]
    assert res.implied_price == pytest.approx(np.mean(vals))
    assert res.upside == pytest.approx(res.implied_price / 2900.0 - 1)


def test_pe_implied_price_uses_median(target):
    # two peers with identical multiples -> median == that multiple, so the
    # P/E-implied equity = median_pe * target PAT
    peers = {
        "A": (_load(fixtures.raw_bundle()), 2900.0),
        "B": (_load(fixtures.raw_bundle()), 2900.0),
    }
    res = peer_comparison(target, "RELIANCE", 2900.0, peers,
                          multiples=["pe"])
    median_pe = res.peer_medians["pe"]
    implied = median_pe * target.sort_index(ascending=False)["pat"].iloc[0] / 676e7
    assert res.implied_prices["pe"] == pytest.approx(implied)


def test_needs_at_least_one_peer(target):
    with pytest.raises(ValueError):
        peer_comparison(target, "RELIANCE", 2900.0, {})
