"""Phase 15 tests — actuarial layer (offline)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from actuarial import (
    merton_pd, pd_from_leverage, stochastic_dcf, simulate_vasicek,
    normal_update, scenario_weighted_value,
)


# --- Merton credit model ----------------------------------------------------
def test_merton_low_leverage_low_pd():
    # huge equity vs tiny debt -> default almost impossible
    r = merton_pd(equity_value=1000, equity_vol=0.3, debt_face=50)
    assert r.default_probability < 0.01
    assert r.distance_to_default > 2


def test_merton_pd_monotonic_in_debt():
    pds = [merton_pd(equity_value=1000, equity_vol=0.35, debt_face=D).default_probability
           for D in (200, 600, 1200, 2000)]
    # more debt -> higher default probability
    assert all(pds[i] <= pds[i + 1] + 1e-9 for i in range(len(pds) - 1))
    assert pds[-1] > pds[0]


def test_merton_pd_monotonic_in_vol():
    lo = merton_pd(1000, 0.20, 800).default_probability
    hi = merton_pd(1000, 0.60, 800).default_probability
    assert hi > lo            # more equity vol -> more default risk


def test_merton_recovers_asset_value():
    r = merton_pd(1000, 0.3, 500, risk_free_rate=0.07)
    # asset value must exceed equity value (assets = equity + risky debt)
    assert r.asset_value > r.equity_value
    assert 0 < r.asset_vol < 0.3      # asset vol below equity vol (leverage)


def test_pd_from_leverage_monotone():
    assert pd_from_leverage(0, 100) < pd_from_leverage(500, 100)


# --- Vasicek stochastic rates ----------------------------------------------
def test_vasicek_mean_reversion():
    # start well above theta; long-run average should pull toward theta
    paths = simulate_vasicek(0.15, kappa=0.8, theta=0.06, sigma=0.005,
                             years=30, n_paths=2000, seed=1)
    assert paths.shape == (2000, 30)
    assert paths[:, -1].mean() < paths[:, 0].mean()      # reverted downward
    assert abs(paths[:, -1].mean() - 0.06) < 0.02         # near theta


def test_stochastic_dcf_distribution():
    fcff = np.array([100.0] * 5)
    res = stochastic_dcf(fcff, net_debt=50, shares_outstanding=10,
                         n_paths=5000, seed=2, current_price=80)
    assert len(res.prices) == 5000
    assert res.p5 <= res.median <= res.p95
    assert 0 <= res.prob_undervalued <= 1


def test_higher_rates_lower_value():
    fcff = np.array([100.0] * 5)
    lo = stochastic_dcf(fcff, net_debt=0, shares_outstanding=10,
                        theta=0.05, r0=0.05, sigma=1e-6, seed=3)
    hi = stochastic_dcf(fcff, net_debt=0, shares_outstanding=10,
                        theta=0.10, r0=0.10, sigma=1e-6, seed=3)
    assert hi.mean < lo.mean


# --- Bayesian updating ------------------------------------------------------
def test_posterior_between_prior_and_obs():
    post = normal_update(prior_mean=0.10, prior_std=0.05,
                         observations=[0.20], obs_std=0.05)
    assert 0.10 < post.mean < 0.20          # shrinks toward the observation
    assert post.std < 0.05                   # uncertainty reduced


def test_more_obs_tightens_posterior():
    one = normal_update(0.10, 0.05, [0.20], 0.05)
    many = normal_update(0.10, 0.05, [0.20] * 10, 0.05)
    assert many.std < one.std                # more data -> tighter
    assert many.mean > one.mean              # pulled closer to 0.20


def test_posterior_mean_formula():
    # equal prior/obs precision, single obs -> posterior mean is the average
    post = normal_update(0.10, 0.05, [0.20], 0.05)
    assert post.mean == pytest.approx(0.15, abs=1e-9)


# --- scenario-weighted valuation -------------------------------------------
class _FakeStress:
    base_intrinsic = 100.0
    market_price = 90.0
    scenarios = {
        "bull": {"intrinsic": 150.0},
        "bear": {"intrinsic": 50.0},
    }


def test_scenario_weighted_expected_value():
    sr = _FakeStress()
    wv = scenario_weighted_value(sr, {"base": 0.5, "bull": 0.25, "bear": 0.25})
    # E = 0.5*100 + 0.25*150 + 0.25*50 = 100
    assert wv.expected_value == pytest.approx(100.0)
    assert abs(sum(wv.probabilities.values()) - 1.0) < 1e-9


def test_scenario_weighted_prob_undervalued():
    sr = _FakeStress()
    # base (100) and bull (150) exceed market 90 -> 0.75 of weight
    wv = scenario_weighted_value(sr, {"base": 0.5, "bull": 0.25, "bear": 0.25})
    assert wv.prob_undervalued == pytest.approx(0.75)


def test_scenario_weights_normalised():
    sr = _FakeStress()
    wv = scenario_weighted_value(sr, {"base": 2, "bull": 1, "bear": 1})  # sum 4
    assert wv.probabilities["base"] == pytest.approx(0.5)
