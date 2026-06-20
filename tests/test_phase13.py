"""Phase 13 tests — portfolio optimization (offline, analytic checks).

Most checks use a *diagonal* covariance matrix, where the optimal portfolios
have closed forms:
    min-variance   weights ∝ 1/σ_i²
    max-Sharpe     weights ∝ μ_i/σ_i²   (rf=0, long-only, μ>0)
    risk-parity    weights ∝ 1/σ_i
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio import (
    min_variance, max_sharpe, mean_variance, risk_parity,
    efficient_frontier, black_litterman, portfolio_performance,
    returns_from_prices, annualized_inputs,
)

NAMES = ["A", "B", "C"]


def _diag_cov(sigmas):
    v = np.array(sigmas) ** 2
    return pd.DataFrame(np.diag(v), index=NAMES, columns=NAMES)


def _mu(vals):
    return pd.Series(vals, index=NAMES)


# --- performance math -------------------------------------------------------
def test_portfolio_performance():
    mu = _mu([0.10, 0.20, 0.15])
    cov = _diag_cov([0.10, 0.20, 0.15])
    w = np.array([0.5, 0.3, 0.2])
    ret, vol, sharpe = portfolio_performance(w, mu, cov, rf=0.05)
    assert ret == pytest.approx(0.5*0.10 + 0.3*0.20 + 0.2*0.15)
    assert vol == pytest.approx(np.sqrt(w @ cov.values @ w))
    assert sharpe == pytest.approx((ret - 0.05) / vol)


# --- min variance -----------------------------------------------------------
def test_min_variance_weights_sum_and_nonneg():
    cov = _diag_cov([0.10, 0.20, 0.30])
    r = min_variance(cov)
    assert r.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (r.weights >= -1e-6).all()


def test_min_variance_diagonal_closed_form():
    sig = [0.10, 0.20, 0.40]
    cov = _diag_cov(sig)
    r = min_variance(cov)
    inv = 1 / np.array(sig) ** 2
    expected = inv / inv.sum()
    assert np.allclose(r.weights.values, expected, atol=1e-4)


def test_min_variance_favours_low_vol_asset():
    cov = _diag_cov([0.10, 0.50, 0.50])
    r = min_variance(cov)
    assert r.weights["A"] > r.weights["B"]
    assert r.weights["A"] > r.weights["C"]


# --- max sharpe -------------------------------------------------------------
def test_max_sharpe_diagonal_closed_form():
    mu = _mu([0.10, 0.20, 0.15])
    sig = [0.20, 0.20, 0.20]
    cov = _diag_cov(sig)
    r = max_sharpe(mu, cov, rf=0.0)
    raw = np.array([0.10, 0.20, 0.15]) / np.array(sig) ** 2
    expected = raw / raw.sum()
    assert np.allclose(r.weights.values, expected, atol=1e-3)
    assert r.sharpe > 0


# --- mean variance ----------------------------------------------------------
def test_mean_variance_hits_target():
    mu = _mu([0.08, 0.16, 0.12])
    cov = _diag_cov([0.15, 0.25, 0.20])
    r = mean_variance(mu, cov, target_return=0.13)
    assert r.expected_return == pytest.approx(0.13, abs=1e-4)
    assert r.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_mean_variance_risk_aversion_runs():
    mu = _mu([0.08, 0.16, 0.12])
    cov = _diag_cov([0.15, 0.25, 0.20])
    lo = mean_variance(mu, cov, risk_aversion=10)   # very risk averse
    hi = mean_variance(mu, cov, risk_aversion=1)    # risk seeking
    # lower risk aversion should accept higher volatility for higher return
    assert hi.expected_return >= lo.expected_return - 1e-9


# --- risk parity ------------------------------------------------------------
def test_risk_parity_equal_risk_contributions():
    cov = _diag_cov([0.10, 0.20, 0.30])
    r = risk_parity(cov)
    w = r.weights.values
    C = cov.values
    rc = w * (C @ w)                 # risk contributions
    assert np.allclose(rc, rc.mean(), rtol=0.05)
    # closed form for diagonal cov: w ∝ 1/σ
    inv = 1 / np.array([0.10, 0.20, 0.30])
    assert np.allclose(w, inv / inv.sum(), atol=1e-2)


# --- efficient frontier -----------------------------------------------------
def test_efficient_frontier_is_a_parabola():
    mu = _mu([0.08, 0.16, 0.12])
    cov = _diag_cov([0.15, 0.25, 0.20])
    ef = efficient_frontier(mu, cov, n_points=11).sort_values("return")
    assert len(ef) >= 5
    vols = ef["volatility"].values
    i = int(np.argmin(vols))            # global minimum-variance point
    # classic mean-variance parabola: vol falls to the min, then rises
    assert i not in (0, len(vols) - 1)                       # interior minimum
    assert np.all(np.diff(vols[:i + 1]) <= 1e-6)             # decreasing branch
    assert np.all(np.diff(vols[i:]) >= -1e-6)                # increasing (efficient) branch


# --- black litterman --------------------------------------------------------
def test_black_litterman_equilibrium_no_views():
    cov = _diag_cov([0.15, 0.20, 0.25])
    w_mkt = pd.Series([0.5, 0.3, 0.2], index=NAMES)
    res, mu_bl = black_litterman(cov, w_mkt, risk_aversion=2.5)
    assert len(mu_bl) == 3
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_black_litterman_with_view():
    cov = _diag_cov([0.15, 0.20, 0.25])
    w_mkt = pd.Series([0.4, 0.3, 0.3], index=NAMES)
    # view: A outperforms by 5%
    P = np.array([[1.0, 0.0, 0.0]])
    Q = np.array([0.05])
    res, mu_bl = black_litterman(cov, w_mkt, P=P, Q=Q)
    # the view should lift A's posterior return above its equilibrium level
    _, mu_eq = black_litterman(cov, w_mkt)
    assert mu_bl["A"] > mu_eq["A"]


# --- input estimation -------------------------------------------------------
def test_inputs_from_prices():
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame({
        "A": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 100)),
        "B": 100 * np.cumprod(1 + rng.normal(0.002, 0.02, 100)),
    }, index=idx)
    rets = returns_from_prices(prices)
    mu, cov = annualized_inputs(rets, periods_per_year=252)
    assert list(mu.index) == ["A", "B"]
    assert cov.shape == (2, 2)
    # B is more volatile than A by construction
    assert cov.loc["B", "B"] > cov.loc["A", "A"]
