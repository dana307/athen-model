"""
Phase 13 — Portfolio Optimization.

Classic mean-variance toolkit on an expected-return vector (mu) and a covariance
matrix (Sigma):

    min_variance     — the global minimum-variance portfolio
    max_sharpe       — the tangency portfolio (max excess-return / vol)
    mean_variance    — target a return, or maximise a risk-aversion utility
    risk_parity      — equalise each asset's risk contribution
    black_litterman  — blend market-equilibrium returns with views (stretch)

Plus an efficient_frontier sweep, and helpers to estimate (mu, Sigma) from a
price history. Optimisation uses scipy SLSQP; long-only by default, with an
optional shorting switch. Everything takes/returns named pandas objects so the
weights stay labelled by ticker.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils.logging import get_logger

log = get_logger("portfolio.optimizer")


# --------------------------------------------------------------------------
@dataclass
class OptimizationResult:
    method: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float

    def summary(self) -> str:
        w = ", ".join(f"{k} {v:.0%}" for k, v in self.weights.items())
        return (f"[{self.method}] ret {self.expected_return:.1%} | "
                f"vol {self.volatility:.1%} | Sharpe {self.sharpe:.2f} | {w}")


# --- input plumbing --------------------------------------------------------
def _names(cov):
    if isinstance(cov, pd.DataFrame):
        return list(cov.columns)
    return [f"A{i}" for i in range(np.asarray(cov).shape[0])]


def _arr(x):
    return np.asarray(x, dtype=float)


def portfolio_performance(weights, mu, cov, rf: float = 0.0):
    """Return (expected_return, volatility, sharpe) for given weights."""
    w = _arr(weights)
    mu, cov = _arr(mu), _arr(cov)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def _bounds(n, long_only, max_weight):
    lo = 0.0 if long_only else -1.0
    hi = max_weight if max_weight is not None else 1.0
    return tuple((lo, hi) for _ in range(n))


def _solve(objective, n, *, long_only=True, max_weight=None, extra_constraints=None):
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if extra_constraints:
        cons.extend(extra_constraints)
    x0 = np.full(n, 1.0 / n)
    res = minimize(objective, x0, method="SLSQP",
                   bounds=_bounds(n, long_only, max_weight),
                   constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-10})
    if not res.success:
        log.warning("optimizer did not fully converge: %s", res.message)
    return res.x


def _result(method, w, names, mu, cov, rf):
    weights = pd.Series(w, index=names).round(6)
    ret, vol, sharpe = portfolio_performance(w, mu, cov, rf)
    out = OptimizationResult(method, weights, ret, vol, sharpe)
    log.info("%s", out.summary())
    return out


# --- optimizers ------------------------------------------------------------
def min_variance(cov, *, mu=None, rf=0.0, long_only=True, max_weight=None):
    names = _names(cov)
    C = _arr(cov)
    mu = np.zeros(len(names)) if mu is None else _arr(mu)
    w = _solve(lambda w: w @ C @ w, len(names),
               long_only=long_only, max_weight=max_weight)
    return _result("min_variance", w, names, mu, C, rf)


def max_sharpe(mu, cov, *, rf=0.0, long_only=True, max_weight=None):
    names = _names(cov)
    m, C = _arr(mu), _arr(cov)

    def neg_sharpe(w):
        ret = w @ m
        vol = np.sqrt(w @ C @ w)
        return -(ret - rf) / vol if vol > 0 else 1e6

    w = _solve(neg_sharpe, len(names), long_only=long_only, max_weight=max_weight)
    return _result("max_sharpe", w, names, m, C, rf)


def mean_variance(mu, cov, *, target_return=None, risk_aversion=None,
                  rf=0.0, long_only=True, max_weight=None):
    """Either hit a target return at min variance, or maximise utility
    mu·w - (risk_aversion/2) w'Σw. Defaults to risk_aversion=3."""
    names = _names(cov)
    m, C = _arr(mu), _arr(cov)

    if target_return is not None:
        extra = [{"type": "eq", "fun": lambda w: w @ m - target_return}]
        w = _solve(lambda w: w @ C @ w, len(names), long_only=long_only,
                   max_weight=max_weight, extra_constraints=extra)
        method = f"mean_variance(target={target_return:.1%})"
    else:
        ra = 3.0 if risk_aversion is None else risk_aversion
        w = _solve(lambda w: 0.5 * ra * (w @ C @ w) - w @ m, len(names),
                   long_only=long_only, max_weight=max_weight)
        method = f"mean_variance(ra={ra:g})"
    return _result(method, w, names, m, C, rf)


def risk_parity(cov, *, mu=None, rf=0.0, long_only=True):
    """Equal risk contribution portfolio.

    Risk contribution of asset i: RC_i = w_i (Σw)_i. We minimise the dispersion
    of RC across assets (long-only; weights normalised to 1).
    """
    names = _names(cov)
    C = _arr(cov)
    mu = np.zeros(len(names)) if mu is None else _arr(mu)

    def dispersion(w):
        port_var = w @ C @ w
        rc = w * (C @ w)                     # risk contributions (unscaled)
        target = port_var / len(w)
        return np.sum((rc - target) ** 2)

    # risk parity needs strictly positive weights
    w = _solve(dispersion, len(names), long_only=True, max_weight=None)
    w = np.clip(w, 1e-9, None)
    w = w / w.sum()
    return _result("risk_parity", w, names, mu, C, rf)


def efficient_frontier(mu, cov, *, n_points=25, rf=0.0, long_only=True):
    """Return a DataFrame of (return, volatility, sharpe) along the frontier."""
    m, C = _arr(mu), _arr(cov)
    names = _names(cov)
    lo, hi = float(m.min()), float(m.max())
    rows = []
    for target in np.linspace(lo, hi, n_points):
        try:
            r = mean_variance(pd.Series(m, index=names),
                              pd.DataFrame(C, index=names, columns=names),
                              target_return=float(target), long_only=long_only, rf=rf)
            rows.append({"target_return": target, "return": r.expected_return,
                         "volatility": r.volatility, "sharpe": r.sharpe})
        except Exception:  # pragma: no cover
            continue
    return pd.DataFrame(rows)


def black_litterman(cov, market_weights, *, risk_aversion=2.5, tau=0.05,
                    P=None, Q=None, omega=None, rf=0.0, long_only=True):
    """Black-Litterman posterior returns, then a max-Sharpe portfolio (stretch).

    Implied equilibrium excess returns: pi = risk_aversion · Σ · w_mkt.
    With views (P, Q, Omega), the posterior combines pi and the views.
    Returns the optimisation on the posterior mu.
    """
    names = _names(cov)
    C = _arr(cov)
    w_mkt = _arr(market_weights)
    pi = risk_aversion * C @ w_mkt                # equilibrium excess returns

    if P is not None and Q is not None:
        P, Q = _arr(P), _arr(Q)
        if omega is None:
            omega = np.diag(np.diag(P @ (tau * C) @ P.T))
        omega = _arr(omega)
        tauC = tau * C
        A = np.linalg.inv(tauC) + P.T @ np.linalg.inv(omega) @ P
        b = np.linalg.inv(tauC) @ pi + P.T @ np.linalg.inv(omega) @ Q
        mu_bl = np.linalg.solve(A, b)
    else:
        mu_bl = pi

    mu_series = pd.Series(mu_bl + rf, index=names)
    res = max_sharpe(mu_series, pd.DataFrame(C, index=names, columns=names),
                     rf=rf, long_only=long_only)
    res.method = "black_litterman"
    return res, mu_series


# --- input estimation ------------------------------------------------------
def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple periodic returns from a price history (columns = tickers)."""
    return prices.sort_index().pct_change().dropna(how="all")


def annualized_inputs(returns: pd.DataFrame, periods_per_year: int = 252):
    """Annualised mean-return vector and covariance matrix from returns."""
    mu = returns.mean() * periods_per_year
    cov = returns.cov() * periods_per_year
    return mu, cov


def fetch_price_history(tickers, *, source="yfinance", period="3y",
                        exchange_suffix=".NS") -> pd.DataFrame:  # pragma: no cover
    """Best-effort daily close history via yfinance (network required)."""
    import yfinance as yf
    syms = [t if "." in t else f"{t}{exchange_suffix}" for t in tickers]
    data = yf.download(syms, period=period, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data.columns = [c.replace(exchange_suffix, "") for c in data.columns]
    return data.dropna(how="all")
