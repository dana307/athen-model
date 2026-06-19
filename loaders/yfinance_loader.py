"""
yfinance loader — Phase 1 default source.

yfinance exposes three annual statements as DataFrames (columns = period-end
dates, rows = line items) plus an `info` dict. The tricky part is that Yahoo's
row labels are inconsistent across companies, so we resolve each canonical
field against a *list of candidate labels* and fall back to derivations
(e.g. EBITDA = EBIT + D&A) when a line item is missing.

Indian tickers use the NSE suffix ".NS" (e.g. RELIANCE.NS).
"""
from __future__ import annotations

import pandas as pd

from config import settings
from loaders.base import BaseLoader, LoaderError
from utils.logging import get_logger

log = get_logger("loaders.yfinance")


# Candidate yfinance row labels for each canonical field, in priority order.
# The first label found in the statement wins.
_INCOME_MAP = {
    "revenue": ["Total Revenue", "Operating Revenue", "Total Revenues"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "ebit": ["EBIT", "Operating Income", "Total Operating Income As Reported"],
    "pat": ["Net Income", "Net Income Common Stockholders",
            "Net Income Continuous Operations"],
}
_BALANCE_MAP = {
    "cash": ["Cash And Cash Equivalents",
             "Cash Cash Equivalents And Short Term Investments"],
    "debt": ["Total Debt"],
    "total_equity": ["Stockholders Equity", "Common Stock Equity",
                     "Total Equity Gross Minority Interest",
                     "Total Stockholder Equity"],
    "working_capital": ["Working Capital"],
    "receivables": ["Accounts Receivable", "Receivables", "Net Receivables",
                    "Gross Accounts Receivable"],
    "inventory": ["Inventory", "Inventories"],
    "shares_outstanding": ["Ordinary Shares Number", "Share Issued"],
    "_current_assets": ["Current Assets", "Total Current Assets"],
    "_current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "_long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "_short_term_debt": ["Current Debt", "Current Debt And Capital Lease Obligation"],
}
_CASHFLOW_MAP = {
    "capex": ["Capital Expenditure", "Purchase Of PPE"],
    "depreciation": ["Depreciation And Amortization",
                     "Depreciation Amortization Depletion",
                     "Depreciation"],
}


def _resolve(stmt: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    """Return the first matching row from a statement, or None."""
    if stmt is None or stmt.empty:
        return None
    for name in candidates:
        if name in stmt.index:
            return stmt.loc[name]
    return None


def _years(stmt: pd.DataFrame) -> list:
    """Map statement columns (period-end dates) to fiscal-year ints."""
    out = []
    for c in stmt.columns:
        try:
            out.append(int(pd.Timestamp(c).year))
        except (ValueError, TypeError):
            out.append(c)
    return out


class YFinanceLoader(BaseLoader):
    source_name = "yfinance"

    def __init__(self, ticker: str, exchange_suffix: str | None = None):
        super().__init__(ticker)
        suffix = (exchange_suffix if exchange_suffix is not None
                  else settings.DEFAULT_EXCHANGE_SUFFIX)
        # don't double-append a suffix if the caller already provided one
        self.yf_symbol = self.ticker if "." in self.ticker else f"{self.ticker}{suffix}"
        self._info: dict = {}

    def market_data(self) -> dict:
        """Current price, beta, and market cap from the cached info dict."""
        info = self._info or {}
        price = (info.get("currentPrice") or info.get("regularMarketPrice")
                 or info.get("previousClose"))
        return {
            "price": float(price) if price else None,
            "beta": float(info["beta"]) if info.get("beta") else None,
            "market_cap": float(info["marketCap"]) if info.get("marketCap") else None,
            "name": info.get("longName") or info.get("shortName"),
        }

    # ------------------------------------------------------------------ fetch
    def fetch_raw(self) -> dict:
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover
            raise LoaderError("yfinance not installed. `pip install yfinance`") from e

        tk = yf.Ticker(self.yf_symbol)
        info = _safe_info(tk)
        self._info = info  # cache for market_data()
        raw = {
            "income": tk.income_stmt,
            "balance": tk.balance_sheet,
            "cashflow": tk.cashflow,
            "info": info,
        }
        if all((v is None or getattr(v, "empty", True))
               for k, v in raw.items() if k != "info"):
            raise LoaderError(
                f"yfinance returned no statements for {self.yf_symbol}. "
                "Likely a bad ticker or blocked network."
            )
        return raw

    # -------------------------------------------------------------- normalize
    def normalize(self, raw: dict) -> pd.DataFrame:
        income = raw["income"]
        balance = raw["balance"]
        cashflow = raw["cashflow"]
        info = raw["info"] or {}

        # use income statement periods as the master year index
        base = income if (income is not None and not income.empty) else balance
        years = _years(base)
        df = pd.DataFrame(index=years)

        # --- direct line items -------------------------------------------
        for field, cands in _INCOME_MAP.items():
            df[field] = _series_to_years(_resolve(income, cands), years)
        for field, cands in _BALANCE_MAP.items():
            df[field] = _series_to_years(_resolve(balance, cands), years)
        for field, cands in _CASHFLOW_MAP.items():
            df[field] = _series_to_years(_resolve(cashflow, cands), years)

        # --- derivations / fallbacks -------------------------------------
        # capex: Yahoo reports it negative (an outflow). Store as positive.
        df["capex"] = df["capex"].abs()

        # EBIT <-> EBITDA reconciliation
        if df["ebitda"].isna().all() and df["ebit"].notna().any():
            df["ebitda"] = df["ebit"] + df["depreciation"].fillna(0)
        if df["ebit"].isna().all() and df["ebitda"].notna().any():
            df["ebit"] = df["ebitda"] - df["depreciation"].fillna(0)

        # total debt fallback from short + long term components
        if df["debt"].isna().all():
            df["debt"] = (df.get("_short_term_debt").fillna(0)
                          + df.get("_long_term_debt").fillna(0))

        # working capital fallback = current assets - current liabilities
        if df["working_capital"].isna().all():
            df["working_capital"] = (df.get("_current_assets")
                                     - df.get("_current_liabilities"))

        # shares outstanding fallback from info dict
        if df["shares_outstanding"].isna().all():
            so = info.get("sharesOutstanding")
            if so:
                df["shares_outstanding"] = float(so)

        # drop helper columns; base loader will coerce to canonical schema
        df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
        return df


def _series_to_years(series: pd.Series | None, years: list) -> list:
    """Align a yfinance row (indexed by date) to our integer-year index."""
    if series is None:
        return [pd.NA] * len(years)
    vals = list(series.values)
    # series order matches statement column order, which we mapped to `years`
    if len(vals) == len(years):
        return vals
    # length mismatch: pad/truncate defensively
    out = vals[: len(years)]
    out += [pd.NA] * (len(years) - len(out))
    return out


def _safe_info(tk) -> dict:
    try:
        return dict(tk.info)
    except Exception as e:  # network/parse issues shouldn't kill the load
        log.warning("could not fetch info dict: %s", repr(e)[:120])
        return {}
