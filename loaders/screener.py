"""
screener.in loader — richer Indian fundamentals (the hybrid plan's second source).

Screener publishes condensed Profit & Loss, Balance Sheet and Cash Flow tables
per company at:

    https://www.screener.in/company/<SYMBOL>/consolidated/

This loader fetches that page, parses the financial tables by section id, maps
the rows onto Athena's canonical schema, and converts ₹-crore figures to
absolute INR. It subclasses BaseLoader exactly like the yfinance loader, so the
valuation engine is unchanged — only the data source differs.

Notes / limitations vs yfinance:
    - Screener's BS is condensed: cash, receivables, inventory and capex are not
      broken out, so those canonical fields are left NaN (downstream code already
      tolerates gaps).
    - Shares outstanding is derived from Market Cap / Current Price in the page's
      top-ratios block.
    - Screener tables are in ₹ crore; values are scaled to absolute INR.

Parsing is separated from fetching (fetch_raw returns HTML; normalize parses it)
so the mapping can be unit-tested offline against a saved HTML fixture.
"""
from __future__ import annotations

import re

import pandas as pd

from config import settings
from loaders.base import BaseLoader, LoaderError
from utils.logging import get_logger

log = get_logger("loaders.screener")

CRORE = 1e7
_BASE_URL = "https://www.screener.in/company/{symbol}/"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Athena research platform)"}

# canonical field -> candidate screener row labels (lower-cased, '+'/'%' stripped)
_PL_MAP = {
    "revenue": ["sales", "revenue", "total revenue"],
    "ebitda": ["operating profit"],
    "depreciation": ["depreciation"],
    "pat": ["net profit"],
}
_BS_MAP = {
    "debt": ["borrowings"],
    "_equity_capital": ["equity capital"],
    "_reserves": ["reserves"],
}


def _clean_num(text: str):
    """Parse a screener cell like '1,234', '-', '12.3%' into a float (or NaN)."""
    t = (text or "").strip().replace(",", "").replace("%", "").replace("₹", "").strip()
    if t in ("", "-"):
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _row_label(cell) -> str:
    txt = cell.get_text(" ", strip=True)
    return re.sub(r"[+%]", "", txt).strip().lower()


def _parse_year(header: str):
    m = re.search(r"(\d{4})", header or "")
    return int(m.group(1)) if m else header


class ScreenerLoader(BaseLoader):
    source_name = "screener"

    def __init__(self, ticker: str, consolidated: bool = True):
        super().__init__(ticker)
        self.consolidated = consolidated
        self._html: str | None = None

    # ------------------------------------------------------------------ fetch
    def fetch_raw(self) -> str:
        import requests

        url = _BASE_URL.format(symbol=self.ticker)
        if self.consolidated:
            url = url + "consolidated/"
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=20)
        except Exception as e:  # network errors
            raise LoaderError(f"screener.in request failed for {self.ticker}: {e}") from e
        if resp.status_code == 404:
            raise LoaderError(f"screener.in has no page for '{self.ticker}'.")
        if resp.status_code != 200:
            raise LoaderError(
                f"screener.in returned HTTP {resp.status_code} for {self.ticker}.")
        self._html = resp.text
        return resp.text

    # -------------------------------------------------------------- normalize
    def normalize(self, raw: str) -> pd.DataFrame:
        try:
            from bs4 import BeautifulSoup
        except ImportError as e:  # pragma: no cover
            raise LoaderError("beautifulsoup4 not installed.") from e

        soup = BeautifulSoup(raw, "html.parser")
        pl_years, pl = _parse_section(soup, "profit-loss")
        bs_years, bs = _parse_section(soup, "balance-sheet")
        if not pl:
            raise LoaderError(
                f"Could not parse a Profit & Loss table for {self.ticker} "
                "(screener layout may have changed or the page is empty).")

        # master year index from the P&L section
        years = pl_years
        df = pd.DataFrame(index=years)

        for field, cands in _PL_MAP.items():
            df[field] = _series_for(pl, pl_years, years, cands)
        for field, cands in _BS_MAP.items():
            df[field] = _series_for(bs, bs_years, years, cands)

        # scale ₹ crore -> absolute INR for monetary fields
        money = ["revenue", "ebitda", "depreciation", "pat", "debt",
                 "_equity_capital", "_reserves"]
        for c in money:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce") * CRORE

        # derivations
        df["ebit"] = df["ebitda"] - df["depreciation"].fillna(0)
        df["total_equity"] = df.get("_equity_capital").fillna(0) + df.get("_reserves").fillna(0)
        df["total_equity"] = df["total_equity"].replace(0, pd.NA)

        # shares outstanding from Market Cap / Current Price (top ratios)
        shares = _shares_outstanding(soup)
        if shares:
            df["shares_outstanding"] = shares

        df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
        return df


# --- parsing helpers -------------------------------------------------------
def _parse_section(soup, section_id: str):
    """Return (years, {row_label: {year: value}}) for a screener section."""
    section = soup.find("section", id=section_id) or soup.find(id=section_id)
    if section is None:
        return [], {}
    table = section.find("table")
    if table is None:
        return [], {}

    head = table.find("thead")
    headers = [th.get_text(strip=True) for th in head.find_all("th")] if head else []
    years = [_parse_year(h) for h in headers[1:]]   # first col is the row label

    rows: dict[str, dict] = {}
    body = table.find("tbody") or table
    for tr in body.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        label = _row_label(cells[0])
        vals = [_clean_num(c.get_text(strip=True)) for c in cells[1:]]
        rows[label] = {y: v for y, v in zip(years, vals)}
    return years, rows


def _series_for(rows: dict, src_years, target_years, candidates):
    for name in candidates:
        if name in rows:
            return [rows[name].get(y, float("nan")) for y in target_years]
    return [float("nan")] * len(target_years)


def _shares_outstanding(soup):
    """Derive shares = Market Cap / Current Price from the top-ratios block."""
    mcap = price = None
    for li in soup.select("#top-ratios li, ul#top-ratios li, li"):
        text = li.get_text(" ", strip=True).lower()
        if "market cap" in text:
            mcap = _first_number(li.get_text(" ", strip=True))
        elif "current price" in text:
            price = _first_number(li.get_text(" ", strip=True))
    if mcap and price and price > 0:
        return mcap * CRORE / price          # market cap is in ₹ crore
    return None


def _first_number(text: str):
    m = re.search(r"[-\d][\d,]*\.?\d*", text.replace("₹", " "))
    return _clean_num(m.group(0)) if m else None
