"""
Phase 8 — Sensitivity analysis.

A DCF's output is dominated by two assumptions: the discount rate (WACC) and the
perpetuity growth rate. This builds the classic valuation grid — intrinsic price
for every (WACC, terminal-growth) pair — so you can see at a glance how fragile
or robust the thesis is. Cells where WACC <= g are invalid (NaN).

The forecast (and thus the FCFF stream) is held fixed; only the discounting and
terminal value change. That keeps the grid a clean function of the two axes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from valuation.dcf import run_dcf
from utils.logging import get_logger

log = get_logger("valuation.sensitivity")


def grid_axes(base_wacc: float, base_growth: float,
              wacc_step: float = 0.01, growth_step: float = 0.005,
              n: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric axes centred on a base case: 2n+1 points each."""
    waccs = base_wacc + wacc_step * np.arange(-n, n + 1)
    growths = base_growth + growth_step * np.arange(-n, n + 1)
    return np.round(waccs, 4), np.round(growths, 4)


def wacc_growth_grid(
    forecast_df: pd.DataFrame,
    wacc_values,
    growth_values,
    *,
    net_debt: float,
    shares_outstanding: float,
    mid_year: bool = False,
) -> pd.DataFrame:
    """Intrinsic price for each (terminal growth × WACC) pair.

    Rows = terminal growth, columns = WACC (matches the roadmap layout).
    """
    rows = {}
    for g in growth_values:
        row = {}
        for w in wacc_values:
            if w <= g:
                row[w] = np.nan
                continue
            res = run_dcf(forecast_df, float(w), net_debt=net_debt,
                          shares_outstanding=shares_outstanding,
                          terminal_growth=float(g), mid_year=mid_year)
            row[w] = res.intrinsic_price
        rows[g] = row
    grid = pd.DataFrame(rows).T
    grid.index.name = "terminal_growth"
    grid.columns.name = "wacc"
    return grid


def save_heatmap(grid: pd.DataFrame, path, *, current_price: float | None = None,
                 title: str = "DCF sensitivity — intrinsic price (₹/share)"):
    """Render the grid as a heatmap PNG (matplotlib, headless)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.4 * len(grid.columns) + 2,
                                    0.8 * len(grid.index) + 2))
    data = grid.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([f"{w:.1%}" for w in grid.columns])
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([f"{g:.1%}" for g in grid.index])
    ax.set_xlabel("WACC")
    ax.set_ylabel("Terminal growth")
    ax.set_title(title)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                txt = f"{v:,.0f}"
                if current_price:
                    txt += "\n" + f"{v/current_price-1:+.0%}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="₹ / share")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info("wrote heatmap %s", path)
    return path
