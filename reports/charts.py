"""
Chart generation for reports (matplotlib, headless).

Each function writes a PNG and returns its path, so the DOCX/PDF renderers can
embed them. Kept separate from the renderers so the same images can feed any
output format.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CR = 1e7  # 1 crore


def revenue_forecast_chart(report, path: Path) -> Path:
    """Historical + forecast revenue (bars) with EBITDA-margin line."""
    hist = report.fundamentals.sort_index()
    fc = report.forecast.sort_index()

    years = [str(y) for y in list(hist.index) + list(fc.index)]
    rev = list(hist["revenue"] / CR) + list(fc["revenue"] / CR)
    split = len(hist)

    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    colors = ["#4C78A8"] * split + ["#9ECAE9"] * (len(years) - split)
    ax1.bar(years, rev, color=colors)
    ax1.set_ylabel("Revenue (₹ cr)")
    ax1.set_title(f"{report.ticker} — revenue: actual (dark) vs forecast (light)")

    # margin line on a second axis
    hm = (hist["ebitda"] / hist["revenue"] * 100).tolist()
    fm = (fc["ebitda"] / fc["revenue"] * 100).tolist()
    ax2 = ax1.twinx()
    ax2.plot(years, hm + fm, color="#E45756", marker="o", label="EBITDA margin %")
    ax2.set_ylabel("EBITDA margin (%)")
    ax1.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def montecarlo_hist(report, path: Path) -> Path:
    mc = report.montecarlo
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(mc.prices, bins=50, color="#54A24B", alpha=0.8)
    ax.axvline(mc.mean, color="#000000", ls="--", label=f"mean ₹{mc.mean:,.0f}")
    ax.axvline(report.market_price, color="#E45756", ls="-",
               label=f"market ₹{report.market_price:,.0f}")
    ax.axvline(mc.p5, color="#888888", ls=":", label=f"5th ₹{mc.p5:,.0f}")
    ax.axvline(mc.p95, color="#888888", ls=":", label=f"95th ₹{mc.p95:,.0f}")
    ax.set_xlabel("Intrinsic value (₹/share)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{report.ticker} — Monte Carlo intrinsic value ({mc.n_sims:,} sims)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def scenario_chart(report, path: Path) -> Path:
    names = list(report.scenarios.keys())
    prices = [report.scenarios[n]["price"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, prices, color=["#54A24B", "#4C78A8", "#E45756"])
    ax.axhline(report.market_price, color="#000000", ls="--",
               label=f"market ₹{report.market_price:,.0f}")
    for b, p in zip(bars, prices):
        ax.text(b.get_x() + b.get_width() / 2, p, f"₹{p:,.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Intrinsic value (₹/share)")
    ax.set_title(f"{report.ticker} — bull / base / bear")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def generate_all(report, out_dir: Path) -> dict:
    """Render every chart for a report; returns {name: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = report.ticker
    charts = {
        "revenue": revenue_forecast_chart(report, out_dir / f"{t}_revenue.png"),
        "montecarlo": montecarlo_hist(report, out_dir / f"{t}_mc.png"),
        "scenarios": scenario_chart(report, out_dir / f"{t}_scenarios.png"),
    }
    # sensitivity heatmap (reuse the Phase 8 renderer)
    try:
        from valuation.sensitivity import grid_axes, wacc_growth_grid, save_heatmap
        waccs, growths = grid_axes(report.wacc.wacc, report.dcf.terminal_growth)
        grid = wacc_growth_grid(report.forecast, waccs, growths,
                                net_debt=report.dcf.net_debt,
                                shares_outstanding=report.dcf.shares_outstanding)
        charts["sensitivity"] = save_heatmap(
            grid, out_dir / f"{t}_sensitivity.png",
            current_price=report.market_price)
    except Exception as e:  # pragma: no cover
        log_path = None
    return {k: str(v) for k, v in charts.items()}
