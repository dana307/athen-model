"""
Phase 11 — Interactive Streamlit dashboard.

Run with:
    streamlit run dashboard/streamlit_app.py

Loads a ticker, shows the financial history, forecast, key ratios, DCF intrinsic
value, Monte Carlo distribution, sensitivity heatmap and comps — with live
sliders for WACC / growth / margin / terminal growth that re-price the stock
instantly (no re-fetch, thanks to caching).
"""
from __future__ import annotations

import sys
from pathlib import Path

# allow `streamlit run dashboard/streamlit_app.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings
from loaders import get_loader, available_sources
from utils.metrics import add_derived_metrics, historical_drivers
from valuation import (
    value_company, intrinsic_price, simulate, MonteCarloConfig,
    wacc_growth_grid, grid_axes, peer_comparison,
)
from risk import assess, RiskLevel

CR = 1e7
RISK_COLOR = {RiskLevel.LOW: "🟢", RiskLevel.MEDIUM: "🟡", RiskLevel.HIGH: "🔴"}


# --------------------------------------------------------------------------
# Data loading (cached so slider moves don't re-fetch)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_company(ticker: str, source: str):
    loader = get_loader(source, ticker)
    df = loader.load()
    md = loader.market_data()
    return df, md


def money_cr(x):
    return "—" if pd.isna(x) else f"₹{x/CR:,.0f}"


# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Athena", layout="wide", page_icon="📈")
    st.title("📈 Athena — Equity Research & Valuation")

    with st.sidebar:
        st.header("Company")
        ticker = st.text_input("Ticker", value="RELIANCE").strip().upper()
        source = st.selectbox("Source", available_sources(), index=0)
        demo = st.checkbox("Use demo data (no network)", value=False,
                           help="Loads a bundled sample so the app works "
                                "even when live data is unavailable.")
        go_btn = st.button("Load", type="primary")

    if not ticker:
        st.info("Enter a ticker in the sidebar and press Load.")
        return

    from dashboard.sample_data import get_sample
    if demo:
        df, md = get_sample(ticker)
    else:
        try:
            df, md = load_company(ticker, source)
        except Exception as e:
            st.warning(f"Live data for {ticker} is unavailable ({e}). "
                       "Showing bundled demo data instead — tick "
                       "**Use demo data** to silence this, or try another ticker.")
            df, md = get_sample(ticker)

    metrics = add_derived_metrics(df)
    profile = historical_drivers(df)
    name = md.get("name") or ticker
    price = md.get("price")

    # --- assumption sliders ----------------------------------------------
    with st.sidebar:
        st.header("Assumptions")
        price = st.number_input("Market price (₹)", value=float(price or 1000.0))
        beta = st.number_input("Beta", value=float(md.get("beta") or 1.0), step=0.05)
        growth = st.slider("Revenue growth", -0.10, 0.40,
                           float(np.clip(profile.revenue_cagr, -0.10, 0.40)), 0.005)
        margin = st.slider("EBITDA margin", 0.0, 0.60,
                           float(np.clip(profile.avg_ebitda_margin, 0.0, 0.60)), 0.005)
        wacc = st.slider("WACC", 0.06, 0.20, 0.12, 0.005)
        tg = st.slider("Terminal growth", 0.00, 0.08, settings.TERMINAL_GROWTH, 0.0025)
        years = st.slider("Forecast years", 3, 10, settings.FORECAST_YEARS)

    st.subheader(f"{name} ({ticker})")

    # --- headline metrics -------------------------------------------------
    live_price = intrinsic_price(df, growth=growth, margin=margin, wacc=wacc,
                                 terminal_growth=tg, years=years)
    upside = live_price / price - 1 if price else float("nan")
    risk = assess(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market price", f"₹{price:,.0f}")
    c2.metric("Intrinsic (live)", f"₹{live_price:,.0f}", f"{upside:+.1%}")
    c3.metric("Revenue CAGR", f"{profile.revenue_cagr:.1%}")
    c4.metric("Risk", f"{RISK_COLOR[risk.overall_level]} {risk.overall_level}")

    tabs = st.tabs(["Financials", "Forecast", "DCF", "Monte Carlo",
                    "Sensitivity", "Risk"])

    # --- Financials -------------------------------------------------------
    with tabs[0]:
        h = metrics.sort_index()
        fig = go.Figure()
        fig.add_bar(x=[str(y) for y in h.index], y=h["revenue"] / CR, name="Revenue (₹cr)")
        fig.add_trace(go.Scatter(
            x=[str(y) for y in h.index], y=h["ebitda_margin"] * 100,
            name="EBITDA margin %", yaxis="y2", mode="lines+markers"))
        fig.update_layout(
            yaxis=dict(title="Revenue (₹ cr)"),
            yaxis2=dict(title="EBITDA margin %", overlaying="y", side="right"),
            legend=dict(orientation="h"), height=420)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(_display_fundamentals(df), width="stretch")

    # --- Forecast ---------------------------------------------------------
    with tabs[1]:
        val = value_company(df, market_price=price, beta=beta,
                            forecast_config=_cfg(years))
        fc = val.forecast.copy()
        show = fc[["revenue", "ebitda", "ebit", "fcff"]] / CR
        show["revenue_growth %"] = (fc["revenue_growth"] * 100).round(1)
        st.dataframe(show.round(0), width="stretch")
        st.caption("FCFF is the stream discounted by the DCF.")

    # --- DCF --------------------------------------------------------------
    with tabs[2]:
        st.markdown(f"### Intrinsic value: **₹{live_price:,.0f}/share** "
                    f"({upside:+.1%} vs market)")
        st.write(f"Driven by your sliders — growth **{growth:.1%}**, "
                 f"margin **{margin:.1%}**, WACC **{wacc:.1%}**, "
                 f"terminal growth **{tg:.1%}**.")
        val = value_company(df, market_price=price, beta=beta,
                            forecast_config=_cfg(years))
        st.write(f"Model WACC (CAPM): **{val.wacc.wacc:.2%}** "
                 f"(Ke {val.wacc.cost_of_equity:.2%}, "
                 f"E/D {val.wacc.weight_equity:.0%}/{val.wacc.weight_debt:.0%})")

    # --- Monte Carlo ------------------------------------------------------
    with tabs[3]:
        mc = simulate(df, MonteCarloConfig(
            n_sims=10_000, years=years,
            revenue_growth=(growth, 0.02), ebitda_margin=(margin, 0.01),
            wacc=(wacc, 0.01), terminal_growth=(tg, 0.005)),
            current_price=price)
        fig = go.Figure()
        fig.add_histogram(x=mc.prices, nbinsx=50, name="intrinsic value")
        fig.add_vline(x=price, line_color="red",
                      annotation_text=f"market ₹{price:,.0f}")
        fig.add_vline(x=mc.mean, line_dash="dash",
                      annotation_text=f"mean ₹{mc.mean:,.0f}")
        fig.update_layout(height=420, xaxis_title="Intrinsic value (₹/share)")
        st.plotly_chart(fig, width="stretch")
        a, b, c = st.columns(3)
        a.metric("Mean / Median", f"₹{mc.mean:,.0f} / ₹{mc.median:,.0f}")
        b.metric("90% CI", f"₹{mc.p5:,.0f}–₹{mc.p95:,.0f}")
        c.metric("P(undervalued)", f"{mc.prob_undervalued:.0%}")

    # --- Sensitivity ------------------------------------------------------
    with tabs[4]:
        val = value_company(df, market_price=price, beta=beta,
                            forecast_config=_cfg(years))
        waccs, growths = grid_axes(wacc, tg)
        grid = wacc_growth_grid(val.forecast, waccs, growths,
                                net_debt=val.dcf.net_debt,
                                shares_outstanding=val.dcf.shares_outstanding)
        fig = go.Figure(data=go.Heatmap(
            z=grid.values,
            x=[f"{w:.1%}" for w in grid.columns],
            y=[f"{g:.1%}" for g in grid.index],
            colorscale="RdYlGn", text=grid.round(0).values,
            texttemplate="%{text}"))
        fig.update_layout(height=460, xaxis_title="WACC",
                          yaxis_title="Terminal growth")
        st.plotly_chart(fig, width="stretch")

    # --- Risk -------------------------------------------------------------
    with tabs[5]:
        st.markdown(f"### Overall: {RISK_COLOR[risk.overall_level]} "
                    f"**{risk.overall_level}** (score {risk.risk_score:.2f}/3)")
        rows = [{"Check": f.label, "Level": str(f.level), "Note": f.note}
                for f in risk.findings]
        st.dataframe(pd.DataFrame(rows), width="stretch")


def _cfg(years):
    from forecasting import ForecastConfig
    return ForecastConfig(years=years)


def _display_fundamentals(df):
    from utils.schema import FIELD_LABELS
    d = df.sort_index(ascending=False).T / CR
    d.loc["shares_outstanding"] = df.sort_index(ascending=False)["shares_outstanding"]
    d.index = [FIELD_LABELS.get(i, i) for i in d.index]
    return d.round(0)


if __name__ == "__main__":
    main()
