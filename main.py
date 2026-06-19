"""
Athena — command-line entrypoint.

Phase 1 usage:
    python main.py --ticker RELIANCE
    python main.py --ticker TCS --source yfinance --save-csv

As later phases land, this CLI grows (--valuation, --report, --dashboard).
For now it runs the ingestion pipeline: fetch -> normalize -> persist -> show.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from config import settings
from forecasting import forecast, ForecastConfig, available as fc_available
from loaders import get_loader, available_sources
from loaders.base import BaseLoader, LoaderError
from utils import database
from utils.schema import FIELD_LABELS
from utils.logging import get_logger
from valuation import (
    value_company, peer_comparison,
    simulate, MonteCarloConfig,
    wacc_growth_grid, grid_axes, save_heatmap,
)
from risk import assess as assess_risk
from reports import generate_report

log = get_logger("athena")


def ingest(ticker: str, source: str, save_csv: bool = False):
    """Run the Phase 1 pipeline. Returns (fundamentals, loader)."""
    loader = get_loader(source, ticker)
    df = loader.load()

    database.save_fundamentals(df, ticker=ticker, source=source)

    if save_csv:
        out = settings.PROCESSED_DIR / f"{ticker.upper()}_{source}.csv"
        df.to_csv(out)
        log.info("wrote %s", out)

    return df, loader


def _print_table(df: pd.DataFrame, ticker: str) -> None:
    show = df.rename(columns=FIELD_LABELS).T
    # display in crores for readability (1 crore = 1e7)
    money_rows = [r for r in show.index if r != FIELD_LABELS["shares_outstanding"]]
    disp = show.copy()
    disp.loc[money_rows] = (disp.loc[money_rows] / 1e7).round(1)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    print(f"\n=== {ticker.upper()} — fundamentals (INR crore; shares in absolute) ===")
    print(disp.to_string())
    print()


def _print_forecast(proj, ticker: str) -> None:
    disp = proj.copy()
    money = ["revenue", "ebitda", "depreciation", "ebit", "capex",
             "working_capital", "change_in_wc", "nopat", "fcff"]
    disp[money] = (disp[money] / 1e7).round(0)
    pct = ["revenue_growth", "ebitda_margin"]
    disp[pct] = (disp[pct] * 100).round(1)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(f"\n=== {ticker.upper()} — forecast (INR crore; growth/margin in %) ===")
    print(disp.T.to_string())
    print()


def _print_valuation(res, ticker: str) -> None:
    w, d = res.wacc, res.dcf
    print(f"\n=== {ticker.upper()} — DCF valuation ===")
    print("Cost of capital")
    print(f"  Cost of equity (CAPM) : {w.cost_of_equity:7.2%}  "
          f"(Rf {w.risk_free_rate:.2%} + β {w.beta:.2f} × ERP {w.equity_risk_premium:.2%})")
    print(f"  Cost of debt (after-tax): {w.cost_of_debt_aftertax:6.2%}")
    print(f"  Weights               : equity {w.weight_equity:.0%} / debt {w.weight_debt:.0%}")
    print(f"  WACC                  : {w.wacc:7.2%}")
    print("Intrinsic value")
    print(f"  PV explicit FCFF      : {d.pv_explicit/1e7:12,.0f} cr")
    print(f"  PV terminal value     : {d.pv_terminal/1e7:12,.0f} cr  "
          f"({d.pv_terminal/d.enterprise_value:.0%} of EV)")
    print(f"  Enterprise value      : {d.enterprise_value/1e7:12,.0f} cr")
    print(f"  (-) Net debt          : {d.net_debt/1e7:12,.0f} cr")
    print(f"  Equity value          : {d.equity_value/1e7:12,.0f} cr")
    print(f"  Intrinsic price       : ₹{d.intrinsic_price:,.0f} / share")
    if d.current_price is not None:
        verdict = "UNDERVALUED" if d.upside > 0 else "OVERVALUED"
        print(f"  Market price          : ₹{d.current_price:,.0f} / share")
        print(f"  Upside / (downside)   : {d.upside:+.1%}   → {verdict}")
    print()


def _print_comps(res, ticker: str) -> None:
    disp = res.table.copy()
    # ratios as readable values; ROE/ROCE/D-E as %/x
    for col in ("ev_sales", "ev_ebitda", "pe", "peg", "debt_to_equity"):
        if col in disp:
            disp[col] = disp[col].round(2)
    for col in ("roe", "roce"):
        if col in disp:
            disp[col] = (disp[col] * 100).round(1)
    pd.set_option("display.width", 160)
    print(f"\n=== {ticker.upper()} — comparable companies "
          f"(ROE/ROCE in %, rest as x) ===")
    print(disp.to_string())
    print("\nImplied value from median peer multiples")
    for m, v in res.implied_prices.items():
        med = res.peer_medians[m]
        if v == v:  # not NaN
            print(f"  {m:10s} (median {med:.2f}x) → ₹{v:,.0f} / share")
    print(f"  {'blended':10s}            → ₹{res.implied_price:,.0f} / share")
    verdict = "UNDERVALUED" if res.upside > 0 else "OVERVALUED"
    print(f"  market price             ₹{res.current_price:,.0f} / share")
    print(f"  upside / (downside)      {res.upside:+.1%}   → {verdict}")
    print()


def _print_montecarlo(mc, ticker: str) -> None:
    a = mc.assumptions
    print(f"\n=== {ticker.upper()} — Monte Carlo ({mc.n_sims:,} sims) ===")
    print("Assumptions (mean ± σ)")
    print(f"  revenue growth : {a['revenue_growth'][0]:.1%} ± {a['revenue_growth'][1]:.1%}")
    print(f"  EBITDA margin  : {a['ebitda_margin'][0]:.1%} ± {a['ebitda_margin'][1]:.1%}")
    print(f"  WACC           : {a['wacc'][0]:.1%} ± {a['wacc'][1]:.1%}")
    print(f"  terminal growth: {a['terminal_growth'][0]:.1%} ± {a['terminal_growth'][1]:.1%}")
    print("Intrinsic value distribution (₹/share)")
    print(f"  mean / median  : ₹{mc.mean:,.0f} / ₹{mc.median:,.0f}")
    print(f"  std deviation  : ₹{mc.std:,.0f}")
    print(f"  5th–95th pctile: ₹{mc.p5:,.0f} – ₹{mc.p95:,.0f}")
    if mc.current_price is not None:
        print(f"  market price   : ₹{mc.current_price:,.0f}")
        print(f"  P(undervalued) : {mc.prob_undervalued:.0%}")
    print()


def _print_sensitivity(grid, ticker: str, price: float) -> None:
    disp = grid.copy()
    disp.index = [f"{g:.1%}" for g in disp.index]
    disp.columns = [f"{w:.1%}" for w in grid.columns]
    pd.set_option("display.width", 160)
    print(f"\n=== {ticker.upper()} — sensitivity: intrinsic ₹/share "
          f"(rows=terminal growth, cols=WACC) ===")
    print(disp.round(0).to_string())
    print(f"(market price ₹{price:,.0f}/share for reference)")
    print()


def _print_risk(report, ticker: str) -> None:
    print(f"\n=== {ticker.upper()} — risk assessment ===")
    print(f"{'Check':<26}{'Level':<9}Note")
    print("-" * 90)
    for f in report.findings:
        print(f"{f.label:<26}{str(f.level):<9}{f.note}")
    if report.not_assessed:
        print(f"\nNot assessed (missing data): {', '.join(report.not_assessed)}")
    print(f"\nOverall risk: {report.overall_level}  "
          f"(score {report.risk_score:.2f} / 3.00)")
    print()


def _load_peer(ticker: str, source: str):
    """Load a peer's fundamentals + market price for comps."""
    loader = get_loader(source, ticker)
    df = loader.load()
    price = loader.market_data().get("price")
    return df, price


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="athena",
        description="Athena — automated equity research & valuation platform.",
    )
    p.add_argument("--ticker", required=True, help="e.g. RELIANCE, TCS, INFY")
    p.add_argument(
        "--source",
        default=settings.DEFAULT_SOURCE,
        choices=available_sources(),
        help=f"data source (default: {settings.DEFAULT_SOURCE})",
    )
    p.add_argument("--save-csv", action="store_true",
                   help="also write the normalized table to data/processed/")
    p.add_argument("--forecast", action="store_true",
                   help="run the Phase 3 forecasting engine")
    p.add_argument("--years", type=int, default=settings.FORECAST_YEARS,
                   help=f"forecast horizon (default: {settings.FORECAST_YEARS})")
    p.add_argument("--revenue-method", default="fade_growth",
                   choices=fc_available("revenue"),
                   help="revenue forecasting method")
    p.add_argument("--margin-method", default="historical_average",
                   choices=fc_available("margin"),
                   help="EBITDA-margin forecasting method")
    # Phase 4-5 valuation
    p.add_argument("--valuation", action="store_true",
                   help="run the DCF valuation (Phases 4-5)")
    p.add_argument("--price", type=float, default=None,
                   help="current market price/share (auto-fetched if omitted)")
    p.add_argument("--beta", type=float, default=None,
                   help="equity beta (auto-fetched if omitted, else 1.0)")
    p.add_argument("--rf", type=float, default=None, help="risk-free rate, e.g. 0.07")
    p.add_argument("--erp", type=float, default=None, help="equity risk premium, e.g. 0.06")
    p.add_argument("--cost-of-debt", type=float, default=None, help="pre-tax cost of debt")
    p.add_argument("--terminal-growth", type=float, default=None,
                   help=f"perpetuity growth (default {settings.TERMINAL_GROWTH})")
    p.add_argument("--mid-year", action="store_true",
                   help="use the mid-year discounting convention")
    # Phase 6 comparables
    p.add_argument("--comps", action="store_true",
                   help="run comparable company analysis (needs --peers)")
    p.add_argument("--peers", default="",
                   help="comma-separated peer tickers, e.g. ONGC,IOC,BPCL")
    # Phase 7 Monte Carlo
    p.add_argument("--montecarlo", action="store_true",
                   help="run Monte Carlo simulation of intrinsic value")
    p.add_argument("--sims", type=int, default=10_000,
                   help="number of Monte Carlo simulations (default 10,000)")
    # Phase 8 sensitivity
    p.add_argument("--sensitivity", action="store_true",
                   help="WACC × terminal-growth sensitivity grid")
    p.add_argument("--heatmap", action="store_true",
                   help="also save the sensitivity grid as a heatmap PNG")
    # Phase 9 risk
    p.add_argument("--risk", action="store_true",
                   help="run the financial risk assessment")
    # Phase 10 report
    p.add_argument("--report", action="store_true",
                   help="generate the full DOCX/PDF research report")
    p.add_argument("--formats", default="docx,pdf",
                   help="report formats, comma-separated (docx,pdf)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        df, loader = ingest(args.ticker, args.source, save_csv=args.save_csv)
    except LoaderError as e:
        log.error("ingestion failed: %s", e)
        return 1
    _print_table(df, args.ticker)
    log.info("Phase 1 complete for %s.", args.ticker.upper())

    cfg = ForecastConfig(
        years=args.years,
        revenue_method=args.revenue_method,
        margin_method=args.margin_method,
    )

    if args.forecast and not args.valuation:
        proj = forecast(df, cfg)
        _print_forecast(proj, args.ticker)
        log.info("Phase 3 forecast complete for %s.", args.ticker.upper())

    # base valuation is shared by --valuation, --montecarlo and --sensitivity
    need_val = args.valuation or args.montecarlo or args.sensitivity
    if need_val:
        md = loader.market_data()
        price = args.price if args.price is not None else md.get("price")
        beta = args.beta if args.beta is not None else (md.get("beta") or 1.0)
        if not price:
            log.error("no market price available — pass --price <value>.")
            return 1
        res = value_company(
            df, market_price=price, beta=beta,
            forecast_config=cfg,
            risk_free_rate=args.rf, equity_risk_premium=args.erp,
            cost_of_debt=args.cost_of_debt, terminal_growth=args.terminal_growth,
            mid_year=args.mid_year,
        )
        if args.valuation:
            if args.forecast:
                _print_forecast(res.forecast, args.ticker)
            _print_valuation(res, args.ticker)
            log.info("Phase 4-5 valuation complete for %s.", args.ticker.upper())

        if args.montecarlo:
            mc = simulate(
                df,
                MonteCarloConfig(n_sims=args.sims, years=args.years,
                                 wacc=(res.wacc.wacc, 0.01),
                                 terminal_growth=(res.dcf.terminal_growth, 0.005)),
                current_price=price,
            )
            _print_montecarlo(mc, args.ticker)
            log.info("Phase 7 Monte Carlo complete for %s.", args.ticker.upper())

        if args.sensitivity:
            waccs, growths = grid_axes(res.wacc.wacc, res.dcf.terminal_growth)
            grid = wacc_growth_grid(
                res.forecast, waccs, growths,
                net_debt=res.dcf.net_debt,
                shares_outstanding=res.dcf.shares_outstanding,
                mid_year=args.mid_year,
            )
            _print_sensitivity(grid, args.ticker, price)
            if args.heatmap:
                out = settings.OUTPUT_DIR / f"{args.ticker.upper()}_sensitivity.png"
                save_heatmap(grid, out, current_price=price)
                log.info("heatmap saved: %s", out)
            log.info("Phase 8 sensitivity complete for %s.", args.ticker.upper())

    if args.risk:
        report = assess_risk(df)
        _print_risk(report, args.ticker)
        log.info("Phase 9 risk assessment complete for %s.", args.ticker.upper())

    if args.report:
        md = loader.market_data()
        price = args.price if args.price is not None else md.get("price")
        beta = args.beta if args.beta is not None else (md.get("beta") or 1.0)
        if not price:
            log.error("no market price available — pass --price <value>.")
            return 1
        report_peers = {}
        for pt in [t.strip() for t in args.peers.split(",") if t.strip()]:
            try:
                pf, pp = _load_peer(pt, args.source)
                if pp:
                    report_peers[pt] = (pf, pp)
            except LoaderError:
                pass
        written = generate_report(
            df, ticker=args.ticker, market_price=price, beta=beta,
            risk_free_rate=args.rf, equity_risk_premium=args.erp,
            terminal_growth=args.terminal_growth,
            company_name=md.get("name"),
            peers=report_peers or None,
            formats=tuple(f.strip() for f in args.formats.split(",") if f.strip()),
        )
        for fmt, pth in written.items():
            print(f"  {fmt.upper()} report → {pth}")
        log.info("Phase 10 report complete for %s.", args.ticker.upper())

    if args.comps:
        peer_tickers = [t.strip() for t in args.peers.split(",") if t.strip()]
        if not peer_tickers:
            log.error("--comps needs --peers TICKER1,TICKER2,...")
            return 1
        md = loader.market_data()
        price = args.price if args.price is not None else md.get("price")
        if not price:
            log.error("no market price for %s — pass --price.", args.ticker)
            return 1
        peers = {}
        for pt in peer_tickers:
            try:
                pf, pp = _load_peer(pt, args.source)
                if pp is None:
                    log.warning("no price for peer %s — skipping", pt)
                    continue
                peers[pt] = (pf, pp)
            except LoaderError as e:
                log.warning("could not load peer %s: %s", pt, e)
        if not peers:
            log.error("no usable peers loaded.")
            return 1
        res = peer_comparison(df, args.ticker, price, peers)
        _print_comps(res, args.ticker)
        log.info("Phase 6 comparables complete for %s.", args.ticker.upper())

    return 0


if __name__ == "__main__":
    sys.exit(main())
