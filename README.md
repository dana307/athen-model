# Athena

[![CI](https://github.com/dana307/athen-model/actions/workflows/ci.yml/badge.svg)](https://github.com/dana307/athen-model/actions/workflows/ci.yml)

**An institutional-grade equity research & valuation platform for Indian equities.**

Athena turns a single ticker into a full research workflow — financial data
ingestion, statement normalization, forecasting, DCF & comparable valuation,
Monte Carlo simulation, risk scoring, and automated reports — from one command:

```bash
python main.py --ticker RELIANCE
```

> Status: **All 15 phases complete** — from data ingestion through the actuarial stretch layer. 121 passing tests, CI on every push.

---

## Motivation

Most retail investors value companies in ad-hoc Excel sheets; analysts rely on
Bloomberg and closed internal tools. Athena bridges that gap with an open,
reproducible, transparent pipeline that could plausibly be used by an analyst,
fund manager, or risk team.

## Architecture

```
Ticker → Loader → Canonical Schema → SQLite → [Forecast → Valuation → Risk] → Reports / Dashboard
```

Every data source implements a common `BaseLoader` contract and emits the same
canonical schema (`utils/schema.py`), so the valuation engine never depends on
where the numbers came from. Sources are swappable by name:

```python
from loaders import get_loader
df = get_loader("yfinance", "RELIANCE").load()   # → canonical fundamentals
```

| Layer        | Module                | Responsibility                                  |
|--------------|-----------------------|-------------------------------------------------|
| Config       | `config/settings.py`  | Paths, default source, valuation defaults       |
| Schema       | `utils/schema.py`     | Canonical field contract (single source of truth)|
| Loaders      | `loaders/`            | Source → canonical DataFrame (`yfinance`, `screener`) |
| Persistence  | `utils/database.py`   | Tidy SQLite store with upsert + pivot-back       |
| Forecasting  | `forecasting/`        | Pluggable driver methods → projected FCFF        |
| Valuation    | `valuation/`          | WACC, DCF, comparables, Monte Carlo, sensitivity |
| Risk         | `risk/`               | Distress-signal checks → Low/Med/High rating     |
| Reports      | `reports/`            | Builder + DOCX/PDF renderers + charts            |
| Portfolio    | `portfolio/`          | Multi-holding analytics + mean-variance optimizer |
| Macro        | `macro/`              | Scenario shocks propagated through the valuation |
| Actuarial    | `actuarial/`          | Merton PD, Vasicek rates, Bayesian updating      |
| Dashboard    | `dashboard/`          | Streamlit app with live assumption sliders       |
| CLI          | `main.py`             | Orchestrates the pipeline                        |

## Data sources

- **yfinance** (default) — free, reliable, uses NSE tickers (`RELIANCE.NS`).
- **screener.in** — richer Indian fundamentals with long history; parses the
  P&L / Balance Sheet tables and maps them onto the canonical schema.

The hybrid strategy delivered: the entire pipeline was built on yfinance, and
screener.in dropped in as a swappable `BaseLoader` without touching anything
downstream. Pick a source by name:

```bash
python main.py --ticker RELIANCE --source yfinance
python main.py --ticker RELIANCE --source screener
```

## Canonical fields

`revenue, ebitda, ebit, pat, depreciation, capex, working_capital, receivables,
inventory, cash, debt, total_equity, shares_outstanding` — all in absolute INR,
indexed by fiscal year (most recent first).

## Quickstart

```bash
pip install -r requirements.txt
python main.py --ticker RELIANCE                    # ingest + persist + print
python main.py --ticker TCS --save-csv              # also write data/processed/TCS_yfinance.csv
python main.py --ticker RELIANCE --forecast         # + 5-yr projected operating model
python main.py --ticker INFY --forecast --years 7 \
    --revenue-method fade_growth --margin-method regression
pytest -q                                           # run the offline test suite
```

### Forecasting (Phase 3)

Each driver is a family of swappable methods (registry pattern):

| Driver        | Methods                                              |
|---------------|------------------------------------------------------|
| Revenue       | `cagr`, `constant_growth`, `fade_growth`, `linear`, `manual` |
| EBITDA margin | `constant`, `latest`, `historical_average`, `regression` |
| Capex / D&A / WC | `percent_of_sales`, `user_defined`                |

```python
from forecasting import forecast, ForecastConfig
proj = forecast(fundamentals, ForecastConfig(
    years=5,
    revenue_method="fade_growth",      # CAGR fading to terminal growth
    margin_method="historical_average",
))
# proj columns: revenue, revenue_growth, ebitda_margin, ebitda, depreciation,
#               ebit, capex, working_capital, change_in_wc, nopat, fcff
```

The engine assembles `revenue → EBITDA → EBIT → NOPAT → FCFF` — the FCFF stream
is exactly what the Phase 5 DCF discounts.

### Valuation (Phases 4–5)

```bash
python main.py --ticker RELIANCE --valuation        # auto-fetches price & beta
python main.py --ticker TCS --valuation \
    --price 3850 --beta 0.9 --rf 0.07 --erp 0.06 --terminal-growth 0.05
```

```python
from valuation import value_company
res = value_company(fundamentals, market_price=2900, beta=1.1,
                    risk_free_rate=0.07, equity_risk_premium=0.06)
res.wacc.summary()           # CAPM + WACC breakdown
res.dcf.summary()            # EV → equity → intrinsic price → upside
```

**Cost of capital (Phase 4):** `Ke = Rf + β·ERP` (CAPM), after-tax `Kd`, and a
market-value-weighted `WACC`. Every intermediate is exposed in `WaccResult` for
auditability; defaults come from `config/settings.py`; everything is overridable.

**DCF (Phase 5):** discounts the forecast FCFF, adds a Gordon-growth terminal
value, and bridges `EV → equity value (− net debt) → intrinsic price/share →
upside vs market`. Optional mid-year convention. Guards enforce `WACC > g` and
`shares > 0`.

### Comparable companies (Phase 6)

The market-based cross-check on the DCF. Computes `EV/Sales, EV/EBITDA, P/E,
PEG, ROE, ROCE, Debt/Equity` for a target and its peers, takes the **median
peer multiple**, and backs out an implied per-share value (per multiple +
blended).

```bash
python main.py --ticker RELIANCE --comps --peers ONGC,IOC,BPCL
```

```python
from valuation import peer_comparison
res = peer_comparison(target_fundamentals, "RELIANCE", target_price=2900,
                      peers={"ONGC": (ongc_df, 270), "IOC": (ioc_df, 165)})
res.table            # every company's multiples
res.implied_price    # blended implied value from median peer multiples
```

### Monte Carlo & sensitivity (Phases 7–8)

```bash
python main.py --ticker RELIANCE --montecarlo --sims 10000
python main.py --ticker RELIANCE --sensitivity --heatmap   # writes output/<T>_sensitivity.png
```

**Monte Carlo (Phase 7):** replaces the DCF's point assumptions with normal
distributions (revenue growth, EBITDA margin, WACC, terminal growth) and runs
N simulations — fully vectorized in NumPy, so 10,000 paths cost milliseconds.
Returns the full intrinsic-value distribution (mean, median, σ, 5th/95th
percentile) and the probability the stock is undervalued. A zero-variance run
exactly reproduces the deterministic DCF (asserted in tests).

**Sensitivity (Phase 8):** the classic WACC × terminal-growth grid of intrinsic
prices (invalid where WACC ≤ g), with an optional RdYlGn heatmap PNG.

```python
from valuation import simulate, MonteCarloConfig, wacc_growth_grid, grid_axes
mc = simulate(fundamentals, MonteCarloConfig(n_sims=10000), current_price=2900)
waccs, growths = grid_axes(base_wacc=0.12, base_growth=0.045)
grid = wacc_growth_grid(forecast_df, waccs, growths, net_debt=nd, shares_outstanding=sh)
```

### Risk engine (Phase 9)

```bash
python main.py --ticker RELIANCE --risk
```

Reads the metrics history and flags six distress signals — **excess leverage,
falling margins, negative free cash flow, rising receivables, inventory buildup,
debt spikes** — each as a Low / Medium / High finding with a plain-English note,
then rolls them up to an overall rating (weakest-link). Each check is a
registered function, so adding a red flag is one decorated function.

```python
from risk import assess
report = assess(fundamentals)
report.overall_level          # LOW / MEDIUM / HIGH
for f in report.findings:
    print(f.label, f.level, f.note)
```

### Automated research report (Phase 10)

One command produces an institutional-style equity research note as **DOCX and
PDF**, with embedded charts: title + recommendation banner, investment thesis,
financial summary, forecast, DCF, comparables, Monte Carlo, bull/base/bear
scenarios, key risks, and a blended fair value with a BUY / HOLD / REDUCE call.

```bash
python main.py --ticker RELIANCE --report --peers TCS,INFY,HCLTECH
# writes output/RELIANCE_Athena.docx and .pdf
```

```python
from reports import generate_report
paths = generate_report(fundamentals, ticker="RELIANCE", market_price=2900,
                        beta=1.1, peers=peers, formats=("docx", "pdf"))
```

The builder (`reports/builder.py`) owns the analysis; the renderers
(`reports/docx_report.py`, `reports/pdf_report.py`) own the formatting, so one
report object drives both outputs **and** the dashboard.

### Interactive dashboard (Phase 11)

```bash
streamlit run dashboard/streamlit_app.py
```

A Streamlit app with **live assumption sliders** (WACC, revenue growth, EBITDA
margin, terminal growth, horizon) that re-price the stock instantly, across six
tabs: Financials, Forecast, DCF, Monte Carlo, Sensitivity heatmap, and Risk.
Data loads are cached, so moving a slider recomputes the valuation without
re-fetching.

### Portfolio analytics (Phase 12)

Aggregates multiple holdings into portfolio-level intrinsic value, sector
allocation, concentration risk (HHI / effective number of holdings / top-N
weight), weighted expected return, and weighted valuation gap.

```bash
python main.py --ticker RELIANCE --portfolio \
    --holdings RELIANCE:100:Energy,TCS:50:IT,INFY:80:IT,HDFCBANK:60:Financials
```

```python
from portfolio import Portfolio, build_holding
p = Portfolio([
    build_holding("RELIANCE", rel_df, market_price=2900, quantity=100, sector="Energy"),
    build_holding("TCS", tcs_df, market_price=3850, quantity=50, sector="IT"),
])
a = p.analyze()
a.portfolio_upside, a.hhi, a.sector_allocation
```

### Portfolio optimization (Phase 13)

Mean-variance toolkit on an expected-return vector and covariance matrix
(estimated from price history): **minimum variance, maximum Sharpe (tangency),
mean-variance (target return / risk-aversion utility), risk parity** (equal risk
contribution), an efficient-frontier sweep, and **Black-Litterman** (stretch) —
which blends market-equilibrium returns with explicit views.

```bash
python main.py --ticker RELIANCE --optimize max_sharpe --rf 0.07 \
    --holdings RELIANCE:40,TCS:30,INFY:20,HDFCBANK:10
```

```python
from portfolio import max_sharpe, min_variance, risk_parity, annualized_inputs, returns_from_prices
mu, cov = annualized_inputs(returns_from_prices(price_history))
res = max_sharpe(mu, cov, rf=0.07)
res.weights            # optimal allocation
res.sharpe             # risk-adjusted performance
```

Solved with scipy SLSQP, long-only by default (optional shorting). Validated
against closed forms: for a diagonal covariance, min-variance weights ∝ 1/σ²,
risk-parity ∝ 1/σ, and Black-Litterman with no views returns the market
portfolio.

### Macro stress testing (Phase 14)

Propagates macro shocks — **oil shock, rate hike, inflation spike, INR
depreciation, recession** — through the valuation drivers (growth, margin,
risk-free rate, ERP, terminal growth), recomputes the WACC, and re-prices the
stock, with **sector-specific transmission**: an oil shock helps Energy
producers but hurts IT; INR depreciation helps IT exporters but hurts Energy.

```bash
python main.py --ticker RELIANCE --stress --sector Energy
python main.py --ticker TCS --stress --sector IT
```

```python
from macro import stress_test
res = stress_test(fundamentals, market_price=2900, sector="Energy")
res.table          # intrinsic value & % impact for each scenario
```

Shocks are stylised and transparent (editable in `macro/scenarios.py`), not
estimated factor betas.

### Actuarial layer (Phase 15, stretch)

Brings probabilistic/actuarial thinking to equity valuation:

```bash
python main.py --ticker RELIANCE --actuarial --sector Energy --equity-vol 0.28
```

- **Default probability** — the **Merton (1974)** structural model: equity is a
  call option on firm assets, so equity value + volatility back out asset value,
  asset volatility, distance to default, and `PD = N(−DD)`.
- **Stochastic discount rates** — the short rate follows a mean-reverting
  **Vasicek** process; each simulated rate path prices the FCFF differently, so
  intrinsic value becomes a distribution driven by rate uncertainty.
- **Scenario-weighted valuation** — probability-weights the Phase 14 stress
  scenarios into an expected intrinsic value, its dispersion, and P(undervalued).
- **Bayesian updating** — a Normal-Normal conjugate update revises an assumption
  (e.g. revenue growth) as evidence arrives, tightening the prior.

```python
from actuarial import merton_pd, stochastic_dcf, normal_update, scenario_weighted_value
pd_  = merton_pd(equity_value=mcap, equity_vol=0.28, debt_face=debt)
dist = stochastic_dcf(forecast["fcff"].values, net_debt=nd, shares_outstanding=sh)
post = normal_update(prior_mean=0.12, prior_std=0.05, observations=realized, obs_std=0.03)
```

## Live demo & deployment

Athena ships ready to go online for free, with **no install required by visitors**:

- **Interactive app** — the Streamlit dashboard deploys to **Streamlit Community
  Cloud** straight from this repo (entrypoint `dashboard/streamlit_app.py`). It
  streams live data with an automatic offline **demo-data fallback**, so the
  public app always works.
- **Landing page** — `docs/index.html` is a static showcase for **GitHub Pages**
  (Settings → Pages → branch `main`, folder `/docs`) with a *Launch live app*
  button.

Step-by-step instructions are in **[DEPLOY.md](DEPLOY.md)**.

## Testing

**111 offline tests** (`tests/test_phase1.py … test_phase15.py`) cover every
phase using yfinance-shaped fixtures — no network required. Beyond plumbing
checks, the suite pins the numerical engines to known truths: the DCF is
cross-checked against an independent NumPy computation, a zero-variance Monte
Carlo reproduces the deterministic DCF, the optimizer is validated against
closed-form diagonal-covariance solutions, Black-Litterman with no views returns
the market portfolio, and the Merton/Vasicek/Bayes models are checked for
monotonicity, mean reversion, and conjugate-update correctness.

```bash
pytest -q
```

## Roadmap

1. ✅ **Data ingestion**
2. ✅ **Financial statement normalization**
3. ✅ **Forecasting engine**
4. ✅ **Cost of capital (CAPM / WACC)**
5. ✅ **FCFF DCF engine**
6. ✅ **Comparable company analysis**
7. ✅ **Monte Carlo simulation**
8. ✅ **Sensitivity analysis**
9. ✅ **Risk engine**
10. ✅ **Automated report generation (DOCX / PDF)**
11. ✅ **Interactive Streamlit dashboard**
12. ✅ **Portfolio analytics**
13. ✅ **Portfolio optimization**
14. ✅ **Macroeconomic stress testing**
15. ✅ **Actuarial layer** *(stretch)* — Merton PD, Vasicek rates, Bayesian updating

## License

MIT (intended).
