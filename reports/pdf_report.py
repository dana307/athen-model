"""
PDF renderer (reportlab Platypus).

Same ResearchReport content as the DOCX renderer, laid out as a flowing PDF with
embedded charts and tables.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    ListFlowable, ListItem,
)

from utils.schema import FIELD_LABELS

CR = 1e7
_REC_HEX = {"BUY": "#1A7F37", "HOLD": "#B58B00", "REDUCE": "#C0392B"}


def _s(text: str) -> str:
    """Helvetica has no rupee glyph — render it as 'Rs '."""
    return str(text).replace("₹", "Rs ")


def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("Rec", parent=s["Title"], fontSize=18))
    s.add(ParagraphStyle("Small", parent=s["Normal"], fontSize=8,
                         textColor=colors.grey))
    return s


def _table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C78A8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F2F6FB")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def build_pdf(report, path: str | Path) -> Path:
    path = Path(path)
    st = _styles()
    story = []

    def H(text, lvl=1):
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(text, st["Heading1" if lvl == 1 else "Heading2"]))

    # --- title ------------------------------------------------------------
    story.append(Paragraph(f"{report.company_name} ({report.ticker})", st["Title"]))
    story.append(Paragraph("Athena — Equity Research Note", st["Italic"]))
    story.append(Paragraph(f"As of {report.as_of} | Currency: {report.currency}",
                           st["Small"]))
    story.append(Spacer(1, 0.3 * cm))

    rec_color = _REC_HEX.get(report.recommendation, "#000000")
    story.append(Paragraph(
        f'<font color="{rec_color}"><b>Recommendation: '
        f'{report.recommendation}</b></font>', st["Rec"]))
    story.append(_table([
        ["Market price", f"Rs {report.market_price:,.0f}"],
        ["Fair value (blended)", f"Rs {report.fair_value:,.0f}"],
        ["Upside / (downside)", f"{report.upside:+.1%}"],
    ], col_widths=[6 * cm, 6 * cm]))

    # --- thesis -----------------------------------------------------------
    H("Investment Thesis")
    story.append(ListFlowable(
        [ListItem(Paragraph(_s(t), st["Normal"])) for t in report.thesis],
        bulletType="bullet"))

    # --- financial summary -----------------------------------------------
    H("Financial Summary")
    story.append(_financial_table(report))

    # --- forecast ---------------------------------------------------------
    H("Forecast")
    if "revenue" in report.charts:
        story.append(Image(report.charts["revenue"], width=15 * cm, height=7.9 * cm))
    story.append(_forecast_table(report))

    # --- DCF --------------------------------------------------------------
    H("DCF Valuation")
    d, w = report.dcf, report.wacc
    story.append(_table([
        ["Cost of equity (CAPM)", f"{w.cost_of_equity:.2%}"],
        ["After-tax cost of debt", f"{w.cost_of_debt_aftertax:.2%}"],
        ["WACC", f"{w.wacc:.2%} (E {w.weight_equity:.0%}/D {w.weight_debt:.0%})"],
        ["PV explicit FCFF (cr)", f"{d.pv_explicit/CR:,.0f}"],
        ["PV terminal value (cr)", f"{d.pv_terminal/CR:,.0f}"],
        ["Enterprise value (cr)", f"{d.enterprise_value/CR:,.0f}"],
        ["Less: net debt (cr)", f"{d.net_debt/CR:,.0f}"],
        ["Equity value (cr)", f"{d.equity_value/CR:,.0f}"],
        ["Intrinsic value", f"Rs {d.intrinsic_price:,.0f}/sh ({d.upside:+.0%})"],
    ], col_widths=[7 * cm, 6 * cm]))

    # --- comparables ------------------------------------------------------
    if report.comps is not None:
        H("Comparable Company Analysis")
        story.append(_comps_table(report.comps))
        story.append(Paragraph(
            f"Implied (blended median multiples): "
            f"Rs {report.comps.implied_price:,.0f}/share "
            f"({report.comps.upside:+.0%})", st["Normal"]))

    # --- Monte Carlo ------------------------------------------------------
    H("Monte Carlo Simulation")
    if "montecarlo" in report.charts:
        story.append(Image(report.charts["montecarlo"], width=15 * cm, height=7.9 * cm))
    mc = report.montecarlo
    story.append(_table([
        ["Mean / median", f"Rs {mc.mean:,.0f} / Rs {mc.median:,.0f}"],
        ["Std deviation", f"Rs {mc.std:,.0f}"],
        ["90% CI", f"Rs {mc.p5:,.0f} - Rs {mc.p95:,.0f}"],
        ["P(undervalued)", f"{mc.prob_undervalued:.0%}"],
    ], col_widths=[6 * cm, 7 * cm]))

    # --- scenarios --------------------------------------------------------
    H("Bull / Base / Bear Scenarios")
    if "scenarios" in report.charts:
        story.append(Image(report.charts["scenarios"], width=11 * cm, height=7.3 * cm))
    story.append(_scenario_table(report))

    # --- risks ------------------------------------------------------------
    H("Key Risks")
    story.append(Paragraph(f"Overall risk rating: <b>{report.risk.overall_level}</b>",
                           st["Normal"]))
    story.append(ListFlowable(
        [ListItem(Paragraph(_s(r), st["Normal"])) for r in report.key_risks],
        bulletType="bullet"))

    # --- recommendation ---------------------------------------------------
    H("Recommendation")
    story.append(Paragraph(
        f"On a blended DCF/peer fair value of Rs {report.fair_value:,.0f} versus a "
        f"market price of Rs {report.market_price:,.0f} ({report.upside:+.1%}), with "
        f"an overall risk rating of {report.risk.overall_level}, Athena's model "
        f"output is <b>{report.recommendation}</b>.", st["Normal"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "Generated automatically by Athena for research/educational purposes only. "
        "Not investment advice.", st["Small"]))

    if "sensitivity" in report.charts:
        H("Appendix — Sensitivity")
        story.append(Image(report.charts["sensitivity"], width=15 * cm, height=10 * cm))

    SimpleDocTemplate(str(path), pagesize=A4,
                      topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                      leftMargin=1.6 * cm, rightMargin=1.6 * cm).build(story)
    return path


# --- table builders --------------------------------------------------------
def _financial_table(report):
    df = report.fundamentals.sort_index(ascending=False)
    years = list(df.index)
    rows = ["revenue", "ebitda", "ebit", "pat", "total_equity", "debt", "cash"]
    data = [["Rs crore"] + [str(y) for y in years]]
    for r in rows:
        line = [FIELD_LABELS[r]]
        for y in years:
            v = df.loc[y, r]
            line.append("—" if v != v else f"{v/CR:,.0f}")
        data.append(line)
    return _table(data)


def _forecast_table(report):
    data = [["Year", "Revenue", "EBITDA", "EBIT", "FCFF", "Growth"]]
    for y, row in report.forecast.iterrows():
        data.append([str(y), f"{row['revenue']/CR:,.0f}", f"{row['ebitda']/CR:,.0f}",
                     f"{row['ebit']/CR:,.0f}", f"{row['fcff']/CR:,.0f}",
                     f"{row['revenue_growth']*100:.1f}%"])
    return _table(data)


def _comps_table(comps):
    data = [["Ticker", "EV/Sales", "EV/EBITDA", "P/E", "ROE%", "D/E"]]
    for tk, row in comps.table.iterrows():
        data.append([str(tk), f"{row['ev_sales']:.1f}", f"{row['ev_ebitda']:.1f}",
                     f"{row['pe']:.1f}", f"{row['roe']*100:.1f}",
                     f"{row['debt_to_equity']:.2f}"])
    return _table(data)


def _scenario_table(report):
    data = [["Scenario", "Intrinsic", "Upside", "Growth", "Margin", "WACC", "Term g"]]
    for name, s in report.scenarios.items():
        a = s["assumptions"]
        data.append([name, f"Rs {s['price']:,.0f}", f"{s['upside']:+.0%}",
                     f"{a['growth']*100:.1f}%", f"{a['margin']*100:.1f}%",
                     f"{a['wacc']*100:.1f}%", f"{a['tg']*100:.1f}%"])
    return _table(data)
