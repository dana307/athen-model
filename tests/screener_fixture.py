"""A screener.in-shaped HTML fixture for testing ScreenerLoader offline.

Mirrors the real page structure: a #top-ratios block for Market Cap / Current
Price, and #profit-loss / #balance-sheet sections each holding a data-table with
year columns. Values are in ₹ crore, as on the live site.
"""
from __future__ import annotations


def _table(rows, years):
    head = "".join(f"<th>{y}</th>" for y in years)
    body = ""
    for label, vals in rows:
        tds = "".join(f"<td>{v}</td>" for v in vals)
        body += f'<tr><td class="text"><button>{label} +</button></td>{tds}</tr>'
    return (f'<table class="data-table"><thead><tr><th></th>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table>")


def screener_html() -> str:
    years = ["Mar 2024", "Mar 2023", "Mar 2022"]
    pl = _table([
        ("Sales", ["900,000", "800,000", "700,000"]),
        ("Expenses", ["740,000", "655,000", "575,000"]),
        ("Operating Profit", ["160,000", "145,000", "125,000"]),
        ("OPM %", ["18%", "18%", "18%"]),
        ("Depreciation", ["50,000", "45,000", "40,000"]),
        ("Net Profit", ["70,000", "66,000", "60,000"]),
    ], years)
    bs = _table([
        ("Equity Capital", ["6,760", "6,760", "6,760"]),
        ("Reserves", ["743,240", "683,240", "623,240"]),
        ("Borrowings", ["320,000", "300,000", "280,000"]),
        ("Total Liabilities", ["1,070,000", "990,000", "910,000"]),
    ], years)
    return f"""
    <html><body>
      <ul id="top-ratios">
        <li><span class="name">Market Cap</span><span class="value">₹ 19,60,400 Cr.</span></li>
        <li><span class="name">Current Price</span><span class="value">₹ 2,900</span></li>
        <li><span class="name">Stock P/E</span><span class="value">28.0</span></li>
      </ul>
      <section id="profit-loss"><h2>Profit &amp; Loss</h2>{pl}</section>
      <section id="balance-sheet"><h2>Balance Sheet</h2>{bs}</section>
    </body></html>
    """
