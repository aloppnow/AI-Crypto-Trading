
from datetime import datetime
import json, os, re
import statistics
import math


MARK_TRADES_START = "<!-- START:TRADE_LOG -->"
MARK_TRADES_END   = "<!-- END:TRADE_LOG -->"
MARK_PORT_START   = "<!-- START:PORTFOLIO -->"
MARK_PORT_END     = "<!-- END:PORTFOLIO -->"
MARK_UPDATED_START= "<!-- START:UPDATED -->"
MARK_UPDATED_END  = "<!-- END:UPDATED -->"

def _render_trades_md(past_trades):
    if not past_trades:
        return "_No recent trades._"
    lines = ["| Time (UTC) | Action | Symbol | Amount ($) | Limit |",
             "|---|---|---:|---:|---:|"]
    for t in past_trades[-10:][::-1]:  # newest first
        time = t.get("timestamp","")
        action = t.get("action","")
        symbol = t.get("symbol","")
        amount = t.get("amount", 0.0)
        limit = t.get("limit", None)
        limit_str = "" if limit in (None, "") else f"{float(limit):.2f}"
        lines.append(f"| {time} | {action} | {symbol} | {float(amount):.2f} | {limit_str} |")
    return "\n".join(lines)

def _render_portfolio_md(positions):
    if not positions:
        return "_No open positions._"
    lines = ["| Symbol | Quantity | Est. Value ($) |",
             "|---|---:|---:|"]
    for p in positions:
        sym = p.get("symbol","")
        qty = float(p.get("quantity", 0.0))
        val = float(p.get("dollar_amount", 0.0))
        lines.append(f"| {sym} | {qty:.8f} | {val:.2f} |")
    return "\n".join(lines)

def _replace_block(content, start_mark, end_mark, replacement):
    block = f"{start_mark}\n{replacement}\n{end_mark}"
    if start_mark in content and end_mark in content:
        return re.sub(
            rf"{re.escape(start_mark)}.*?{re.escape(end_mark)}",
            block,
            content,
            flags=re.DOTALL
        )
    # if markers missing, append at end
    return content.rstrip() + f"\n\n{block}\n"

def write_readme(past_trades_path="past_trades/past_trades.json", positions=None, readme_path="README.md"):
    # load past trades
    past_trades = []
    if os.path.exists(past_trades_path):
        try:
            with open(past_trades_path, "r", encoding="utf-8") as f:
                past_trades = json.load(f)
        except Exception:
            past_trades = []

    trades_md = _render_trades_md(past_trades)
    portfolio_md = _render_portfolio_md(positions or [])

    # ensure file exists with markers
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# Crypto Bot\n\n## Current Portfolio\n\n" +
                    f"{MARK_PORT_START}\n{MARK_PORT_END}\n\n" +
                    "## Recent Trades (last 10)\n\n" +
                    f"{MARK_TRADES_START}\n{MARK_TRADES_END}\n\n" +
                    f"{MARK_UPDATED_START}\n{MARK_UPDATED_END}\n"
                    )

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = _replace_block(content, MARK_TRADES_START, MARK_TRADES_END, trades_md)
    content = _replace_block(content, MARK_PORT_START, MARK_PORT_END, portfolio_md)

    footer = f"_Last updated: {datetime.utcnow().isoformat(timespec='seconds')}Z_"
    content = _replace_block(content, MARK_UPDATED_START, MARK_UPDATED_END, footer)

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
