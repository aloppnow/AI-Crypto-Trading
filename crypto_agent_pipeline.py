import os
import re
import requests
from datetime import datetime, timedelta
import feedparser
import ssl, certifi
from openai import OpenAI
from dotenv import load_dotenv
from utils_readme import write_readme
import utils_readme
from coinbase.rest import RESTClient
from uuid import uuid4
import json
import statistics
import math
from typing import List, Tuple
from decimal import Decimal, getcontext


load_dotenv()
ssl_context = ssl.create_default_context(cafile=certifi.where())
coinbase_api_key = os.getenv("COINBASE_API_KEY")
coinbase_api_secret = os.getenv("COINBASE_API_SECRET")



# === Config ===

RESERVE_USD = 10.0  # keep this unspent; adjust or set to 0 if you prefer

symbols = ["BTC", "ETH", "SOL", "DOGE", "ADA", "AVAX", "LINK", "SHIB"]
# === Candidate universe (broad, all USD pairs you’re okay trading) ===
CANDIDATES = ["BTC", "ETH", "SOL", "DOGE", "ADA", "AVAX", "LINK", "SHIB", "MATIC", "ARB", "APT"]


# Build a temporary map for candidates so the ranking code can query candles
PRODUCTS_CANDIDATE_MAP = {s: f"{s}-USD" for s in CANDIDATES}

# Pick the most liquid/active N symbols right now
try:
    symbols = select_active_symbols(CANDIDATES, top_k=8, min_usd=5_000_000.0)
    print("[rank] Selected symbols:", symbols)
except Exception as e:
    print("[rank] Falling back to default candidates due to error:", e)
    symbols = CANDIDATES[:8]

# FINAL map used by the rest of the bot
PRODUCTS = {s: f"{s}-USD" for s in symbols}



# ---------- Liquidity/volatility ranking ----------
def _usd_volume_and_vol(symbol: str, hours: int = 24) -> Tuple[float, float]:
    """
    Estimate last-24h USD liquidity and volatility for SYMBOL-USD using 1h candles.
    Returns (usd_volume_sum, volatility_score), where volatility_score is stdev of hourly returns (%).
    """
    product_id = PRODUCTS_CANDIDATE_MAP[symbol]  # use candidate map (set below)
    end_ts = int(datetime.now().timestamp())
    start_ts = end_ts - hours * 60 * 60

    # 1h candles for last 24h
    resp = CB_CLIENT.get_candles(product_id, start=start_ts, end=end_ts, granularity="ONE_HOUR", limit=hours+2)
    candles = _candles_list(resp)
    if not candles:
        return 0.0, 0.0

    # Sum of (volume * close) across the window (approx USD notional)
    usd_volume = 0.0
    closes = []
    for c in candles:
        try:
            usd_volume += float(c["volume"]) * float(c["close"])
            closes.append(float(c["close"]))
        except (TypeError, ValueError):
            pass

    # Volatility as stdev of hourly % returns
    rets = []
    for i in range(1, len(closes)):
        if closes[i-1] > 0:
            rets.append(100.0 * (closes[i] - closes[i-1]) / closes[i-1])
    vol = statistics.pstdev(rets) if len(rets) >= 2 else 0.0

    return usd_volume, vol

def select_active_symbols(candidates: List[str], top_k: int = 8, min_usd: float = 1_000_000.0) -> List[str]:
    """
    Rank candidates by 24h USD volume (desc), then by volatility (desc).
    Filter out anything below min_usd notional.
    """
    scored = []
    for sym in candidates:
        try:
            usd_vol, vol = _usd_volume_and_vol(sym, hours=24)
            if usd_vol >= min_usd:
                scored.append((sym, usd_vol, vol))
        except Exception as e:
            print("[rank] Failed for {sym}:", e)

    if not scored:
        # Fallback: return the original list if nothing passes the filter
        return candidates[:top_k]

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)  # by usd_vol, then vol
    return [s[0] for s in scored[:top_k]]

# === Coinbase client and helpers ===
CB_CLIENT = RESTClient(coinbase_api_key, coinbase_api_secret)
PRODUCTS = {s: f"{s}-USD" for s in PRODUCTS}
def _order_id(): return str(uuid4())


def _compute_features(candles, news_items):
    """
    candles: list of dicts with start, open, close, high, low, volume (sorted newest last)
    news_items: list of dicts with sentiment
    """
    if not candles:
        return {}

    closes = [float(c["close"]) for c in candles]
    volumes = [float(c["volume"]) for c in candles]

    last_price = closes[-1]

    def pct_change(n):
        if len(closes) <= n:
            return 0.0
        return ((closes[-1] - closes[-n]) / closes[-n]) * 100.0

    # 1h = 6 bars, 3h = 18 bars, 12h = 72 bars (for 10-min candles)
    c1h = pct_change(6)
    c3h = pct_change(18)
    c12h = pct_change(72)

    # SMA cross — fast=6 bars (~1h), slow=18 bars (~3h)
    sma_fast = sum(closes[-6:]) / min(6, len(closes))
    sma_slow = sum(closes[-18:]) / min(18, len(closes))
    sma_crossover = sma_fast > sma_slow

    # Volume z-score over last 72 bars (~12h)
    if len(volumes) >= 10:
        mean_vol = statistics.mean(volumes[-72:]) if len(volumes) >= 72 else statistics.mean(volumes)
        stdev_vol = statistics.pstdev(volumes[-72:]) if len(volumes) >= 72 else statistics.pstdev(volumes)
        vol_z = (volumes[-1] - mean_vol) / stdev_vol if stdev_vol > 0 else 0.0
    else:
        vol_z = 0.0

    # Avg sentiment
    if news_items:
        sent_avg = sum([float(n.get("sentiment", 0)) for n in news_items]) / len(news_items)
    else:
        sent_avg = 0.0

    return {
        "price": round(last_price, 2),
        "1h_change_pct": round(c1h, 2),
        "3h_change_pct": round(c3h, 2),
        "12h_change_pct": round(c12h, 2),
        "sma_fast": round(sma_fast, 2),
        "sma_slow": round(sma_slow, 2),
        "sma_crossover": sma_crossover,
        "volume_1h": round(sum(volumes[-6:]), 2),
        "vol_zscore": round(vol_z, 2),
        "sentiment_avg": round(sent_avg, 2)
    }

# ---- Coinbase model -> dict helpers ----
def _asdict(obj):
    """Best-effort convert coinbase-advanced-py models (Pydantic v1/v2) to plain dicts."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):  # Pydantic v2
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):        # Pydantic v1
        try:
            return obj.dict()
        except Exception:
            pass
    try:
        return dict(obj)
    except Exception:
        try:
            return vars(obj)
        except Exception:
            return {"value": str(obj)}

def _candles_list(resp):
    """Extract a list of candle dicts from get_candles() response (supports models or dicts)."""
    d = _asdict(resp)
    candles = d.get("candles") or getattr(resp, "candles", []) or []
    out = []
    for c in candles:
        cd = _asdict(c)
        out.append({
            "start": cd.get("start"),
            "high": float(cd.get("high")) if cd.get("high") is not None else None,
            "low": float(cd.get("low")) if cd.get("low") is not None else None,
            "open": float(cd.get("open")) if cd.get("open") is not None else None,
            "close": float(cd.get("close")) if cd.get("close") is not None else None,
            "volume": float(cd.get("volume")) if cd.get("volume") is not None else None,
        })
    return out

def _cb_price(symbol):
    product_id = PRODUCTS[symbol]
    resp = CB_CLIENT.get_candles(product_id, start=None, end=None, granularity="ONE_MINUTE", limit=1)
    candles = _candles_list(resp)
    if not candles:
        return 0.0
    return float(candles[0]["close"] or 0.0)

# Helper functions to buy right amount


def _buying_power_usd() -> float:
    """USD available for Advanced Trade after reserve."""
    avail = float(get_balance() or 0.0)
    bp = max(0.0, avail - RESERVE_USD)
    return round(bp, 2)

def _cap_to_buying_power(symbol: str, quote_usd: float) -> float:
    """
    Clamp requested quote amount to available buying power and product min notional.
    Returns 0 if we still can't meet minimum.
    """
    bp = _buying_power_usd()
    if bp <= 0:
        return 0.0
    # Clamp to buying power first
    clamped = min(float(quote_usd), bp)
    # Quantize & enforce per-product min
    clamped_q = _ensure_quote_amount(symbol, clamped)
    # If min notional > buying power, we still can't place
    specs = _product_specs(symbol)
    if clamped_q < (specs["min_market_funds"] or 1.0):
        return 0.0
    return clamped_q

getcontext().prec = 28  # high precision for quantization

def _as_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _round_to_increment(value: float, increment: float) -> float:
    if increment <= 0:
        return value
    v = Decimal(str(value))
    inc = Decimal(str(increment))
    # floor to the nearest allowed tick (Coinbase requires truncation, not rounding)
    return float((v // inc) * inc)

_prod_cache = {}

def _product_specs(symbol):
    """Return dict with base_increment, quote_increment, min_market_funds, base_min_size for symbol."""
    if symbol in _prod_cache:
        return _prod_cache[symbol]
    pid = PRODUCTS[symbol]
    prod = _asdict(CB_CLIENT.get_product(pid))
    specs = {
        "base_increment": _as_float(prod.get("base_increment", 0.0)) or _as_float(prod.get("base_increment_value", 0.0)) or 0.0,
        "quote_increment": _as_float(prod.get("quote_increment", 0.0)) or 0.0,
        "min_market_funds": _as_float(prod.get("min_market_funds", 0.0)) or _as_float(prod.get("min_order_value", 0.0)) or 1.0,
        "base_min_size": _as_float(prod.get("base_min_size", 0.0)) or _as_float(prod.get("min_order_size", 0.0)) or 0.0,
    }
    _prod_cache[symbol] = specs
    return specs

def _ensure_quote_amount(symbol, quote_usd: float) -> float:
    """Respect product min notional & quote_increment."""
    specs = _product_specs(symbol)
    q = max(quote_usd, specs["min_market_funds"] or 1.0)
    inc = specs["quote_increment"] or 0.01  # fallback 1 cent
    q_adj = _round_to_increment(q, inc)
    # guard: if rounding down hit 0 due to very small increment, bump to min
    if q_adj < specs["min_market_funds"]:
        q_adj = specs["min_market_funds"]
    return float(q_adj)

def _base_from_quote_quantized(symbol, quote_usd: float, price: float) -> float:
    """Convert quote→base and quantize to base_increment and base_min_size."""
    specs = _product_specs(symbol)
    base = 0.0 if price <= 0 else (float(quote_usd) / float(price))
    base_q = _round_to_increment(base, specs["base_increment"] or 0.0)
    # enforce base_min_size
    if specs["base_min_size"] and base_q < specs["base_min_size"]:
        base_q = specs["base_min_size"]
        # and quantize again to increment
        base_q = _round_to_increment(base_q, specs["base_increment"] or 0.0)
    return float(base_q)

def _log_http_error(prefix, err):
    try:
        resp = getattr(err, "response", None)
        print(prefix, "-", err)
        if resp is not None:
            print("Status:", resp.status_code)
            try:
                print("Body:", resp.text)
            except Exception:
                pass
    except Exception:
        pass

# === Helpers: historical candles & sentiment ===
def _cb_candles(product_id, start_ts, end_ts, granularity="FIVE_MINUTE"):
    """
    Fetch candles between start_ts and end_ts with the given granularity.
    Coinbase returns up to a limit; we slice in chunks of ~6 hours for safety.
    """
    results = []
    # 6 hours in seconds
    step = 6 * 60 * 60
    cursor = int(start_ts)
    while cursor < end_ts:
        chunk_end = min(cursor + step, end_ts)
        resp = CB_CLIENT.get_candles(
            product_id,
            start=cursor,
            end=chunk_end,
            granularity=granularity,
            limit=500
        )
        candles = _candles_list(resp)
        results.extend(candles)
        cursor = chunk_end
    # Sort by start ascending and de-dupe by start
    seen = set()
    deduped = []
    for c in sorted(results, key=lambda x: x["start"]):
        if c["start"] in seen:
            continue
        seen.add(c["start"])
        deduped.append(c)
    return deduped

def _aggregate_to_10min(candles_5m):
    """
    Convert 5-minute candles to 10-minute by merging pairs.
    Each item of candles_5m is dict with keys: start, open, close, high, low, volume
    Returns list of 10-minute candles with same keys.
    """
    out = []
    # Ensure sorted by start
    c = sorted(candles_5m, key=lambda x: x["start"])
    for i in range(0, len(c)-1, 2):
        a = c[i]
        b = c[i+1]
        out.append({
            "start": a["start"],
            "open": a["open"],
            "close": b["close"],
            "high": max(float(a["high"]), float(b["high"])),
            "low": min(float(a["low"]), float(b["low"])),
            "volume": float(a["volume"]) + float(b["volume"]),
        })
    return out

_POS_WORDS = {
    "beat","beats","beating","surge","surges","surging","rally","rallies","bull","bullish","upgrade","upgrades","breakout",
    "record","top","growth","strong","soar","soars","soaring","pop","pops","rebound","rebounds","rebounding","approve","approved",
    "positive","gain","gains","gaining","win","wins","winning","expand","expands","expanding","adopt","adoption",
}
_NEG_WORDS = {
    "miss","misses","warning","warns","plunge","plunges","plunging","drop","drops","dropping","bear","bearish","downgrade","downgrades",
    "lawsuit","ban","bans","banned","hack","hacked","exploit","exploit","scam","scams","fraud","fraudulent","negative","loss",
    "loses","losing","concern","concerns","concerned","crash","crashes","crashing","liquidation","selloff","sell-off",
}

def _simple_sentiment(text):
    t = (text or "").lower()
    pos = sum(1 for w in _POS_WORDS if w in t)
    neg = sum(1 for w in _NEG_WORDS if w in t)
    if pos == 0 and neg == 0:
        return 0.0
    score = (pos - neg) / max(pos + neg, 1)
    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, score))

def _yahoo_symbol(symbol):
    # Map to Yahoo tickers for RSS; many use -USD suffix
    if symbol == "BTC":
        return "BTC-USD"
    return f"{symbol}-USD"

# === GitHub Actions Job Summary ===
def _md_table_positions(positions):
    if not positions:
        return "_No open positions._"
    lines = ["| Symbol | Quantity | Est. Value ($) |",
             "|---|---:|---:|"]
    for p in positions:
        lines.append(f"| {p['symbol']} | {float(p['quantity']):.8f} | {float(p['dollar_amount']):.2f} |")
    return "\n".join(lines)

def _md_table_trades(past_trades_path):
    import json, os
    if not os.path.exists(past_trades_path):
        return "_No trades logged yet._"
    try:
        with open(past_trades_path, "r", encoding="utf-8") as f:
            trades = json.load(f)
    except Exception:
        return "_Could not read trade log._"
    if not trades:
        return "_No trades logged yet._"
    lines = ["| Time (UTC) | Action | Symbol | Amount ($) | Limit |",
             "|---|---|---:|---:|---:|"]
    for t in trades[-10:][::-1]:
        ts = t.get("timestamp","")
        action = t.get("action","")
        sym = t.get("symbol","")
        amt = float(t.get("amount",0.0))
        lim = t.get("limit", None)
        lim_str = "" if lim in (None,"") else f"{float(lim):.2f}"
        lines.append(f"| {ts} | {action} | {sym} | {amt:.2f} | {lim_str} |")
    return "\n".join(lines)

def write_job_summary(past_trades_path, positions, last_command=None, last_explanation=None):
    import os
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return  # Not running inside GitHub Actions or summary not supported

    parts = []
    parts.append(f"# Crypto Bot Run • {datetime.utcnow().isoformat(timespec='seconds')}Z")
    if last_command:
        parts.append("")
        parts.append("**Model Output:**")
        parts.append("")
        parts.append(f"`{last_command}`")
    if last_explanation:
        parts.append("")
        parts.append("**Explanation:**")
        parts.append("")
        parts.append(last_explanation.strip())

    parts.append("")
    parts.append("## Recent Trades (last 10)")
    parts.append(_md_table_trades(past_trades_path))

    parts.append("")
    parts.append("## Current Portfolio")
    parts.append(_md_table_positions(positions))

    md = "\n".join(parts) + "\n"

    try:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(md)
    except Exception as e:
        print("Failed to write GITHUB_STEP_SUMMARY:", e)

# === Trade log ===
LOG_DIR = "past_trades"
LOG_FILE = os.path.join(LOG_DIR, "past_trades.json")
os.makedirs(LOG_DIR, exist_ok=True)
try:
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        past_trades = json.load(f)
except Exception:
    past_trades = []

def record_trade(action, symbol, amount, limit=None):
    info = {
        "action": action, "symbol": symbol,
        "amount": float(amount), "limit": float(limit) if limit is not None else None,
        "timestamp": datetime.now().isoformat()
    }
    past_trades.append(info)
    if len(past_trades) > 10:
        del past_trades[:-10]
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(past_trades, f)

# === Execution wrappers (Coinbase) ===
def buy_crypto_price(symbol, amount):
    product_id = PRODUCTS[symbol]
    amt = _cap_to_buying_power(symbol, float(amount))

    # 🔹 Log the request vs capped value
    print("[BUY] {symbol} request=${amount:.2f} → capped=${amt:.2f} | USD avail={_buying_power_usd():.2f}")


    if amt <= 0:
        print("[BUY] Skipped: insufficient buying power. USD avail={_buying_power_usd():.2f}")
        return
    try:
        res = CB_CLIENT.create_order(
            client_order_id=_order_id(),
            product_id=product_id,
            side="BUY",
            order_configuration={"market_market_ioc": {"quote_size": f"{amt:.2f}"}},
        )
        record_trade("buy_crypto_price", symbol, amt)
        print(res)
    except Exception as e:
        _log_http_error("BUY market failed", e)
        raise

def sell_crypto_price(symbol, amount):
    product_id = PRODUCTS[symbol]
    # For sells we size by requested quote but clamp to position size too (optional)
    px = _cb_price(symbol)
    amt = _cap_to_buying_power(symbol, float(amount))  # keeps behavior symmetric

    # 🔹 Log the request vs capped value
    print("[BUY] {symbol} request=${amount:.2f} → capped=${amt:.2f} | USD avail={_buying_power_usd():.2f}")


    if amt <= 0:
        # If you want sells to always execute up to position size, compute base from position instead.
        print("[SELL] Skipped: insufficient buying power context; requested={amount}, px={px}")
        return
    base_size = _base_from_quote_quantized(symbol, amt, px)
    if base_size <= 0:
        print("[SELL] Skipped: base_size quantized to 0. requested={amount}, px={px}")
        return
    try:
        res = CB_CLIENT.create_order(
            client_order_id=_order_id(),
            product_id=product_id,
            side="SELL",
            order_configuration={"market_market_ioc": {"base_size": f"{base_size}"}},
        )
        record_trade("sell_crypto_price", symbol, amt)
        print(res)
    except Exception as e:
        _log_http_error("SELL market failed", e)
        raise

def buy_crypto_limit(symbol, amount, limit):
    product_id = PRODUCTS[symbol]
    limit = float(limit)
    amt = _cap_to_buying_power(symbol, float(amount))

    # 🔹 Log the request vs capped value
    print("[BUY] {symbol} request=${amount:.2f} → capped=${amt:.2f} | USD avail={_buying_power_usd():.2f}")

    if amt <= 0:
        print("[BUY LIMIT] Skipped: insufficient buying power. USD avail={_buying_power_usd():.2f}")
        return
    base_size = _base_from_quote_quantized(symbol, amt, limit)
    if base_size <= 0:
        print("[BUY LIMIT] Skipped: base_size quantized to 0.")
        return
    try:
        res = CB_CLIENT.create_order(
            client_order_id=_order_id(),
            product_id=product_id,
            side="BUY",
            order_configuration={
                "limit_limit_gtc": {
                    "base_size": f"{base_size}",
                    "limit_price": f"{limit:.2f}",
                    "post_only": False,
                }
            },
        )
        record_trade("buy_crypto_limit", symbol, amt, limit)
        print(res)
    except Exception as e:
        _log_http_error("BUY limit failed", e)
        raise

def sell_crypto_limit(symbol, amount, limit):
    product_id = PRODUCTS[symbol]
    limit = float(limit)
    amt = _cap_to_buying_power(symbol, float(amount))

    # 🔹 Log the request vs capped value
    print("[BUY] {symbol} request=${amount:.2f} → capped=${amt:.2f} | USD avail={_buying_power_usd():.2f}")

    if amt <= 0:
        print("[SELL LIMIT] Skipped: insufficient buying power. USD avail={_buying_power_usd():.2f}")
        return
    base_size = _base_from_quote_quantized(symbol, amt, limit)
    if base_size <= 0:
        print("[SELL LIMIT] Skipped: base_size quantized to 0.")
        return
    try:
        res = CB_CLIENT.create_order(
                client_order_id=_order_id(),
                product_id=product_id,
                side="SELL",
                order_configuration={
                    "limit_limit_gtc": {
                        "base_size": f"{base_size}",
                        "limit_price": f"{limit:.2f}",
                        "post_only": False,
                    }
                },
        )
        record_trade("sell_crypto_limit", symbol, amt, limit)
        print(res)
    except Exception as e:
        _log_http_error("SELL limit failed", e)
        raise

def cancel_order(orderId):
    res = CB_CLIENT.cancel_orders(order_ids=[orderId])
    print(res)

# === Account state (Coinbase) ===
def get_balance():
    data = _asdict(CB_CLIENT.get_accounts())
    accounts = data.get("accounts") or getattr(CB_CLIENT.get_accounts(), "accounts", []) or []
    for a in accounts:
        ad = _asdict(a)
        if ad.get("currency") == "USD":
            try:
                avail = _asdict(ad.get("available_balance") or {})
                return float(avail.get("value", 0.0))
            except Exception:
                return 0.0
    return 0.0

def get_positions():
    data = _asdict(CB_CLIENT.get_accounts())
    accounts = data.get("accounts") or getattr(CB_CLIENT.get_accounts(), "accounts", []) or []
    positions = []
    for a in accounts:
        ad = _asdict(a)
        cur = ad.get("currency")
        if cur == "USD" or cur not in PRODUCTS:
            continue
        try:
            avail = _asdict(ad.get("available_balance") or {})
            qty = float(avail.get("value", 0.0))
        except Exception:
            qty = 0.0
        if qty <= 0:
            continue
        px = _cb_price(cur)
        positions.append({"symbol": cur, "quantity": qty, "dollar_amount": qty * px})
    return positions

def get_open_orders():
    data = _asdict(CB_CLIENT.list_orders(limit=50))
    orders = data.get("orders") or getattr(CB_CLIENT.list_orders(limit=50), "orders", []) or []
    out = []
    for o in orders:
        od = _asdict(o)
        cfg = _asdict(od.get("order_configuration") or {})
        out.append({
            "id": od.get("order_id"),
            "type": (od.get("order_type") or "").lower(),
            "side": (od.get("side") or "").lower(),
            "quantity": cfg.get("base_size") or cfg.get("quote_size") or "",
            "price": cfg.get("limit_price") or "",
        })
    return out

# === Market data ===
def get_crypto_infos():
    infos = {}
    start_date = int((datetime.now() - timedelta(minutes=2)).timestamp())
    end_date = int(datetime.now().timestamp())
    granularity = "ONE_MINUTE"
    limit = 1
    for symbol in PRODUCTS:
        product_id = f"{symbol}-USD"
        resp = CB_CLIENT.get_candles(product_id, start=start_date, end=end_date, granularity=granularity, limit=limit)
        first_candle = _candles_list(resp)
        if first_candle:
            c = first_candle[0]
            infos[symbol] = {
                "symbol": symbol,
                "high_price": c["high"],
                "low_price": c["low"],
                "open_price": c["open"],
                "close_price": c["close"],
                "volume": c["volume"],
            }
    return infos

def get_historical_data():
    """
    Return last 7 days of 10-minute candles per symbol using 5-minute data aggregated.
    Structure: { "BTC": [ {start, open, close, high, low, volume}, ... ], ... }
    """
    out = {}
    end_ts = int(datetime.now().timestamp())
    start_ts = end_ts - 7*24*60*60  # 7 days
    for symbol in PRODUCTS:
        product_id = PRODUCTS[symbol]
        try:
            c5 = _cb_candles(product_id, start_ts, end_ts, granularity="FIVE_MINUTE")
            c10 = _aggregate_to_10min(c5)
            out[symbol] = c10
        except Exception as e:
            print("Historical fetch failed for {symbol}:", e)
            out[symbol] = []
    return out

def get_all_crypto_news():
    """
    Fetch top 3 Yahoo Finance headlines per symbol and attach sentiment score [-1,1].
    """
    news = {}
    for symbol in PRODUCTS:
        ysym = _yahoo_symbol(symbol)
        url = f'http://finance.yahoo.com/rss/headline?s={ysym}'
        try:
            parsed = feedparser.parse(url)
            items = parsed.entries[:6]  # fetch a few, then filter
            # de-dup titles and keep top 3
            seen = set()
            rows = []
            for it in items:
                title = getattr(it, "title", "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                rows.append({"title": title, "sentiment": _simple_sentiment(title)})
                if len(rows) >= 3:
                    break
            news[symbol] = rows
        except Exception as e:
            print("News fetch failed for", symbol, e)
            news[symbol] = []
    return news

PROMPT_FOR_AI = """You are an advanced cryptocurrency trading AI focused on aggressive, short-term gains with moderate risk tolerance.
Objective: Maximize returns over the next 30 days, trading only: "BTC", "ETH", "SOL", "DOGE", "ADA", "AVAX", "LINK", "SHIB", "MATIC", "ARB", "APT"
Data Provided: Crypto Info, Balance, Open Orders, Positions, Historical Data (10-minute), News (with sentiment).
Current time: {now}. Call frequency: every 30 minutes.
Strategy: Use technicals, volume, breakouts, reversals, sentiment; allow scalping; up to 60% concentration; fast reversals.
Risk: Stop >8% unless strong bullish case; cool-off 60 min unless price moved >2%; do nothing if no edge. Leave a balance of $10 in the account.
Output ONE command line, then '## Explanation' and a short explanation.
Valid commands:
buy_crypto_price(SYMBOL, AMOUNT)
buy_crypto_limit(SYMBOL, AMOUNT, LIMIT)
sell_crypto_price(SYMBOL, AMOUNT)
sell_crypto_limit(SYMBOL, AMOUNT, LIMIT)
cancel_order(ORDER_ID)
do_nothing()
"""

def parse_ai_response(response_text):
    cleaned = response_text.strip()
    if "## Explanation" in cleaned:
        command_part, explanation_part = cleaned.split("## Explanation", 1)
        return command_part.strip(), explanation_part.strip()
    return cleaned, ""

def _compact_positions(positions: list, max_items: int = 12):
    out = []
    for p in (positions or [])[:max_items]:
        out.append({
            "s": p.get("symbol"),
            "q": round(float(p.get("quantity", 0)), 8),
            "usd": round(float(p.get("dollar_amount", 0)), 2),
        })
    return out

def _compact_history(hist: dict, max_points_per_symbol: int = 36):
    """Keep only the most recent N 10-min candles per symbol and round numbers."""
    out = {}
    for sym, rows in (hist or {}).items():
        tail = (rows or [])[-max_points_per_symbol:]
        out[sym] = [
            {
                "t": r.get("start"),
                "o": round(float(r.get("open", 0)), 2),
                "c": round(float(r.get("close", 0)), 2),
                "h": round(float(r.get("high", 0)), 2),
                "l": round(float(r.get("low", 0)), 2),
                "v": round(float(r.get("volume", 0)), 2),
            }
            for r in tail
        ]
    return out

def _compact_news(news: dict, per_symbol: int = 2, max_len: int = 120):
    """Keep only top N headlines per symbol and trim text length."""
    out = {}
    for sym, items in (news or {}).items():
        mini = []
        for it in (items or [])[:per_symbol]:
            title = (it.get("title") or "")[:max_len]
            mini.append({"t": title, "s": float(it.get("sentiment", 0))})
        out[sym] = mini
    return out

def _compact_orders(open_orders: list, max_items: int = 5):
    out = []
    for o in (open_orders or [])[:max_items]:
        out.append({
            "id": o.get("id"),
            "type": o.get("type"),
            "side": o.get("side"),
            "qty": o.get("quantity"),
            "px": o.get("price"),
        })
    return out

def _compact_positions(positions: list, max_items: int = 12):
    out = []
    for p in (positions or [])[:max_items]:
        out.append({
            "s": p.get("symbol"),
            "q": round(float(p.get("quantity", 0)), 8),
            "usd": round(float(p.get("dollar_amount", 0)), 2),
        })
    return out

def _compact_crypto_info(crypto_info: dict):
    out = {}
    for sym, ci in (crypto_info or {}).items():
        out[sym] = {
            "h": ci.get("high_price"),
            "l": ci.get("low_price"),
            "o": ci.get("open_price"),
            "c": ci.get("close_price"),
            "v": ci.get("volume"),
        }
    return out

def get_trade_advice():
    balance = get_balance()
    positions = get_positions()
    news = get_all_crypto_news()
    open_orders = get_open_orders()

    # Historical 10-min candles for ~12h (72 bars) for feature calc
    end_ts = int(datetime.now().timestamp())
    start_ts = end_ts - 12 * 60 * 60
    historical_data = {}
    for symbol in PRODUCTS:
        product_id = PRODUCTS[symbol]
        try:
            c5 = _cb_candles(product_id, start_ts, end_ts, granularity="FIVE_MINUTE")
            c10 = _aggregate_to_10min(c5)
            historical_data[symbol] = c10
        except Exception as e:
            print("Historical fetch failed for {symbol}:", e)
            historical_data[symbol] = []

    # Build feature set
    features = {}
    for sym in PRODUCTS:
        feats = _compute_features(historical_data[sym], news.get(sym, []))
        features[sym] = feats

    payload = {
        "features": features,
        "balance": round(float(balance), 2),
        "positions": _compact_positions(positions),
        "open_orders": _compact_orders(open_orders),
        "now": datetime.now().isoformat(),
        "call_frequency": "30m"
    }

    prompt = PROMPT_FOR_AI.format(now=datetime.now().isoformat())
    user_prompt = "Inputs (JSON):\n" + json.dumps(payload, separators=(",",":"))

    client = OpenAI()
    response = client.responses.create(
        model="gpt-5-nano",
        input=f"{prompt}\n\n{user_prompt}"
    )
    return response.output_text

def execute_response(response):
    command, explanation = parse_ai_response(response)
    match = re.match(r'(\w+)\((.*?)\)', command)
    if not match:
        print("No valid command found. Full text:", response)
        return
    cmd = match.group(1)
    args = [arg.strip().strip('"') for arg in match.group(2).split(',')] if match.group(2).strip() else []

    command_map = {
        "buy_crypto_price": buy_crypto_price,
        "buy_crypto_limit": buy_crypto_limit,
        "sell_crypto_price": sell_crypto_price,
        "sell_crypto_limit": sell_crypto_limit,
        "cancel_order": cancel_order,
        "do_nothing": lambda *a, **k: None,
    }
    func = command_map.get(cmd)
    if not func:
        print("Unknown command:", cmd)
        return
    try:
        func(*args)
    except TypeError as e:
        print("Argument mismatch for", cmd, "args:", args, "error:", e)
    if explanation:
        print("Explanation:", explanation)

if __name__ == "__main__":
    print("----------------------------------------")
    print("Starting new iteration at.", datetime.now().isoformat())
    advice = get_trade_advice()
    execute_response(advice)

    # Update README with post-trade positions
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parent
    LOG_DIR = REPO_ROOT / "past_trades"
    LOG_FILE = LOG_DIR / "past_trades.json"
    README_PATH = REPO_ROOT / "README.md"
    try:
        positions_post = get_positions()
    except Exception as e:
        print("Could not refresh positions:", e)
        positions_post = []
    write_readme(
        past_trades_path=str(LOG_FILE),
        positions=positions_post,
        readme_path=str(README_PATH),
    )