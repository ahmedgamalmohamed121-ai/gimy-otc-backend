"""
Institutional Signal Generator
================================
- Fetches real OHLCV data from Yahoo Finance for indicator calculations
- Uses live price from ws_engine (WebSocket / fallback simulator) for current_price
- Computes RSI(14), EMA(9/21) crossover, Stochastic(14,3), Support/Resistance
- Applies strict SKIP rules to avoid signals near S/R boundaries
"""

import asyncio
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
import logging
from datetime import datetime, timedelta
from backend.services.quotex_ws import ws_engine, TOP_5_OTC_ASSETS

logger = logging.getLogger("SignalGenerator")
cairo_tz = pytz.timezone("Africa/Cairo")

YF_TICKERS = {
    "EUR/USD (OTC)": "EURUSD=X",
    "GBP/USD (OTC)": "GBPUSD=X",
    "USD/JPY (OTC)": "USDJPY=X",
    "USD/BRL (OTC)": "USDBRL=X",
}

# ─────────────── Indicator helpers ───────────────

def _rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if not np.isnan(val) else 50.0


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3):
    if len(close) < k_period:
        return 50.0, 50.0
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * ((close - lowest_low) / denom)
    d = k.rolling(d_period).mean()
    kv = float(k.iloc[-1]) if not np.isnan(k.iloc[-1]) else 50.0
    dv = float(d.iloc[-1]) if not np.isnan(d.iloc[-1]) else 50.0
    return kv, dv


def _ema(series: pd.Series, span: int) -> float:
    if len(series) < span:
        return float(series.iloc[-1]) if len(series) else 0.0
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def _support_resistance(high: pd.Series, low: pd.Series, window: int = 20):
    recent_high = high.tail(window)
    recent_low  = low.tail(window)
    resistance  = float(recent_high.max())
    support     = float(recent_low.min())
    return support, resistance

# ─────────────── Yahoo Finance fetch ───────────────

async def _fetch_ohlcv(ticker: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(
        None,
        lambda: yf.download(ticker, period=period, interval=interval, progress=False)
    )
    if df.empty:
        return df
    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df.tail(300)  # keep last 300 candles max


async def _get_market_context(asset: str) -> dict | None:
    """
    Returns market context dict with real indicators for an asset.
    Falls back gracefully on any error.
    """
    ticker = YF_TICKERS.get(asset)
    if not ticker:
        return None

    try:
        df = await _fetch_ohlcv(ticker, period="5d", interval="5m")
        if df.empty or len(df) < 30:
            logger.warning(f"Not enough OHLCV data for {asset} ({len(df)} rows)")
            return None

        close = df["Close"].astype(float)
        high  = df["High"].astype(float)
        low   = df["Low"].astype(float)

        rsi_val     = _rsi(close, 14)
        ema9        = _ema(close, 9)
        ema21       = _ema(close, 21)
        stoch_k, stoch_d = _stochastic(high, low, close, 14, 3)
        support, resistance = _support_resistance(high, low, window=50)

        # Get live price from ws_engine (overrides Yahoo last close)
        live_price = ws_engine.live_data.get(asset, {}).get("current_price", 0.0)
        current_price = live_price if live_price > 0 else float(close.iloc[-1])

        trend = "CALL" if ema9 > ema21 else "PUT"
        trend_strength = abs(ema9 - ema21) / ema21 * 10000  # in pips-equivalent

        logger.info(
            f"[CTX] {asset} | Price: {current_price:.5f} | "
            f"EMA9: {ema9:.5f} EMA21: {ema21:.5f} | "
            f"RSI: {rsi_val:.1f} | Stoch-K: {stoch_k:.1f} | "
            f"Trend: {trend} | S: {support:.5f} R: {resistance:.5f}"
        )

        return {
            "asset": asset,
            "price": current_price,
            "ema9": ema9,
            "ema21": ema21,
            "rsi": rsi_val,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "support": support,
            "resistance": resistance,
            "trend": trend,
            "trend_strength": trend_strength,
        }

    except Exception as e:
        logger.error(f"Failed to compute market context for {asset}: {e}")
        return None


# ─────────────── Signal logic ───────────────

def _should_skip(ctx: dict, price: float) -> bool:
    """True if price is too close to S/R — skip entry."""
    spread = ctx["resistance"] - ctx["support"]
    if spread <= 0:
        return False
    dist_to_res = (ctx["resistance"] - price) / spread
    dist_to_sup = (price - ctx["support"]) / spread
    # Skip if within 5% of S/R zone
    if ctx["trend"] == "CALL" and dist_to_res < 0.05:
        return True
    if ctx["trend"] == "PUT" and dist_to_sup < 0.05:
        return True
    return False


def _build_signal(sig_id: int, ctx: dict, entry_time: datetime, sim_price: float) -> dict:
    action = ctx["trend"]
    lvl = (
        f"SUPPORT: {ctx['support']:.5f}"
        if action == "CALL"
        else f"RESISTANCE: {ctx['resistance']:.5f}"
    )
    return {
        "id": sig_id,
        "pair": ctx["asset"],
        "time": entry_time.strftime("%H:%M"),
        "action": action,
        "rsi": round(ctx["rsi"]),
        "adx": round(ctx["stoch_k"]),
        "level": lvl,
        "lvl": lvl,
        "current_price": f"{sim_price:.5f}",
    }


# ─────────────── Public API ───────────────

async def generate_asset_signals(asset: str, count: int = 15, ssid: str = None) -> list[dict]:
    ctx = await _get_market_context(asset)
    if not ctx:
        logger.warning(f"No market context for {asset}")
        return []

    signals = []
    now = datetime.now(cairo_tz)
    entry_time = (now + timedelta(minutes=3)).replace(second=0, microsecond=0)

    slope = (ctx["ema9"] - ctx["ema21"])  # per-candle drift
    sim_price = ctx["price"]
    attempts = 0

    while len(signals) < count and attempts < 300:
        attempts += 1
        sim_price += slope * 0.01  # small per-minute drift

        if not _should_skip(ctx, sim_price):
            signals.append(_build_signal(len(signals) + 1, ctx, entry_time, sim_price))
            entry_time += timedelta(minutes=3)
        else:
            entry_time += timedelta(minutes=1)

    logger.info(f"Generated {len(signals)} signals for {asset}")
    return signals


async def generate_mixed_signals(assets_list: list[str], count: int = 15, ssid: str = None) -> list[dict]:
    # Fetch context for all assets concurrently
    contexts = await asyncio.gather(*[_get_market_context(a) for a in assets_list])
    valid_contexts = [c for c in contexts if c is not None]

    if not valid_contexts:
        logger.error("No valid market contexts available for mixed signals")
        return []

    # Sort by trend strength — strongest trend gets priority
    valid_contexts.sort(key=lambda c: c["trend_strength"], reverse=True)

    signals = []
    now = datetime.now(cairo_tz)
    entry_time = (now + timedelta(minutes=3)).replace(second=0, microsecond=0)

    # Round-robin across contexts, weighted by trend_strength
    ctx_cycle = valid_contexts.copy()
    attempts = 0

    while len(signals) < count and attempts < 300:
        attempts += 1
        # Pick context in round-robin
        ctx = ctx_cycle[len(signals) % len(ctx_cycle)]
        slope = (ctx["ema9"] - ctx["ema21"])
        sim_price = ctx["price"] + slope * 0.01 * attempts

        if not _should_skip(ctx, sim_price):
            signals.append(_build_signal(len(signals) + 1, ctx, entry_time, sim_price))
            entry_time += timedelta(minutes=3)
        else:
            entry_time += timedelta(minutes=1)

    logger.info(f"Generated {len(signals)} mixed signals across {len(valid_contexts)} assets")
    return signals
