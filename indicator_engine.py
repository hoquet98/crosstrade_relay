"""
Server-side indicator enrichment engine.

Pulls 1-min and 5-min bars from ai_bars, computes a full suite of
technical indicators using indicators.py, and returns a standardized
payload dict that any bot can filter and send to the AI.

Any strategy that sends a basic signal (BUY/SELL + instrument) gets
the full indicator suite computed here — no Pine Script needed.
"""

import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

import database as db
import indicators as ind

logger = logging.getLogger("trade_relay")

# Minimum bars needed for reliable indicator computation
MIN_BARS_1M = 200
MIN_BARS_5M = 50


# ══════════════════════════════════════════════════════════════════════════════
# BAR LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_bars(instrument: str, limit: int = 300) -> pd.DataFrame:
    """Load recent 1-min bars from ai_bars into a pandas DataFrame."""
    conn = db.get_connection()
    rows = conn.execute(
        """SELECT timestamp, open, high, low, close, volume
           FROM ai_bars WHERE instrument = ?
           ORDER BY timestamp DESC LIMIT ?""",
        (instrument, limit)
    ).fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    data = []
    for r in rows:
        data.append({
            "time": pd.Timestamp(r["timestamp"]),
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["volume"] or 0.0,
        })

    df = pd.DataFrame(data)
    df = df.set_index("time").sort_index()
    # Filter out bad bars (zeros)
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    return df


def _resample_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-min bars into 5-min bars."""
    if df_1m.empty:
        return pd.DataFrame()
    df_5m = df_1m.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return df_5m


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_indicators(instrument: str) -> dict:
    """Compute the full indicator suite from stored bars.

    Returns a flat dict with all indicator values for the latest bar,
    matching the field names used by the AI Gate payload format.
    Returns empty dict if insufficient data.
    """
    df = _load_bars(instrument, limit=300)
    if len(df) < MIN_BARS_1M:
        logger.warning(f"Insufficient bars for {instrument}: {len(df)} < {MIN_BARS_1M}")
        return {}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    result = {}

    # ── OHLCV (latest bar) ──
    result["open"] = round(close.iloc[-1], 2)  # current bar
    result["high"] = round(high.iloc[-1], 2)
    result["low"] = round(low.iloc[-1], 2)
    result["close"] = round(close.iloc[-1], 2)
    result["price"] = round(close.iloc[-1], 2)
    result["volume"] = float(volume.iloc[-1])
    result["open"] = round(open_.iloc[-1], 2)

    # ── Last 5 closes ──
    if len(close) >= 5:
        result["last_5_closes"] = [round(float(c), 2) for c in close.iloc[-5:].values]

    # ── ZLEMA (period=8) ──
    zlema_val = ind.zlema(close, 8)
    result["zlema"] = round(float(zlema_val.iloc[-1]), 2)
    zl_slope = float(zlema_val.iloc[-1] - zlema_val.iloc[-2]) if len(zlema_val) > 1 else 0
    result["zl_slope"] = round(zl_slope, 4)
    trend_up = zl_slope > 0
    trend_down = zl_slope < 0
    result["f_zlema_trend_bull"] = trend_up
    result["f_zlema_trend_bear"] = trend_down

    # ZLEMA flips
    if len(zlema_val) >= 3:
        prev_slope = float(zlema_val.iloc[-2] - zlema_val.iloc[-3])
        result["zlema_flipped_bull"] = trend_up and prev_slope <= 0
        result["zlema_flipped_bear"] = trend_down and prev_slope >= 0
        result["zlema_went_neutral"] = (not trend_up and not trend_down and
                                         (prev_slope > 0 or prev_slope < 0))
    else:
        result["zlema_flipped_bull"] = False
        result["zlema_flipped_bear"] = False
        result["zlema_went_neutral"] = False

    # ── WaveTrend ──
    wt1, wt2 = ind.wavetrend(high, low, close, channel_len=6, avg_len=9, ma_len=2)
    result["wt1"] = round(float(wt1.iloc[-1]), 2) if not np.isnan(wt1.iloc[-1]) else 0
    result["wt2"] = round(float(wt2.iloc[-1]), 2) if not np.isnan(wt2.iloc[-1]) else 0
    result["wt_bull_cross"] = bool(ind.crossover(wt1, wt2).iloc[-1])
    result["wt_bear_cross"] = bool(ind.crossunder(wt1, wt2).iloc[-1])

    # WT momentum flags
    wt_ob = 50.0
    wt_ob_short = 55.0
    mom_bull = result["wt1"] < wt_ob  # not overbought
    result["f_wt_mom_bull"] = mom_bull  # simplified — full version checks sq_mom too
    result["f_wt_mom_bear"] = result["wt1"] > wt_ob_short

    # ── Squeeze Momentum ──
    sq_mom, sq_on = ind.squeeze_momentum(high, low, close,
                                          bb_period=20, bb_mult=2.0,
                                          kc_period=20, kc_mult=1.5)
    result["sq_mom"] = round(float(sq_mom.iloc[-1]), 4) if not np.isnan(sq_mom.iloc[-1]) else 0
    result["squeeze_on"] = bool(sq_on.iloc[-1])
    result["f_squeeze_off"] = not bool(sq_on.iloc[-1])

    # Momentum flips
    if len(sq_mom) >= 2:
        mom_now = float(sq_mom.iloc[-1])
        mom_prev = float(sq_mom.iloc[-2])
        result["mom_flipped_bull"] = mom_prev < 0 and mom_now > 0
        result["mom_flipped_bear"] = mom_prev > 0 and mom_now < 0
    else:
        result["mom_flipped_bull"] = False
        result["mom_flipped_bear"] = False

    # Update f_wt_mom flags with squeeze momentum
    result["f_wt_mom_bull"] = result["f_wt_mom_bull"] and result["sq_mom"] > 0
    result["f_wt_mom_bear"] = result["f_wt_mom_bear"] and result["sq_mom"] < 0

    # ── Efficiency Ratio ──
    er_val = ind.efficiency_ratio(close, 8)
    result["er_val"] = round(float(er_val.iloc[-1]), 4) if not np.isnan(er_val.iloc[-1]) else 0
    result["f_trending"] = result["er_val"] > 0.2

    # ── UT Bot ──
    ut_trail, ut_buy, ut_sell = ind.ut_bot(close, high, low, key_value=0.4, atr_period=5)
    result["ut_trail"] = round(float(ut_trail.iloc[-1]), 2) if not np.isnan(ut_trail.iloc[-1]) else 0
    result["ut_flipped_bull"] = bool(ut_buy.iloc[-1]) if not np.isnan(ut_buy.iloc[-1]) else False
    result["ut_flipped_bear"] = bool(ut_sell.iloc[-1]) if not np.isnan(ut_sell.iloc[-1]) else False

    # ── EMAs ──
    ema20 = ind.ema(close, 20)
    ema50 = ind.ema(close, 50)
    result["ema20"] = round(float(ema20.iloc[-1]), 2) if not np.isnan(ema20.iloc[-1]) else 0
    result["ema50"] = round(float(ema50.iloc[-1]), 2) if not np.isnan(ema50.iloc[-1]) else 0
    result["price_above_ema20"] = float(close.iloc[-1]) > result["ema20"]
    result["price_above_ema50"] = float(close.iloc[-1]) > result["ema50"]

    # EMA20 slope
    if len(ema20) >= 6:
        result["ema20_slope5"] = round(float(ema20.iloc[-1] - ema20.iloc[-6]), 4)
    else:
        result["ema20_slope5"] = 0

    # ── HTF Bias (from 5-min bars) ──
    df_5m = _resample_to_5m(df)
    if len(df_5m) >= 50:
        htf_ema20 = ind.ema(df_5m["close"], 20)
        htf_ema50 = ind.ema(df_5m["close"], 50)
        if not np.isnan(htf_ema20.iloc[-1]) and not np.isnan(htf_ema50.iloc[-1]):
            if float(htf_ema20.iloc[-1]) > float(htf_ema50.iloc[-1]):
                result["htf_bias"] = "bullish"
            elif float(htf_ema20.iloc[-1]) < float(htf_ema50.iloc[-1]):
                result["htf_bias"] = "bearish"
            else:
                result["htf_bias"] = "neutral"
        else:
            result["htf_bias"] = "neutral"
    else:
        result["htf_bias"] = "neutral"

    # ── RSI ──
    rsi14 = ind.rsi(close, 14)
    result["rsi14"] = round(float(rsi14.iloc[-1]), 2) if not np.isnan(rsi14.iloc[-1]) else 50

    # RSI divergence (simplified)
    if len(low) >= 10 and len(rsi14) >= 10:
        price_low_10 = float(low.iloc[-10:].min())
        price_high_10 = float(high.iloc[-10:].max())
        rsi_low_10 = float(rsi14.iloc[-10:].min())
        rsi_high_10 = float(rsi14.iloc[-10:].max())
        result["rsi_bull_div"] = float(low.iloc[-1]) <= price_low_10 and float(rsi14.iloc[-1]) > rsi_low_10
        result["rsi_bear_div"] = float(high.iloc[-1]) >= price_high_10 and float(rsi14.iloc[-1]) < rsi_high_10
    else:
        result["rsi_bull_div"] = False
        result["rsi_bear_div"] = False

    # ── CCI ──
    cci20 = ind.cci(close, 20)
    result["cci20"] = round(float(cci20.iloc[-1]), 2) if not np.isnan(cci20.iloc[-1]) else 0

    # ── Stoch RSI ──
    stoch_k, stoch_d = ind.stochastic_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3)
    result["stoch_rsi_k"] = round(float(stoch_k.iloc[-1]), 2) if not np.isnan(stoch_k.iloc[-1]) else 50
    result["stoch_rsi_d"] = round(float(stoch_d.iloc[-1]), 2) if not np.isnan(stoch_d.iloc[-1]) else 50
    result["stoch_in_ob"] = result["stoch_rsi_k"] > 80
    result["stoch_in_os"] = result["stoch_rsi_k"] < 20
    result["stoch_bull_cross"] = bool(ind.crossover(stoch_k, stoch_d).iloc[-1])
    result["stoch_bear_cross"] = bool(ind.crossunder(stoch_k, stoch_d).iloc[-1])

    # ── ATR ──
    atr14 = ind.atr(high, low, close, 14)
    result["atr14"] = round(float(atr14.iloc[-1]), 4) if not np.isnan(atr14.iloc[-1]) else 0

    # ATR consumed percentage
    # Find session open (first bar of the day)
    today = df.index[-1].normalize()
    today_bars = df[df.index >= today]
    if len(today_bars) > 0:
        session_open = float(today_bars["open"].iloc[0])
        session_move = abs(float(close.iloc[-1]) - session_open)
        result["atr_consumed_pct"] = round(session_move / result["atr14"], 4) if result["atr14"] > 0 else 0
    else:
        result["atr_consumed_pct"] = 0

    # ── Bollinger %B ──
    bb_upper, bb_mid, bb_lower = ind.bollinger_bands(close, 20, 2.0)
    bb_range = float(bb_upper.iloc[-1] - bb_lower.iloc[-1]) if not np.isnan(bb_upper.iloc[-1]) else 1
    result["bb_pctb"] = round((float(close.iloc[-1]) - float(bb_lower.iloc[-1])) / bb_range, 4) if bb_range > 0 else 0.5

    # ── Bar structure ──
    bar_range = float(high.iloc[-1] - low.iloc[-1])
    bar_body = abs(float(close.iloc[-1] - open_.iloc[-1]))
    result["bar_body_pct"] = round(bar_body / bar_range, 4) if bar_range > 0 else 0
    result["bar_upper_wick_pct"] = round((float(high.iloc[-1]) - max(float(close.iloc[-1]), float(open_.iloc[-1]))) / bar_range, 4) if bar_range > 0 else 0
    result["bar_lower_wick_pct"] = round((min(float(close.iloc[-1]), float(open_.iloc[-1])) - float(low.iloc[-1])) / bar_range, 4) if bar_range > 0 else 0
    result["bar_is_bullish"] = float(close.iloc[-1]) > float(open_.iloc[-1])
    result["bar_range_vs_atr"] = round(bar_range / result["atr14"], 4) if result["atr14"] > 0 else 1

    # ── Volume ratio ──
    vol_avg_20 = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else 1
    result["vol_ratio"] = round(float(volume.iloc[-1]) / vol_avg_20, 2) if vol_avg_20 > 0 else 1

    # ── Previous 3-bar pattern ──
    if len(close) >= 4:
        p = ""
        p += "H" if float(close.iloc[-3]) >= float(close.iloc[-4]) else "L"
        p += "H" if float(close.iloc[-2]) >= float(close.iloc[-3]) else "L"
        p += "H" if float(close.iloc[-1]) >= float(close.iloc[-2]) else "L"
        result["prev_3_pattern"] = p
    else:
        result["prev_3_pattern"] = "---"

    # ── Session high/low distance ──
    if len(today_bars) > 0:
        sess_high = float(today_bars["high"].max())
        sess_low = float(today_bars["low"].min())
        tick_size = 0.25  # default for NQ/MNQ
        result["dist_sess_high_ticks"] = round((float(close.iloc[-1]) - sess_high) / tick_size, 1)
        result["dist_sess_low_ticks"] = round((float(close.iloc[-1]) - sess_low) / tick_size, 1)
    else:
        result["dist_sess_high_ticks"] = 0
        result["dist_sess_low_ticks"] = 0

    # ── Swing high/low bars since ──
    bars_since_swing_high = 0
    bars_since_swing_low = 0
    for i in range(2, min(50, len(high))):
        idx = -i
        if float(high.iloc[idx]) == float(high.iloc[-min(50, len(high)):].max()):
            bars_since_swing_high = i - 1
            break
    for i in range(2, min(50, len(low))):
        idx = -i
        if float(low.iloc[idx]) == float(low.iloc[-min(50, len(low)):].min()):
            bars_since_swing_low = i - 1
            break
    result["bars_since_swing_high"] = bars_since_swing_high
    result["bars_since_swing_low"] = bars_since_swing_low

    # ── Consolidation range ──
    if len(high) >= 10:
        cons_high = float(high.iloc[-10:].max())
        cons_low = float(low.iloc[-10:].min())
        tick_size = 0.25
        result["cons_range_ticks"] = round((cons_high - cons_low) / tick_size, 1)
        result["cons_range_vs_atr"] = round((cons_high - cons_low) / result["atr14"], 4) if result["atr14"] > 0 else 1
    else:
        result["cons_range_ticks"] = 0
        result["cons_range_vs_atr"] = 0

    # ── Confluence scoring ──
    # Layer 1: ZLEMA trend
    # Layer 2: WT + Momentum
    # Layer 3: Squeeze off
    # Layer 4: Trending (ER)
    bull_count = sum([
        result["f_zlema_trend_bull"],
        result["f_wt_mom_bull"],
        result["f_squeeze_off"],
        result["f_trending"],
    ])
    bear_count = sum([
        result["f_zlema_trend_bear"],
        result["f_wt_mom_bear"],
        result["f_squeeze_off"],
        result["f_trending"],
    ])
    result["bull_confluence"] = bull_count
    result["bear_confluence"] = bear_count

    # ── Signal generation (UT Bot cross) ──
    result["long_signal"] = result["ut_flipped_bull"] and bull_count >= 2
    result["short_signal"] = result["ut_flipped_bear"] and bear_count >= 2

    # ── Exit warnings ──
    result["long_exit_warn"] = result["wt1"] > wt_ob
    result["short_exit_warn"] = result["wt1"] < -45.0

    # ── Session timing ──
    now = datetime.now(timezone(timedelta(hours=-5)))  # ET
    hhmm = now.hour * 100 + now.minute
    if hhmm < 1000:
        result["session_bucket"] = "open_drive"
    elif hhmm < 1130:
        result["session_bucket"] = "morning"
    elif hhmm < 1330:
        result["session_bucket"] = "midday_chop"
    elif hhmm < 1550:
        result["session_bucket"] = "afternoon"
    else:
        result["session_bucket"] = "close"

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    result["day_of_week"] = dow_names[now.weekday()]

    maint_start = 17 * 60  # 5pm ET in minutes
    current_mins = now.hour * 60 + now.minute
    result["mins_to_maintenance"] = max(0, maint_start - current_mins)

    # ── Bar time ──
    result["bar_time_et"] = f"{now.hour:02d}:{now.minute:02d} ET"
    result["bar_time_hhmm"] = hhmm

    return result


def enrich_payload(payload: dict, instrument: str) -> dict:
    """Enrich a basic signal payload with server-computed indicators.

    If the payload already has indicators (from Pine Script), they are
    preserved. Server-computed indicators fill in any missing fields.
    """
    computed = compute_all_indicators(instrument)
    if not computed:
        return payload

    # Server-computed values are defaults — Pine Script values override
    merged = {**computed, **{k: v for k, v in payload.items() if v is not None}}
    return merged
