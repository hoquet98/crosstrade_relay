"""
QTP201 Super RSI Scalper [Elite] v1.5.0 — Single-file strategy module.

Converted from Pine Script to Python/VectorBT Pro.

Contains everything needed to backtest the QTP201 strategy:
- QTP201Params dataclass (all strategy parameters, independent long/short sides)
- OMA indicator (Jurik-style triple-smoothing with optional adaptive period)
- Indicator computation (Supertrend, RSI, CCI, ATR volatility filter, OMA)
- Signal detection — Numba compiled
  • Long:  Supertrend bullish flip + RSI in range + OMA bullish
  • Short: CCI extreme pullback (crosses below exit level) + RSI above threshold + OMA bearish
- Trade simulation — Numba compiled
  • Independent ATM per side (SL/TP/trail/BE with 1-bar delay, duration BE, max bars exit)
  • Pyramiding = 0 (only one position at a time, either side)
  • Daily loss limit, lockout bars (per side), session/exclude windows
- Parameter mapping and Strategy class interface

Standard interface for generic run_study.py:
  Strategy.PARAMS_CLASS — the dataclass
  Strategy.map_params(full_params) — DB dict → dataclass kwargs
  Strategy.compute_indicators(data, params) — pre-compute once per window
  Strategy.run_combo(data, indicators, params, w_start, w_end, w_type) — run one combo
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from numba import njit

from strategies.window_result import WindowResult

# Order record dtype — matches vectorbtpro's order_dt but avoids the heavy import
order_dt = np.dtype([
    ('id', np.int64), ('col', np.int64), ('idx', np.int64),
    ('size', np.float64), ('price', np.float64), ('fees', np.float64), ('side', np.int64),
])

from strategies.window_result import WindowResult


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QTP201Params:
    """All strategy parameters. Defaults match Pine Script v1.5.0 optimised values."""

    # ── Global ──
    allow_longs: bool = True
    allow_shorts: bool = True

    # ── Volatility Filter ──
    vol_filter_enable: bool = False
    vol_atr_period: int = 14
    vol_atr_mult: float = 2.0

    # ── OMA (shared) ──
    oma_enable: bool = True
    oma_len: int = 10
    oma_speed: float = 2.5
    oma_close_period: int = 1
    oma_adaptive: bool = True

    # ── Global Exit Management ──
    use_max_bars: bool = False
    max_bars_in_trade: int = 60
    use_dur_be: bool = False
    dur_be_bars: int = 10
    dur_be_offset: int = 5
    use_be_confirm: bool = False

    # ── Daily Loss Limit ──
    dlc_enable: bool = False
    dlc_ticks: int = 500

    # ── Long — Signal ──
    l_st_period: int = 14
    l_st_mult: float = 3.0
    l_rsi_period: int = 14
    l_rsi_max: float = 60.0
    l_lock_bars: int = 0
    l_lock_on_exit: bool = False

    # ── Long — Session ──
    l_use_session: bool = False
    l_session_tz: str = "America/New_York"
    l_session_hours: str = "0930-1600"
    l_use_exclude1: bool = False
    l_exclude_hours1: str = "1545-1600"
    l_use_exclude2: bool = False
    l_exclude_hours2: str = "0100-0400"
    l_use_exclude3: bool = False
    l_exclude_hours3: str = "1200-1300"
    l_use_exclude4: bool = False
    l_exclude_hours4: str = "0800-0930"
    l_use_cme_maint: bool = True

    # ── Long — Day Filter ──
    l_use_day_filter: bool = False
    l_day_sun: bool = False
    l_day_mon: bool = True
    l_day_tue: bool = True
    l_day_wed: bool = True
    l_day_thu: bool = True
    l_day_fri: bool = True
    l_day_sat: bool = False

    # ── Long — ATM (ticks) ──
    l_stop_ticks: int = 225
    l_target_ticks: int = 150
    l_use_trail: bool = True
    l_trail_trigger: int = 90
    l_trail_offset: int = 15
    l_use_be: bool = True
    l_be_trigger: int = 80
    l_be_offset: int = 20

    # ── Short — Signal ──
    s_cci_period: int = 30
    s_cci_extreme: float = 130.0
    s_cci_exit: float = 50.0
    s_rsi_period: int = 12
    s_rsi_min: float = 49.0
    s_lock_bars: int = 10
    s_cci_lookback: int = 5
    s_lock_on_exit: bool = False

    # ── Short — Session ──
    s_use_session: bool = False
    s_session_tz: str = "America/New_York"
    s_session_hours: str = "0930-1600"
    s_use_exclude1: bool = False
    s_exclude_hours1: str = "1545-1600"
    s_use_exclude2: bool = False
    s_exclude_hours2: str = "0100-0400"
    s_use_exclude3: bool = False
    s_exclude_hours3: str = "1200-1300"
    s_use_exclude4: bool = False
    s_exclude_hours4: str = "0800-0930"
    s_use_cme_maint: bool = True

    # ── Short — Day Filter ──
    s_use_day_filter: bool = False
    s_day_sun: bool = False
    s_day_mon: bool = True
    s_day_tue: bool = True
    s_day_wed: bool = True
    s_day_thu: bool = True
    s_day_fri: bool = True
    s_day_sat: bool = False

    # ── Short — ATM (ticks) ──
    s_stop_ticks: int = 230
    s_target_ticks: int = 130
    s_use_trail: bool = True
    s_trail_trigger: int = 100
    s_trail_offset: int = 1
    s_use_be: bool = True
    s_be_trigger: int = 95
    s_be_offset: int = 5

    # ── Futures config ──
    tick_size: float = 0.10       # GC=0.10, ES=0.25
    point_value: float = 100.0    # GC=100, ES=50
    commission: float = 5.0
    slippage: int = 2             # ticks
    initial_capital: float = 50000.0


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _sma(arr, period):
    """Simple Moving Average."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = np.mean(arr[i - period + 1:i + 1])
    return out


def _ema(arr, period):
    """Exponential Moving Average — TV-compatible (SMA seed)."""
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = np.mean(arr[:period])
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _rma(arr, period):
    """Wilder's Moving Average (RMA) — TV-compatible (SMA seed, alpha = 1/period)."""
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = np.mean(arr[:period])
    alpha = 1.0 / period
    for i in range(period, n):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _wilder_atr(high, low, close, period):
    """Wilder's ATR (RMA smoothing) — matches TradingView ta.atr()."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    atr_out = np.full(n, np.nan)
    if n < period:
        return atr_out

    # Seed with SMA
    atr_out[period - 1] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period, n):
        atr_out[i] = alpha * tr[i] + (1 - alpha) * atr_out[i - 1]
    return atr_out


def _rsi(close, period):
    """RSI — TV-compatible (RMA smoothing for up/down)."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    # Seed with SMA over first `period` changes
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss == 0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    alpha = 1.0 / period
    for i in range(period, len(gain)):
        avg_gain = alpha * gain[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * loss[i] + (1 - alpha) * avg_loss
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            out[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def _cci(close, period):
    """CCI — TV-compatible: CCI = (src - sma(src)) / (0.015 * meandev).

    Pine script uses ta.cci(close, period) — source is close, not hlc3.
    """
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean_val = np.mean(window)
        mean_dev = np.mean(np.abs(window - mean_val))
        if mean_dev == 0:
            out[i] = 0.0
        else:
            out[i] = (close[i] - mean_val) / (0.015 * mean_dev)
    return out


def _supertrend(high, low, close, atr_period, factor):
    """Supertrend — TV-compatible.

    Returns (st_line, st_dir) where:
      st_dir = -1 means bullish (price above ST), +1 means bearish (below ST).
      (Pine convention: -1 = bullish, +1 = bearish)
    """
    n = len(close)
    atr = _wilder_atr(high, low, close, atr_period)

    hl2 = (high + low) / 2.0
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    st_dir = np.ones(n, dtype=np.int32)  # +1 = bearish
    st_line = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(atr[i]):
            continue

        basic_upper = hl2[i] + factor * atr[i]
        basic_lower = hl2[i] - factor * atr[i]

        # Ratchet upper band down (only tighten, never widen)
        if i > 0 and not np.isnan(upper_band[i - 1]):
            upper_band[i] = min(basic_upper, upper_band[i - 1]) if close[i - 1] <= upper_band[i - 1] else basic_upper
        else:
            upper_band[i] = basic_upper

        # Ratchet lower band up (only tighten)
        if i > 0 and not np.isnan(lower_band[i - 1]):
            lower_band[i] = max(basic_lower, lower_band[i - 1]) if close[i - 1] >= lower_band[i - 1] else basic_lower
        else:
            lower_band[i] = basic_lower

        # Direction logic
        if i == 0 or np.isnan(st_line[i - 1]):
            st_dir[i] = -1 if close[i] > upper_band[i] else 1
        else:
            prev_dir = st_dir[i - 1]
            if prev_dir == 1:  # was bearish
                if close[i] > upper_band[i]:
                    st_dir[i] = -1  # flip to bullish
                else:
                    st_dir[i] = 1
            else:  # was bullish (-1)
                if close[i] < lower_band[i]:
                    st_dir[i] = 1  # flip to bearish
                else:
                    st_dir[i] = -1

        st_line[i] = lower_band[i] if st_dir[i] == -1 else upper_band[i]

    return st_line, st_dir


def _oma(src, length, speed, adaptive):
    """OMA — Jurik-style triple-smoothing with optional adaptive period.

    Matches the Pine Script _oma() function exactly:
      Three cascaded Jurik EMA passes producing v1, v2, then final = 1.5*e5 - 0.5*e6.
    """
    n = len(src)
    out = np.full(n, np.nan)

    e1 = src[0] if n > 0 else 0.0
    e2 = e1
    e3 = e1
    e4 = e1
    e5 = e1
    e6 = e1

    for i in range(n):
        if np.isnan(src[i]):
            continue

        avg_period = length
        noise = 1e-11
        min_per = avg_period / 2.0
        max_per = min_per * 5.0
        end_per = int(np.ceil(max_per))

        if adaptive:
            # Signal: |src[i] - src[i - endPer]|
            ref_idx = max(0, i - end_per)
            sig = abs(src[i] - src[ref_idx])

            # Noise: sum of |src[i] - src[i-k]| for k=1..endPer
            noise_sum = 0.0
            for k in range(1, end_per + 1):
                prev_idx = max(0, i - k)
                noise_sum += abs(src[i] - src[prev_idx])
            if noise_sum > 0:
                noise = noise_sum
            avg_period = int(np.ceil((sig / noise) * (max_per - min_per) + min_per))

        alpha = (2.0 + speed) / (1.0 + speed + avg_period)

        # Three cascaded Jurik EMA passes
        e1 = e1 + alpha * (src[i] - e1)
        e2 = e2 + alpha * (e1 - e2)
        v1 = 1.5 * e1 - 0.5 * e2

        e3 = e3 + alpha * (v1 - e3)
        e4 = e4 + alpha * (e3 - e4)
        v2 = 1.5 * e3 - 0.5 * e4

        e5 = e5 + alpha * (v2 - e5)
        e6 = e6 + alpha * (e5 - e6)

        out[i] = 1.5 * e5 - 0.5 * e6

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_indicators(data: pd.DataFrame, params: QTP201Params):
    """Pre-compute all indicators from OHLCV data.

    Returns a dict of numpy arrays (1D, same length as data).
    """
    close = data['close'].values.astype(np.float64)
    high = data['high'].values.astype(np.float64)
    low = data['low'].values.astype(np.float64)
    open_ = data['open'].values.astype(np.float64)
    n = len(close)

    # ── OMA (shared) ──
    oma_high = _oma(high, params.oma_len, params.oma_speed, params.oma_adaptive)
    oma_low = _oma(low, params.oma_len, params.oma_speed, params.oma_adaptive)
    oma_close = _oma(close, params.oma_close_period, params.oma_speed, params.oma_adaptive)

    # JFGHLA: tracks OMA regime flip level
    jfghla = np.full(n, np.nan)
    oma_trend = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if np.isnan(oma_close[i]):
            continue

        prev_high = oma_high[i - 1] if i > 0 and not np.isnan(oma_high[i - 1]) else np.nan
        prev_low = oma_low[i - 1] if i > 0 and not np.isnan(oma_low[i - 1]) else np.nan

        if not np.isnan(prev_high) and oma_close[i] > prev_high:
            jfghla[i] = oma_low[i]
        elif not np.isnan(prev_low) and oma_close[i] < prev_low:
            jfghla[i] = oma_high[i]
        elif i > 0 and not np.isnan(jfghla[i - 1]):
            jfghla[i] = jfghla[i - 1]
        else:
            jfghla[i] = oma_high[i]  # default

        if oma_close[i] > jfghla[i]:
            oma_trend[i] = 1   # bullish
        elif oma_close[i] < jfghla[i]:
            oma_trend[i] = -1  # bearish
        else:
            oma_trend[i] = 0

    # ── Supertrend (long side) ──
    l_st_line, l_st_dir = _supertrend(high, low, close, params.l_st_period, params.l_st_mult)

    # ── RSI (long side) ──
    l_rsi = _rsi(close, params.l_rsi_period)

    # ── CCI (short side) ──
    s_cci = _cci(close, params.s_cci_period)

    # ── RSI (short side) ──
    s_rsi = _rsi(close, params.s_rsi_period)

    # ── ATR volatility filter ──
    vol_atr = _wilder_atr(high, low, close, params.vol_atr_period)
    vol_atr_avg = _sma(vol_atr, 20)
    vol_ok = np.ones(n, dtype=np.bool_)
    if params.vol_filter_enable:
        for i in range(n):
            if not np.isnan(vol_atr[i]) and not np.isnan(vol_atr_avg[i]):
                if vol_atr[i] > params.vol_atr_mult * vol_atr_avg[i]:
                    vol_ok[i] = False

    return {
        'close': close, 'high': high, 'low': low, 'open': open_,
        'oma_trend': oma_trend,
        'l_st_dir': l_st_dir,
        'l_rsi': l_rsi,
        's_cci': s_cci,
        's_rsi': s_rsi,
        'vol_ok': vol_ok,
        'n': n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MASKS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_session(s):
    """Parse "HHMM-HHMM" to (start_minutes, end_minutes)."""
    s = s.replace(":", "")
    parts = s.split('-')
    start = int(parts[0][:2]) * 60 + int(parts[0][2:])
    end = int(parts[1][:2]) * 60 + int(parts[1][2:])
    return start, end


def _in_session(minutes, session_str):
    """Check if each minute-of-day falls within a session window."""
    s, e = _parse_session(session_str)
    if s < e:
        return (minutes >= s) & (minutes < e)
    else:
        return (minutes >= s) | (minutes < e)


def _get_tz_minutes_and_dow(index, tz_str):
    """Convert index to minutes-of-day and day-of-week in the specified timezone."""
    from zoneinfo import ZoneInfo
    if tz_str == "Exchange":
        tz_str = "America/New_York"  # default for futures
    try:
        localized = index.tz_convert(ZoneInfo(tz_str))
    except Exception:
        localized = index
    minutes = localized.hour * 60 + localized.minute
    dow = localized.dayofweek  # Mon=0, Sun=6
    return minutes, dow


def compute_session_masks(index, params: QTP201Params):
    """Compute per-side session/exclude/day masks.

    Returns (l_session_ok, s_session_ok) — bool arrays, True = can trade.
    """
    from zoneinfo import ZoneInfo
    n = len(index)

    # ── Long side ──
    l_ok = np.ones(n, dtype=np.bool_)
    l_minutes, l_dow = _get_tz_minutes_and_dow(index, params.l_session_tz)

    if params.l_use_session:
        l_ok &= _in_session(l_minutes, params.l_session_hours)

    if params.l_use_exclude1:
        l_ok &= ~_in_session(l_minutes, params.l_exclude_hours1)
    if params.l_use_exclude2:
        l_ok &= ~_in_session(l_minutes, params.l_exclude_hours2)
    if params.l_use_exclude3:
        l_ok &= ~_in_session(l_minutes, params.l_exclude_hours3)
    if params.l_use_exclude4:
        l_ok &= ~_in_session(l_minutes, params.l_exclude_hours4)

    if params.l_use_cme_maint:
        # CME halt: 16:00-17:00 Chicago time
        chi_minutes, _ = _get_tz_minutes_and_dow(index, "America/Chicago")
        l_ok &= ~_in_session(chi_minutes, "1600-1700")

    if params.l_use_day_filter:
        # Python: Mon=0..Sun=6
        day_map = np.array([
            params.l_day_mon, params.l_day_tue, params.l_day_wed,
            params.l_day_thu, params.l_day_fri, params.l_day_sat, params.l_day_sun
        ], dtype=np.bool_)
        l_day_ok = day_map[l_dow]
        l_ok &= l_day_ok

    # ── Short side ──
    s_ok = np.ones(n, dtype=np.bool_)
    s_minutes, s_dow = _get_tz_minutes_and_dow(index, params.s_session_tz)

    if params.s_use_session:
        s_ok &= _in_session(s_minutes, params.s_session_hours)

    if params.s_use_exclude1:
        s_ok &= ~_in_session(s_minutes, params.s_exclude_hours1)
    if params.s_use_exclude2:
        s_ok &= ~_in_session(s_minutes, params.s_exclude_hours2)
    if params.s_use_exclude3:
        s_ok &= ~_in_session(s_minutes, params.s_exclude_hours3)
    if params.s_use_exclude4:
        s_ok &= ~_in_session(s_minutes, params.s_exclude_hours4)

    if params.s_use_cme_maint:
        chi_minutes, _ = _get_tz_minutes_and_dow(index, "America/Chicago")
        s_ok &= ~_in_session(chi_minutes, "1600-1700")

    if params.s_use_day_filter:
        day_map = np.array([
            params.s_day_mon, params.s_day_tue, params.s_day_wed,
            params.s_day_thu, params.s_day_fri, params.s_day_sat, params.s_day_sun
        ], dtype=np.bool_)
        s_day_ok = day_map[s_dow]
        s_ok &= s_day_ok

    return l_ok, s_ok


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL DETECTION (Numba compiled)
# ═══════════════════════════════════════════════════════════════════════════════

@njit
def detect_signals(
    close, high, low, n,
    l_st_dir, l_rsi, oma_trend,
    s_cci, s_rsi,
    vol_ok, l_session_ok, s_session_ok,
    # Signal params
    oma_enable,
    l_rsi_max,
    s_cci_extreme, s_cci_exit, s_rsi_min, s_cci_lookback,
    allow_longs, allow_shorts,
):
    """Detect raw long/short signals (before lockout/position gating).

    Long signal:  Supertrend flips bullish + RSI in [45, l_rsi_max] + OMA bullish
    Short signal: CCI was > extreme within lookback, now crosses below exit + RSI > min + OMA bearish

    Returns (long_sig, short_sig) bool arrays.
    """
    long_sig = np.zeros(n, dtype=np.bool_)
    short_sig = np.zeros(n, dtype=np.bool_)

    for i in range(1, n):
        # ── Long: Supertrend bullish flip ──
        if allow_longs and l_session_ok[i] and vol_ok[i]:
            # Flip: dir was +1 (bearish), now -1 (bullish)
            st_flip_bull = (l_st_dir[i] == -1) and (l_st_dir[i - 1] == 1)

            rsi_ok = (not np.isnan(l_rsi[i])) and (l_rsi[i] > 45.0) and (l_rsi[i] < l_rsi_max)
            oma_long_ok = (not oma_enable) or (oma_trend[i] == 1)

            if st_flip_bull and rsi_ok and oma_long_ok:
                long_sig[i] = True

        # ── Short: CCI extreme pullback ──
        if allow_shorts and s_session_ok[i] and vol_ok[i]:
            # Was CCI above extreme within last N bars?
            was_extreme = False
            for k in range(1, s_cci_lookback + 1):
                if i - k >= 0 and not np.isnan(s_cci[i - k]):
                    if s_cci[i - k] > s_cci_extreme:
                        was_extreme = True
                        break

            # CCI crosses below exit level (current < exit, previous >= exit)
            now_below = (not np.isnan(s_cci[i])) and (s_cci[i] < s_cci_exit)
            prev_above = (not np.isnan(s_cci[i - 1])) and (s_cci[i - 1] >= s_cci_exit)

            rsi_ok = (not np.isnan(s_rsi[i])) and (s_rsi[i] > s_rsi_min)
            oma_short_ok = (not oma_enable) or (oma_trend[i] == -1)

            if was_extreme and now_below and prev_above and rsi_ok and oma_short_ok:
                short_sig[i] = True

    return long_sig, short_sig


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL WRAPPER (for VBT adapter — unpacks dict args for Numba function)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_signals_from_dict(indicators, session, params):
    """Unpack indicator dict and session tuple, call Numba detect_signals().

    This is the bridge between the VBT adapter (which passes dicts) and the
    Numba-compiled detect_signals() which needs individual arrays.
    """
    l_session_ok, s_session_ok = session
    return detect_signals(
        indicators['close'], indicators['high'], indicators['low'], indicators['n'],
        indicators['l_st_dir'], indicators['l_rsi'], indicators['oma_trend'],
        indicators['s_cci'], indicators['s_rsi'],
        indicators['vol_ok'], l_session_ok, s_session_ok,
        params.oma_enable,
        params.l_rsi_max,
        params.s_cci_extreme, params.s_cci_exit, params.s_rsi_min, params.s_cci_lookback,
        params.allow_longs, params.allow_shorts,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE SIMULATION (Numba compiled)
# ═══════════════════════════════════════════════════════════════════════════════

@njit
def run_simulation(
    open_, high, low, close, n,
    long_sig, short_sig,
    tick, point_value, commission, slippage_ticks, init_capital,
    # Long ATM
    l_stop_ticks, l_target_ticks,
    l_use_trail, l_trail_trigger, l_trail_offset,
    l_use_be, l_be_trigger, l_be_offset,
    l_lock_bars, l_lock_on_exit,
    # Short ATM
    s_stop_ticks, s_target_ticks,
    s_use_trail, s_trail_trigger, s_trail_offset,
    s_use_be, s_be_trigger, s_be_offset,
    s_lock_bars, s_lock_on_exit,
    # Global exit
    use_max_bars, max_bars_in_trade,
    use_dur_be, dur_be_bars, dur_be_offset,
    use_be_confirm,
    # DLC
    dlc_enable, dlc_ticks,
    # Bars per day (for DLC day boundary)
    bars_per_day,
):
    """Run the full simulation with independent long/short ATM management.

    Pyramiding = 0: only one position at a time (either side).
    Returns (order_records, order_count).
    """
    max_orders = n * 2
    order_records = np.empty(max_orders, dtype=order_dt)
    order_count = 0

    # Position state
    position = 0.0       # >0 long, <0 short, 0 flat
    entry_price = 0.0
    bars_in_trade = 0

    # ATM state
    sl_price = 0.0
    tp_price = 0.0
    trail_active = False
    trail_peak = 0.0
    be_triggered = False

    # BE confirmation delay (1-bar)
    be_cond_prev = False

    # Lockout counters (per side)
    l_lock_counter = 999
    s_lock_counter = 999

    # DLC state
    daily_pnl = 0.0
    last_day = -1
    dlc_blocked = False

    # Cash tracking
    cash = init_capital

    # Slippage in price
    slip_price = slippage_ticks * tick

    for i in range(n):
        # ── Day boundary reset for DLC ──
        current_day = i // bars_per_day
        if current_day != last_day:
            daily_pnl = 0.0
            dlc_blocked = False
            last_day = current_day

        # ── Increment lockout counters ──
        l_lock_counter += 1
        s_lock_counter += 1

        # ── Exit checks (before entries, using current bar OHLC) ──
        if position != 0.0:
            is_long = position > 0
            exited = False
            exit_price = 0.0
            bars_in_trade += 1

            # Max bars exit
            if use_max_bars and bars_in_trade >= max_bars_in_trade:
                exit_price = close[i]
                exited = True

            # Check SL
            if not exited:
                if is_long and low[i] <= sl_price:
                    exit_price = max(sl_price, low[i])
                    # Apply slippage (adverse)
                    exit_price -= slip_price
                    exited = True
                elif not is_long and high[i] >= sl_price:
                    exit_price = min(sl_price, high[i])
                    exit_price += slip_price
                    exited = True

            # Check TP (only if SL didn't fire)
            if not exited:
                if is_long and high[i] >= tp_price:
                    exit_price = min(tp_price, high[i])
                    exit_price -= slip_price  # conservative TP fill
                    exited = True
                elif not is_long and low[i] <= tp_price:
                    exit_price = max(tp_price, low[i])
                    exit_price += slip_price
                    exited = True

            # Check Trail
            if not exited and trail_active:
                if is_long:
                    trail_peak = max(trail_peak, high[i])
                    trail_sl = trail_peak - (l_trail_offset if is_long else s_trail_offset) * tick
                    if low[i] <= trail_sl:
                        exit_price = max(trail_sl, low[i])
                        exit_price -= slip_price
                        exited = True
                else:
                    trail_peak = min(trail_peak, low[i])
                    trail_sl = trail_peak + s_trail_offset * tick
                    if high[i] >= trail_sl:
                        exit_price = min(trail_sl, high[i])
                        exit_price += slip_price
                        exited = True

            # ── Update ATM triggers (set pending for next bar — 1-bar delay) ──
            if not exited:
                # Compute MFE in ticks
                if is_long:
                    mfe_ticks = (high[i] - entry_price) / tick
                else:
                    mfe_ticks = (entry_price - low[i]) / tick

                # BE trigger
                be_met = False
                be_trigger_val = l_be_trigger if is_long else s_be_trigger
                be_offset_val = l_be_offset if is_long else s_be_offset
                use_be = l_use_be if is_long else s_use_be

                if use_be and not be_triggered:
                    be_met = mfe_ticks >= be_trigger_val
                    be_ready = be_met and (not use_be_confirm or be_cond_prev)
                    if be_ready:
                        be_triggered = True
                        if is_long:
                            new_sl = entry_price + be_offset_val * tick
                            if new_sl > sl_price:
                                sl_price = new_sl
                        else:
                            new_sl = entry_price - be_offset_val * tick
                            if new_sl < sl_price:
                                sl_price = new_sl

                be_cond_prev = be_met

                # Duration-based BE
                if use_dur_be and not be_triggered and bars_in_trade >= dur_be_bars:
                    be_triggered = True
                    if is_long:
                        dur_stop = entry_price + dur_be_offset * tick
                        if dur_stop > sl_price:
                            sl_price = dur_stop
                    else:
                        dur_stop = entry_price - dur_be_offset * tick
                        if dur_stop < sl_price:
                            sl_price = dur_stop

                # Trail activation
                use_trail = l_use_trail if is_long else s_use_trail
                trail_trig_val = l_trail_trigger if is_long else s_trail_trigger
                if use_trail and not trail_active:
                    if mfe_ticks >= trail_trig_val:
                        trail_active = True
                        trail_peak = high[i] if is_long else low[i]

            if exited:
                # Record exit order
                size = abs(position)
                if is_long:
                    pnl = (exit_price - entry_price) * point_value * size - commission
                else:
                    pnl = (entry_price - exit_price) * point_value * size - commission

                side = 1 if is_long else 0  # 0=buy(cover), 1=sell(close long)
                if order_count < max_orders:
                    order_records[order_count]['id'] = order_count
                    order_records[order_count]['col'] = 0
                    order_records[order_count]['idx'] = i
                    order_records[order_count]['size'] = point_value
                    order_records[order_count]['price'] = exit_price
                    order_records[order_count]['fees'] = commission
                    order_records[order_count]['side'] = side
                    order_count += 1

                cash += pnl + commission
                daily_pnl += pnl

                # Lockout-on-exit: reset counter when position closes
                if is_long and l_lock_on_exit and l_lock_bars > 0:
                    l_lock_counter = 0
                if not is_long and s_lock_on_exit and s_lock_bars > 0:
                    s_lock_counter = 0

                position = 0.0
                entry_price = 0.0
                bars_in_trade = 0
                trail_active = False
                be_triggered = False
                be_cond_prev = False

                # DLC check
                if dlc_enable and daily_pnl <= -(dlc_ticks * tick * point_value):
                    dlc_blocked = True

        # ── Entry checks (only when flat) ──
        if position == 0.0 and not dlc_blocked:
            l_can_trade = (l_lock_bars == 0) or (l_lock_counter > l_lock_bars)
            s_can_trade = (s_lock_bars == 0) or (s_lock_counter > s_lock_bars)

            entered = False

            # Long entry
            if long_sig[i] and l_can_trade and not entered:
                fill_price = close[i] + slip_price  # adverse slippage
                position = 1.0
                entry_price = fill_price
                bars_in_trade = 0
                sl_price = fill_price - l_stop_ticks * tick
                tp_price = fill_price + l_target_ticks * tick
                trail_active = False
                trail_peak = fill_price
                be_triggered = False
                be_cond_prev = False
                l_lock_counter = 0
                entered = True

                if order_count < max_orders:
                    order_records[order_count]['id'] = order_count
                    order_records[order_count]['col'] = 0
                    order_records[order_count]['idx'] = i
                    order_records[order_count]['size'] = point_value
                    order_records[order_count]['price'] = fill_price
                    order_records[order_count]['fees'] = commission
                    order_records[order_count]['side'] = 0  # buy
                    order_count += 1

            # Short entry (only if didn't just enter long)
            if short_sig[i] and s_can_trade and not entered:
                fill_price = close[i] - slip_price  # adverse slippage
                position = -1.0
                entry_price = fill_price
                bars_in_trade = 0
                sl_price = fill_price + s_stop_ticks * tick
                tp_price = fill_price - s_target_ticks * tick
                trail_active = False
                trail_peak = fill_price
                be_triggered = False
                be_cond_prev = False
                s_lock_counter = 0
                entered = True

                if order_count < max_orders:
                    order_records[order_count]['id'] = order_count
                    order_records[order_count]['col'] = 0
                    order_records[order_count]['idx'] = i
                    order_records[order_count]['size'] = point_value
                    order_records[order_count]['price'] = fill_price
                    order_records[order_count]['fees'] = commission
                    order_records[order_count]['side'] = 1  # sell (short)
                    order_count += 1

    # Close any open position at end
    if position != 0.0:
        exit_price = close[n - 1]
        is_long = position > 0
        side = 1 if is_long else 0
        if order_count < max_orders:
            order_records[order_count]['id'] = order_count
            order_records[order_count]['col'] = 0
            order_records[order_count]['idx'] = n - 1
            order_records[order_count]['size'] = point_value
            order_records[order_count]['price'] = exit_price
            order_records[order_count]['fees'] = commission
            order_records[order_count]['side'] = side
            order_count += 1

    return order_records[:order_count], order_count


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

def map_params(full_params):
    """Map DB full_params dict to QTP201Params kwargs."""
    p = full_params
    return dict(
        allow_longs=p.get('Allow Longs', True),
        allow_shorts=p.get('Allow Shorts', True),

        vol_filter_enable=p.get('Vol Filter Enable', False),
        vol_atr_period=int(p.get('Vol ATR Period', 14)),
        vol_atr_mult=float(p.get('Vol ATR Multiplier', 2.0)),

        oma_enable=p.get('OMA Enable', True),
        oma_len=int(p.get('OMA Period', 10)),
        oma_speed=float(p.get('OMA Speed', 2.5)),
        oma_close_period=int(p.get('OMA Close Period', 1)),
        oma_adaptive=p.get('OMA Adaptive', True),

        use_max_bars=p.get('Max Bars Exit', False),
        max_bars_in_trade=int(p.get('Max Bars in Trade', 60)),
        use_dur_be=p.get('Duration BE Enable', False),
        dur_be_bars=int(p.get('Duration BE Bars', 10)),
        dur_be_offset=int(p.get('Duration BE Offset', 5)),
        use_be_confirm=p.get('BE Confirm Delay', False),

        dlc_enable=p.get('Daily Loss Limit Enable', False),
        dlc_ticks=int(p.get('Daily Loss Limit (Ticks)', 500)),

        l_st_period=int(p.get('L Supertrend Period', 14)),
        l_st_mult=float(p.get('L Supertrend Mult', 3.0)),
        l_rsi_period=int(p.get('L RSI Period', 14)),
        l_rsi_max=float(p.get('L RSI Max', 60.0)),
        l_lock_bars=int(p.get('L Lockout Bars', 0)),
        l_lock_on_exit=p.get('L Lock on Exit', False),

        l_use_session=p.get('L Session Enable', False),
        l_session_tz=p.get('L Timezone', 'America/New_York'),
        l_session_hours=p.get('L Trading Hours', '0930-1600'),
        l_use_exclude1=p.get('L Exclude 1 Enable', False),
        l_exclude_hours1=p.get('L Exclude Hours 1', '1545-1600'),
        l_use_exclude2=p.get('L Exclude 2 Enable', False),
        l_exclude_hours2=p.get('L Exclude Hours 2', '0100-0400'),
        l_use_exclude3=p.get('L Exclude 3 Enable', False),
        l_exclude_hours3=p.get('L Exclude Hours 3', '1200-1300'),
        l_use_exclude4=p.get('L Exclude 4 Enable', False),
        l_exclude_hours4=p.get('L Exclude Hours 4', '0800-0930'),
        l_use_cme_maint=p.get('L CME Maintenance', True),

        l_use_day_filter=p.get('L Day Filter Enable', False),
        l_day_sun=p.get('L Sunday', False),
        l_day_mon=p.get('L Monday', True),
        l_day_tue=p.get('L Tuesday', True),
        l_day_wed=p.get('L Wednesday', True),
        l_day_thu=p.get('L Thursday', True),
        l_day_fri=p.get('L Friday', True),
        l_day_sat=p.get('L Saturday', False),

        l_stop_ticks=int(p.get('L Stop Ticks', 225)),
        l_target_ticks=int(p.get('L Target Ticks', 150)),
        l_use_trail=p.get('L Trail Enable', True),
        l_trail_trigger=int(p.get('L Trail Trigger', 90)),
        l_trail_offset=int(p.get('L Trail Offset', 15)),
        l_use_be=p.get('L BE Enable', True),
        l_be_trigger=int(p.get('L BE Trigger', 80)),
        l_be_offset=int(p.get('L BE Offset', 20)),

        s_cci_period=int(p.get('S CCI Period', 30)),
        s_cci_extreme=float(p.get('S CCI Extreme', 130.0)),
        s_cci_exit=float(p.get('S CCI Exit', 50.0)),
        s_rsi_period=int(p.get('S RSI Period', 12)),
        s_rsi_min=float(p.get('S RSI Min', 49.0)),
        s_lock_bars=int(p.get('S Lockout Bars', 10)),
        s_cci_lookback=int(p.get('S CCI Lookback', 5)),
        s_lock_on_exit=p.get('S Lock on Exit', False),

        s_use_session=p.get('S Session Enable', False),
        s_session_tz=p.get('S Timezone', 'America/New_York'),
        s_session_hours=p.get('S Trading Hours', '0930-1600'),
        s_use_exclude1=p.get('S Exclude 1 Enable', False),
        s_exclude_hours1=p.get('S Exclude Hours 1', '1545-1600'),
        s_use_exclude2=p.get('S Exclude 2 Enable', False),
        s_exclude_hours2=p.get('S Exclude Hours 2', '0100-0400'),
        s_use_exclude3=p.get('S Exclude 3 Enable', False),
        s_exclude_hours3=p.get('S Exclude Hours 3', '1200-1300'),
        s_use_exclude4=p.get('S Exclude 4 Enable', False),
        s_exclude_hours4=p.get('S Exclude Hours 4', '0800-0930'),
        s_use_cme_maint=p.get('S CME Maintenance', True),

        s_use_day_filter=p.get('S Day Filter Enable', False),
        s_day_sun=p.get('S Sunday', False),
        s_day_mon=p.get('S Monday', True),
        s_day_tue=p.get('S Tuesday', True),
        s_day_wed=p.get('S Wednesday', True),
        s_day_thu=p.get('S Thursday', True),
        s_day_fri=p.get('S Friday', True),
        s_day_sat=p.get('S Saturday', False),

        s_stop_ticks=int(p.get('S Stop Ticks', 230)),
        s_target_ticks=int(p.get('S Target Ticks', 130)),
        s_use_trail=p.get('S Trail Enable', True),
        s_trail_trigger=int(p.get('S Trail Trigger', 100)),
        s_trail_offset=int(p.get('S Trail Offset', 1)),
        s_use_be=p.get('S BE Enable', True),
        s_be_trigger=int(p.get('S BE Trigger', 95)),
        s_be_offset=int(p.get('S BE Offset', 5)),

        tick_size=float(p.get('Tick Size', 0.10)),
        point_value=float(p.get('Point Value', 100.0)),
        commission=float(p.get('Commission', 5.0)),
        slippage=int(p.get('Slippage', 2)),
        initial_capital=float(p.get('Initial Capital', 50000.0)),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# COMBO RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_combo(data, indicators, session, params, combo_idx, w_start, w_end, w_type,
              target_start_idx=None, n_bars=None):
    """Run a single combo on pre-computed indicators. Returns WindowResult or None."""
    ind = indicators
    l_session_ok, s_session_ok = session
    n_full = ind['n']
    tick = params.tick_size

    # Find target window start index
    if target_start_idx is None:
        target_start_idx = np.searchsorted(data.index, pd.Timestamp(w_start, tz='UTC'))

    # Detect signals
    long_sig, short_sig = detect_signals(
        ind['close'], ind['high'], ind['low'], n_full,
        ind['l_st_dir'], ind['l_rsi'], ind['oma_trend'],
        ind['s_cci'], ind['s_rsi'],
        ind['vol_ok'], l_session_ok, s_session_ok,
        params.oma_enable,
        params.l_rsi_max,
        params.s_cci_extreme, params.s_cci_exit, params.s_rsi_min, params.s_cci_lookback,
        params.allow_longs, params.allow_shorts,
    )

    # Run simulation
    order_records, oc = run_simulation(
        ind['open'], ind['high'], ind['low'], ind['close'], n_full,
        long_sig, short_sig,
        tick, params.point_value, params.commission, params.slippage, params.initial_capital,
        # Long ATM
        params.l_stop_ticks, params.l_target_ticks,
        params.l_use_trail, params.l_trail_trigger, params.l_trail_offset,
        params.l_use_be, params.l_be_trigger, params.l_be_offset,
        params.l_lock_bars, params.l_lock_on_exit,
        # Short ATM
        params.s_stop_ticks, params.s_target_ticks,
        params.s_use_trail, params.s_trail_trigger, params.s_trail_offset,
        params.s_use_be, params.s_be_trigger, params.s_be_offset,
        params.s_lock_bars, params.s_lock_on_exit,
        # Global exit
        params.use_max_bars, params.max_bars_in_trade,
        params.use_dur_be, params.dur_be_bars, params.dur_be_offset,
        params.use_be_confirm,
        # DLC
        params.dlc_enable, params.dlc_ticks,
        # Bars per day (GC 1min = 1380 bars/day)
        1380,
    )

    # Extract trades in target window
    total_pnl = 0.0
    total_trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    max_dd = 0.0
    peak_eq = params.initial_capital
    equity = params.initial_capital
    best_t = 0.0
    worst_t = 0.0

    for j in range(0, oc - 1, 2):
        ei = int(order_records[j]['idx'])
        if ei < target_start_idx:
            continue
        ep = order_records[j]['price']
        xp = order_records[j + 1]['price']
        if order_records[j]['side'] == 0:
            pnl_t = (xp - ep) * params.point_value - params.commission * 2
        else:
            pnl_t = (ep - xp) * params.point_value - params.commission * 2
        total_pnl += pnl_t
        total_trades += 1
        if pnl_t > 0:
            wins += 1
            gross_profit += pnl_t
        else:
            gross_loss += abs(pnl_t)
        best_t = max(best_t, pnl_t)
        worst_t = min(worst_t, pnl_t)
        equity += pnl_t
        peak_eq = max(peak_eq, equity)
        dd = (peak_eq - equity) / peak_eq * 100 if peak_eq > 0 else 0
        max_dd = max(max_dd, dd)

    wr = (wins / total_trades * 100) if total_trades > 0 else 0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0)
    pf = min(pf, 999.0)

    target_mask = (data.index >= pd.Timestamp(w_start, tz='UTC')) & \
                  (data.index <= pd.Timestamp(w_end, tz='UTC'))

    return WindowResult(
        combo_index=combo_idx,
        window_start=w_start, window_end=w_end, window_type=w_type,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl / params.initial_capital * 100,
        max_drawdown=max_dd, max_drawdown_pct=max_dd,
        win_rate=wr, profit_factor=pf, total_trades=total_trades,
        avg_trade=total_pnl / total_trades if total_trades > 0 else 0,
        best_trade=best_t, worst_trade=worst_t,
        bars=n_bars if n_bars is not None else int(target_mask.sum()), duration_ms=0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY CLASS (Standard interface)
# ═══════════════════════════════════════════════════════════════════════════════

class Strategy:
    """Standard interface for generic study runner."""
    PARAMS_CLASS = QTP201Params
    NAME = 'qtp201_super_rsi_scalper'

    @staticmethod
    def map_params(full_params):
        """Map database parameter names to QTP201Params kwargs."""
        return map_params(full_params)

    @staticmethod
    def make_params(full_params):
        """Create QTP201Params from database full_params dict."""
        return QTP201Params(**map_params(full_params))

    @staticmethod
    def compute_indicators(data, params):
        """Pre-compute indicators for a data window. Called once per window."""
        return compute_indicators(data, params)

    @staticmethod
    def compute_session(data, params):
        """Compute session masks. Called once per window."""
        return compute_session_masks(data.index, params)

    @staticmethod
    def run_combo(data, indicators, session, params, combo_idx, w_start, w_end, w_type,
                  target_start_idx=None, n_bars=None):
        """Run a single combo on pre-computed indicators. Returns WindowResult or None."""
        return run_combo(data, indicators, session, params, combo_idx, w_start, w_end, w_type,
                         target_start_idx, n_bars)
