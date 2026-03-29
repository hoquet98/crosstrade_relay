"""
VECTOR Pattern Strategy v12 — Single-file strategy module.

Contains everything needed to backtest the VECTOR strategy:
- VectorParams dataclass (all strategy parameters)
- Indicator computation (VECTOR line, candle analysis, ARD, squeeze, session masks)
- Pattern detection (cross, bounce, continuation) — Numba compiled
- ARD filter — Numba compiled
- Trade simulation (SL/TP/trail/BE with 1-bar delay, DLC, EOD, re-entry) — Numba compiled
- Parameter mapping from database format
- Single combo runner

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

# Order record dtype — replaces order_dt to avoid vectorbtpro dependency
order_dt = np.dtype([
    ('id', np.int64), ('col', np.int64), ('idx', np.int64),
    ('size', np.float64), ('price', np.float64), ('fees', np.float64), ('side', np.int64),
])

from strategies.window_result import WindowResult


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VectorParams:
    """All strategy parameters. Defaults match GC optimized baseline v10.2.0."""

    # ── VECTOR computation ──
    vector_length: int = 20
    tick_size: float = 0.10       # GC=0.10, ES=0.25

    # ── Cross Pattern ──
    enable_cross: bool = True
    cross_priority: int = 65
    cross_min_penetration: float = 1.0
    cross_max_penetration: float = 52.0
    cross_bar_filter: bool = True
    cross_min_bar_ticks: float = 14.0
    cross_body_filter: bool = True
    cross_min_body_ratio: float = 0.75
    cross_close_filter: bool = True
    cross_close_strength: float = 0.50
    cross_slope_filter: bool = True
    cross_min_slope_ticks: float = 14.0

    # ── Bounce Pattern ──
    enable_bounce: bool = True
    bounce_priority: int = 50
    bounce_touch_ticks: float = 45.0
    bounce_min_reversal_ticks: float = 10.0
    bounce_bar_filter: bool = True
    bounce_min_bar_ticks: float = 11.0
    bounce_close_filter: bool = True
    bounce_close_strength: float = 0.85
    bounce_slope_filter: bool = True
    bounce_min_slope_ticks: float = 22.0

    # ── Continuation Pattern ──
    enable_cont: bool = True
    cont_priority: int = 85
    cont_window: int = 20
    cont_min_dist_ticks: float = 18.0
    cont_max_dist_ticks: float = 25.0
    cont_slope_filter: bool = True
    cont_min_slope_ticks: float = 2.0

    # ── ATM (tick-based exits) ──
    stop_ticks: int = 100
    target_ticks: int = 200
    trail_enable: bool = True
    trail_trigger: int = 140
    trail_offset: int = 5
    be_enable: bool = True
    be_trigger: int = 90
    be_offset: int = 20

    # ── Session / Exclude ──
    session_enable: bool = False
    session_hours: str = "0930-1600"
    exclude_enable: bool = True
    exclude_hours: str = "1445-1800"
    exclude2_enable: bool = True
    exclude_hours2: str = "0600-0630"
    session_tz: str = "America/New_York"
    eod_enable: bool = True
    eod_hour: int = 15
    eod_minute: int = 50

    # ── Global ──
    lock_bars: int = 10
    allow_longs: bool = True
    allow_shorts: bool = True

    # ── Squeeze Filter ──
    squeeze_enable: bool = False

    # ── ARD (Adverse Regime Detector) ──
    ard_enable: bool = True
    ard_activation: str = "Any One"
    ard_mode: str = "Block All"
    ard_var_threshold: float = 1.4
    ard_dsi_threshold: float = 0.65
    ard_dsi_lookback: int = 10
    ard_map_threshold: float = 1.4
    ard_map_lookback: int = 10

    # ── Daily Loss Cap ──
    dlc_enable: bool = True
    dlc_ticks: int = 150

    # ── Futures config ──
    point_value: float = 100.0
    commission: float = 5.0
    initial_capital: float = 1000000.0


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

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


def _sma(arr, period):
    """Simple Moving Average."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = np.mean(arr[i - period + 1:i + 1])
    return out


def _rolling_std(arr, period):
    """Rolling standard deviation (ddof=0 to match TV)."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = arr[i - period + 1:i + 1]
        out[i] = np.std(window)
    return out


def _lowest(arr, period):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = np.min(arr[i - period + 1:i + 1])
    return out


def _highest(arr, period):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = np.max(arr[i - period + 1:i + 1])
    return out


def _bars_since(condition, n):
    out = np.full(n, 999, dtype=np.int32)
    last = -999
    for i in range(n):
        if condition[i]:
            last = i
        out[i] = i - last if last >= 0 else 999
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


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_indicators(data: pd.DataFrame, vector_length: int = 20):
    """Pre-compute all indicators from OHLCV data.

    Returns a dict of numpy arrays (1D, same length as data).
    All arrays are price-derived and parameter-independent
    (except vector_length which rarely varies).
    """
    close = data['close'].values
    high = data['high'].values
    low = data['low'].values
    open_ = data['open'].values
    n = len(close)

    # VECTOR line: highest(lowest(low, N), N)
    vector_line = _highest(_lowest(low, vector_length), vector_length)
    vector_slope = np.zeros(n)
    vector_slope[1:] = vector_line[1:] - vector_line[:-1]
    abs_slope = np.abs(vector_slope)

    # Cross detection
    bull_cross = np.zeros(n, dtype=bool)
    bear_cross = np.zeros(n, dtype=bool)
    for i in range(1, n):
        bull_cross[i] = close[i] > vector_line[i] and close[i - 1] <= vector_line[i - 1]
        bear_cross[i] = close[i] < vector_line[i] and close[i - 1] >= vector_line[i - 1]

    # Bars since last cross (for continuation pattern)
    bars_since_cross = _bars_since(bull_cross | bear_cross, n)

    # Candle analysis
    bar_range = high - low
    body_size = np.abs(close - open_)
    safe_range = np.where(bar_range > 0, bar_range, np.nan)
    body_ratio = body_size / safe_range
    body_ratio = np.nan_to_num(body_ratio, nan=0.5)
    close_position = (close - low) / safe_range
    close_position = np.nan_to_num(close_position, nan=0.5)
    bull_candle = close > open_
    bear_candle = close < open_

    # Distance from VECTOR line
    dist_from_vec = np.abs(close - vector_line)

    # ARD components (Wilder's ATR)
    short_atr = _wilder_atr(high, low, close, 5)
    long_atr = _wilder_atr(high, low, close, 20)
    var_ratio = np.where(long_atr > 0, short_atr / long_atr, 0.0)

    # DSI: rolling mean of stress bars
    stress_bar = (close < vector_line) & bear_candle
    # MAP: avg down range / avg up range
    down_range = np.where(bear_candle, bar_range, 0.0)
    up_range = np.where(bull_candle, bar_range, 0.0)

    # Squeeze filter: BB inside KC = squeeze active
    ema8 = _ema(close, 8)
    ema13 = _ema(close, 13)
    ema21 = _ema(close, 21)
    bb_upper = ema13 + 2.0 * _rolling_std(close, 13)
    kc_mid = _sma(close, 13)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    kc_shift = _ema(tr, 13)

    delta_high = bb_upper - (kc_mid + kc_shift * 1.0)
    delta_mid = bb_upper - (kc_mid + kc_shift * 1.5)

    # Squeeze type: 3=High, 2=Mid, 0=None
    sq_type = np.where(delta_high <= 0, 3, np.where(delta_mid <= 0, 2, 0))

    # Ideal squeeze: EMAs aligned + squeeze >= Mid
    sq_ideal_bull = ((ema8 > ema13) & (ema13 > ema21) &
                     (ema13 > np.roll(ema13, 1)) & (ema21 > np.roll(ema21, 1)) &
                     (sq_type >= 2))
    sq_ideal_bear = ((ema8 < ema13) & (ema13 < ema21) &
                     (ema13 < np.roll(ema13, 1)) & (ema21 < np.roll(ema21, 1)) &
                     (sq_type >= 2))
    squeeze_ok = sq_ideal_bull | sq_ideal_bear

    return {
        'close': close, 'high': high, 'low': low, 'open': open_,
        'vector_line': vector_line,
        'vector_slope': vector_slope,
        'abs_slope': abs_slope,
        'bull_cross': bull_cross,
        'bear_cross': bear_cross,
        'bars_since_cross': bars_since_cross,
        'bar_range': bar_range,
        'body_ratio': body_ratio,
        'close_position': close_position,
        'bull_candle': bull_candle,
        'bear_candle': bear_candle,
        'dist_from_vec': dist_from_vec,
        'var_ratio': var_ratio,
        'stress_bar': stress_bar.astype(np.float64),
        'down_range': down_range,
        'up_range': up_range,
        'squeeze_ok': squeeze_ok,
        'n': n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MASK
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_session(s):
    parts = s.split('-')
    start = int(parts[0][:2]) * 60 + int(parts[0][2:])
    end = int(parts[1][:2]) * 60 + int(parts[1][2:])
    return start, end


def compute_session_mask(index, params):
    """Compute session/exclude filter mask. True = can trade."""
    from zoneinfo import ZoneInfo
    n = len(index)
    mask = np.ones(n, dtype=bool)

    try:
        ny = index.tz_convert(ZoneInfo(params.session_tz))
        minutes = ny.hour * 60 + ny.minute
    except Exception:
        minutes = index.hour * 60 + index.minute

    if params.exclude_enable:
        s, e = _parse_session(params.exclude_hours)
        if s < e:
            mask &= ~((minutes >= s) & (minutes < e))
        else:
            mask &= ~((minutes >= s) | (minutes < e))

    if params.exclude2_enable:
        s, e = _parse_session(params.exclude_hours2)
        if s < e:
            mask &= ~((minutes >= s) & (minutes < e))
        else:
            mask &= ~((minutes >= s) | (minutes < e))

    # EOD mask
    eod_mask = np.zeros(n, dtype=bool)
    if params.eod_enable:
        eod_min = params.eod_hour * 60 + params.eod_minute
        eod_mask = (minutes == eod_min)

    return mask, eod_mask, minutes


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN DETECTION (Numba compiled)
# ═══════════════════════════════════════════════════════════════════════════════

@njit
def detect_patterns(
    close, high, low, open_, vector_line, vector_slope, abs_slope,
    bull_cross, bear_cross, bars_since_cross,
    bar_range, body_ratio, close_position, bull_candle, bear_candle,
    dist_from_vec, tick,
    # Cross params
    enable_cross, cross_priority, cross_min_pen, cross_max_pen,
    cross_bar_filter, cross_min_bar, cross_body_filter, cross_min_body,
    cross_close_filter, cross_close_str, cross_slope_filter, cross_min_slope,
    # Bounce params
    enable_bounce, bounce_priority, bounce_touch, bounce_min_rev,
    bounce_bar_filter, bounce_min_bar, bounce_close_filter, bounce_close_str,
    bounce_slope_filter, bounce_min_slope,
    # Cont params
    enable_cont, cont_priority, cont_window, cont_min_dist, cont_max_dist,
    cont_slope_filter, cont_min_slope,
):
    """Detect Cross/Bounce/Continuation patterns. Returns (long_signal, short_signal) bool arrays."""
    n = len(close)
    long_sig = np.zeros(n, dtype=np.bool_)
    short_sig = np.zeros(n, dtype=np.bool_)

    for i in range(1, n):
        if np.isnan(vector_line[i]):
            continue

        long_score = 0
        short_score = 0

        # ── Cross Pattern ──
        if enable_cross:
            dist = dist_from_vec[i]
            min_d = cross_min_pen * tick
            max_d = cross_max_pen * tick
            dist_ok = dist >= min_d and dist <= max_d
            bar_ok = (not cross_bar_filter) or (bar_range[i] >= cross_min_bar * tick)
            body_ok = (not cross_body_filter) or (body_ratio[i] >= cross_min_body)
            slope_ok = (not cross_slope_filter) or (abs_slope[i] >= cross_min_slope * tick)

            if bull_cross[i] and dist_ok and bar_ok and slope_ok:
                close_ok = (not cross_close_filter) or (close_position[i] >= cross_close_str)
                bull_body_ok = (not cross_body_filter) or (body_ok and bull_candle[i])
                if close_ok and bull_body_ok and cross_priority > long_score:
                    long_score = cross_priority

            if bear_cross[i] and dist_ok and bar_ok and slope_ok:
                close_ok = (not cross_close_filter) or (close_position[i] <= (1 - cross_close_str))
                bear_body_ok = (not cross_body_filter) or (body_ok and bear_candle[i])
                if close_ok and bear_body_ok and cross_priority > short_score:
                    short_score = cross_priority

        # ── Bounce Pattern ──
        if enable_bounce:
            touch = bounce_touch * tick
            min_rev = bounce_min_rev * tick
            min_bar = bounce_min_bar * tick
            min_slope = bounce_min_slope * tick

            # Bull bounce: touched from below
            touched_below = (low[i] <= vector_line[i] + touch) and (low[i] >= vector_line[i] - touch * 2)
            strong_bull = bull_candle[i] and ((close[i] - low[i]) >= min_rev) and (close[i] > vector_line[i])
            # Bear bounce: touched from above
            touched_above = (high[i] >= vector_line[i] - touch) and (high[i] <= vector_line[i] + touch * 2)
            strong_bear = bear_candle[i] and ((high[i] - close[i]) >= min_rev) and (close[i] < vector_line[i])

            bar_ok = (not bounce_bar_filter) or (bar_range[i] >= min_bar)
            slope_ok = (not bounce_slope_filter) or (abs_slope[i] >= min_slope)
            base_ok = bar_ok and slope_ok

            if touched_below and strong_bull and base_ok and not bull_cross[i]:
                close_ok = (not bounce_close_filter) or (close_position[i] >= bounce_close_str)
                if close_ok and bounce_priority > long_score:
                    long_score = bounce_priority

            if touched_above and strong_bear and base_ok and not bear_cross[i]:
                close_ok = (not bounce_close_filter) or (close_position[i] <= (1 - bounce_close_str))
                if close_ok and bounce_priority > short_score:
                    short_score = bounce_priority

        # ── Continuation Pattern ──
        if enable_cont:
            dist = dist_from_vec[i]
            min_d = cont_min_dist * tick
            max_d = cont_max_dist * tick
            dist_ok = dist >= min_d and dist <= max_d
            window_ok = bars_since_cross[i] <= cont_window and bars_since_cross[i] > 1

            if dist_ok and window_ok:
                slope_bull = (not cont_slope_filter) or (vector_slope[i] > cont_min_slope * tick)
                slope_bear = (not cont_slope_filter) or (vector_slope[i] < -cont_min_slope * tick)

                if slope_bull and close[i] > vector_line[i] and cont_priority > long_score:
                    long_score = cont_priority
                if slope_bear and close[i] < vector_line[i] and cont_priority > short_score:
                    short_score = cont_priority

        # Resolve: long wins ties
        if long_score > 0 and short_score > 0:
            if long_score >= short_score:
                short_score = 0
            else:
                long_score = 0

        long_sig[i] = long_score > 0
        short_sig[i] = short_score > 0

    return long_sig, short_sig


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL WRAPPER (for VBT adapter — unpacks dict args for Numba function)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_signals_from_dict(indicators, session, params):
    """Unpack indicator dict and session tuple, call Numba detect_patterns().

    This is the bridge between the VBT adapter (which passes dicts) and the
    Numba-compiled detect_patterns() which needs individual arrays.
    Does NOT affect backtesting — run_combo() calls detect_patterns() directly.
    """
    ind = indicators
    return detect_patterns(
        ind['close'], ind['high'], ind['low'], ind['open'],
        ind['vector_line'], ind['vector_slope'], ind['abs_slope'],
        ind['bull_cross'], ind['bear_cross'], ind['bars_since_cross'],
        ind['bar_range'], ind['body_ratio'], ind['close_position'],
        ind['bull_candle'], ind['bear_candle'],
        ind['dist_from_vec'], params.tick_size,
        params.enable_cross, params.cross_priority, params.cross_min_penetration,
        params.cross_max_penetration, params.cross_bar_filter, params.cross_min_bar_ticks,
        params.cross_body_filter, params.cross_min_body_ratio,
        params.cross_close_filter, params.cross_close_strength,
        params.cross_slope_filter, params.cross_min_slope_ticks,
        params.enable_bounce, params.bounce_priority, params.bounce_touch_ticks,
        params.bounce_min_reversal_ticks, params.bounce_bar_filter, params.bounce_min_bar_ticks,
        params.bounce_close_filter, params.bounce_close_strength,
        params.bounce_slope_filter, params.bounce_min_slope_ticks,
        params.enable_cont, params.cont_priority, params.cont_window,
        params.cont_min_dist_ticks, params.cont_max_dist_ticks,
        params.cont_slope_filter, params.cont_min_slope_ticks,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ARD FILTER (Numba compiled)
# ═══════════════════════════════════════════════════════════════════════════════

@njit
def compute_ard(var_ratio, stress_bar, down_range, up_range, n,
                var_threshold, dsi_threshold, dsi_lookback,
                map_threshold, map_lookback, activation, mode):
    """Compute ARD filter. Returns (long_ok, short_ok) bool arrays."""
    long_ok = np.ones(n, dtype=np.bool_)
    short_ok = np.ones(n, dtype=np.bool_)

    for i in range(max(dsi_lookback, map_lookback), n):
        var_pass = var_ratio[i] >= var_threshold

        # DSI: rolling mean of stress bars
        dsi = 0.0
        for j in range(i - dsi_lookback + 1, i + 1):
            dsi += stress_bar[j]
        dsi /= dsi_lookback
        dsi_pass = dsi >= dsi_threshold

        # MAP: avg down / avg up
        avg_down = 0.0
        avg_up = 0.0
        for j in range(i - map_lookback + 1, i + 1):
            avg_down += down_range[j]
            avg_up += up_range[j]
        avg_down /= map_lookback
        avg_up /= map_lookback
        map_val = avg_down / avg_up if avg_up > 0 else 1.0
        map_pass = map_val >= map_threshold

        # Activation: 1=Any One, 2=Any Two, 3=All Three, 0=VAR Only
        count = int(var_pass) + int(dsi_pass) + int(map_pass)
        if activation == 0:  # VAR Only
            adverse = var_pass
        elif activation == 1:  # Any One
            adverse = count >= 1
        elif activation == 2:  # Any Two
            adverse = count >= 2
        else:  # All Three
            adverse = count >= 3

        if adverse:
            long_ok[i] = False
            if mode == 0:  # Block All
                short_ok[i] = False
            # mode == 1: Block Longs Only — short_ok stays True

    return long_ok, short_ok


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE SIMULATION (Numba compiled)
# ═══════════════════════════════════════════════════════════════════════════════

@njit
def run_simulation(
    open_, high, low, close, n,
    long_sig, short_sig, session_ok, eod_mask, long_ok, short_ok,
    tick, point_value, commission, init_capital,
    stop_ticks, target_ticks,
    trail_enable, trail_trigger, trail_offset,
    be_enable, be_trigger, be_offset,
    lock_bars, allow_longs, allow_shorts,
    squeeze_enable, squeeze_ok,
    dlc_enable, dlc_ticks, bars_per_day,
):
    """Run the full simulation with Pine Script-compatible execution.

    Returns order_records array compatible with VBT Portfolio.
    """
    # Pre-allocate order records (max 2 orders per bar: exit + entry)
    max_orders = n * 2
    order_records = np.empty(max_orders, dtype=order_dt)
    order_count = 0

    # Position state
    position = 0.0       # >0 long, <0 short, 0 flat
    entry_price = 0.0
    entry_bar = 0

    # ATM state
    sl_price = 0.0
    tp_price = 0.0
    trail_active = False
    trail_peak = 0.0
    be_triggered = False

    # Pending ATM update (1-bar delay)
    pending_sl = np.nan
    pending_be = False
    pending_be_price = 0.0

    # Lockout
    lock_counter = lock_bars + 1  # start unlocked

    # DLC state
    daily_pnl = 0.0
    last_day = -1
    dlc_blocked = False

    # Cash tracking
    cash = init_capital

    for i in range(n):
        # ── Day boundary reset for DLC ──
        current_day = i // bars_per_day
        if current_day != last_day:
            daily_pnl = 0.0
            dlc_blocked = False
            last_day = current_day

        # ── Apply pending ATM updates (1-bar delay) ──
        if not np.isnan(pending_sl):
            sl_price = pending_sl
            pending_sl = np.nan

        if pending_be and not be_triggered:
            be_triggered = True
            if position > 0:
                sl_price = entry_price + be_offset * tick
            elif position < 0:
                sl_price = entry_price - be_offset * tick
            pending_be = False

        # ── Exit checks (before entries, using current bar OHLC) ──
        if position != 0.0:
            is_long = position > 0
            exited = False
            exit_price = 0.0
            exit_reason = 0  # 0=none, 1=sl, 2=tp, 3=trail, 4=eod, 5=reentry

            # Check SL
            if is_long and low[i] <= sl_price:
                exit_price = max(sl_price, low[i])
                exited = True
                exit_reason = 1
            elif not is_long and high[i] >= sl_price:
                exit_price = min(sl_price, high[i])
                exited = True
                exit_reason = 1

            # Check TP (only if SL didn't fire)
            if not exited:
                if is_long and high[i] >= tp_price:
                    exit_price = min(tp_price, high[i])
                    exited = True
                    exit_reason = 2
                elif not is_long and low[i] <= tp_price:
                    exit_price = max(tp_price, low[i])
                    exited = True
                    exit_reason = 2

            # Check Trail (only if SL/TP didn't fire)
            if not exited and trail_enable and trail_active:
                if is_long:
                    trail_peak = max(trail_peak, high[i])
                    trail_sl = trail_peak - trail_offset * tick
                    if low[i] <= trail_sl:
                        exit_price = max(trail_sl, low[i])
                        exited = True
                        exit_reason = 3
                else:
                    trail_peak = min(trail_peak, low[i])
                    trail_sl = trail_peak + trail_offset * tick
                    if high[i] >= trail_sl:
                        exit_price = min(trail_sl, high[i])
                        exited = True
                        exit_reason = 3

            # Check EOD
            if not exited and eod_mask[i]:
                exit_price = close[i]
                exited = True
                exit_reason = 4

            # Update MFE-based triggers (set pending for next bar — 1-bar delay)
            if not exited and not be_triggered and be_enable:
                if is_long:
                    mfe_ticks = (high[i] - entry_price) / tick
                else:
                    mfe_ticks = (entry_price - low[i]) / tick
                if mfe_ticks >= be_trigger:
                    pending_be = True

            # Activate trail (pending for next bar)
            if not exited and trail_enable and not trail_active:
                if is_long:
                    mfe_ticks = (high[i] - entry_price) / tick
                else:
                    mfe_ticks = (entry_price - low[i]) / tick
                if mfe_ticks >= trail_trigger:
                    trail_active = True
                    trail_peak = high[i] if is_long else low[i]

            if exited:
                # Record exit order
                size = abs(position)
                if is_long:
                    pnl = (exit_price - entry_price) * point_value * size - commission
                else:
                    pnl = (entry_price - exit_price) * point_value * size - commission

                # Sell order for long exit, buy for short exit
                side = 1 if is_long else 0  # 0=buy, 1=sell
                if order_count < max_orders:
                    order_records[order_count]['id'] = order_count
                    order_records[order_count]['col'] = 0
                    order_records[order_count]['idx'] = i
                    order_records[order_count]['size'] = point_value
                    order_records[order_count]['price'] = exit_price
                    order_records[order_count]['fees'] = commission
                    order_records[order_count]['side'] = side
                    order_count += 1

                cash += pnl + commission  # add back commission since PnL already subtracted it
                daily_pnl += pnl
                position = 0.0
                entry_price = 0.0
                trail_active = False
                be_triggered = False
                pending_be = False
                pending_sl = np.nan

                # DLC check
                if dlc_enable and daily_pnl <= -(dlc_ticks * tick * point_value):
                    dlc_blocked = True

        # ── Entry checks ──
        can_trade = (lock_bars == 0) or (lock_counter > lock_bars)
        squeeze_gate = (not squeeze_enable) or squeeze_ok[i]
        can_enter = can_trade and session_ok[i] and not dlc_blocked and squeeze_gate

        # Same-direction re-entry: close existing + reopen
        if position > 0 and long_sig[i] and long_ok[i] and allow_longs and can_enter:
            # Close existing long at current close
            size = abs(position)
            pnl = (close[i] - entry_price) * point_value * size - commission
            if order_count < max_orders:
                order_records[order_count]['id'] = order_count
                order_records[order_count]['col'] = 0
                order_records[order_count]['idx'] = i
                order_records[order_count]['size'] = point_value
                order_records[order_count]['price'] = close[i]
                order_records[order_count]['fees'] = commission
                order_records[order_count]['side'] = 1  # sell
                order_count += 1
            cash += pnl + commission
            daily_pnl += pnl
            position = 0.0
            # Fall through to re-enter below

        elif position < 0 and short_sig[i] and short_ok[i] and allow_shorts and can_enter:
            # Close existing short at current close
            size = abs(position)
            pnl = (entry_price - close[i]) * point_value * size - commission
            if order_count < max_orders:
                order_records[order_count]['id'] = order_count
                order_records[order_count]['col'] = 0
                order_records[order_count]['idx'] = i
                order_records[order_count]['size'] = point_value
                order_records[order_count]['price'] = close[i]
                order_records[order_count]['fees'] = commission
                order_records[order_count]['side'] = 0  # buy (cover)
                order_count += 1
            cash += pnl + commission
            daily_pnl += pnl
            position = 0.0

        # New entry (or re-entry after close above)
        if position == 0.0 and can_enter:
            entered = False

            if long_sig[i] and long_ok[i] and allow_longs:
                # Long entry at close
                position = 1.0
                entry_price = close[i]
                entry_bar = i
                sl_price = entry_price - stop_ticks * tick
                tp_price = entry_price + target_ticks * tick
                trail_active = False
                trail_peak = entry_price
                be_triggered = False
                pending_be = False
                pending_sl = np.nan
                lock_counter = 0
                entered = True

                if order_count < max_orders:
                    order_records[order_count]['id'] = order_count
                    order_records[order_count]['col'] = 0
                    order_records[order_count]['idx'] = i
                    order_records[order_count]['size'] = point_value
                    order_records[order_count]['price'] = close[i]
                    order_records[order_count]['fees'] = commission
                    order_records[order_count]['side'] = 0  # buy
                    order_count += 1

            elif short_sig[i] and short_ok[i] and allow_shorts:
                # Short entry at close
                position = -1.0
                entry_price = close[i]
                entry_bar = i
                sl_price = entry_price + stop_ticks * tick
                tp_price = entry_price - target_ticks * tick
                trail_active = False
                trail_peak = entry_price
                be_triggered = False
                pending_be = False
                pending_sl = np.nan
                lock_counter = 0
                entered = True

                if order_count < max_orders:
                    order_records[order_count]['id'] = order_count
                    order_records[order_count]['col'] = 0
                    order_records[order_count]['idx'] = i
                    order_records[order_count]['size'] = point_value
                    order_records[order_count]['price'] = close[i]
                    order_records[order_count]['fees'] = commission
                    order_records[order_count]['side'] = 1  # sell (short)
                    order_count += 1

            # Reversal: opposite direction signal while in position
            if not entered and position == 0.0:
                pass  # already flat from exit above

        # ── Lockout counter ──
        if position == 0.0 or lock_counter <= lock_bars:
            lock_counter += 1

    # Close any open position at end
    if position != 0.0:
        exit_price = close[n - 1]
        size = abs(position)
        is_long = position > 0
        side = 1 if is_long else 0
        if order_count < max_orders:
            order_records[order_count]['id'] = order_count
            order_records[order_count]['col'] = 0
            order_records[order_count]['idx'] = n - 1
            order_records[order_count]['size'] = size
            order_records[order_count]['price'] = exit_price
            order_records[order_count]['fees'] = commission
            order_records[order_count]['side'] = side
            order_count += 1

    return order_records[:order_count], order_count


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

def map_params(full_params):
    """Map DB full_params dict to VectorParams kwargs."""
    p = full_params
    return dict(
        vector_length=p.get('VECTOR Length', 20),
        tick_size=0.10,
        enable_cross=p.get('Cross Enable', True),
        cross_priority=p.get('Cross Priority', 65),
        cross_min_penetration=float(p.get('Cross Min Penetration', 1.0)),
        cross_max_penetration=float(p.get('Cross Max Penetration', 52.0)),
        cross_bar_filter=p.get('Cross Bar Filter', True),
        cross_min_bar_ticks=float(p.get('Cross Min Bar Ticks', 14.0)),
        cross_body_filter=p.get('Cross Body Filter', True),
        cross_min_body_ratio=float(p.get('Cross Min Body Ratio', 0.75)),
        cross_close_filter=p.get('Cross Close Filter', True),
        cross_close_strength=float(p.get('Cross Close Strength', 0.50)),
        cross_slope_filter=p.get('Cross Slope Filter', True),
        cross_min_slope_ticks=float(p.get('Cross Min Slope Ticks', 14.0)),
        enable_bounce=p.get('Bounce Enable', True),
        bounce_priority=p.get('Bounce Priority', 50),
        bounce_touch_ticks=float(p.get('Bounce Touch Zone Ticks', 45.0)),
        bounce_min_reversal_ticks=float(p.get('Bounce Min Reversal Ticks', 10.0)),
        bounce_bar_filter=p.get('Bounce Bar Filter', True),
        bounce_min_bar_ticks=float(p.get('Bounce Min Bar Ticks', 11.0)),
        bounce_close_filter=p.get('Bounce Close Filter', True),
        bounce_close_strength=float(p.get('Bounce Close Strength', 0.85)),
        bounce_slope_filter=p.get('Bounce Slope Filter', True),
        bounce_min_slope_ticks=float(p.get('Bounce Min Slope Ticks', 22.0)),
        enable_cont=p.get('Cont Enable', True),
        cont_priority=p.get('Cont Priority', 85),
        cont_window=p.get('Cont Window Bars', 20),
        cont_min_dist_ticks=float(p.get('Cont Min Distance Ticks', 18.0)),
        cont_max_dist_ticks=float(p.get('Cont Max Distance Ticks', 25.0)),
        cont_slope_filter=p.get('Cont Slope Filter', True),
        cont_min_slope_ticks=float(p.get('Cont Min Slope Ticks', 2.0)),
        stop_ticks=int(p.get('Stop Ticks', 100)),
        target_ticks=int(p.get('Target Ticks', 200)),
        trail_enable=p.get('Trail Enable', True),
        trail_trigger=int(p.get('Trail Trigger', 140)),
        trail_offset=int(p.get('Trail Offset', 5)),
        be_enable=p.get('BE Enable', True),
        be_trigger=int(p.get('BE Trigger', 90)),
        be_offset=int(p.get('BE Offset', 20)),
        lock_bars=int(p.get('Lockout Bars', 10)),
        allow_longs=p.get('Allow Longs', True),
        allow_shorts=p.get('Allow Shorts', True),
        squeeze_enable=p.get('Squeeze Filter Enable', False),
        ard_enable=p.get('Adverse Regime Enable', True),
        ard_activation=p.get('Activation Mode', 'Any One'),
        ard_mode=p.get('Response Mode', 'Block All'),
        ard_var_threshold=float(p.get('VAR Threshold', 1.4)),
        ard_dsi_threshold=float(p.get('DSI Threshold', 0.65)),
        ard_dsi_lookback=int(p.get('DSI Lookback', 10)),
        ard_map_threshold=float(p.get('MAP Threshold', 1.4)),
        ard_map_lookback=int(p.get('MAP Lookback', 10)),
        dlc_enable=p.get('Daily Loss Cap Enable', True),
        dlc_ticks=int(p.get('Daily Loss Cap (Ticks)', 150)),
        eod_enable=p.get('EOD Close Enable', True),
        eod_hour=int(p.get('EOD Hour', 15)),
        eod_minute=int(p.get('EOD Minute', 50)),
        exclude_enable=p.get('Exclude Enable', True),
        exclude_hours=p.get('Exclude Hours 1', '1445-1800'),
        exclude2_enable=p.get('Exclude 2 Enable', True),
        exclude_hours2=p.get('Exclude Hours 2', '0600-0630'),
        point_value=100.0,
        commission=5.0,
        initial_capital=1000000.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# COMBO RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

# ARD mapping constants
_ARD_MODE_MAP = {"Block All": 0, "Block Longs Only": 1}
_ARD_ACT_MAP = {"VAR Only": 0, "Any One": 1, "Any Two": 2, "All Three": 3}


def run_combo(data, indicators, session, params, combo_idx, w_start, w_end, w_type,
              target_start_idx=None, n_bars=None):
    """Run a single combo on pre-computed indicators. Returns WindowResult or None.

    Args:
        data: DataFrame with OHLCV for the full window (including lookback).
        indicators: dict from compute_indicators() for this window.
        session: tuple (session_ok, eod_mask) from compute_session_mask().
        params: VectorParams instance for this combo.
        combo_idx: integer combo index.
        w_start: window start date string (e.g. '2025-01-01').
        w_end: window end date string (e.g. '2025-02-01').
        w_type: window type string ('month' or 'week').
        target_start_idx: pre-computed index of window start (optional, avoids recomputing per combo).
        n_bars: pre-computed bar count in target window (optional).
    """
    ind = indicators
    sess_ok, eod_m = session[0], session[1]
    n_full = ind['n']
    tick = params.tick_size

    # Find target window start index (use pre-computed if available)
    if target_start_idx is None:
        target_start_idx = np.searchsorted(data.index, pd.Timestamp(w_start, tz='UTC'))

    # Detect patterns with this combo's params
    long_sig, short_sig = detect_patterns(
        ind['close'], ind['high'], ind['low'], ind['open'],
        ind['vector_line'], ind['vector_slope'], ind['abs_slope'],
        ind['bull_cross'], ind['bear_cross'], ind['bars_since_cross'],
        ind['bar_range'], ind['body_ratio'], ind['close_position'],
        ind['bull_candle'], ind['bear_candle'], ind['dist_from_vec'], tick,
        params.enable_cross, params.cross_priority, params.cross_min_penetration,
        params.cross_max_penetration, params.cross_bar_filter, params.cross_min_bar_ticks,
        params.cross_body_filter, params.cross_min_body_ratio,
        params.cross_close_filter, params.cross_close_strength,
        params.cross_slope_filter, params.cross_min_slope_ticks,
        params.enable_bounce, params.bounce_priority, params.bounce_touch_ticks,
        params.bounce_min_reversal_ticks, params.bounce_bar_filter, params.bounce_min_bar_ticks,
        params.bounce_close_filter, params.bounce_close_strength,
        params.bounce_slope_filter, params.bounce_min_slope_ticks,
        params.enable_cont, params.cont_priority, params.cont_window,
        params.cont_min_dist_ticks, params.cont_max_dist_ticks,
        params.cont_slope_filter, params.cont_min_slope_ticks,
    )

    # ARD filter
    if params.ard_enable:
        long_ok, short_ok = compute_ard(
            ind['var_ratio'], ind['stress_bar'], ind['down_range'], ind['up_range'], n_full,
            params.ard_var_threshold, params.ard_dsi_threshold, params.ard_dsi_lookback,
            params.ard_map_threshold, params.ard_map_lookback,
            _ARD_ACT_MAP.get(params.ard_activation, 1),
            _ARD_MODE_MAP.get(params.ard_mode, 0))
    else:
        long_ok = np.ones(n_full, dtype=np.bool_)
        short_ok = np.ones(n_full, dtype=np.bool_)

    # Run simulation on full window (with lookback for indicator warmup)
    order_records, oc = run_simulation(
        ind['open'], ind['high'], ind['low'], ind['close'], n_full,
        long_sig, short_sig, sess_ok, eod_m, long_ok, short_ok,
        tick, params.point_value, params.commission, params.initial_capital,
        params.stop_ticks, params.target_ticks,
        params.trail_enable, params.trail_trigger, params.trail_offset,
        params.be_enable, params.be_trigger, params.be_offset,
        params.lock_bars, params.allow_longs, params.allow_shorts,
        params.squeeze_enable, ind['squeeze_ok'],
        params.dlc_enable, params.dlc_ticks, 1380,
    )

    # Extract trades in target window only
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

    # Count target bars
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
    PARAMS_CLASS = VectorParams
    NAME = 'vector_v12'

    @staticmethod
    def map_params(full_params):
        """Map database parameter names to VectorParams kwargs."""
        return map_params(full_params)

    @staticmethod
    def make_params(full_params):
        """Create VectorParams from database full_params dict."""
        return VectorParams(**map_params(full_params))

    @staticmethod
    def compute_indicators(data, params):
        """Pre-compute indicators for a data window. Called once per window."""
        return compute_indicators(data, params.vector_length)

    @staticmethod
    def compute_session(data, params):
        """Compute session mask. Called once per window."""
        return compute_session_mask(data.index, params)

    @staticmethod
    def run_combo(data, indicators, session, params, combo_idx, w_start, w_end, w_type,
                  target_start_idx=None, n_bars=None):
        """Run a single combo on pre-computed indicators. Returns WindowResult or None."""
        return run_combo(data, indicators, session, params, combo_idx, w_start, w_end, w_type,
                         target_start_idx, n_bars)
