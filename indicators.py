"""
Technical indicators library — TradingView-compatible calculations.

All functions accept pandas Series and return pandas Series.
NaN handling matches TradingView behavior (initial NaN period for lookback).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rma(series: pd.Series, period) -> pd.Series:
    """Wilder's smoothing (RMA) — TradingView-compatible."""
    period = int(period)
    values = series.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)

    if n < period:
        return pd.Series(result, index=series.index)

    start = 0
    while start < n and np.isnan(values[start]):
        start += 1

    seed_end = start + period
    if seed_end > n:
        return pd.Series(result, index=series.index)

    result[seed_end - 1] = np.mean(values[start:seed_end])

    alpha = 1.0 / period
    for i in range(seed_end, n):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]

    return pd.Series(result, index=series.index)


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (TradingView-compatible)."""
    values = series.values.astype(float)
    n = len(values)
    alpha = 2.0 / (period + 1)
    result = np.full(n, np.nan)

    if n < period:
        return pd.Series(result, index=series.index)

    start = 0
    while start < n and np.isnan(values[start]):
        start += 1

    seed_end = start + period
    if seed_end > n:
        return pd.Series(result, index=series.index)

    result[seed_end - 1] = np.mean(values[start:seed_end])

    for i in range(seed_end, n):
        if np.isnan(values[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]

    return pd.Series(result, index=series.index)


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average."""
    half_period = int(period / 2)
    sqrt_period = round(np.sqrt(period))
    wma_half = wma(series, half_period)
    wma_full = wma(series, period)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_period)


def vwma(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """Volume Weighted Moving Average."""
    return (close * volume).rolling(window=period, min_periods=period).sum() / \
           volume.rolling(window=period, min_periods=period).sum()


def zlema(series: pd.Series, period: int) -> pd.Series:
    """Zero-Lag EMA."""
    lag = int((period - 1) / 2)
    adjusted = series + (series - series.shift(lag))
    return ema(adjusted, period)


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (TradingView-compatible)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    gain_vals = gain.values.astype(float)
    loss_vals = loss.values.astype(float)
    n = len(gain_vals)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    if n < period + 1:
        return pd.Series(np.full(n, np.nan), index=series.index)

    avg_gain[period] = np.nanmean(gain_vals[1:period + 1])
    avg_loss[period] = np.nanmean(loss_vals[1:period + 1])

    alpha = 1.0 / period
    for i in range(period + 1, n):
        avg_gain[i] = alpha * gain_vals[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * loss_vals[i] + (1 - alpha) * avg_loss[i - 1]

    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(result, index=series.index)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD. Returns: (macd_line, signal_line, histogram)"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, k_smooth: int = 1, d_smooth: int = 3):
    """Stochastic Oscillator. Returns: (k_line, d_line)"""
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k_line = sma(raw_k, k_smooth) if k_smooth > 1 else raw_k
    d_line = sma(k_line, d_smooth)
    return k_line, d_line


def stochastic_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14,
                   k_smooth: int = 3, d_smooth: int = 3):
    """Stochastic RSI. Returns (k_line, d_line) in range 0-100."""
    rsi_val = rsi(series, rsi_period)
    lowest_rsi = rsi_val.rolling(window=stoch_period, min_periods=stoch_period).min()
    highest_rsi = rsi_val.rolling(window=stoch_period, min_periods=stoch_period).max()
    stoch_rsi = 100 * (rsi_val - lowest_rsi) / (highest_rsi - lowest_rsi)
    k_line = sma(stoch_rsi, k_smooth) if k_smooth > 1 else stoch_rsi
    d_line = sma(k_line, d_smooth)
    return k_line, d_line


def cci(source: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index (TradingView-compatible)."""
    source_sma = sma(source, period)
    mean_dev = source.rolling(window=period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (source - source_sma) / (0.015 * mean_dev)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R."""
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (TradingView-compatible, Wilder's smoothing)."""
    tr = true_range(high, low, close)
    return _rma(tr, period)


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands. Returns: (upper, middle, lower)"""
    middle = sma(series, period)
    rolling_std = series.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    return upper, middle, lower


def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                     ema_period: int = 20, atr_period: int = 10, multiplier: float = 1.5):
    """Keltner Channels. Returns: (upper, middle, lower)"""
    middle = ema(close, ema_period)
    atr_val = atr(high, low, close, atr_period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return upper, middle, lower


def bollinger_pctb(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger %B — position of price within the bands (0=lower, 1=upper)."""
    upper, _, lower = bollinger_bands(series, period, std_dev)
    return (series - lower) / (upper - lower)


# ---------------------------------------------------------------------------
# Trend Strength
# ---------------------------------------------------------------------------

def efficiency_ratio(series: pd.Series, period: int = 10) -> pd.Series:
    """Kaufman Efficiency Ratio. 1.0=trending, 0.0=noise."""
    direction = (series - series.shift(period)).abs()
    volatility = series.diff().abs().rolling(window=period, min_periods=period).sum()
    return direction / volatility


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Average Directional Index (TradingView-compatible). Returns: (adx_line, plus_di, minus_di)"""
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low

    plus_dm[(plus_dm < 0) | (plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < 0) | (minus_dm < plus_dm)] = 0

    atr_val = atr(high, low, close, period)
    smooth_plus_dm = _rma(plus_dm, period)
    smooth_minus_dm = _rma(minus_dm, period)

    plus_di = 100.0 * smooth_plus_dm / atr_val
    minus_di = 100.0 * smooth_minus_dm / atr_val

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_line = _rma(dx, period)

    return adx_line, plus_di, minus_di


def choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> pd.Series:
    """Choppiness Index. High (>61.8)=choppy, Low (<38.2)=trending."""
    tr = true_range(high, low, close)
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    hh = high.rolling(window=period, min_periods=period).max()
    ll = low.rolling(window=period, min_periods=period).min()
    return 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)


def squeeze_momentum(high: pd.Series, low: pd.Series, close: pd.Series,
                     bb_period: int = 20, bb_mult: float = 2.0,
                     kc_period: int = 20, kc_mult: float = 1.5):
    """Squeeze Momentum. Returns: (momentum_histogram, squeeze_on)"""
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, bb_period, bb_mult)
    kc_upper, kc_mid, kc_lower = keltner_channels(high, low, close, kc_period, kc_period, kc_mult)

    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    midline = (high.rolling(kc_period, min_periods=kc_period).max() +
               low.rolling(kc_period, min_periods=kc_period).min()) / 2
    avg_mid = (midline + bb_mid) / 2
    momentum = close - avg_mid

    return momentum, squeeze_on


# ---------------------------------------------------------------------------
# Trend / UT Bot
# ---------------------------------------------------------------------------

def ut_bot(close: pd.Series, high: pd.Series, low: pd.Series,
           key_value: float = 1.0, atr_period: int = 10):
    """UT Bot Alerts. Returns: (trail_stop, buy_signal, sell_signal)"""
    atr_val = atr(high, low, close, atr_period)
    loss = key_value * atr_val

    n = len(close)
    trail = np.full(n, np.nan)

    for i in range(atr_period + 1, n):
        if np.isnan(close.iloc[i]) or np.isnan(loss.iloc[i]):
            continue

        c = close.iloc[i]
        l = loss.iloc[i]

        if np.isnan(trail[i - 1]):
            trail[i] = c - l
            continue

        prev = trail[i - 1]

        if c > prev:
            trail[i] = max(prev, c - l)
        elif c < prev:
            trail[i] = min(prev, c + l)
        else:
            trail[i] = c - l

    trail_s = pd.Series(trail, index=close.index)
    buy_signal = crossover(close, trail_s)
    sell_signal = crossunder(close, trail_s)

    return trail_s, buy_signal, sell_signal


def wavetrend(high: pd.Series, low: pd.Series, close: pd.Series,
              channel_len: int = 9, avg_len: int = 12, ma_len: int = 3):
    """WaveTrend Oscillator. Returns: (wt1, wt2)"""
    hlc3 = (high + low + close) / 3.0
    esa_val = ema(hlc3, channel_len)
    de = ema((hlc3 - esa_val).abs(), channel_len)
    ci = (hlc3 - esa_val) / (0.015 * de)
    wt1 = ema(ci, avg_len)
    wt2 = sma(wt1, ma_len)
    return wt1, wt2


# ---------------------------------------------------------------------------
# Utility / Crossover
# ---------------------------------------------------------------------------

def crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """True when series_a crosses above series_b."""
    return (series_a > series_b) & (series_a.shift(1) <= series_b.shift(1))


def crossunder(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """True when series_a crosses below series_b."""
    return (series_a < series_b) & (series_a.shift(1) >= series_b.shift(1))


def highest(series: pd.Series, period: int) -> pd.Series:
    """Highest value over period."""
    return series.rolling(window=period, min_periods=period).max()


def lowest(series: pd.Series, period: int) -> pd.Series:
    """Lowest value over period."""
    return series.rolling(window=period, min_periods=period).min()


def change(series: pd.Series, period: int = 1) -> pd.Series:
    """Change over period."""
    return series.diff(period)
