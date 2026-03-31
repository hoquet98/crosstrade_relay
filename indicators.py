"""
Technical indicators library â TradingView-compatible calculations.

All functions accept pandas Series and return pandas Series.
NaN handling matches TradingView behavior (initial NaN period for lookback).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rma(series: pd.Series, period) -> pd.Series:
    """Wilder's smoothing (RMA) â TradingView-compatible.

    Used internally by RSI, ATR, ADX. Seeds with SMA of the first `period`
    values, then applies exponential smoothing with alpha = 1/period.
    Returns NaN for the first `period-1` bars.
    """
    period = int(period)  # ensure integer (keltner_channels passes float)
    values = series.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)

    if n < period:
        return pd.Series(result, index=series.index)

    # Find first window of `period` non-NaN values for SMA seed
    # (handles TR which has NaN at index 0 due to shift)
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
    """Exponential Moving Average (TradingView-compatible).

    TV initializes EMA with SMA of the first `period` values, then applies
    exponential smoothing from bar `period` onwards. Returns NaN for the
    first `period-1` bars â matching TV's ta.ema() exactly.

    Handles leading NaN values (e.g. when chaining ema(ema(src, p), p))
    by skipping to the first non-NaN value before seeding.
    """
    values = series.values.astype(float)
    n = len(values)
    alpha = 2.0 / (period + 1)
    result = np.full(n, np.nan)

    if n < period:
        return pd.Series(result, index=series.index)

    # Skip leading NaN values (needed for chained EMA calls)
    start = 0
    while start < n and np.isnan(values[start]):
        start += 1

    seed_end = start + period
    if seed_end > n:
        return pd.Series(result, index=series.index)

    # Seed with SMA of first `period` non-NaN values (matches TV)
    result[seed_end - 1] = np.mean(values[start:seed_end])

    # Exponential smoothing from seed onwards
    for i in range(seed_end, n):
        if np.isnan(values[i]):
            result[i] = result[i - 1]  # hold last value through NaN gaps
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
    sqrt_period = int(np.sqrt(period))
    wma_half = wma(series, half_period)
    wma_full = wma(series, period)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_period)


def vwma(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """Volume Weighted Moving Average."""
    return (close * volume).rolling(window=period, min_periods=period).sum() / \
           volume.rolling(window=period, min_periods=period).sum()


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (TradingView-compatible).

    TV uses Wilder's smoothing (RMA with alpha=1/period), seeded with SMA
    of the first `period` gain/loss values. Returns NaN for the first
    `period` bars.

    Previous implementation used pandas ewm() which seeds differently,
    causing divergence on early bars vs TradingView's ta.rsi().
    """
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

    # Seed with SMA of first `period` values (bars 1..period, since bar 0 is NaN from diff)
    avg_gain[period] = np.nanmean(gain_vals[1:period + 1])
    avg_loss[period] = np.nanmean(loss_vals[1:period + 1])

    # Wilder's smoothing (RMA): alpha = 1/period
    alpha = 1.0 / period
    for i in range(period + 1, n):
        avg_gain[i] = alpha * gain_vals[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * loss_vals[i] + (1 - alpha) * avg_loss[i - 1]

    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(result, index=series.index)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD (Moving Average Convergence Divergence).

    Returns: (macd_line, signal_line, histogram)
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, k_smooth: int = 1, d_smooth: int = 3):
    """Stochastic Oscillator.

    Returns: (k_line, d_line)
    """
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k_line = sma(raw_k, k_smooth) if k_smooth > 1 else raw_k
    d_line = sma(k_line, d_smooth)
    return k_line, d_line


def cci(source: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index (TradingView-compatible).

    Matches TV's ta.cci(source, period). Pass any source series:
      - close for CCI on close prices: cci(close, 20)
      - hlc3 for traditional CCI: cci((high+low+close)/3, 20)

    BREAKING CHANGE: Previous signature was cci(high, low, close, period)
    which computed hlc3 internally. Now accepts a single source series
    to exactly match TV's ta.cci() API.
    """
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


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
        period: int = 14) -> pd.Series:
    """Money Flow Index."""
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume

    positive_mf = pd.Series(0.0, index=close.index)
    negative_mf = pd.Series(0.0, index=close.index)

    tp_diff = tp.diff()
    positive_mf[tp_diff > 0] = raw_mf[tp_diff > 0]
    negative_mf[tp_diff < 0] = raw_mf[tp_diff < 0]

    pos_sum = positive_mf.rolling(window=period, min_periods=period).sum()
    neg_sum = negative_mf.rolling(window=period, min_periods=period).sum()

    mf_ratio = pos_sum / neg_sum
    return 100.0 - (100.0 / (1.0 + mf_ratio))


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change (percentage)."""
    shifted = series.shift(period)
    return ((series - shifted) / shifted) * 100.0


def cmo(series: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator."""
    delta = series.diff()
    up_sum = delta.clip(lower=0).rolling(window=period, min_periods=period).sum()
    down_sum = (-delta).clip(lower=0).rolling(window=period, min_periods=period).sum()
    return 100.0 * (up_sum - down_sum) / (up_sum + down_sum)


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
    """Average True Range (TradingView-compatible, Wilder's smoothing).

    TV uses RMA (Wilder's smoothing) seeded with SMA of first `period` TR values.
    """
    tr = true_range(high, low, close)
    return _rma(tr, period)


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands.

    Returns: (upper, middle, lower)
    """
    middle = sma(series, period)
    rolling_std = series.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    return upper, middle, lower


def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                     ema_period: int = 20, atr_period: int = 10, multiplier: float = 1.5):
    """Keltner Channels.

    Returns: (upper, middle, lower)
    """
    middle = ema(close, ema_period)
    atr_val = atr(high, low, close, atr_period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return upper, middle, lower


def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20):
    """Donchian Channels.

    Returns: (upper, lower, middle)
    """
    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2
    return upper, lower, middle


def bollinger_pctb(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger %B â position of price within the bands (0 = lower, 1 = upper)."""
    upper, _, lower = bollinger_bands(series, period, std_dev)
    return (series - lower) / (upper - lower)


def bollinger_width(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger Bandwidth â band width as fraction of middle band."""
    upper, middle, lower = bollinger_bands(series, period, std_dev)
    return (upper - lower) / middle


# ---------------------------------------------------------------------------
# Trend Strength
# ---------------------------------------------------------------------------

def efficiency_ratio(series: pd.Series, period: int = 10) -> pd.Series:
    """Kaufman Efficiency Ratio.

    Measures trend efficiency: 1.0 = perfectly trending, 0.0 = pure noise.
    Direction (net change) divided by volatility (sum of absolute bar changes).
    """
    direction = (series - series.shift(period)).abs()
    volatility = series.diff().abs().rolling(window=period, min_periods=period).sum()
    return direction / volatility


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Average Directional Index (TradingView-compatible).

    Uses SMA-seeded Wilder's smoothing for DM, ATR, and DX â matching
    TV's ta.dmi() / ta.adx() calculations.

    Returns: (adx_line, plus_di, minus_di)
    """
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


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0):
    """Supertrend indicator.

    Returns: (supertrend_line, direction)  direction: 1=bullish, -1=bearish
    """
    atr_val = atr(high, low, close, period)
    hl2 = (high + low) / 2

    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    supertrend_arr = np.zeros(len(close))
    direction_arr = np.ones(len(close))

    supertrend_arr[:period] = np.nan
    direction_arr[:period] = np.nan

    for i in range(period, len(close)):
        if np.isnan(upper_band.iloc[i]) or np.isnan(lower_band.iloc[i]):
            supertrend_arr[i] = np.nan
            direction_arr[i] = np.nan
            continue

        if close.iloc[i] > supertrend_arr[i - 1]:
            supertrend_arr[i] = max(lower_band.iloc[i],
                                     supertrend_arr[i - 1] if direction_arr[i - 1] == 1 else lower_band.iloc[i])
            direction_arr[i] = 1
        else:
            supertrend_arr[i] = min(upper_band.iloc[i],
                                     supertrend_arr[i - 1] if direction_arr[i - 1] == -1 else upper_band.iloc[i])
            direction_arr[i] = -1

    return pd.Series(supertrend_arr, index=close.index), pd.Series(direction_arr, index=close.index)


def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
             tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
    """Ichimoku Cloud.

    Returns: (tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span)
    Note: senkou spans are NOT shifted forward (caller handles displacement).
    """
    tenkan_sen = (high.rolling(tenkan, min_periods=tenkan).max() +
                  low.rolling(tenkan, min_periods=tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun, min_periods=kijun).max() +
                 low.rolling(kijun, min_periods=kijun).min()) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b_val = (high.rolling(senkou_b, min_periods=senkou_b).max() +
                         low.rolling(senkou_b, min_periods=senkou_b).min()) / 2
    chikou_span = close  # shifted back by kijun periods by caller

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b_val, chikou_span


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (volume * direction).cumsum()


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
    """Change over period (series - series[period ago])."""
    return series.diff(period)


def pivothigh(series: pd.Series, left_bars: int, right_bars: int) -> pd.Series:
    """Pivot high detection."""
    result = pd.Series(np.nan, index=series.index)
    for i in range(left_bars, len(series) - right_bars):
        val = series.iloc[i]
        left_slice = series.iloc[i - left_bars:i]
        right_slice = series.iloc[i + 1:i + 1 + right_bars]
        if (val >= left_slice).all() and (val >= right_slice).all():
            result.iloc[i + right_bars] = val
    return result


def pivotlow(series: pd.Series, left_bars: int, right_bars: int) -> pd.Series:
    """Pivot low detection."""
    result = pd.Series(np.nan, index=series.index)
    for i in range(left_bars, len(series) - right_bars):
        val = series.iloc[i]
        left_slice = series.iloc[i - left_bars:i]
        right_slice = series.iloc[i + 1:i + 1 + right_bars]
        if (val <= left_slice).all() and (val <= right_slice).all():
            result.iloc[i + right_bars] = val
    return result


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# NEW INDICATORS â Batch conversion from Pine Source + from scratch
# Added for comprehensive strategy sweep
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ


# ---------------------------------------------------------------------------
# Moving Averages (new)
# ---------------------------------------------------------------------------

def dema(series: pd.Series, period: int) -> pd.Series:
    """Double Exponential Moving Average.

    DEMA = 2 * EMA(src, len) - EMA(EMA(src, len), len)
    Reduces lag compared to standard EMA.
    """
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    return 2 * ema1 - ema2


def tema(series: pd.Series, period: int) -> pd.Series:
    """Triple Exponential Moving Average.

    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    Even less lag than DEMA.
    """
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3


def zlema(series: pd.Series, period: int) -> pd.Series:
    """Zero-Lag EMA.

    Compensates for EMA lag by pre-adjusting the source:
    lag = (period - 1) / 2
    adjusted_src = src + (src - src[lag])
    zlema = EMA(adjusted_src, period)
    """
    lag = int((period - 1) / 2)
    adjusted = series + (series - series.shift(lag))
    return ema(adjusted, period)


def t3(series: pd.Series, period: int = 5, vfactor: float = 0.618) -> pd.Series:
    """Tillson T3 â 6x cascaded EMA with volume factor.

    Ported from Pine Script: indicators/pine_source/T3 Oscillator.pine
    Extremely smooth moving average with minimal lag.
    The volume factor (default 0.618) controls responsiveness.

    Returns
    -------
    pd.Series
        T3 smoothed values
    """
    c1 = -vfactor ** 3
    c2 = 3 * vfactor ** 2 + 3 * vfactor ** 3
    c3 = -6 * vfactor ** 2 - 3 * vfactor - 3 * vfactor ** 3
    c4 = 1 + 3 * vfactor + vfactor ** 3 + 3 * vfactor ** 2

    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    e4 = ema(e3, period)
    e5 = ema(e4, period)
    e6 = ema(e5, period)

    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def alma(series: pd.Series, period: int = 9, offset: float = 0.85,
         sigma: float = 6.0) -> pd.Series:
    """Arnaud Legoux Moving Average (ALMA).

    Gaussian-weighted MA with adjustable offset and sigma.
    offset=0.85 biases toward recent data, sigma=6 controls smoothness.
    """
    m = offset * (period - 1)
    s = period / sigma
    weights = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(period)])
    weights = weights / weights.sum()

    return series.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights), raw=True
    )


# ---------------------------------------------------------------------------
# Ehlers Filters (from Pine Source)
# ---------------------------------------------------------------------------

def ehlers_super_smoother(series: pd.Series, period: int = 10) -> pd.Series:
    """Ehlers Super Smoother Filter.

    Ported from Pine Script: indicators/pine_source/Ehlers Super Smoother filter.pine
    2-pole recursive filter that removes high-frequency noise while
    preserving low-frequency trend information. Much less lag than
    equivalent-period SMA/EMA.

    Flags: LOOP_BASED
    """
    pi = np.pi
    values = series.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)

    a1 = np.exp(-1.414 * pi / period)
    b1 = 2 * a1 * np.cos(1.414 * 2 * pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    for i in range(2, n):
        if np.isnan(values[i]) or np.isnan(values[i - 1]):
            continue
        prev1 = result[i - 1] if not np.isnan(result[i - 1]) else values[i - 1]
        prev2 = result[i - 2] if not np.isnan(result[i - 2]) else values[i - 1]
        result[i] = c1 * (values[i] + values[i - 1]) / 2 + c2 * prev1 + c3 * prev2

    return pd.Series(result, index=series.index)


def ehlers_instantaneous_trendline(series: pd.Series, alpha: float = 0.07):
    """Ehlers Instantaneous Trendline.

    Ported from Pine Script: indicators/pine_source/Ehlers Instantaneous Trendline.pine
    Recursive filter that estimates the instantaneous trend.
    Returns (itrend, trigger) where trigger = 2*itrend - itrend[2].
    Signal: trigger > itrend = bullish, trigger < itrend = bearish.

    Flags: LOOP_BASED

    Returns
    -------
    (pd.Series, pd.Series)
        (itrend_line, trigger_line)
    """
    values = series.values.astype(float)
    n = len(values)
    itrend = np.full(n, np.nan)

    for i in range(2, n):
        if i < 7:
            itrend[i] = (values[i] + 2 * values[i - 1] + values[i - 2]) / 4
        else:
            prev1 = itrend[i - 1] if not np.isnan(itrend[i - 1]) else values[i - 1]
            prev2 = itrend[i - 2] if not np.isnan(itrend[i - 2]) else values[i - 2]
            itrend[i] = ((alpha - alpha ** 2 / 4) * values[i]
                         + 0.5 * alpha ** 2 * values[i - 1]
                         - (alpha - 0.75 * alpha ** 2) * values[i - 2]
                         + 2 * (1 - alpha) * prev1
                         - (1 - alpha) ** 2 * prev2)

    itrend_s = pd.Series(itrend, index=series.index)
    trigger_s = 2 * itrend_s - itrend_s.shift(2)
    return itrend_s, trigger_s


# ---------------------------------------------------------------------------
# Oscillators (new)
# ---------------------------------------------------------------------------

def stochastic_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14,
                   k_smooth: int = 3, d_smooth: int = 3):
    """Stochastic RSI.

    RSI passed through the stochastic formula, then smoothed.
    Returns (k_line, d_line) in range 0-100.
    """
    rsi_val = rsi(series, rsi_period)
    lowest_rsi = rsi_val.rolling(window=stoch_period, min_periods=stoch_period).min()
    highest_rsi = rsi_val.rolling(window=stoch_period, min_periods=stoch_period).max()
    stoch_rsi = 100 * (rsi_val - lowest_rsi) / (highest_rsi - lowest_rsi)
    k_line = sma(stoch_rsi, k_smooth) if k_smooth > 1 else stoch_rsi
    d_line = sma(k_line, d_smooth)
    return k_line, d_line


def trix(series: pd.Series, period: int = 18) -> pd.Series:
    """TRIX â Triple Exponential Average rate of change.

    Rate of change of a triple-smoothed EMA. Oscillator that filters noise.
    Positive = bullish momentum, Negative = bearish.
    """
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return 10000 * (ema3 - ema3.shift(1)) / ema3.shift(1)


def wavetrend(high: pd.Series, low: pd.Series, close: pd.Series,
              channel_len: int = 9, avg_len: int = 12, ma_len: int = 3):
    """WaveTrend Oscillator (LazyBear).

    Core oscillator from VuManChu Cipher B. Measures normalized price
    deviation from its smoothed average using channeling.

    Ported from: indicators/pine_source/VuManChu_Cipher_B_Divergences.pine

    Returns
    -------
    (pd.Series, pd.Series)
        (wt1, wt2) â wt1 is the fast line, wt2 is the signal (SMA of wt1)
    """
    hlc3 = (high + low + close) / 3.0
    esa = ema(hlc3, channel_len)
    de = ema((hlc3 - esa).abs(), channel_len)
    ci = (hlc3 - esa) / (0.015 * de)
    wt1 = ema(ci, avg_len)
    wt2 = sma(wt1, ma_len)
    return wt1, wt2


def schaff_trend_cycle(series: pd.Series, length: int = 10,
                       fast: int = 23, slow: int = 50,
                       factor: float = 0.5) -> pd.Series:
    """Schaff Trend Cycle (STC).

    Double-stochastic of MACD. Oscillates 0-100.
    Rising through 25 = bullish, falling through 75 = bearish.
    Ported from: VuManChu Cipher B Pine source (f_tc function).

    Flags: LOOP_BASED
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_val = ema_fast - ema_slow

    values = macd_val.values.astype(float)
    n = len(values)
    delta = np.full(n, np.nan)
    stc = np.full(n, np.nan)

    for i in range(length, n):
        window = values[i - length + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 2:
            continue
        lo = np.min(valid)
        hi = np.max(valid)
        gamma = ((values[i] - lo) / (hi - lo) * 100) if hi > lo else (delta[i - 1] if not np.isnan(delta[i - 1]) else 0)
        if np.isnan(delta[i - 1]):
            delta[i] = gamma
        else:
            delta[i] = delta[i - 1] + factor * (gamma - delta[i - 1])

    # Second stochastic pass
    for i in range(length, n):
        if np.isnan(delta[i]):
            continue
        window = delta[max(0, i - length + 1):i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 2:
            continue
        lo = np.min(valid)
        hi = np.max(valid)
        eta = ((delta[i] - lo) / (hi - lo) * 100) if hi > lo else (stc[i - 1] if not np.isnan(stc[i - 1]) else 0)
        if np.isnan(stc[i - 1]):
            stc[i] = eta
        else:
            stc[i] = stc[i - 1] + factor * (eta - stc[i - 1])

    return pd.Series(stc, index=series.index)


def fisher_transform(high: pd.Series, low: pd.Series, period: int = 9):
    """Fisher Transform.

    Converts price into a Gaussian normal distribution.
    Sharp peaks at extremes, clear turning points.

    Returns
    -------
    (pd.Series, pd.Series)
        (fisher, trigger) â trigger is fisher[1]
    """
    hl2 = (high + low) / 2.0
    lowest_hl2 = hl2.rolling(window=period, min_periods=period).min()
    highest_hl2 = hl2.rolling(window=period, min_periods=period).max()

    raw = 2 * ((hl2 - lowest_hl2) / (highest_hl2 - lowest_hl2) - 0.5)
    raw = raw.clip(-0.999, 0.999)  # avoid log(0) or log(negative)

    values = raw.values.astype(float)
    n = len(values)
    fisher = np.full(n, np.nan)

    for i in range(period, n):
        if np.isnan(values[i]):
            continue
        prev = fisher[i - 1] if not np.isnan(fisher[i - 1]) else 0.0
        fisher[i] = 0.5 * np.log((1 + values[i]) / (1 - values[i])) + 0.5 * prev

    fisher_s = pd.Series(fisher, index=high.index)
    trigger_s = fisher_s.shift(1)
    return fisher_s, trigger_s


def awesome_oscillator(high: pd.Series, low: pd.Series,
                       fast: int = 5, slow: int = 34) -> pd.Series:
    """Awesome Oscillator (Bill Williams).

    Difference between 5-period and 34-period SMA of midpoint (hl2).
    Positive = bullish momentum, negative = bearish.
    """
    midpoint = (high + low) / 2.0
    return sma(midpoint, fast) - sma(midpoint, slow)


def aroon(high: pd.Series, low: pd.Series, period: int = 25):
    """Aroon Indicator.

    Measures time since last highest high / lowest low.
    Aroon Up > 70 = strong uptrend, Aroon Down > 70 = strong downtrend.

    Returns
    -------
    (pd.Series, pd.Series)
        (aroon_up, aroon_down)
    """
    n = len(high)
    up_arr = np.full(n, np.nan)
    down_arr = np.full(n, np.nan)

    for i in range(period, n):
        window_high = high.iloc[i - period:i + 1].values
        window_low = low.iloc[i - period:i + 1].values
        bars_since_high = period - np.argmax(window_high)
        bars_since_low = period - np.argmin(window_low)
        up_arr[i] = ((period - bars_since_high) / period) * 100
        down_arr[i] = ((period - bars_since_low) / period) * 100

    return pd.Series(up_arr, index=high.index), pd.Series(down_arr, index=low.index)


def vortex(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Vortex Indicator.

    Measures positive and negative trend movement using true range.
    VI+ > VI- = uptrend, VI- > VI+ = downtrend. Crossovers are signals.

    Returns
    -------
    (pd.Series, pd.Series)
        (vi_plus, vi_minus)
    """
    tr = true_range(high, low, close)
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()

    sum_tr = tr.rolling(window=period, min_periods=period).sum()
    sum_vmp = vm_plus.rolling(window=period, min_periods=period).sum()
    sum_vmm = vm_minus.rolling(window=period, min_periods=period).sum()

    vi_plus = sum_vmp / sum_tr
    vi_minus = sum_vmm / sum_tr
    return vi_plus, vi_minus


def choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> pd.Series:
    """Choppiness Index.

    Measures whether market is trending or ranging.
    High values (>61.8) = choppy/ranging, Low values (<38.2) = trending.
    Useful as a filter â only enter trades when CHOP is low.
    """
    tr = true_range(high, low, close)
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    hh = high.rolling(window=period, min_periods=period).max()
    ll = low.rolling(window=period, min_periods=period).min()
    return 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)


def balance_of_power(open_: pd.Series, high: pd.Series, low: pd.Series,
                     close: pd.Series, period: int = 14) -> pd.Series:
    """Balance of Power (BOP).

    Measures the strength of buyers vs sellers.
    BOP = SMA((close - open) / (high - low), period)
    Range: -1 to +1. Positive = buyers dominate.
    """
    raw = (close - open_) / (high - low)
    raw = raw.replace([np.inf, -np.inf], 0).fillna(0)
    return sma(raw, period)


def mass_index(high: pd.Series, low: pd.Series,
               ema_period: int = 9, sum_period: int = 25) -> pd.Series:
    """Mass Index.

    Detects trend reversals by measuring range expansion/contraction.
    A 'reversal bulge' occurs when MI rises above 27 then falls below 26.5.
    """
    hl_range = high - low
    ema1 = ema(hl_range, ema_period)
    ema2 = ema(ema1, ema_period)
    ratio = ema1 / ema2
    return ratio.rolling(window=sum_period, min_periods=sum_period).sum()


def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                        p1: int = 7, p2: int = 14, p3: int = 28) -> pd.Series:
    """Ultimate Oscillator (Larry Williams).

    Multi-timeframe weighted momentum oscillator. Range 0-100.
    Combines buying pressure over 3 periods with different weights.
    """
    prev_close = close.shift(1)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = true_range(high, low, close)

    avg1 = bp.rolling(p1, min_periods=p1).sum() / tr.rolling(p1, min_periods=p1).sum()
    avg2 = bp.rolling(p2, min_periods=p2).sum() / tr.rolling(p2, min_periods=p2).sum()
    avg3 = bp.rolling(p3, min_periods=p3).sum() / tr.rolling(p3, min_periods=p3).sum()

    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7


def connors_rsi(series: pd.Series, rsi_period: int = 3,
                streak_period: int = 2, rank_period: int = 100) -> pd.Series:
    """Connors RSI (CRSI).

    Combines: RSI + Streak RSI + Percent Rank of ROC.
    Designed for mean-reversion. Range 0-100.
    Low values = oversold, high values = overbought.

    Flags: LOOP_BASED
    """
    # 1. Standard RSI
    rsi_val = rsi(series, rsi_period)

    # 2. Streak RSI â streak of consecutive up/down closes
    diff = series.diff()
    values = diff.values
    n = len(values)
    streak = np.zeros(n)
    for i in range(1, n):
        if values[i] > 0:
            streak[i] = max(streak[i - 1], 0) + 1
        elif values[i] < 0:
            streak[i] = min(streak[i - 1], 0) - 1
        else:
            streak[i] = 0
    streak_s = pd.Series(streak, index=series.index)
    streak_rsi = rsi(streak_s, streak_period)

    # 3. Percent rank of 1-bar ROC
    roc1 = series.pct_change()
    pct_rank = roc1.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x[-1:] > x[:-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else 50,
        raw=True
    )

    return (rsi_val + streak_rsi + pct_rank) / 3


def kst(series: pd.Series, r1: int = 10, r2: int = 15, r3: int = 20, r4: int = 30,
        s1: int = 10, s2: int = 10, s3: int = 10, s4: int = 15,
        signal_period: int = 9):
    """Know Sure Thing (KST) by Martin Pring.

    Multi-timeframe ROC oscillator. Combines 4 smoothed ROC values.
    KST cross above signal = bullish, below = bearish.

    Returns
    -------
    (pd.Series, pd.Series)
        (kst_line, signal_line)
    """
    roc1 = roc(series, r1)
    roc2 = roc(series, r2)
    roc3 = roc(series, r3)
    roc4 = roc(series, r4)

    kst_line = (sma(roc1, s1) * 1 + sma(roc2, s2) * 2 +
                sma(roc3, s3) * 3 + sma(roc4, s4) * 4)
    signal_line = sma(kst_line, signal_period)
    return kst_line, signal_line


# ---------------------------------------------------------------------------
# Volatility (new)
# ---------------------------------------------------------------------------

def squeeze_momentum(high: pd.Series, low: pd.Series, close: pd.Series,
                     bb_period: int = 20, bb_mult: float = 2.0,
                     kc_period: int = 20, kc_mult: float = 1.5):
    """Squeeze Momentum Indicator (John Carter).

    Detects when Bollinger Bands are inside Keltner Channels (squeeze).
    When squeeze fires, a breakout is imminent.

    Returns
    -------
    (pd.Series, pd.Series)
        (momentum_histogram, squeeze_on)
        momentum_histogram: positive = bullish, negative = bearish
        squeeze_on: True when BB is inside KC (squeeze active)
    """
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, bb_period, bb_mult)

    # Keltner Channels
    kc_upper, kc_mid, kc_lower = keltner_channels(high, low, close, kc_period, kc_period, kc_mult)

    # Squeeze detection
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    # Momentum (linear regression of close - midline(highest+lowest+BB_mid+KC_mid)/4)
    # Simplified: use close - average of BB and KC midlines
    midline = (highest(high, kc_period) + lowest(low, kc_period)) / 2
    avg_mid = (midline + bb_mid) / 2
    momentum = close - avg_mid

    return momentum, squeeze_on


# ---------------------------------------------------------------------------
# Trend / Direction (new â from Pine Source)
# ---------------------------------------------------------------------------

def waddah_attar_explosion(close: pd.Series, fast: int = 20, slow: int = 40,
                           sensitivity: int = 150, bb_period: int = 20,
                           bb_mult: float = 2.0):
    """Waddah Attar Explosion (WAE).

    Ported from: indicators/pine_source/Deadzone Pro by Daviddtech.pine
    Combines MACD momentum change with Bollinger Band width as a deadzone.
    trend_up > deadzone = bullish momentum, trend_down > deadzone = bearish.

    Returns
    -------
    (pd.Series, pd.Series, pd.Series)
        (trend_up, trend_down, deadzone)
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_val = ema_fast - ema_slow
    macd_diff = (macd_val - macd_val.shift(1)) * sensitivity

    bb_upper, bb_mid, bb_lower = bollinger_bands(close, bb_period, bb_mult)
    deadzone = bb_upper - bb_lower

    trend_up = macd_diff.clip(lower=0)
    trend_down = (-macd_diff).clip(lower=0)

    return trend_up, trend_down, deadzone


def trend_akkam(high: pd.Series, low: pd.Series, close: pd.Series,
                atr_range: int = 100, atr_factor: float = 6.0):
    """Trend Akkam â ATR-based adaptive trailing stop.

    Ported from: indicators/pine_source/Trend Akkam.pine
    Uses ATR * factor as dynamic stop distance. Stays on one side of price
    until price crosses through. Similar to Supertrend but uses open-based logic.

    Returns
    -------
    (pd.Series, pd.Series)
        (trstop_line, direction)  direction: 1=bullish, -1=bearish

    Flags: LOOP_BASED
    """
    atr_val = atr(high, low, close, atr_range)
    delta_stop = atr_val * atr_factor

    n = len(close)
    trstop = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    for i in range(atr_range + 1, n):
        if np.isnan(close.iloc[i]) or np.isnan(delta_stop.iloc[i]):
            continue
        c = close.iloc[i]
        ds = delta_stop.iloc[i]

        if np.isnan(trstop[i - 1]):
            trstop[i] = c - ds
            direction[i] = 1
            continue

        prev = trstop[i - 1]
        prev_c = close.iloc[i - 1]

        if prev_c < prev and c < prev:
            trstop[i] = min(prev, c + ds)
            direction[i] = -1
        elif prev_c > prev and c > prev:
            trstop[i] = max(prev, c - ds)
            direction[i] = 1
        elif c > prev:
            trstop[i] = c - ds
            direction[i] = 1
        else:
            trstop[i] = c + ds
            direction[i] = -1

    return pd.Series(trstop, index=close.index), pd.Series(direction, index=close.index)


def ut_bot(close: pd.Series, high: pd.Series, low: pd.Series,
           key_value: float = 1.0, atr_period: int = 10):
    """UT Bot Alerts â ATR Trailing Stop System.

    Ported from: indicators/pine_source/UT BOT Alerts.pine
    Calculates a dynamic trailing stop based on ATR.
    Generates buy/sell signals when price crosses the trail.

    Returns
    -------
    (pd.Series, pd.Series, pd.Series)
        (trail_stop, buy_signal, sell_signal)

    Flags: LOOP_BASED
    """
    atr_val = atr(high, low, close, atr_period)
    loss = key_value * atr_val

    n = len(close)
    trail = np.full(n, np.nan)

    for i in range(atr_period + 1, n):
        if np.isnan(close.iloc[i]) or np.isnan(loss.iloc[i]):
            continue

        c = close.iloc[i]
        c1 = close.iloc[i - 1]
        l = loss.iloc[i]

        if np.isnan(trail[i - 1]):
            trail[i] = c - l
            continue

        prev = trail[i - 1]

        if c > prev and c1 > prev:
            trail[i] = max(prev, c - l)
        elif c < prev and c1 < prev:
            trail[i] = min(prev, c + l)
        elif c > prev:
            trail[i] = c - l
        else:
            trail[i] = c + l

    trail_s = pd.Series(trail, index=close.index)
    buy_signal = crossover(close, trail_s)
    sell_signal = crossunder(close, trail_s)

    return trail_s, buy_signal, sell_signal


# ---------------------------------------------------------------------------
# Market Regime / Specialized Oscillators (from Pine Source)
# ---------------------------------------------------------------------------

def williams_vix_fix(close: pd.Series, low: pd.Series,
                     period: int = 22, bb_period: int = 20,
                     bb_mult: float = 2.0):
    """Williams Vix Fix â Market Bottom Detector.

    Ported from: indicators/pine_source/CM Williams Vix Fix Market Bottoms.pine
    Synthetic VIX that detects market bottoms using highest-close drop.
    High readings = fear/capitulation = potential bottom.

    Returns
    -------
    (pd.Series, pd.Series, pd.Series)
        (wvf, upper_band, is_bottom)
        wvf: the synthetic VIX values
        upper_band: BB upper band on wvf (used as threshold)
        is_bottom: True when wvf >= upper_band (potential bottom)
    """
    highest_close = close.rolling(window=period, min_periods=period).max()
    wvf = ((highest_close - low) / highest_close) * 100

    wvf_sma = sma(wvf, bb_period)
    wvf_std = wvf.rolling(window=bb_period, min_periods=bb_period).std(ddof=0)
    upper_band = wvf_sma + bb_mult * wvf_std

    is_bottom = wvf >= upper_band

    return wvf, upper_band, is_bottom


def tops_bottoms_cvi(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 3, bull_level: float = -0.51,
                     bear_level: float = 0.43):
    """Tops/Bottoms CVI (Cumulative Volume Index variant).

    Ported from: indicators/pine_source/Tops and Bottoms.pine
    Normalizes price deviation from midline by volatility.
    Crossing bull_level from below = potential bottom signal.
    Crossing bear_level from above = potential top signal.

    Returns
    -------
    (pd.Series, pd.Series, pd.Series)
        (cvi, bull_signal, bear_signal)
    """
    hl2 = (high + low) / 2.0
    val_c = sma(hl2, period)
    vol = sma(atr(high, low, close, period), period)
    cvi = (close - val_c) / (vol * np.sqrt(period))

    # Signal: was below bull_level, now crossed above
    was_bull = cvi.shift(1) <= bull_level
    bull_signal = was_bull & (cvi > bull_level)

    # Signal: was above bear_level, now crossed below
    was_bear = cvi.shift(1) >= bear_level
    bear_signal = was_bear & (cvi < bear_level)

    return cvi, bull_signal, bear_signal


def impulse_macd(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 34) -> pd.Series:
    """Impulse MACD (LazyBear).

    Ported from: indicators/pine_source/Impulse MACD.pine
    Uses SMMA of high/low as channel + ZLEMA of hlc3 as midline.
    Histogram: positive when above channel, negative when below.
    """
    hlc3 = (high + low + close) / 3.0

    # SMMA = EMA with alpha=1/period = RMA
    hi_smma = _rma(high, period)
    lo_smma = _rma(low, period)

    # ZLEMA of hlc3
    mi = zlema(hlc3, period)

    # Histogram: above upper channel = positive, below lower = negative
    result = np.where(mi > hi_smma, mi - hi_smma,
                      np.where(mi < lo_smma, mi - lo_smma, 0.0))
    return pd.Series(result, index=close.index)


def trendilo(series: pd.Series, smooth: int = 7, period: int = 20) -> pd.Series:
    """Trendilo â ALMA-based trend oscillator.

    Ported from: indicators/pine_source/Confirmation Trendilo.pine
    Percentage change smoothed with ALMA, then compared to RMS bands.
    Positive = bullish trend, Negative = bearish trend.
    """
    pch = (series - series.shift(smooth)) / series * 100
    smoothed = alma(pch, period)
    return smoothed


def dorsey_inertia(high: pd.Series, low: pd.Series, close: pd.Series,
                   rvi_period: int = 14, smooth_period: int = 20) -> pd.Series:
    """Dorsey Inertia â RVI smoothed with linear regression.

    Ported from: indicators/pine_source/Confirmation Dorsey iniertia.pine
    RVI (Relative Volatility Index) measures whether volatility is rising
    in upward or downward price moves. Smoothed by linreg for trend confirmation.
    Above 50 = bullish inertia, below 50 = bearish.
    """
    delta = close.diff()
    std_val = close.rolling(rvi_period, min_periods=rvi_period).std()
    up_vol = pd.Series(np.where(delta > 0, std_val, 0), index=close.index)
    down_vol = pd.Series(np.where(delta <= 0, std_val, 0), index=close.index)

    up_ema = ema(up_vol, rvi_period)
    down_ema = ema(down_vol, rvi_period)

    rvi = 100 * up_ema / (up_ema + down_ema)
    rvi = rvi.fillna(50.0)

    # Smooth with SMA as linear regression proxy (faster, nearly identical)
    return sma(rvi, smooth_period)


def price_distance_to_ma(series: pd.Series, period: int = 20,
                         use_ema: bool = False) -> pd.Series:
    """Price Distance to MA (percentage).

    Ported from: indicators/pine_source/Price Distance to MA.pine
    Measures how far price is from its moving average as a percentage.
    Extreme positive = overbought, extreme negative = oversold.
    Mean-reversion signal when returning to zero.
    """
    ma = ema(series, period) if use_ema else sma(series, period)
    return (series / ma - 1) * 100


def parabolic_sar(high: pd.Series, low: pd.Series,
                  start: float = 0.02, increment: float = 0.02,
                  maximum: float = 0.2) -> pd.Series:
    """Parabolic SAR (Welles Wilder).

    Classic trend-following stop-and-reverse indicator.
    When price crosses SAR, trend reverses.

    Flags: LOOP_BASED

    Returns
    -------
    pd.Series
        SAR values (below price in uptrend, above in downtrend)
    """
    n = len(high)
    sar = np.full(n, np.nan)
    af = start
    is_long = True
    ep = high.iloc[0]
    sar_val = low.iloc[0]

    for i in range(2, n):
        if np.isnan(high.iloc[i]) or np.isnan(low.iloc[i]):
            continue

        prev_sar = sar_val
        sar_val = prev_sar + af * (ep - prev_sar)

        if is_long:
            # Ensure SAR doesn't go above prior two lows
            sar_val = min(sar_val, low.iloc[i - 1])
            if i >= 3:
                sar_val = min(sar_val, low.iloc[i - 2])

            if low.iloc[i] < sar_val:
                # Reverse to short
                is_long = False
                sar_val = ep
                ep = low.iloc[i]
                af = start
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + increment, maximum)
        else:
            # Ensure SAR doesn't go below prior two highs
            sar_val = max(sar_val, high.iloc[i - 1])
            if i >= 3:
                sar_val = max(sar_val, high.iloc[i - 2])

            if high.iloc[i] > sar_val:
                # Reverse to long
                is_long = True
                sar_val = ep
                ep = high.iloc[i]
                af = start
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + increment, maximum)

        sar[i] = sar_val

    return pd.Series(sar, index=high.index)


def relative_vigor_index(open_: pd.Series, high: pd.Series,
                         low: pd.Series, close: pd.Series,
                         period: int = 10):
    """Relative Vigor Index (RVI).

    Measures the conviction of a trend by comparing close-open to high-low
    over a period. Uses triangular weighting.

    Returns
    -------
    (pd.Series, pd.Series)
        (rvi_line, signal_line)
    """
    co = close - open_
    hl = high - low

    # Triangular weighted sum (weights: 1,2,2,1)
    num = (co + 2 * co.shift(1) + 2 * co.shift(2) + co.shift(3)) / 6
    den = (hl + 2 * hl.shift(1) + 2 * hl.shift(2) + hl.shift(3)) / 6

    sum_num = num.rolling(window=period, min_periods=period).sum()
    sum_den = den.rolling(window=period, min_periods=period).sum()

    rvi_line = sum_num / sum_den
    signal_line = (rvi_line + 2 * rvi_line.shift(1) + 2 * rvi_line.shift(2) + rvi_line.shift(3)) / 6

    return rvi_line, signal_line
