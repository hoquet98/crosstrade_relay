# Indicator Reference Guide

All indicators are computed server-side from stored 1-min bars. They are available for:
- Conditional entry/exit rules (Form View or Code View)
- AI prompt context
- Bot indicator selection
- Live Values panel in the bot wizard

---

## Whale Pressure Indicator (WPI)

Detects institutional accumulation/distribution by combining where price closed within the bar (CLV), candle body conviction, and ATR-normalized range.

### Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `whale_pressure` | number | -100 to +100 | Smoothed whale pressure. Positive = buying, negative = selling |
| `whale_pressure_raw` | number | -100 to +100 | Raw unsmoothed pressure (reacts faster, more noise) |
| `whale_zone` | number | -2, -1, 0, +1, +2 | Zone classification (see below) |
| `whale_conviction` | number | any integer | Consecutive bars in same direction. +5 = 5 bars of buying |
| `whale_momentum` | number | any | Rate of change of pressure. Positive = pressure increasing |
| `whale_buying` | bool | | Zone is +1 or +2 (buy or strong buy) |
| `whale_selling` | bool | | Zone is -1 or -2 (sell or strong sell) |
| `whale_strong_buy` | bool | | Zone is +2 |
| `whale_strong_sell` | bool | | Zone is -2 |
| `whale_neutral` | bool | | Zone is 0 |
| `whale_flipped_bull` | bool | | Zone just changed from neutral/sell to buy |
| `whale_flipped_bear` | bool | | Zone just changed from neutral/buy to sell |
| `whale_went_neutral` | bool | | Zone just returned to 0 from any direction |
| `whale_early_buy` | bool | | Raw pressure spike suggests buying incoming (1-2 bars early) |
| `whale_early_sell` | bool | | Raw pressure spike suggests selling incoming (1-2 bars early) |

### Zones

| Zone | Value | Meaning | whale_pressure range |
|------|-------|---------|---------------------|
| Strong Buy | +2 | Heavy institutional buying | >= 60 |
| Buy | +1 | Moderate buying pressure | >= 25 |
| Neutral | 0 | No clear direction | -25 to +25 |
| Sell | -1 | Moderate selling pressure | <= -25 |
| Strong Sell | -2 | Heavy institutional selling | <= -60 |

### Common Condition Patterns

**Entry: Only enter longs when whales are buying**
```
whale_buying == true AND whale_conviction > 2
```

**Entry: Only enter when whales confirm direction**
```
# Long entry
whale_zone >= 1 AND whale_momentum > 0

# Short entry
whale_zone <= -1 AND whale_momentum < 0
```

**Exit: Whale flipped against your position**
```
# If long, exit when whales flip bearish
whale_flipped_bear == true

# If short, exit when whales flip bullish
whale_flipped_bull == true
```

**Exit: Whale pressure went neutral (lost conviction)**
```
whale_went_neutral == true AND ticks_pnl > 20
```

**Early warning entry filter**
```
whale_early_buy == true AND rsi14 < 50
```

---

## Institutional Quality Gate Filters

Five institutional-grade filters derived from the VECTOR Pattern Strategy v12. Each produces boolean gates (pass/fail) that can be used as entry conditions.

### Filter 1: Squeeze Gate

Detects BB/KC compression with EMA stack alignment. Compression = energy coiling for breakout.

| Field | Type | Description |
|-------|------|-------------|
| `squeeze_type` | number (0-3) | 0=none, 1=low, 2=mid, 3=high compression |
| `squeeze_ideal_bull` | bool | EMA 8>13>21 stack + mid/high compression |
| `squeeze_ideal_bear` | bool | EMA 8<13<21 stack + mid/high compression |
| `squeeze_gate_ok` | bool | Either ideal bull or bear (compression with direction) |

**Usage:**
```
# Only enter when squeeze is firing with direction
squeeze_gate_ok == true

# Stronger: require high compression
squeeze_type >= 2
```

### Filter 2: Volatility Regime Gate

ATR percentile rank — where current volatility sits in recent history. When vol is in the 85th+ percentile, fixed-tick stops are structurally inadequate.

| Field | Type | Description |
|-------|------|-------------|
| `vol_percentile` | number (0-100) | ATR percentile rank. 85 = higher than 85% of recent bars |
| `vol_regime_adverse` | bool | True when vol_percentile >= 85 |

**Usage:**
```
# Block entries during elevated volatility
vol_regime_adverse == false

# More nuanced: allow if percentile is moderate
vol_percentile < 80
```

### Filter 3: VWAP Gate

Session VWAP with standard deviation bands. Direction confirms entries align with session flow. Bands block overextended entries (chasing).

| Field | Type | Description |
|-------|------|-------------|
| `vwap_price` | number | Session VWAP price level |
| `vwap_above` | bool | Price is above session VWAP |
| `vwap_sigma_dist` | number | Distance from VWAP in standard deviations (negative = below) |
| `vwap_long_ok` | bool | Above VWAP AND not overextended (within +2 sigma) |
| `vwap_short_ok` | bool | Below VWAP AND not overextended (within -2 sigma) |

**Usage:**
```
# Long entry: price above VWAP but not chasing
vwap_long_ok == true

# Short entry: price below VWAP
vwap_short_ok == true

# Check overextension directly
vwap_sigma_dist between -2 2
```

### Filter 4: Delta Gate (Order Flow)

Uses CVD bar delta from the Tastytrade data feed. Positive delta = buyers dominant, negative = sellers.

| Field | Type | Description |
|-------|------|-------------|
| `bar_delta` | number | Current bar's net volume delta (buy - sell) |
| `delta_bull` | bool | Bar delta is positive (buyers winning) |
| `delta_bear` | bool | Bar delta is negative (sellers winning) |

**Usage:**
```
# Confirm long entry with buying flow
delta_bull == true

# Confirm short entry with selling flow
delta_bear == true
```

### Filter 5: Relative Volume (RVOL) Gate

Measures signal bar participation relative to recent average. Filters out signals on thin/quiet bars.

| Field | Type | Description |
|-------|------|-------------|
| `vol_ratio` | number | Current volume / 20-bar average. 1.0 = average, 1.5 = 50% above |
| `rvol_ok` | bool | vol_ratio >= 1.0 (at least average volume) |

**Usage:**
```
# Require decent volume participation
rvol_ok == true

# Require strong volume
vol_ratio > 1.5
```

---

## Core Indicators

### Trend

| Field | Type | Description |
|-------|------|-------------|
| `htf_bias` | string | Higher timeframe bias: "bullish", "bearish", "neutral" (5-min EMA20 vs EMA50) |
| `f_zlema_trend_bull` | bool | ZLEMA slope positive (micro-trend up) |
| `f_zlema_trend_bear` | bool | ZLEMA slope negative (micro-trend down) |
| `f_trending` | bool | Efficiency ratio > 0.2 (market is trending, not choppy) |
| `price_above_ema20` | bool | Close > EMA(20) |
| `price_above_ema50` | bool | Close > EMA(50) |
| `supertrend_bull` | bool | Supertrend direction is bullish |
| `adx14` | number (0-100) | ADX strength. >20 = trending, >40 = strong trend |
| `adx_trending` | bool | ADX > 20 |
| `macd_bull` | bool | MACD line above signal line |
| `not_choppy` | bool | Choppiness index < 55 (trending market) |
| `choppiness` | number (0-100) | Choppiness index. <38 = strong trend, >62 = choppy |

### Oscillators

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `rsi14` | number | 0-100 | RSI 14-period. <30 = oversold, >70 = overbought |
| `stoch_rsi_k` | number | 0-100 | Stoch RSI %K. <20 = oversold, >80 = overbought |
| `stoch_in_ob` | bool | | Stoch RSI K > 80 |
| `stoch_in_os` | bool | | Stoch RSI K < 20 |
| `cci20` | number | any | CCI 20. >100 = overbought, <-100 = oversold |
| `wt1` | number | any | WaveTrend line 1. >50 = overbought, <-50 = oversold |
| `williams_r` | number | -100 to 0 | Williams %R. >-20 = overbought, <-80 = oversold |
| `mfi14` | number | 0-100 | Money Flow Index. >80 = overbought, <20 = oversold |

### Volatility

| Field | Type | Description |
|-------|------|-------------|
| `atr14` | number | Average True Range (14-period). Measures volatility in price units |
| `bb_pctb` | number (0-1) | Bollinger %B. 0 = at lower band, 1 = at upper band |
| `bar_body_pct` | number (0-1) | Bar body as fraction of range. 1.0 = full body (strong conviction) |
| `bar_is_bullish` | bool | Close > Open |
| `er_val` | number (0-1) | Efficiency ratio. 1.0 = perfectly trending, 0.0 = pure noise |

### Session & Timing

| Field | Type | Description |
|-------|------|-------------|
| `session_bucket` | string | "open_drive", "morning", "midday_chop", "afternoon", "close" |
| `day_of_week` | string | "Mon", "Tue", "Wed", "Thu", "Fri" |
| `bar_time_hhmm` | number | Time as HHMM integer (e.g., 945 = 9:45 AM ET) |
| `mins_to_maintenance` | number | Minutes until CME 5pm ET maintenance |

### CVD (Cumulative Volume Delta)

| Field | Type | Description |
|-------|------|-------------|
| `cvd_trend` | string | "rising", "falling", "flat" (based on 5-min delta) |
| `cvd_divergence` | string | "bullish_div", "bearish_div", "none" (price vs CVD mismatch) |
| `cvd_1m_delta` | number | CVD change over last 1 minute |
| `cvd_3m_delta` | number | CVD change over last 3 minutes |
| `cvd_5m_delta` | number | CVD change over last 5 minutes |

### Exit-Only Fields (available in exit conditions)

| Field | Type | Description |
|-------|------|-------------|
| `exit_score` | number | Server-computed exit pressure score (sum of flip/cross points) |
| `ticks_pnl` | number | Unrealized P&L in ticks |
| `dollar_pnl` | number | Unrealized P&L in dollars |
| `bars_in_trade` | number | How many bars since entry |

---

## Example Condition Strategies

### Conservative Long Entry
```
htf_bias == bullish AND f_trending == true AND rsi14 < 60
AND whale_buying == true AND vwap_long_ok == true AND rvol_ok == true
```

### Aggressive Short Entry
```
whale_strong_sell == true AND stoch_in_ob == true AND delta_bear == true
```

### Profit-Tiered Exit (Long)
```
# Take profit when whale flips and you're up decent
whale_flipped_bear == true AND ticks_pnl > 40

# Or whale went neutral with good profit
whale_went_neutral == true AND ticks_pnl > 80

# Hard stop
ticks_pnl < -100
```

### Time-Based Exit
```
bars_in_trade > 30 AND ticks_pnl < 10
```
