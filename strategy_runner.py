"""
Strategy Runner — executes Python strategies server-side using stored bar data.

Runs on every bar close (triggered by CVD WebSocket candle completion).
Strategies produce entry signals; AI Gate confirms entries; exits can be
strategy-managed or AI-managed.

Bot mode: "python" in ai_bots table.
"""

import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional

import database as db

logger = logging.getLogger("strategy_runner")


# ---------------------------------------------------------------------------
# Strategy Registry — add new strategies here
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY = {}


def register_strategy(name: str):
    """Decorator to register a strategy class."""
    def wrapper(cls):
        STRATEGY_REGISTRY[name] = cls
        return cls
    return wrapper


class BaseStrategy:
    """Base class for all Python strategies."""
    name: str = "base"
    description: str = ""

    def generate_signals(self, data: pd.DataFrame, params: dict) -> dict:
        """Compute indicators and return signal dict for the latest bar.

        Args:
            data: OHLCV DataFrame with at least 200+ bars of history.
                  Columns: open, high, low, close, volume
                  Index: datetime
            params: Strategy parameters from bot config.

        Returns:
            dict with keys matching Pine Script payload format:
            {
                "long_signal": bool,
                "short_signal": bool,
                "bull_confluence": int,
                "bear_confluence": int,
                # ... any indicator values for AI context
            }
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Example: UT Bot + RSI + ADX + EMA + Choppiness (from nq_utbot_strategy.py)
# ---------------------------------------------------------------------------

@register_strategy("ut_bot_trend")
class UtBotTrendStrategy(BaseStrategy):
    name = "ut_bot_trend"
    description = "UT Bot trend-follower with RSI + ADX + EMA(200) + Choppiness filter"

    def generate_signals(self, data: pd.DataFrame, params: dict) -> dict:
        from indicators import (rsi, adx, atr, ema, choppiness_index,
                                ut_bot, crossover, crossunder,
                                efficiency_ratio, bollinger_pctb,
                                stochastic_rsi, cci, wavetrend,
                                squeeze_momentum)

        close = data['close']
        high = data['high']
        low = data['low']
        volume = data.get('volume', pd.Series(0, index=data.index))

        # Parameters with defaults
        ut_key = params.get("ut_key", 3.0)
        ut_atr_period = params.get("ut_atr_period", 10)
        rsi_period = params.get("rsi_period", 14)
        rsi_oversold = params.get("rsi_oversold", 50.0)
        rsi_overbought = params.get("rsi_overbought", 70.0)
        adx_period = params.get("adx_period", 14)
        adx_threshold = params.get("adx_threshold", 20.0)
        ema_period = params.get("ema_period", 200)
        chop_period = params.get("chop_period", 14)
        chop_threshold = params.get("chop_threshold", 55.0)

        # Core UT Bot signals
        trail_stop, buy_sig, sell_sig = ut_bot(close, high, low,
                                                key_value=ut_key,
                                                atr_period=ut_atr_period)

        # Filters
        rsi_val = rsi(close, rsi_period)
        adx_val, di_plus, di_minus = adx(high, low, close, adx_period)
        trending = adx_val > adx_threshold

        ema_val = ema(close, ema_period)
        above_ema = close > ema_val

        chop_val = choppiness_index(high, low, close, chop_period)
        not_choppy = chop_val < chop_threshold

        # Additional indicators for AI context
        ema20 = ema(close, 20)
        ema50 = ema(close, 50)
        rsi14 = rsi(close, 14)
        atr14 = atr(high, low, close, 14)
        er_val = efficiency_ratio(close, 8)
        bb_pctb = bollinger_pctb(close, 20, 2.0)
        stoch_k, stoch_d = stochastic_rsi(close, 14, 14, 3, 3)
        cci20 = cci(close, 20)
        wt1, wt2 = wavetrend(high, low, close, 6, 9, 2)
        sq_mom, sq_on = squeeze_momentum(high, low, close, 20, 2.0, 20, 1.5)

        # Get last bar values
        i = -1  # latest bar

        # Confluence scoring (matches Pine Script layers)
        f1_bull = bool(close.iloc[i] > ema_val.iloc[i]) if not np.isnan(ema_val.iloc[i]) else False
        f1_bear = bool(close.iloc[i] < ema_val.iloc[i]) if not np.isnan(ema_val.iloc[i]) else False

        wt_not_ob = bool(wt1.iloc[i] < 50) if not np.isnan(wt1.iloc[i]) else True
        wt_above_short = bool(wt1.iloc[i] > 55) if not np.isnan(wt1.iloc[i]) else False
        mom_bull = bool(sq_mom.iloc[i] > 0) if not np.isnan(sq_mom.iloc[i]) else False
        mom_bear = bool(sq_mom.iloc[i] < 0) if not np.isnan(sq_mom.iloc[i]) else False

        f2_bull = wt_not_ob and mom_bull
        f2_bear = wt_above_short and mom_bear
        f3_bull = not bool(sq_on.iloc[i]) if not isinstance(sq_on.iloc[i], (bool, np.bool_)) else not sq_on.iloc[i]
        f3_bear = f3_bull
        f4_bull = bool(er_val.iloc[i] > 0.2) if not np.isnan(er_val.iloc[i]) else False
        f4_bear = f4_bull

        bull_count = sum([f1_bull, f2_bull, f3_bull, f4_bull])
        bear_count = sum([f1_bear, f2_bear, f3_bear, f4_bear])

        # Entry signals
        long_entry = (bool(buy_sig.iloc[i])
                      and not np.isnan(rsi_val.iloc[i])
                      and rsi_val.iloc[i] > rsi_oversold
                      and rsi_val.iloc[i] < rsi_overbought
                      and bool(trending.iloc[i])
                      and bool(above_ema.iloc[i])
                      and bool(not_choppy.iloc[i]))

        short_entry = (bool(sell_sig.iloc[i])
                       and not np.isnan(rsi_val.iloc[i])
                       and rsi_val.iloc[i] > 65
                       and bool(trending.iloc[i])
                       and bool(di_minus.iloc[i] > di_plus.iloc[i]))

        # Flip detection
        zlema_flipped_bull = bool(close.iloc[i] > ema20.iloc[i] and close.iloc[-2] <= ema20.iloc[-2]) if len(close) > 1 else False
        zlema_flipped_bear = bool(close.iloc[i] < ema20.iloc[i] and close.iloc[-2] >= ema20.iloc[-2]) if len(close) > 1 else False
        ut_flipped_bull = bool(buy_sig.iloc[i]) if not np.isnan(buy_sig.iloc[i]) else False
        ut_flipped_bear = bool(sell_sig.iloc[i]) if not np.isnan(sell_sig.iloc[i]) else False
        wt_bull_cross = bool(wt1.iloc[i] > wt2.iloc[i] and wt1.iloc[-2] <= wt2.iloc[-2]) if len(wt1) > 1 else False
        wt_bear_cross = bool(wt1.iloc[i] < wt2.iloc[i] and wt1.iloc[-2] >= wt2.iloc[-2]) if len(wt1) > 1 else False
        stoch_bull_cross = bool(stoch_k.iloc[i] > stoch_d.iloc[i] and stoch_k.iloc[-2] <= stoch_d.iloc[-2]) if len(stoch_k) > 1 else False
        stoch_bear_cross = bool(stoch_k.iloc[i] < stoch_d.iloc[i] and stoch_k.iloc[-2] >= stoch_d.iloc[-2]) if len(stoch_k) > 1 else False

        # Time context — use proper ET conversion
        now = datetime.now(timezone.utc)
        # ET offset: -5 (EST) or -4 (EDT). March-Nov is EDT.
        month = now.month
        is_dst = 3 <= month <= 10  # approximate DST
        et_offset = 4 if is_dst else 5
        et_hour = (now.hour - et_offset) % 24
        et_minute = now.minute
        hhmm = et_hour * 100 + et_minute
        bar_time_et = f"{et_hour:02d}:{et_minute:02d} ET"
        bar_time_unix = int(now.timestamp())

        if hhmm < 1000:
            session_bucket = "open_drive"
        elif hhmm < 1130:
            session_bucket = "morning"
        elif hhmm < 1330:
            session_bucket = "midday_chop"
        elif hhmm < 1500:
            session_bucket = "afternoon"
        else:
            session_bucket = "close"

        # Minutes to CME maintenance (5pm ET = 1700)
        maint_start_mins = 17 * 60
        current_mins = et_hour * 60 + et_minute
        mins_to_maint = maint_start_mins - current_mins if maint_start_mins > current_mins else 0

        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_of_week = dow_names[now.weekday()]

        # HTF bias (use EMA20 vs EMA50 as proxy)
        htf_bullish = ema20.iloc[i] > ema50.iloc[i] if not (np.isnan(ema20.iloc[i]) or np.isnan(ema50.iloc[i])) else False
        htf_bearish = ema20.iloc[i] < ema50.iloc[i] if not (np.isnan(ema20.iloc[i]) or np.isnan(ema50.iloc[i])) else False
        htf_bias = "bullish" if htf_bullish else ("bearish" if htf_bearish else "neutral")

        def _safe(val):
            """Convert numpy value to Python scalar, NaN to 0."""
            if isinstance(val, (np.bool_, bool)):
                return bool(val)
            if isinstance(val, (np.floating, float)):
                return 0.0 if np.isnan(val) else round(float(val), 4)
            if isinstance(val, (np.integer, int)):
                return int(val)
            return val

        # Build payload matching Pine Script format
        return {
            "long_signal": long_entry,
            "short_signal": short_entry,
            "bull_confluence": bull_count,
            "bear_confluence": bear_count,
            "long_exit_warn": bool(wt1.iloc[i] > 50) if not np.isnan(wt1.iloc[i]) else False,
            "short_exit_warn": bool(wt1.iloc[i] < -45) if not np.isnan(wt1.iloc[i]) else False,
            # Layer flags
            "f_zlema_trend_bull": f1_bull,
            "f_zlema_trend_bear": f1_bear,
            "f_wt_mom_bull": f2_bull,
            "f_wt_mom_bear": f2_bear,
            "f_squeeze_off": f3_bull,
            "f_trending": f4_bull,
            # Indicators
            "zlema": _safe(ema20.iloc[i]),
            "zl_slope": _safe(ema20.iloc[i] - ema20.iloc[-2]) if len(ema20) > 1 else 0.0,
            "zlema_flipped_bull": zlema_flipped_bull,
            "zlema_flipped_bear": zlema_flipped_bear,
            "zlema_went_neutral": False,
            "wt1": _safe(wt1.iloc[i]),
            "wt2": _safe(wt2.iloc[i]),
            "wt_bull_cross": wt_bull_cross,
            "wt_bear_cross": wt_bear_cross,
            "sq_mom": _safe(sq_mom.iloc[i]),
            "squeeze_on": bool(sq_on.iloc[i]),
            "mom_flipped_bull": False,
            "mom_flipped_bear": False,
            "er_val": _safe(er_val.iloc[i]),
            "ut_trail": _safe(trail_stop.iloc[i]),
            "ut_flipped_bull": ut_flipped_bull,
            "ut_flipped_bear": ut_flipped_bear,
            "ema20": _safe(ema20.iloc[i]),
            "ema50": _safe(ema50.iloc[i]),
            "ema20_slope5": _safe(ema20.iloc[i] - ema20.iloc[-6]) if len(ema20) > 5 else 0.0,
            "price_above_ema20": bool(close.iloc[i] > ema20.iloc[i]) if not np.isnan(ema20.iloc[i]) else False,
            "price_above_ema50": bool(close.iloc[i] > ema50.iloc[i]) if not np.isnan(ema50.iloc[i]) else False,
            "htf_bias": htf_bias,
            "rsi14": _safe(rsi14.iloc[i]),
            "rsi_bull_div": False,
            "rsi_bear_div": False,
            "cci20": _safe(cci20.iloc[i]),
            "stoch_rsi_k": _safe(stoch_k.iloc[i]),
            "stoch_rsi_d": _safe(stoch_d.iloc[i]),
            "stoch_in_ob": bool(stoch_k.iloc[i] > 80) if not np.isnan(stoch_k.iloc[i]) else False,
            "stoch_in_os": bool(stoch_k.iloc[i] < 20) if not np.isnan(stoch_k.iloc[i]) else False,
            "stoch_bull_cross": stoch_bull_cross,
            "stoch_bear_cross": stoch_bear_cross,
            "atr14": _safe(atr14.iloc[i]),
            "atr_consumed_pct": 0.0,
            "bb_pctb": _safe(bb_pctb.iloc[i]),
            "bar_range_vs_atr": _safe((high.iloc[i] - low.iloc[i]) / atr14.iloc[i]) if not np.isnan(atr14.iloc[i]) and atr14.iloc[i] > 0 else 1.0,
            "vol_ratio": 1.0,
            "bar_body_pct": _safe(abs(close.iloc[i] - data['open'].iloc[i]) / max(high.iloc[i] - low.iloc[i], 0.01)),
            "bar_upper_wick_pct": _safe((high.iloc[i] - max(close.iloc[i], data['open'].iloc[i])) / max(high.iloc[i] - low.iloc[i], 0.01)),
            "bar_lower_wick_pct": _safe((min(close.iloc[i], data['open'].iloc[i]) - low.iloc[i]) / max(high.iloc[i] - low.iloc[i], 0.01)),
            "bar_is_bullish": bool(close.iloc[i] > data['open'].iloc[i]),
            "prev_3_pattern": "",
            "vwap_dist_ticks": 0.0,
            "dist_sess_high_ticks": 0.0,
            "dist_sess_low_ticks": 0.0,
            "bars_since_swing_high": 0,
            "bars_since_swing_low": 0,
            "cons_range_ticks": 0.0,
            "cons_range_vs_atr": 0.0,
            "last_5_closes": [_safe(close.iloc[j]) for j in range(-5, 0)],
            "open": _safe(data['open'].iloc[i]),
            "high": _safe(high.iloc[i]),
            "low": _safe(low.iloc[i]),
            "close": _safe(close.iloc[i]),
            "volume": _safe(volume.iloc[i]),
            "session_bucket": session_bucket,
            "day_of_week": day_of_week,
            "mins_to_maintenance": mins_to_maint,
            "bar_time_unix": bar_time_unix,
            "bar_time_et": bar_time_et,
            "bar_time_hhmm": hhmm,
            "price": _safe(close.iloc[i]),
        }


# ---------------------------------------------------------------------------
# Strategy Runner — called on each bar close
# ---------------------------------------------------------------------------

def get_bars_as_dataframe(instrument: str, limit: int = 300) -> Optional[pd.DataFrame]:
    """Load stored bars from ai_bars table into a pandas DataFrame."""
    import ai_gate
    bars = ai_gate.get_bars(instrument, limit)
    if not bars or len(bars) < 50:
        return None

    records = []
    for b in bars:
        ts = b.get("timestamp") or b.get("time")
        if isinstance(ts, str):
            ts = pd.Timestamp(ts)
        elif isinstance(ts, (int, float)):
            ts = pd.Timestamp(ts, unit='s')
        records.append({
            "time": ts,
            "open": b["open"],
            "high": b["high"],
            "low": b["low"],
            "close": b["close"],
            "volume": b.get("volume", 0),
        })

    df = pd.DataFrame(records)
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df


async def run_python_bots():
    """Run all enabled Python-mode bots against latest bar data.

    Called by CVD on each 1-min bar close.
    """
    import ai_gate

    conn = db.get_connection()
    bots = conn.execute("""
        SELECT * FROM ai_bots WHERE mode = 'python' AND enabled = 1
    """).fetchall()
    conn.close()

    if not bots:
        return

    for bot in bots:
        try:
            bot_id = bot["bot_id"]
            strategy_name = bot.get("strategy_name") or bot.get("source_bot") or "ut_bot_trend"
            instrument = "MNQ1!"  # default for now
            account = bot["account"]
            strategy_tag = bot["strategy_tag"]

            # Get strategy class
            strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
            if not strategy_cls:
                logger.warning(f"Bot {bot_id}: strategy '{strategy_name}' not found in registry")
                continue

            strategy = strategy_cls()

            # Load bar data — use NT8 instrument format for bar storage
            # Try common formats
            df = None
            for inst_name in ["MNQ JUN26", "MNQ 06-26", "MNQ1!"]:
                df = get_bars_as_dataframe(inst_name, limit=300)
                if df is not None and len(df) >= 50:
                    break

            if df is None or len(df) < 50:
                logger.debug(f"Bot {bot_id}: not enough bar data yet ({len(df) if df is not None else 0} bars)")
                continue

            # Parse strategy params from bot config
            params = {}
            if bot.get("config_json"):
                import json
                try:
                    params = json.loads(bot["config_json"])
                except:
                    pass

            # Run strategy
            signal = strategy.generate_signals(df, params)

            # Build payload matching Pine Script webhook format
            payload = {
                "relay_user": bot.get("relay_user", "titon"),
                "relay_id": bot_id,
                "account": account,
                "qty": 1,
                "strategy_tag": strategy_tag or "PyBot",
                "strategy": strategy_name,
                "strategy_type": "scalp",
                "instrument": instrument,
                "timeframe": "1",
                "target_ticks": params.get("target_ticks", 50),
                "stop_ticks": params.get("stop_ticks", 30),
                "hard_stop_ticks": params.get("hard_stop_ticks", 100),
                "rr": 1.67,
                "tick_value": 2,
            }
            payload.update(signal)

            # Feed through AI Gate
            await ai_gate.process_bar(payload)

        except Exception as e:
            logger.error(f"Bot {bot['bot_id']}: strategy runner error: {e}", exc_info=True)
