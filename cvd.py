# cvd.py — Real-time Cumulative Volume Delta via CrossTrade WebSocket
#
# Connects to CT WebSocket, subscribes to instrument quotes,
# tracks CVD by comparing last price to bid/ask each second.
# Keeps rolling 5-minute history for trend/divergence metrics.

import os
import json
import logging
import asyncio
import time
from collections import deque
from datetime import datetime, timezone

import ai_gate

logger = logging.getLogger("trade_relay")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CT_WS_URL = "wss://app.crosstrade.io/ws/stream"
CT_API_KEY = os.environ.get("CT_API_KEY", "")
HISTORY_SECONDS = 300  # 5 minutes of snapshots

# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

_cvd_state: dict[str, dict] = {}
# Rolling history: deque of (timestamp, cvd_value, price) per instrument
_cvd_history: dict[str, deque] = {}
# 1-minute candle aggregation
_candles: dict[str, list] = {}        # completed candles per instrument
_current_candle: dict[str, dict] = {} # in-progress candle per instrument
MAX_CANDLES = 480                      # keep 8 hours of 1-min bars
_subscribed_instruments: set[str] = set()
_ws_task: asyncio.Task | None = None
_ws_connected = False


# ══════════════════════════════════════════════════════════════════════════════
# METRICS (computed from rolling history)
# ══════════════════════════════════════════════════════════════════════════════

def _get_history(instrument: str) -> deque:
    if instrument not in _cvd_history:
        _cvd_history[instrument] = deque(maxlen=HISTORY_SECONDS)
    return _cvd_history[instrument]


def _cvd_at_seconds_ago(history: deque, seconds: int) -> int | None:
    """Get CVD value from N seconds ago. Returns None if not enough data."""
    if not history:
        return None
    now = time.time()
    target = now - seconds
    # Walk backwards to find closest snapshot
    for ts, cvd_val, price in reversed(history):
        if ts <= target:
            return cvd_val
    return None


def _price_at_seconds_ago(history: deque, seconds: int) -> float | None:
    """Get price from N seconds ago."""
    if not history:
        return None
    now = time.time()
    target = now - seconds
    for ts, cvd_val, price in reversed(history):
        if ts <= target:
            return price
    return None


def compute_metrics(instrument: str) -> dict:
    """Compute CVD trend metrics from rolling history."""
    state = _cvd_state.get(instrument)
    if not state:
        return {
            "cvd_1m_delta": 0,
            "cvd_3m_delta": 0,
            "cvd_5m_delta": 0,
            "cvd_trend": "flat",
            "cvd_divergence": "none"
        }

    history = _get_history(instrument)
    current_cvd = state.get("cvd", 0)
    current_price = state.get("last", 0)

    # Deltas
    cvd_1m_ago = _cvd_at_seconds_ago(history, 60)
    cvd_3m_ago = _cvd_at_seconds_ago(history, 180)
    cvd_5m_ago = _cvd_at_seconds_ago(history, 300)

    cvd_1m_delta = (current_cvd - cvd_1m_ago) if cvd_1m_ago is not None else 0
    cvd_3m_delta = (current_cvd - cvd_3m_ago) if cvd_3m_ago is not None else 0
    cvd_5m_delta = (current_cvd - cvd_5m_ago) if cvd_5m_ago is not None else 0

    # Trend (based on 5m delta, fall back to 3m, then 1m)
    delta_for_trend = cvd_5m_delta or cvd_3m_delta or cvd_1m_delta
    if delta_for_trend > 50:
        cvd_trend = "rising"
    elif delta_for_trend < -50:
        cvd_trend = "falling"
    else:
        cvd_trend = "flat"

    # Divergence: price vs CVD direction mismatch over 3 minutes
    price_3m_ago = _price_at_seconds_ago(history, 180)
    cvd_divergence = "none"
    if price_3m_ago is not None and cvd_3m_ago is not None:
        price_rising = current_price > price_3m_ago
        price_falling = current_price < price_3m_ago
        cvd_rising = cvd_3m_delta > 50
        cvd_falling = cvd_3m_delta < -50

        if price_rising and cvd_falling:
            cvd_divergence = "bearish_div"  # price up but sellers dominant
        elif price_falling and cvd_rising:
            cvd_divergence = "bullish_div"  # price down but buyers dominant

    return {
        "cvd_1m_delta": cvd_1m_delta,
        "cvd_3m_delta": cvd_3m_delta,
        "cvd_5m_delta": cvd_5m_delta,
        "cvd_trend": cvd_trend,
        "cvd_divergence": cvd_divergence
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_cvd(instrument: str) -> dict:
    """Get current CVD state + metrics for an instrument."""
    state = _cvd_state.get(instrument)
    if not state:
        return {
            "instrument": instrument,
            "cvd": 0,
            "last": 0,
            "bid": 0,
            "ask": 0,
            "volume": 0,
            "direction": "neutral",
            "delta_1s": 0,
            "cvd_1m_delta": 0,
            "cvd_3m_delta": 0,
            "cvd_5m_delta": 0,
            "cvd_trend": "flat",
            "cvd_divergence": "none",
            "connected": _ws_connected,
            "updated_at": None
        }
    metrics = compute_metrics(instrument)
    return {
        "instrument": state.get("instrument", instrument),
        "cvd": state.get("cvd", 0),
        "last": state.get("last", 0),
        "bid": state.get("bid", 0),
        "ask": state.get("ask", 0),
        "volume": state.get("volume", 0),
        "direction": state.get("direction", "neutral"),
        "delta_1s": state.get("delta_1s", 0),
        **metrics,
        "connected": _ws_connected,
        "updated_at": state.get("updated_at")
    }


def get_all_cvd() -> list[dict]:
    """Get CVD state + metrics for all tracked instruments."""
    return [get_cvd(inst) for inst in _subscribed_instruments]


# ══════════════════════════════════════════════════════════════════════════════
# QUOTE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _process_quote(quote: dict):
    """Process a single quote update and update CVD."""
    instrument = quote.get("instrument", "")
    if not instrument:
        return

    last = quote.get("last", 0)
    bid = quote.get("bid", 0)
    ask = quote.get("ask", 0)
    volume = quote.get("volume", 0)

    if instrument not in _cvd_state:
        _cvd_state[instrument] = {
            "instrument": instrument,
            "cvd": 0,
            "last": last,
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "prev_volume": volume,
            "direction": "neutral",
            "delta_1s": 0,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        return

    state = _cvd_state[instrument]
    prev_volume = state.get("prev_volume", volume)

    # Volume delta since last update
    vol_delta = volume - prev_volume if volume > prev_volume else 0

    # Classify: trade at ask = buyer, trade at bid = seller
    direction = "neutral"
    delta = 0
    if last and bid and ask and vol_delta > 0:
        if last >= ask:
            direction = "buy"
            delta = vol_delta
        elif last <= bid:
            direction = "sell"
            delta = -vol_delta
        else:
            direction = "neutral"
            delta = 0

    state["cvd"] += delta
    state["last"] = last
    state["bid"] = bid
    state["ask"] = ask
    state["volume"] = volume
    state["prev_volume"] = volume
    state["direction"] = direction
    state["delta_1s"] = delta
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    # Append to rolling history
    history = _get_history(instrument)
    history.append((time.time(), state["cvd"], last))

    # Update 1-minute candle
    _update_candle(instrument, last, delta)


# ══════════════════════════════════════════════════════════════════════════════
# 1-MINUTE CANDLE AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def _current_minute() -> int:
    """Get current time floored to the minute as unix timestamp."""
    now = int(time.time())
    return now - (now % 60)


def _update_candle(instrument: str, price: float, cvd_delta: int):
    """Update the current 1-minute candle with a new tick."""
    minute_ts = _current_minute()

    if instrument not in _candles:
        _candles[instrument] = []

    candle = _current_candle.get(instrument)

    # New minute — close previous candle, start new one
    if candle is None or candle["time"] != minute_ts:
        # Save completed candle (skip flat bars — no price movement = market closed)
        if candle is not None:
            is_flat = (candle["open"] == candle["high"] == candle["low"] == candle["close"])

            if not is_flat:
                _candles[instrument].append(candle)
                # Trim to max
                if len(_candles[instrument]) > MAX_CANDLES:
                    _candles[instrument] = _candles[instrument][-MAX_CANDLES:]

                # Persist completed candle to database
                try:
                    state = _cvd_state.get(instrument, {})
                    candle_ts_iso = datetime.fromtimestamp(candle["time"], tz=timezone.utc).isoformat()
                    ai_gate.save_bar(
                        instrument=instrument,
                        timestamp=candle_ts_iso,
                        o=candle["open"], h=candle["high"],
                        l=candle["low"], c=candle["close"],
                        volume=state.get("volume", 0),
                        cvd=state.get("cvd", 0),
                        cvd_delta=candle.get("cvd_delta", 0),
                    )
                except Exception as e:
                    logger.warning(f"CVD: failed to save bar to DB: {e}")

                # Aggregate into 5-min bars
                try:
                    ai_gate.aggregate_5m_bar(instrument)
                except Exception as e:
                    logger.warning(f"CVD: 5m aggregation error: {e}")

                # Trigger Python strategy bots on bar close
                try:
                    import strategy_runner
                    asyncio.ensure_future(strategy_runner.run_python_bots())
                except Exception as e:
                    logger.warning(f"CVD: strategy runner error: {e}")

        # Start new candle
        _current_candle[instrument] = {
            "time": minute_ts,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "cvd_delta": cvd_delta,
        }
    else:
        # Update current candle
        candle["high"] = max(candle["high"], price)
        candle["low"] = min(candle["low"], price)
        candle["close"] = price
        candle["cvd_delta"] += cvd_delta


def get_candles(instrument: str) -> list[dict]:
    """Get completed 1-min candles + current in-progress candle."""
    completed = _candles.get(instrument, [])
    current = _current_candle.get(instrument)
    if current:
        return completed + [current]
    return completed


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET CLIENT
# ══════════════════════════════════════════════════════════════════════════════

async def _ws_loop(api_key: str, instruments: list[str]):
    """Main WebSocket loop — connects, subscribes, processes quotes."""
    global _ws_connected

    try:
        import websockets
    except ImportError:
        logger.error("CVD: websockets package not installed. Run: pip install websockets")
        return

    while True:
        try:
            logger.info(f"CVD: connecting to {CT_WS_URL}...")
            async with websockets.connect(
                CT_WS_URL,
                additional_headers={"Authorization": f"Bearer {api_key}"},
                ping_interval=30,
                ping_timeout=10
            ) as ws:
                _ws_connected = True
                logger.info(f"CVD: connected. Subscribing to {instruments}")

                subscribe_msg = json.dumps({
                    "action": "subscribe",
                    "instruments": instruments
                })
                await ws.send(subscribe_msg)

                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        if data.get("type") == "marketData":
                            for quote in data.get("quotes", []):
                                _process_quote(quote)
                    except json.JSONDecodeError:
                        pass

        except Exception as e:
            _ws_connected = False
            logger.warning(f"CVD: WebSocket disconnected: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


def _load_historical_bars(instruments: list[str]):
    """Load persisted bars from database to populate in-memory candle buffer."""
    try:
        for inst in instruments:
            bars = ai_gate.get_bars(inst, MAX_CANDLES)
            if bars:
                _candles[inst] = []
                for bar in bars:
                    # Convert ISO timestamp to unix seconds
                    try:
                        ts = int(datetime.fromisoformat(bar["timestamp"]).timestamp())
                    except (ValueError, TypeError):
                        continue
                    _candles[inst].append({
                        "time": ts,
                        "open": bar["open"],
                        "high": bar["high"],
                        "low": bar["low"],
                        "close": bar["close"],
                        "cvd_delta": bar.get("cvd_delta", 0),
                    })
                logger.info(f"CVD: loaded {len(_candles[inst])} historical bars for {inst}")
    except Exception as e:
        logger.warning(f"CVD: failed to load historical bars: {e}")


def start(api_key: str, instruments: list[str]):
    """Start the CVD WebSocket client as a background task."""
    global _ws_task, _subscribed_instruments

    if not api_key:
        logger.warning("CVD: no API key provided, skipping")
        return

    if not instruments:
        logger.warning("CVD: no instruments to track, skipping")
        return

    _subscribed_instruments = set(instruments)
    _load_historical_bars(instruments)
    _ws_task = asyncio.create_task(_ws_loop(api_key, instruments))
    logger.info(f"CVD: started tracking {instruments}")


def stop():
    """Stop the CVD WebSocket client."""
    global _ws_task, _ws_connected
    if _ws_task:
        _ws_task.cancel()
        _ws_task = None
    _ws_connected = False


def reset(instrument: str = None):
    """Reset CVD to zero."""
    if instrument:
        if instrument in _cvd_state:
            _cvd_state[instrument]["cvd"] = 0
            _cvd_state[instrument]["delta_1s"] = 0
        if instrument in _cvd_history:
            _cvd_history[instrument].clear()
    else:
        for state in _cvd_state.values():
            state["cvd"] = 0
            state["delta_1s"] = 0
        for hist in _cvd_history.values():
            hist.clear()
