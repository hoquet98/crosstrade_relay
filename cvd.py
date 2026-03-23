# cvd.py — Real-time Cumulative Volume Delta via CrossTrade WebSocket
#
# Connects to CT WebSocket, subscribes to instrument quotes,
# tracks CVD by comparing last price to bid/ask each second.

import os
import json
import logging
import asyncio
from datetime import datetime, timezone

logger = logging.getLogger("trade_relay")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CT_WS_URL = "wss://app.crosstrade.io/ws/stream"
CT_API_KEY = os.environ.get("CT_API_KEY", "")

# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

# Per-instrument CVD state
_cvd_state: dict[str, dict] = {}
_subscribed_instruments: set[str] = set()
_ws_task: asyncio.Task | None = None
_ws_connected = False


def get_cvd(instrument: str) -> dict:
    """Get current CVD state for an instrument."""
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
            "connected": _ws_connected,
            "updated_at": None
        }
    return {**state, "connected": _ws_connected}


def get_all_cvd() -> list[dict]:
    """Get CVD state for all tracked instruments."""
    return [get_cvd(inst) for inst in _subscribed_instruments]


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
            # Between bid and ask — split or neutral
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

                # Subscribe
                subscribe_msg = json.dumps({
                    "action": "subscribe",
                    "instruments": instruments
                })
                await ws.send(subscribe_msg)

                # Process messages
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


def start(api_key: str, instruments: list[str]):
    """Start the CVD WebSocket client as a background task.
    Call this from the relay lifespan.
    """
    global _ws_task, _subscribed_instruments

    if not api_key:
        logger.warning("CVD: no API key provided, skipping")
        return

    if not instruments:
        logger.warning("CVD: no instruments to track, skipping")
        return

    _subscribed_instruments = set(instruments)
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
    else:
        for state in _cvd_state.values():
            state["cvd"] = 0
            state["delta_1s"] = 0
