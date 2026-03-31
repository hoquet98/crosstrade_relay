"""
Tastytrade Market Data Feed — real-time tick streaming via DXLink.

Replaces/supplements the NT8 QTPDataFeed indicator for market data.
Streams Quote + TimeAndSale events for configured futures instruments.
Auto-detects front month contracts using roll dates from master_instruments.

Feeds into the same CVD/candle infrastructure as the NT8 feed (cvd.py).
"""

import os
import asyncio
import logging
from datetime import datetime, timezone, date

import database as db

logger = logging.getLogger("trade_relay")

# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

_tt_task: asyncio.Task | None = None
_tt_connected = False
_tt_symbols: dict[str, str] = {}  # root -> streamer_symbol mapping


# ══════════════════════════════════════════════════════════════════════════════
# CONTRACT RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

async def _resolve_contracts(session, product_codes: list[str]) -> dict[str, str]:
    """Resolve product codes to front-month streamer symbols.

    Uses roll_date from master_instruments to decide front vs next month.
    Returns: {root_symbol: streamer_symbol}
    """
    from tastytrade.instruments import Future

    symbols = {}
    today = date.today()

    for code in product_codes:
        try:
            futures = await Future.get(session, product_codes=[code])
            active = sorted(
                [f for f in futures if f.is_tradeable],
                key=lambda f: f.expiration_date
            )
            if not active:
                logger.warning(f"TT: No active contracts for {code}")
                continue

            # Check roll date from DB
            instrument = db.get_instrument(code)
            roll_date_str = instrument.get("roll_date") if instrument else None

            if roll_date_str:
                try:
                    roll_date = date.fromisoformat(roll_date_str)
                    if today >= roll_date and len(active) > 1:
                        # Past roll date — use next month
                        chosen = active[1]
                        logger.info(f"TT: {code} past roll date ({roll_date_str}), using next: {chosen.streamer_symbol}")
                    else:
                        chosen = active[0]
                except ValueError:
                    chosen = active[0]
            else:
                chosen = active[0]

            symbols[code] = chosen.streamer_symbol
            logger.info(f"TT: {code} -> {chosen.symbol} (streamer: {chosen.streamer_symbol}, expires: {chosen.expiration_date})")

        except Exception as e:
            logger.error(f"TT: Failed to resolve {code}: {e}")

    return symbols


# ══════════════════════════════════════════════════════════════════════════════
# TICK PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _streamer_to_root(streamer_symbol: str) -> str:
    """Extract root from streamer symbol: '/MNQM26:XCME' -> 'MNQ'"""
    s = streamer_symbol.lstrip("/")
    root = ""
    for ch in s:
        if ch.isalpha():
            root += ch
        else:
            break
    return root.upper()


def _streamer_to_display(streamer_symbol: str) -> str:
    """Convert streamer symbol to display name: '/MNQM26:XCME' -> 'MNQ JUN26'"""
    # Map month codes to names
    months = {'F': 'JAN', 'G': 'FEB', 'H': 'MAR', 'J': 'APR', 'K': 'MAY',
              'M': 'JUN', 'N': 'JUL', 'Q': 'AUG', 'U': 'SEP', 'V': 'OCT',
              'X': 'NOV', 'Z': 'DEC'}
    s = streamer_symbol.lstrip("/").split(":")[0]
    root = _streamer_to_root(streamer_symbol)
    # Extract month code and year from remainder
    remainder = s[len(root):]  # e.g., "M26"
    if len(remainder) >= 2:
        month_code = remainder[0].upper()
        year = remainder[1:]
        month_name = months.get(month_code, month_code)
        return f"{root} {month_name}{year}"
    return root


def _process_quote(quote):
    """Process a Quote event — update bid/ask in CVD state."""
    import cvd

    symbol = quote.event_symbol
    display_name = _streamer_to_display(symbol)

    bid = float(quote.bid_price) if quote.bid_price else 0
    ask = float(quote.ask_price) if quote.ask_price else 0

    if bid <= 0 and ask <= 0:
        return

    # Update CVD state
    if display_name not in cvd._cvd_state:
        cvd._cvd_state[display_name] = {
            "cvd": 0, "last": 0, "bid": 0, "ask": 0,
            "volume": 0, "direction": "neutral", "delta_1s": 0,
        }
    cvd._subscribed_instruments.add(display_name)

    state = cvd._cvd_state[display_name]
    state["bid"] = bid
    state["ask"] = ask
    state["updated_at"] = datetime.now(timezone.utc).isoformat()


def _process_time_and_sale(tick):
    """Process a TimeAndSale tick — update CVD, price, candle."""
    import cvd
    import time as _time

    symbol = tick.event_symbol
    display_name = _streamer_to_display(symbol)

    price = float(tick.price) if tick.price else 0
    size = int(tick.size) if tick.size else 0
    bid = float(tick.bid_price) if tick.bid_price else 0
    ask = float(tick.ask_price) if tick.ask_price else 0

    if price <= 0:
        return

    # Determine trade direction from aggressor_side
    side = str(tick.aggressor_side).upper() if tick.aggressor_side else ""
    if side == "BUY" or side == "BUYER":
        delta = size
    elif side == "SELL" or side == "SELLER":
        delta = -size
    else:
        # Fall back to bid/ask comparison
        if ask > 0 and price >= ask:
            delta = size
        elif bid > 0 and price <= bid:
            delta = -size
        else:
            delta = 0

    # Update CVD state
    if display_name not in cvd._cvd_state:
        cvd._cvd_state[display_name] = {
            "cvd": 0, "last": 0, "bid": 0, "ask": 0,
            "volume": 0, "direction": "neutral", "delta_1s": 0,
        }
    cvd._subscribed_instruments.add(display_name)

    state = cvd._cvd_state[display_name]
    state["last"] = price
    if bid > 0:
        state["bid"] = bid
    if ask > 0:
        state["ask"] = ask
    state["cvd"] = state.get("cvd", 0) + delta
    state["delta_1s"] = delta
    state["direction"] = "buy" if delta > 0 else "sell" if delta < 0 else "neutral"
    state["volume"] = (state.get("volume", 0) or 0) + size
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    # Update rolling history for CVD metrics
    history = cvd._get_history(display_name)
    history.append((_time.time(), state["cvd"], price))

    # Update 1-min candle aggregation
    cvd._update_candle(display_name, price, delta)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING LOOP
# ══════════════════════════════════════════════════════════════════════════════

async def _stream_loop(client_secret: str, refresh_token: str,
                       product_codes: list[str]):
    """Main streaming loop — connects, subscribes, processes events."""
    global _tt_connected, _tt_symbols

    from tastytrade import Session, DXLinkStreamer
    from tastytrade.dxfeed import Quote, TimeAndSale

    while True:
        try:
            logger.info("TT: Creating session...")
            session = Session(
                provider_secret=client_secret,
                refresh_token=refresh_token,
            )
            await session.refresh(force=True)
            logger.info("TT: Session authenticated")

            # Resolve front-month contracts
            _tt_symbols = await _resolve_contracts(session, product_codes)
            if not _tt_symbols:
                logger.error("TT: No contracts resolved, retrying in 60s")
                await asyncio.sleep(60)
                continue

            streamer_symbols = list(_tt_symbols.values())
            logger.info(f"TT: Subscribing to {len(streamer_symbols)} instruments: {streamer_symbols}")

            async with DXLinkStreamer(session) as streamer:
                await streamer.subscribe(Quote, streamer_symbols)
                await streamer.subscribe(TimeAndSale, streamer_symbols)
                _tt_connected = True
                logger.info("TT: Streaming started")

                # Process events in parallel
                quote_task = asyncio.create_task(_listen_quotes(streamer))
                tas_task = asyncio.create_task(_listen_ticks(streamer))

                # Wait for either task to finish (shouldn't unless error)
                done, pending = await asyncio.wait(
                    [quote_task, tas_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                for task in done:
                    if task.exception():
                        logger.error(f"TT: Stream task error: {task.exception()}")

        except asyncio.CancelledError:
            logger.info("TT: Stream cancelled")
            _tt_connected = False
            return
        except Exception as e:
            logger.error(f"TT: Stream error: {e}, reconnecting in 10s")
            _tt_connected = False
            await asyncio.sleep(10)


async def _listen_quotes(streamer):
    """Listen for Quote events."""
    from tastytrade.dxfeed import Quote
    async for quote in streamer.listen(Quote):
        try:
            _process_quote(quote)
        except Exception as e:
            logger.debug(f"TT: Quote processing error: {e}")


async def _listen_ticks(streamer):
    """Listen for TimeAndSale tick events."""
    from tastytrade.dxfeed import TimeAndSale
    async for tick in streamer.listen(TimeAndSale):
        try:
            _process_time_and_sale(tick)
        except Exception as e:
            logger.debug(f"TT: Tick processing error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def start(product_codes: list[str] = None):
    """Start the Tastytrade data feed as a background task.

    Reads credentials from settings DB. Subscribes to front-month contracts
    for each product code.
    """
    global _tt_task

    client_secret = db.get_setting("tastytrade_client_secret")
    refresh_token = db.get_setting("tastytrade_refresh_token")

    if not client_secret or not refresh_token:
        logger.warning("TT: Missing credentials (client_secret or refresh_token)")
        return False

    if product_codes is None:
        # Default instruments from master_instruments
        instruments = db.get_instruments(active_only=True)
        product_codes = list(set(i["root"] for i in instruments))
        # Filter to ones we care about
        product_codes = [c for c in product_codes if c in
                         {"MNQ", "NQ", "MES", "ES", "GC", "MGC", "CL", "MCL",
                          "RTY", "M2K", "SI", "SIL"}]

    if not product_codes:
        logger.warning("TT: No product codes to subscribe")
        return False

    # Load historical bars
    try:
        import cvd
        conn = db.get_connection()
        rows = conn.execute("SELECT DISTINCT instrument FROM ai_bars ORDER BY instrument").fetchall()
        conn.close()
        for r in rows:
            cvd._load_historical_bars([r["instrument"]])
    except Exception as e:
        logger.warning(f"TT: Failed to load historical bars: {e}")

    _tt_task = asyncio.create_task(_stream_loop(client_secret, refresh_token, product_codes))
    logger.info(f"TT: Feed started for {product_codes}")
    return True


def stop():
    """Stop the Tastytrade data feed."""
    global _tt_task, _tt_connected
    if _tt_task:
        _tt_task.cancel()
        _tt_task = None
    _tt_connected = False


def is_connected() -> bool:
    return _tt_connected


def get_symbols() -> dict:
    return dict(_tt_symbols)
