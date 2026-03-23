# ai_gate.py — AI Gate v2: LLM-powered trade decision engine
#
# Single process_bar() entry point receives every bar from Pine Script.
# Server owns the state machine: FLAT → ENTER → MANAGE → EXIT.
# Computes exit scores, P&L, and safety nets server-side.

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Awaitable

import httpx

import database as db

logger = logging.getLogger("trade_relay")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_TIMEOUT = 15.0

AI_DRY_RUN = os.environ.get("AI_DRY_RUN", "1").lower() not in ("0", "false", "no")

# ══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT SPECS — tick_size and point_value for P&L computation
# tick_size  = minimum price increment
# point_value = dollar value of 1.0 point move for 1 contract
# dollar_per_tick = point_value * tick_size
# ══════════════════════════════════════════════════════════════════════════════

INSTRUMENTS = {
    # E-mini / Micro Nasdaq
    "NQ":   {"tick_size": 0.25, "point_value": 20.0},
    "MNQ":  {"tick_size": 0.25, "point_value": 2.0},
    # E-mini / Micro S&P 500
    "ES":   {"tick_size": 0.25, "point_value": 50.0},
    "MES":  {"tick_size": 0.25, "point_value": 5.0},
    # E-mini / Micro Russell 2000
    "RTY":  {"tick_size": 0.10, "point_value": 50.0},
    "M2K":  {"tick_size": 0.10, "point_value": 5.0},
    # Gold
    "GC":   {"tick_size": 0.10, "point_value": 100.0},
    "MGC":  {"tick_size": 0.10, "point_value": 10.0},
    # Silver
    "SI":   {"tick_size": 0.005, "point_value": 5000.0},
    "SIL":  {"tick_size": 0.005, "point_value": 1000.0},
    # Crude Oil
    "CL":   {"tick_size": 0.01, "point_value": 1000.0},
    "MCL":  {"tick_size": 0.01, "point_value": 100.0},
    # E-mini / Micro Dow
    "YM":   {"tick_size": 1.0,  "point_value": 5.0},
    "MYM":  {"tick_size": 1.0,  "point_value": 0.5},
}


def get_instrument_spec(instrument: str) -> dict:
    """Look up tick_size and point_value from instrument string.
    Handles TradingView formats like 'MNQ1!', 'NQ1!', 'MNQM2026', etc.
    """
    # Extract root symbol: strip trailing digits, '!', month codes
    root = ""
    for ch in instrument.upper():
        if ch.isalpha():
            root += ch
        else:
            break

    # Try exact match first, then common aliases
    if root in INSTRUMENTS:
        return INSTRUMENTS[root]

    # Micro aliases (M2K -> M2K, already in table)
    # Try without leading 'M' for micro variants not in table
    if root.startswith("M") and root[1:] in INSTRUMENTS:
        return INSTRUMENTS[root[1:]]

    # Default: assume NQ-like specs and warn
    logger.warning(f"Unknown instrument '{instrument}' (root: {root}) — using MNQ defaults")
    return {"tick_size": 0.25, "point_value": 2.0}


def compute_dollar_pnl(instrument: str, direction: str,
                       entry_price: float, exit_price: float, qty: int = 1) -> tuple[float, float, float]:
    """Compute P&L for a trade. Returns (points_pnl, dollar_pnl, ticks_pnl).

    points_pnl = raw price difference
    ticks_pnl  = points / tick_size (actual tick count)
    dollar_pnl = points * point_value * qty
    """
    spec = get_instrument_spec(instrument)
    if direction == "long":
        points = exit_price - entry_price
    else:
        points = entry_price - exit_price

    ticks = points / spec["tick_size"]
    dollars = points * spec["point_value"] * qty
    return round(points, 4), round(dollars, 2), round(ticks, 1)


# Position management
# Defaults — overridden by payload values if present
DEFAULT_HARD_STOP_TICKS = 100
DEFAULT_MAX_BARS_HOLD = 60
DEFAULT_EARLY_EXIT_TICKS = -60
DEFAULT_EARLY_EXIT_SCORE = 3
STALE_TIMEOUT_SECONDS = 300  # 5 minutes
AI_MANAGE_INTERVAL = 5       # call AI every N bars even if nothing happened

# Routing fields to strip from AI payload
ROUTING_KEYS = {"relay_user", "relay_id", "account", "qty",
                "order_type", "tif", "out_of_sync", "sync_strategy",
                "strategy_tag"}

# Required fields from Pine v2.0.0
REQUIRED_BAR_FIELDS = {"relay_user", "relay_id", "account", "instrument"}

# Per-instrument locks to prevent race conditions
_locks: dict[str, asyncio.Lock] = {}


def _get_lock(key: str) -> asyncio.Lock:
    if key not in _locks:
        _locks[key] = asyncio.Lock()
    return _locks[key]


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════

_migrated = False


def init_db():
    global _migrated
    if _migrated:
        return
    conn = db.get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_gate_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
            relay_user      TEXT,
            relay_id        TEXT,
            account         TEXT,
            instrument      TEXT,
            signal          TEXT,
            strategy        TEXT,
            confluence_score INTEGER,
            alert_type      TEXT DEFAULT 'bar',
            ai_decision     TEXT,
            ai_reason       TEXT,
            ai_latency_ms   INTEGER,
            relay_result    TEXT,
            relay_details   TEXT,
            payload_json    TEXT,
            ai_raw_response TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_positions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            relay_user      TEXT NOT NULL,
            relay_id        TEXT NOT NULL,
            account         TEXT NOT NULL,
            instrument      TEXT NOT NULL,
            direction       TEXT NOT NULL,
            entry_price     REAL,
            tick_value      REAL,
            strategy        TEXT,
            opened_at       TEXT NOT NULL,
            bar_count       INTEGER DEFAULT 0,
            last_bar_at     TEXT,
            last_pnl_tier   TEXT DEFAULT 'none',
            UNIQUE(relay_user, account, instrument)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            relay_user      TEXT NOT NULL,
            relay_id        TEXT NOT NULL,
            account         TEXT NOT NULL,
            instrument      TEXT NOT NULL,
            strategy        TEXT,
            direction       TEXT NOT NULL,
            entry_price     REAL,
            exit_price      REAL,
            tick_value      REAL,
            ticks_pnl       REAL,
            dollar_pnl      REAL,
            bars_held       INTEGER,
            exit_reason     TEXT,
            opened_at       TEXT,
            closed_at       TEXT,
            status          TEXT DEFAULT 'open'
        )
    """)
    # Migrations for existing tables
    cols = [r[1] for r in conn.execute("PRAGMA table_info(ai_gate_logs)").fetchall()]
    if "alert_type" not in cols:
        conn.execute("ALTER TABLE ai_gate_logs ADD COLUMN alert_type TEXT DEFAULT 'bar'")

    pos_cols = [r[1] for r in conn.execute("PRAGMA table_info(ai_positions)").fetchall()]
    if "last_bar_at" not in pos_cols:
        conn.execute("ALTER TABLE ai_positions ADD COLUMN last_bar_at TEXT")
    if "last_pnl_tier" not in pos_cols:
        conn.execute("ALTER TABLE ai_positions ADD COLUMN last_pnl_tier TEXT DEFAULT 'none'")
    if "bar_count" not in pos_cols and "bars_managed" in pos_cols:
        conn.execute("ALTER TABLE ai_positions RENAME COLUMN bars_managed TO bar_count")
    elif "bar_count" not in pos_cols:
        conn.execute("ALTER TABLE ai_positions ADD COLUMN bar_count INTEGER DEFAULT 0")

    conn.commit()
    conn.close()
    _migrated = True


# ══════════════════════════════════════════════════════════════════════════════
# POSITION DB OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _set_position(relay_user: str, relay_id: str, account: str,
                  instrument: str, direction: str, entry_price: float,
                  tick_value: float, strategy: str):
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    conn = db.get_connection()
    conn.execute("""
        INSERT INTO ai_positions
            (relay_user, relay_id, account, instrument, direction,
             entry_price, tick_value, strategy, opened_at, bar_count,
             last_bar_at, last_pnl_tier)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 'none')
        ON CONFLICT(relay_user, account, instrument) DO UPDATE SET
            relay_id = excluded.relay_id,
            direction = excluded.direction,
            entry_price = excluded.entry_price,
            tick_value = excluded.tick_value,
            strategy = excluded.strategy,
            opened_at = excluded.opened_at,
            bar_count = 0,
            last_bar_at = excluded.last_bar_at,
            last_pnl_tier = 'none'
    """, (relay_user, relay_id, account, instrument, direction,
          entry_price, tick_value, strategy, now, now))
    conn.commit()
    conn.close()


def _get_position(relay_user: str, account: str, instrument: str) -> dict | None:
    init_db()
    conn = db.get_connection()
    row = conn.execute(
        "SELECT * FROM ai_positions WHERE relay_user = ? AND account = ? AND instrument = ?",
        (relay_user, account, instrument)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def clear_position(relay_user: str, account: str, instrument: str):
    init_db()
    conn = db.get_connection()
    conn.execute(
        "DELETE FROM ai_positions WHERE relay_user = ? AND account = ? AND instrument = ?",
        (relay_user, account, instrument)
    )
    conn.commit()
    conn.close()


def _update_position_bar(relay_user: str, account: str, instrument: str,
                         pnl_tier: str = None):
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    conn = db.get_connection()
    if pnl_tier:
        conn.execute("""
            UPDATE ai_positions
            SET bar_count = bar_count + 1, last_bar_at = ?, last_pnl_tier = ?
            WHERE relay_user = ? AND account = ? AND instrument = ?
        """, (now, pnl_tier, relay_user, account, instrument))
    else:
        conn.execute("""
            UPDATE ai_positions
            SET bar_count = bar_count + 1, last_bar_at = ?
            WHERE relay_user = ? AND account = ? AND instrument = ?
        """, (now, relay_user, account, instrument))
    conn.commit()
    conn.close()


def list_positions(relay_user: str = None) -> list[dict]:
    init_db()
    conn = db.get_connection()
    if relay_user:
        rows = conn.execute(
            "SELECT * FROM ai_positions WHERE relay_user = ?", (relay_user,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM ai_positions").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# TRADE DB OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _open_trade(relay_user: str, relay_id: str, account: str,
                instrument: str, strategy: str, direction: str,
                entry_price: float) -> int:
    init_db()
    spec = get_instrument_spec(instrument)
    conn = db.get_connection()
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute("""
        INSERT INTO ai_trades
            (relay_user, relay_id, account, instrument, strategy,
             direction, entry_price, tick_value, opened_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
    """, (relay_user, relay_id, account, instrument, strategy,
          direction, entry_price, spec["point_value"], now))
    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return trade_id


def _close_trade(relay_user: str, account: str, instrument: str,
                 exit_price: float, exit_reason: str, bars_held: int = 0,
                 ticks_pnl: float = None, dollar_pnl: float = None):
    init_db()
    conn = db.get_connection()
    now = datetime.now(timezone.utc).isoformat()

    row = conn.execute("""
        SELECT id, direction, entry_price FROM ai_trades
        WHERE relay_user = ? AND account = ? AND instrument = ? AND status = 'open'
        ORDER BY id DESC LIMIT 1
    """, (relay_user, account, instrument)).fetchone()

    if row:
        trade = dict(row)

        # Compute P&L from instrument specs if not provided
        if ticks_pnl is None or dollar_pnl is None:
            ep = trade["entry_price"] or 0
            _, d_pnl, t_pnl = compute_dollar_pnl(
                instrument, trade["direction"], ep, exit_price
            )
            if ticks_pnl is None:
                ticks_pnl = t_pnl
            if dollar_pnl is None:
                dollar_pnl = d_pnl

        conn.execute("""
            UPDATE ai_trades SET
                exit_price = ?, ticks_pnl = ?, dollar_pnl = ?,
                bars_held = ?, exit_reason = ?, closed_at = ?, status = 'closed'
            WHERE id = ?
        """, (exit_price, round(ticks_pnl, 2), round(dollar_pnl, 2),
              bars_held, exit_reason, now, trade["id"]))

    conn.commit()
    conn.close()


def get_trades(relay_user: str = None, limit: int = 50) -> list[dict]:
    init_db()
    conn = db.get_connection()
    if relay_user:
        rows = conn.execute(
            "SELECT * FROM ai_trades WHERE relay_user = ? ORDER BY id DESC LIMIT ?",
            (relay_user, limit)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM ai_trades ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_trade_stats(relay_user: str = None) -> dict:
    init_db()
    conn = db.get_connection()
    where = "WHERE relay_user = ? AND status = 'closed'" if relay_user else "WHERE status = 'closed'"
    params = (relay_user,) if relay_user else ()

    row = conn.execute(f"""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN dollar_pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN dollar_pnl <= 0 THEN 1 ELSE 0 END) as losses,
            SUM(dollar_pnl) as total_pnl,
            AVG(dollar_pnl) as avg_pnl,
            MAX(dollar_pnl) as best_trade,
            MIN(dollar_pnl) as worst_trade,
            AVG(bars_held) as avg_bars
        FROM ai_trades {where}
    """, params).fetchone()

    conn.close()
    if row:
        d = dict(row)
        d["win_rate"] = round(d["wins"] / d["total"] * 100, 1) if d["total"] and d["total"] > 0 else 0
        return d
    return {"total": 0, "wins": 0, "losses": 0, "total_pnl": 0}


# ══════════════════════════════════════════════════════════════════════════════
# LOG DB OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _log_decision(
    relay_user: str, relay_id: str, account: str, instrument: str,
    signal: str, strategy: str, confluence_score: int,
    ai_decision: str, ai_reason: str, ai_latency_ms: int,
    relay_result: str = None, relay_details: str = None,
    payload_json: str = None, ai_raw_response: str = None,
    alert_type: str = "bar"
):
    init_db()
    conn = db.get_connection()
    conn.execute("""
        INSERT INTO ai_gate_logs
            (timestamp, relay_user, relay_id, account, instrument,
             signal, strategy, confluence_score, alert_type,
             ai_decision, ai_reason, ai_latency_ms,
             relay_result, relay_details, payload_json, ai_raw_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        relay_user, relay_id, account, instrument,
        signal, strategy, confluence_score, alert_type,
        ai_decision, ai_reason, ai_latency_ms,
        relay_result, relay_details, payload_json, ai_raw_response
    ))
    conn.commit()
    conn.close()


def get_logs(relay_user: str = None, limit: int = 50) -> list[dict]:
    init_db()
    conn = db.get_connection()
    if relay_user:
        rows = conn.execute(
            "SELECT * FROM ai_gate_logs WHERE relay_user = ? ORDER BY id DESC LIMIT ?",
            (relay_user, limit)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM ai_gate_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats(relay_user: str = None) -> dict:
    init_db()
    conn = db.get_connection()
    where = "WHERE relay_user = ?" if relay_user else ""
    params = (relay_user,) if relay_user else ()

    rows = conn.execute(f"""
        SELECT ai_decision, COUNT(*) as count, AVG(ai_latency_ms) as avg_latency_ms
        FROM ai_gate_logs {where} GROUP BY ai_decision
    """, params).fetchall()

    total = conn.execute(
        f"SELECT COUNT(*) as total FROM ai_gate_logs {where}", params
    ).fetchone()

    conn.close()
    stats = {row["ai_decision"]: {"count": row["count"], "avg_latency_ms": round(row["avg_latency_ms"] or 0)}
             for row in rows}
    stats["total"] = total["total"] if total else 0
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# P&L + EXIT SCORE COMPUTATION (server-side)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_pnl(instrument: str, direction: str, entry_price: float,
                 current_close: float, qty: int = 1) -> tuple[float, float, float]:
    """Returns (points_pnl, dollar_pnl, ticks_pnl) for a position."""
    return compute_dollar_pnl(instrument, direction, entry_price, current_close, qty)


def _pnl_tier(ticks: float) -> str:
    """Categorize P&L into tiers for threshold-based exit logic."""
    if ticks < -15:
        return "deep_loss"
    elif ticks < 0:
        return "loss"
    elif ticks < 10:
        return "scratch"
    elif ticks < 20:
        return "small_win"
    elif ticks < 50:
        return "medium_win"
    elif ticks < 100:
        return "good_win"
    elif ticks < 150:
        return "great_win"
    else:
        return "runner"


def _compute_exit_score(payload: dict, direction: str) -> int:
    """Compute exit pressure score from raw indicator flip flags.

    For LONG positions, bearish flips add pressure.
    For SHORT positions, bullish flips add pressure.
    """
    score = 0
    is_long = (direction == "long")

    # ZLEMA trend flipped against position: +3
    if is_long and payload.get("zlema_flipped_bear"):
        score += 3
    elif not is_long and payload.get("zlema_flipped_bull"):
        score += 3

    # UT Bot trail flipped against position: +3
    if is_long and payload.get("ut_flipped_bear"):
        score += 3
    elif not is_long and payload.get("ut_flipped_bull"):
        score += 3

    # ZLEMA went neutral: +2
    if payload.get("zlema_went_neutral"):
        score += 2

    # Stoch RSI crossed against position: +2
    if is_long and payload.get("stoch_bear_cross"):
        score += 2
    elif not is_long and payload.get("stoch_bull_cross"):
        score += 2

    # Squeeze momentum flipped against: +2
    if is_long and payload.get("mom_flipped_bear"):
        score += 2
    elif not is_long and payload.get("mom_flipped_bull"):
        score += 2

    # Exit warning (WT extreme): +2
    if is_long and payload.get("long_exit_warn"):
        score += 2
    elif not is_long and payload.get("short_exit_warn"):
        score += 2

    # Stoch RSI in extreme zone: +1
    if is_long and payload.get("stoch_in_ob"):
        score += 1
    elif not is_long and payload.get("stoch_in_os"):
        score += 1

    return score


def _should_call_ai_manage(exit_score: int, bar_count: int,
                           ticks_pnl: float, prev_tier: str) -> bool:
    """Decide whether to spend an AI call this bar."""
    # Always call if exit pressure detected
    if exit_score > 0:
        return True
    # Periodic checkpoint
    if bar_count > 0 and bar_count % AI_MANAGE_INTERVAL == 0:
        return True
    # P&L tier changed (crossed a threshold)
    current_tier = _pnl_tier(ticks_pnl)
    if current_tier != prev_tier:
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# ANTHROPIC API
# ══════════════════════════════════════════════════════════════════════════════

def _get_cvd_context(instrument: str) -> dict:
    """Get CVD metrics to inject into AI payload."""
    try:
        import cvd
        metrics = cvd.compute_metrics(instrument)
        state = cvd.get_cvd(instrument)
        return {
            "cvd": state.get("cvd", 0),
            "cvd_1m_delta": metrics.get("cvd_1m_delta", 0),
            "cvd_3m_delta": metrics.get("cvd_3m_delta", 0),
            "cvd_5m_delta": metrics.get("cvd_5m_delta", 0),
            "cvd_trend": metrics.get("cvd_trend", "flat"),
            "cvd_divergence": metrics.get("cvd_divergence", "none"),
        }
    except Exception:
        return {}


def _build_entry_message(payload: dict) -> str:
    instrument = payload.get("instrument", "")
    cvd_data = _get_cvd_context(instrument)
    indicator_data = {**cvd_data}
    for k, v in payload.items():
        if k not in ROUTING_KEYS:
            indicator_data[k] = v
    return f"Signal payload:\n{json.dumps(indicator_data, indent=2)}"


def _build_manage_message(payload: dict, position: dict,
                          exit_score: int, ticks_pnl: float,
                          dollar_pnl: float, bar_count: int) -> str:
    """Build enriched message with server-computed fields for manage decisions."""
    instrument = position.get("instrument", payload.get("instrument", ""))
    cvd_data = _get_cvd_context(instrument)
    enriched = {
        "position_direction": position["direction"],
        "entry_price": position["entry_price"],
        "current_price": payload.get("close", payload.get("price", 0)),
        "unrealized_ticks": ticks_pnl,
        "unrealized_dollars": dollar_pnl,
        "exit_score": exit_score,
        "bars_in_trade": bar_count,
        "hard_stop_ticks": payload.get("hard_stop_ticks", DEFAULT_HARD_STOP_TICKS),
        "max_bars_hold": payload.get("max_bars_hold", DEFAULT_MAX_BARS_HOLD),
        **cvd_data,
    }
    # Add all indicators from payload
    for k, v in payload.items():
        if k not in ROUTING_KEYS and k not in enriched:
            enriched[k] = v
    return f"Position management payload:\n{json.dumps(enriched, indent=2)}"


# ══════════════════════════════════════════════════════════════════════════════
# CROSSTRADE POSITION SYNC
# ══════════════════════════════════════════════════════════════════════════════

CT_API_BASE = "https://app.crosstrade.io/v1/api"


def _extract_root_symbol(instrument: str) -> str:
    """Extract root symbol from any instrument format.
    'MNQ1!' -> 'MNQ', 'MNQ JUN26' -> 'MNQ', 'MNQM2026' -> 'MNQ'
    """
    root = ""
    for ch in instrument.upper():
        if ch.isalpha():
            root += ch
        else:
            break
    return root


async def _get_ct_position(relay_user: str, account: str, instrument: str) -> dict | None:
    """Query CrossTrade API for current position on this instrument.
    Returns position dict or None if flat/error.
    Matches by root symbol since TV uses 'MNQ1!' but NT8 uses 'MNQ JUN26'.
    """
    user = db.get_user(relay_user)
    if not user:
        return None

    ct_key = user["crosstrade_key"]
    our_root = _extract_root_symbol(instrument)

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{CT_API_BASE}/accounts/{account}/positions",
                headers={
                    "Authorization": f"Bearer {ct_key}",
                    "Content-Type": "application/json"
                }
            )

        if resp.status_code != 200:
            logger.warning(f"CT API returned {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()
        positions = data.get("positions", [])

        for pos in positions:
            ct_instrument = pos.get("instrument", "")
            ct_root = _extract_root_symbol(ct_instrument)
            if ct_root == our_root:
                return pos

        return None  # No matching position = flat

    except Exception as e:
        logger.warning(f"CT position check failed: {e}")
        return None


async def sync_after_entry(relay_user: str, account: str, instrument: str):
    """After entering a trade, check CT for the real fill price and update our records."""
    ct_pos = await _get_ct_position(relay_user, account, instrument)
    if not ct_pos:
        logger.warning(f"[{relay_user}] Sync: CT shows no position after entry — may not have filled")
        return

    real_fill = ct_pos.get("averagePrice", 0)
    ct_direction = ct_pos.get("marketPosition", "").lower()
    ct_qty = ct_pos.get("quantity", 0)

    # Update our position with real fill price
    our_pos = _get_position(relay_user, account, instrument)
    if our_pos and real_fill and real_fill != our_pos["entry_price"]:
        conn = db.get_connection()
        conn.execute("""
            UPDATE ai_positions SET entry_price = ?
            WHERE relay_user = ? AND account = ? AND instrument = ?
        """, (real_fill, relay_user, account, instrument))
        # Also update the open trade record
        conn.execute("""
            UPDATE ai_trades SET entry_price = ?
            WHERE relay_user = ? AND account = ? AND instrument = ?
            AND status = 'open'
        """, (real_fill, relay_user, account, instrument))
        conn.commit()
        conn.close()
        logger.info(
            f"[{relay_user}] Sync: updated entry price {our_pos['entry_price']} → {real_fill} "
            f"(CT: {ct_direction} {ct_qty} @ {real_fill})"
        )


async def sync_position_check(relay_user: str, account: str, instrument: str,
                               our_direction: str) -> str:
    """Check if CT position matches our state. Returns 'synced', 'ct_flat', or 'mismatch'.
    Called periodically during manage to detect drift.
    """
    ct_pos = await _get_ct_position(relay_user, account, instrument)

    if ct_pos is None:
        # CT shows flat but we think we're in a position
        logger.warning(
            f"[{relay_user}] Sync: CT is FLAT but we track {our_direction} — position closed externally"
        )
        return "ct_flat"

    ct_direction = ct_pos.get("marketPosition", "").lower()
    if ct_direction != our_direction:
        logger.warning(
            f"[{relay_user}] Sync: direction mismatch — we track {our_direction}, CT shows {ct_direction}"
        )
        return "mismatch"

    return "synced"


async def sync_after_exit(relay_user: str, account: str, instrument: str):
    """After exiting, confirm CT is actually flat."""
    ct_pos = await _get_ct_position(relay_user, account, instrument)
    if ct_pos:
        ct_direction = ct_pos.get("marketPosition", "").lower()
        ct_qty = ct_pos.get("quantity", 0)
        logger.error(
            f"[{relay_user}] Sync: CT still shows {ct_direction} {ct_qty} after our exit! "
            f"Position may not have closed in NT8."
        )
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# ANTHROPIC API
# ══════════════════════════════════════════════════════════════════════════════

async def _call_anthropic(user_msg: str, system_prompt: str) -> tuple[str, str, int, str]:
    """Call Anthropic API. Returns: (decision, reason, latency_ms, raw_response)"""
    if not ANTHROPIC_API_KEY:
        return "ERROR", "ANTHROPIC_API_KEY not set", 0, ""

    request_body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 150,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_msg}]
    }

    start = datetime.now(timezone.utc)

    try:
        async with httpx.AsyncClient(timeout=ANTHROPIC_TIMEOUT) as client:
            resp = await client.post(
                ANTHROPIC_URL,
                json=request_body,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
            )

        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)

        if resp.status_code != 200:
            error_text = resp.text[:300]
            logger.error(f"Anthropic API {resp.status_code}: {error_text}")
            return "ERROR", f"API returned {resp.status_code}", latency_ms, error_text

        data = resp.json()
        raw_text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                raw_text += block.get("text", "")

        decision = "UNKNOWN"
        reason = raw_text.strip()

        for line in raw_text.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("DECISION:"):
                val = line.split(":", 1)[1].strip().upper()
                if val == "EXIT":
                    decision = "EXIT"
                elif val == "HOLD":
                    decision = "HOLD"
                elif "AGREE" in val and "DISAGREE" not in val:
                    decision = "AGREE"
                elif "DISAGREE" in val:
                    decision = "DISAGREE"
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        return decision, reason, latency_ms, raw_text

    except httpx.TimeoutException:
        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        logger.error(f"Anthropic API timeout after {latency_ms}ms")
        return "TIMEOUT", "API call timed out", latency_ms, ""
    except Exception as e:
        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        logger.error(f"Anthropic API error: {e}")
        return "ERROR", str(e), latency_ms, ""


# ══════════════════════════════════════════════════════════════════════════════
# CROSSTRADE ORDER SENDING
# ══════════════════════════════════════════════════════════════════════════════

def _build_ct_payload(ct_key: str, account: str, instrument: str,
                      action: str, qty: int, market_position: str,
                      prev_market_position: str, strategy_tag: str) -> str:
    """Build semicolon-delimited CrossTrade payload."""
    # CrossTrade requires a non-empty strategy_tag
    if not strategy_tag:
        strategy_tag = "AIGate"
    lines = [
        f"key={ct_key}",
        "command=PLACE",
        f"account={account}",
        f"instrument={instrument}",
        f"action={action}",
        f"qty={qty}",
        "order_type=MARKET",
        "tif=DAY",
        f"strategy_tag={strategy_tag}",
        "sync_strategy=true",
        f"market_position={market_position}",
        f"prev_market_position={prev_market_position}",
        "out_of_sync=flatten",
    ]
    return "\n".join(f"{line};" for line in lines)


async def _send_to_crosstrade(relay_user: str, account: str, instrument: str,
                               action: str, qty: int, market_position: str,
                               prev_market_position: str, strategy_tag: str) -> dict:
    """Send order directly to CrossTrade webhook. Returns status dict."""
    user = db.get_user(relay_user)
    if not user:
        logger.error(f"[{relay_user}] Cannot send to CT — user not found")
        return {"result": "error", "details": f"User not found: {relay_user}"}

    ct_key = user["crosstrade_key"]
    ct_url = user["ct_webhook_url"]

    payload_text = _build_ct_payload(
        ct_key, account, instrument, action, qty,
        market_position, prev_market_position, strategy_tag
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                ct_url,
                content=payload_text,
                headers={"Content-Type": "text/plain"}
            )
        logger.info(f"[{relay_user}] CT order sent: {action} {qty} {instrument} → {resp.status_code}")
        return {"result": "sent", "ct_status": resp.status_code}
    except Exception as e:
        logger.error(f"[{relay_user}] CT send failed: {e}")
        return {"result": "error", "details": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# FORCE CLOSE (unified close path — used by safety nets, AI, stale watchdog)
# ══════════════════════════════════════════════════════════════════════════════

async def _force_close(position: dict, exit_price: float, reason: str,
                       payload: dict = None):
    """Close a position. Always closes the trade, clears position, sends to CT if live."""
    relay_user = position["relay_user"]
    relay_id = position["relay_id"]
    account = position["account"]
    instrument = position["instrument"]
    direction = position["direction"]
    bar_count = position.get("bar_count", 0)

    points_pnl, dollar_pnl, ticks_pnl = _compute_pnl(
        instrument, direction, position["entry_price"], exit_price
    )

    logger.info(
        f"[{relay_user}/{relay_id}] CLOSING {direction} {instrument}: "
        f"{reason} | P&L: {ticks_pnl} ticks (${dollar_pnl}) | bars: {bar_count}"
    )

    # Close trade record
    _close_trade(relay_user, account, instrument, exit_price, reason,
                 bar_count, ticks_pnl=ticks_pnl, dollar_pnl=dollar_pnl)

    # Clear position
    clear_position(relay_user, account, instrument)

    # Send flatten to CrossTrade if live
    if not AI_DRY_RUN:
        exit_action = "sell" if direction == "long" else "buy"
        strategy_tag = ""
        if payload:
            strategy_tag = payload.get("strategy_tag", payload.get("relay_id", relay_id))
        await _send_to_crosstrade(
            relay_user, account, instrument,
            action=exit_action,
            qty=int(payload.get("qty", 1)) if payload else 1,
            market_position="flat",
            prev_market_position=direction,
            strategy_tag=strategy_tag
        )
        # Sync: confirm CT is actually flat after exit
        await asyncio.sleep(2)
        await sync_after_exit(relay_user, account, instrument)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

AI_ENTRY_PROMPT = """You are a trade conviction filter for algorithmic futures trading. You receive a JSON payload of technical indicator values computed at the moment a trading strategy fires a signal. Your job is to decide whether the signal has enough edge to execute.

You are NOT predicting whether price will go up or down. You are evaluating whether the CURRENT INDICATOR CONTEXT supports this specific signal direction and trade profile.

## Decision Framework

AGREE when:
- The majority of indicators align with the signal direction
- Momentum, trend, and volatility context support the trade thesis
- The risk/reward profile is reasonable given current conditions
- Session timing and market structure don't present obvious headwinds

DISAGREE when:
- Key indicators contradict the signal direction (e.g., LONG signal but HTF bearish, RSI overbought, price at session highs)
- Volatility is extreme and working against the trade (e.g., ATR already overextended for a mean-reversion scalp)
- Session timing is unfavorable (midday chop for momentum trades, close for new positions)
- Candle structure on the signal bar is adverse (e.g., LONG but strong upper wick rejection)
- Consolidation is too tight (no volatility to capture) or too wide (no clear direction)

## Strategy Type Calibration
- **scalp**: Be more permissive on trend alignment. Mean-reversion against short-term extremes is valid. Focus on volatility, overextension, and candle structure.
- **swing**: Require trend alignment. HTF bias must match signal direction. Reject counter-trend entries unless divergence is strong.
- **trend**: Require strong trend alignment across all timeframes. Only agree when momentum, trend, and structure are all aligned.

## Response Format
Respond with EXACTLY this format — no extra text, no markdown:

DECISION: AGREE
REASON: [one concise sentence explaining why]

or

DECISION: DISAGREE
REASON: [one concise sentence explaining why]"""


AI_MANAGE_PROMPT = """You are a trade exit manager for algorithmic futures trading. You receive a JSON payload with the current indicator state, position P&L, and a server-computed exit_score.

Safety nets (hard stop, time stop) have already been checked — if you are being asked, the position is NOT in immediate danger. Your job is to evaluate whether the trade thesis is intact or deteriorating.

## Exit Score (server-computed)
The exit_score reflects how many indicators have flipped against the position:
| Condition                              | Points |
|----------------------------------------|--------|
| ZLEMA trend flipped against position   | +3     |
| UT Bot trail flipped against position  | +3     |
| ZLEMA went neutral                     | +2     |
| Stoch RSI crossed against position     | +2     |
| Squeeze momentum flipped against       | +2     |
| Exit warning fired (WT extreme)        | +2     |
| Stoch RSI hit extreme zone             | +1     |

## Profit-Tiered Exit Thresholds
The bigger the profit, the less adversity you tolerate:

| Unrealized Profit        | Exit Score Needed |
|--------------------------|-------------------|
| Under 10 ticks           | Don't exit — too small to take |
| 10-20 ticks              | Score >= 5        |
| 20-50 ticks              | Score >= 4        |
| 50-100 ticks             | Score >= 3        |
| 100-150 ticks            | Score >= 2        |
| 150+ ticks               | Score >= 1 (hair trigger) |

## Your Decision
Use the exit_score AND the indicator context together. You can override:
- HOLD even with a high score if indicators show the adverse move is exhausting
- EXIT even with a low score if you see a clear structural breakdown

## Response Format
Respond with EXACTLY this format — no extra text, no markdown:

DECISION: EXIT
REASON: [one concise sentence]

or

DECISION: HOLD
REASON: [one concise sentence]"""


# ══════════════════════════════════════════════════════════════════════════════
# CORE: process_bar() — single entry point for every bar
# ══════════════════════════════════════════════════════════════════════════════

async def process_bar(payload: dict, body_text: str):
    """Main entry point. Called once per bar from Pine v2.0.0.

    State machine:
        FLAT + signal    → call AI for entry decision
        FLAT + no signal → skip (no cost)
        IN_POSITION      → compute exit score, check safety, maybe call AI
    """
    relay_user = payload.get("relay_user", "")
    relay_id = payload.get("relay_id", "")
    account = payload.get("account", "")
    instrument = payload.get("instrument", "")
    strategy = payload.get("strategy", "unknown")

    if not relay_user or not account or not instrument:
        logger.warning(f"Bar missing required fields: {payload.keys()}")
        return

    # Serialize per instrument to prevent race conditions
    lock_key = f"{relay_user}:{account}:{instrument}"
    async with _get_lock(lock_key):
        position = _get_position(relay_user, account, instrument)

        if position is None:
            # === FLAT — check for entry signals ===
            long_sig = payload.get("long_signal", False)
            short_sig = payload.get("short_signal", False)

            if long_sig or short_sig:
                direction = "long" if long_sig else "short"
                await _handle_entry(payload, body_text, direction)
            # else: flat, no signal — nothing to do
        else:
            # === IN POSITION — manage ===
            await _handle_manage(payload, body_text, position)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def _handle_entry(payload: dict, body_text: str, direction: str):
    """Flat + signal detected. Ask AI whether to enter."""
    relay_user = payload["relay_user"]
    relay_id = payload["relay_id"]
    account = payload["account"]
    instrument = payload["instrument"]
    strategy = payload.get("strategy", "unknown")
    confluence = payload.get("bull_confluence" if direction == "long" else "bear_confluence", 0)

    logger.info(f"[{relay_user}/{relay_id}] Signal: {direction.upper()} {instrument} (confluence {confluence})")

    # Call AI for entry decision
    user_msg = _build_entry_message(payload)
    ai_decision, ai_reason, ai_latency_ms, ai_raw = await _call_anthropic(
        user_msg, AI_ENTRY_PROMPT
    )

    logger.info(f"[{relay_user}/{relay_id}] AI: {ai_decision} ({ai_latency_ms}ms) — {ai_reason}")

    relay_result = None
    relay_details = None

    if ai_decision == "AGREE":
        entry_price = payload.get("close", payload.get("price", 0))
        spec = get_instrument_spec(instrument)

        # Open position + trade record
        _set_position(relay_user, relay_id, account, instrument,
                      direction, entry_price, spec["point_value"], strategy)
        _open_trade(relay_user, relay_id, account, instrument, strategy,
                    direction, entry_price)

        if AI_DRY_RUN:
            relay_result = "dry_run"
            relay_details = f"AI agreed — DRY RUN {direction.upper()} @ {entry_price}"
        else:
            entry_action = "buy" if direction == "long" else "sell"
            ct_response = await _send_to_crosstrade(
                relay_user, account, instrument,
                action=entry_action,
                qty=int(payload.get("qty", 1)),
                market_position=direction,
                prev_market_position="flat",
                strategy_tag=payload.get("strategy_tag", relay_id)
            )
            relay_result = ct_response.get("result", "unknown")
            relay_details = f"CT {entry_action} {direction} → {ct_response}"
            # Sync: get real fill price from CT after a short delay
            await asyncio.sleep(2)
            await sync_after_entry(relay_user, account, instrument)

        logger.info(f"[{relay_user}/{relay_id}] ENTERED {direction.upper()} @ {entry_price} ({relay_result})")

    elif ai_decision == "DISAGREE":
        relay_result = "ai_rejected"
        relay_details = f"AI disagreed: {ai_reason}"

    elif ai_decision == "TIMEOUT":
        relay_result = "ai_timeout"
        relay_details = f"AI timed out — signal skipped"

    else:
        relay_result = "ai_error"
        relay_details = f"AI error: {ai_reason}"

    _log_decision(
        relay_user=relay_user, relay_id=relay_id,
        account=account, instrument=instrument,
        signal=direction.upper(), strategy=strategy,
        confluence_score=confluence,
        ai_decision=ai_decision, ai_reason=ai_reason,
        ai_latency_ms=ai_latency_ms,
        relay_result=relay_result, relay_details=relay_details,
        payload_json=body_text, ai_raw_response=ai_raw,
        alert_type="entry"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MANAGE HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def _handle_manage(payload: dict, body_text: str, position: dict):
    """In position. Compute P&L, exit score, check safety nets, maybe call AI."""
    relay_user = position["relay_user"]
    relay_id = position["relay_id"]
    account = position["account"]
    instrument = position["instrument"]
    direction = position["direction"]
    strategy = position.get("strategy", "unknown")
    bar_count = (position.get("bar_count", 0) or 0) + 1
    prev_tier = position.get("last_pnl_tier", "none")

    current_price = payload.get("close", payload.get("price", 0))
    entry_price = position["entry_price"]

    # Server-side computations
    points_pnl, dollar_pnl, ticks_pnl = _compute_pnl(instrument, direction, entry_price, current_price)
    exit_score = _compute_exit_score(payload, direction)
    current_tier = _pnl_tier(ticks_pnl)

    # Update position bar count + last seen
    _update_position_bar(relay_user, account, instrument, pnl_tier=current_tier)

    logger.info(
        f"[{relay_user}/{relay_id}] BAR {bar_count}: {direction} {instrument} "
        f"P&L={ticks_pnl:.1f}t (${dollar_pnl:.2f}) exit_score={exit_score}"
    )

    # --- CT position sync (every 5 bars in live mode) ---
    if not AI_DRY_RUN and bar_count > 0 and bar_count % AI_MANAGE_INTERVAL == 0:
        sync_status = await sync_position_check(relay_user, account, instrument, direction)
        if sync_status == "ct_flat":
            # CT closed the position externally — clean up our state
            reason = "SYNC: CT position is flat — closed externally"
            _close_trade(relay_user, account, instrument, current_price, reason, bar_count,
                         ticks_pnl=ticks_pnl, dollar_pnl=dollar_pnl)
            clear_position(relay_user, account, instrument)
            _log_decision(
                relay_user=relay_user, relay_id=relay_id,
                account=account, instrument=instrument,
                signal="EXIT", strategy=strategy, confluence_score=exit_score,
                ai_decision="SYNC_EXIT", ai_reason=reason,
                ai_latency_ms=0, relay_result="sync_exit",
                relay_details=reason, payload_json=body_text,
                alert_type="manage"
            )
            return

    # --- Safety nets (no AI call) — thresholds from strategy payload ---
    hard_stop = payload.get("hard_stop_ticks", DEFAULT_HARD_STOP_TICKS)
    max_bars = payload.get("max_bars_hold", DEFAULT_MAX_BARS_HOLD)
    early_exit_ticks = payload.get("early_exit_ticks", DEFAULT_EARLY_EXIT_TICKS)
    early_exit_score = payload.get("early_exit_score", DEFAULT_EARLY_EXIT_SCORE)

    # Hard stop
    if ticks_pnl <= -abs(hard_stop):
        reason = f"HARD STOP: {ticks_pnl:.1f} ticks (limit: -{hard_stop})"
        await _force_close(position, current_price, reason, payload)
        _log_decision(
            relay_user=relay_user, relay_id=relay_id,
            account=account, instrument=instrument,
            signal="EXIT", strategy=strategy, confluence_score=exit_score,
            ai_decision="SAFETY_EXIT", ai_reason=reason,
            ai_latency_ms=0, relay_result="safety_exit",
            relay_details=reason, payload_json=body_text,
            alert_type="manage"
        )
        return

    # Time stop
    if bar_count >= max_bars:
        reason = f"TIME STOP: {bar_count} bars (limit: {max_bars})"
        await _force_close(position, current_price, reason, payload)
        _log_decision(
            relay_user=relay_user, relay_id=relay_id,
            account=account, instrument=instrument,
            signal="EXIT", strategy=strategy, confluence_score=exit_score,
            ai_decision="SAFETY_EXIT", ai_reason=reason,
            ai_latency_ms=0, relay_result="safety_exit",
            relay_details=reason, payload_json=body_text,
            alert_type="manage"
        )
        return

    # Early exit: losing N+ ticks AND high exit score
    if ticks_pnl <= early_exit_ticks and exit_score >= early_exit_score:
        reason = f"EARLY EXIT: {ticks_pnl:.1f} ticks with exit_score={exit_score}"
        await _force_close(position, current_price, reason, payload)
        _log_decision(
            relay_user=relay_user, relay_id=relay_id,
            account=account, instrument=instrument,
            signal="EXIT", strategy=strategy, confluence_score=exit_score,
            ai_decision="SAFETY_EXIT", ai_reason=reason,
            ai_latency_ms=0, relay_result="safety_exit",
            relay_details=reason, payload_json=body_text,
            alert_type="manage"
        )
        return

    # --- API cost gate: should we call AI this bar? ---
    if not _should_call_ai_manage(exit_score, bar_count, ticks_pnl, prev_tier):
        return  # Nothing interesting — skip AI call, save money

    # --- Call AI ---
    user_msg = _build_manage_message(payload, position, exit_score,
                                     ticks_pnl, dollar_pnl, bar_count)
    ai_decision, ai_reason, ai_latency_ms, ai_raw = await _call_anthropic(
        user_msg, AI_MANAGE_PROMPT
    )

    logger.info(
        f"[{relay_user}/{relay_id}] AI manage: {ai_decision} ({ai_latency_ms}ms) — {ai_reason}"
    )

    relay_result = None
    relay_details = None

    if ai_decision == "EXIT":
        reason = f"AI EXIT: {ai_reason}"
        await _force_close(position, current_price, reason, payload)
        relay_result = "exit"
        relay_details = f"AI EXIT — P&L: {ticks_pnl:.1f} ticks (${dollar_pnl:.2f})"

    elif ai_decision == "HOLD":
        relay_result = "hold"
        relay_details = f"Holding — bar {bar_count}, P&L: {ticks_pnl:.1f} ticks"

    elif ai_decision == "TIMEOUT":
        relay_result = "hold_timeout"
        relay_details = "AI timed out — holding by default"

    else:
        relay_result = "hold_error"
        relay_details = f"AI error — holding: {ai_reason}"

    _log_decision(
        relay_user=relay_user, relay_id=relay_id,
        account=account, instrument=instrument,
        signal=ai_decision, strategy=strategy,
        confluence_score=exit_score,
        ai_decision=ai_decision, ai_reason=ai_reason,
        ai_latency_ms=ai_latency_ms,
        relay_result=relay_result, relay_details=relay_details,
        payload_json=body_text, ai_raw_response=ai_raw,
        alert_type="manage"
    )


# ══════════════════════════════════════════════════════════════════════════════
# STALE POSITION WATCHDOG
# ══════════════════════════════════════════════════════════════════════════════

async def stale_position_watchdog():
    """Background task: close positions that haven't received bar data.
    Runs every 60 seconds. Catches orphaned positions from Pine crashes,
    chart closures, or network issues.
    """
    while True:
        await asyncio.sleep(60)
        try:
            positions = list_positions()
            now = datetime.now(timezone.utc)
            for pos in positions:
                last_bar = pos.get("last_bar_at")
                if not last_bar:
                    continue
                try:
                    last_dt = datetime.fromisoformat(last_bar)
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                    age = (now - last_dt).total_seconds()
                except (ValueError, TypeError):
                    continue

                if age > STALE_TIMEOUT_SECONDS:
                    # Use last known price as exit price (best we can do)
                    exit_price = pos.get("entry_price", 0)
                    reason = f"STALE TIMEOUT: no bar data for {age:.0f}s"
                    logger.warning(
                        f"[{pos['relay_user']}/{pos['relay_id']}] {reason} — closing {pos['instrument']}"
                    )
                    await _force_close(pos, exit_price, reason)
        except Exception as e:
            logger.error(f"Stale watchdog error: {e}")


def startup_recovery():
    """On server restart, close any positions with stale last_bar_at."""
    init_db()
    positions = list_positions()
    now = datetime.now(timezone.utc)
    for pos in positions:
        last_bar = pos.get("last_bar_at")
        if not last_bar:
            # No last_bar_at means position was created before migration
            # Close it as stale
            exit_price = pos.get("entry_price", 0)
            _close_trade(pos["relay_user"], pos["account"], pos["instrument"],
                         exit_price, "STALE: server restart (no last_bar_at)", 0)
            clear_position(pos["relay_user"], pos["account"], pos["instrument"])
            logger.warning(f"Startup recovery: closed stale position {pos['instrument']}")
            continue

        try:
            last_dt = datetime.fromisoformat(last_bar)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            age = (now - last_dt).total_seconds()
        except (ValueError, TypeError):
            age = STALE_TIMEOUT_SECONDS + 1

        if age > STALE_TIMEOUT_SECONDS:
            exit_price = pos.get("entry_price", 0)
            _close_trade(pos["relay_user"], pos["account"], pos["instrument"],
                         exit_price, f"STALE: server restart ({age:.0f}s old)", 0)
            clear_position(pos["relay_user"], pos["account"], pos["instrument"])
            logger.warning(f"Startup recovery: closed stale position {pos['instrument']} ({age:.0f}s old)")