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
import indicator_engine

logger = logging.getLogger("trade_relay")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
ANTHROPIC_TIMEOUT = 30.0

# Model registry: model_id -> (api_key_env, base_url)
AI_MODELS = {
    "claude-sonnet-4-20250514": {
        "label": "Claude Sonnet 4",
        "base_url": "https://api.anthropic.com/v1/messages",
        "key_env": "ANTHROPIC_API_KEY",
    },
    "MiniMax-M2.7": {
        "label": "MiniMax M2.7",
        "base_url": "https://api.minimax.io/anthropic/v1/messages",
        "key_env": "MINIMAX_API_KEY",
    },
}
DEFAULT_AI_MODEL = "MiniMax-M2.7"

def _get_model_config(model_id: str) -> dict:
    """Get API config for a model. Falls back to default."""
    cfg = AI_MODELS.get(model_id, AI_MODELS[DEFAULT_AI_MODEL])
    key_env = cfg["key_env"]
    api_key = os.environ.get(key_env, "")
    return {"base_url": cfg["base_url"], "api_key": api_key, "model": model_id}

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

# Fields that should ALWAYS be included regardless of indicator selection
ALWAYS_INCLUDE_FIELDS = {
    "relay_user", "relay_id", "account", "instrument", "strategy",
    "strategy_type", "timeframe", "qty", "strategy_tag",
    "target_ticks", "stop_ticks", "hard_stop_ticks", "rr", "tick_value",
    "long_signal", "short_signal",  # signals always needed for state machine
    "close", "price", "open", "high", "low", "volume",  # OHLCV always needed
}

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

    # Bot configuration table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_bots (
            bot_id          TEXT PRIMARY KEY,
            mode            TEXT NOT NULL DEFAULT 'normal',
            source_bot      TEXT,
            relay_id        TEXT,
            account         TEXT NOT NULL,
            strategy_tag    TEXT NOT NULL DEFAULT 'AIGate',
            entry_prompt    TEXT,
            manage_prompt   TEXT,
            enabled         INTEGER NOT NULL DEFAULT 1,
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # Add strategy_name and config_json columns if missing
    bot_cols = [r[1] for r in conn.execute("PRAGMA table_info(ai_bots)").fetchall()]
    if "strategy_name" not in bot_cols:
        conn.execute("ALTER TABLE ai_bots ADD COLUMN strategy_name TEXT")
    if "config_json" not in bot_cols:
        conn.execute("ALTER TABLE ai_bots ADD COLUMN config_json TEXT")
    if "relay_user" not in bot_cols:
        conn.execute("ALTER TABLE ai_bots ADD COLUMN relay_user TEXT DEFAULT 'titon'")
    if "ai_model" not in bot_cols:
        conn.execute(f"ALTER TABLE ai_bots ADD COLUMN ai_model TEXT DEFAULT '{DEFAULT_AI_MODEL}'")

    # Strategy indicator registry
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_strategy_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT NOT NULL,
            field_name TEXT NOT NULL,
            field_type TEXT NOT NULL DEFAULT 'number',
            category TEXT NOT NULL,
            description TEXT,
            UNIQUE(strategy_id, field_name)
        )
    """)

    # Bot indicator selections
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_bot_indicators (
            bot_id TEXT NOT NULL,
            field_name TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY(bot_id, field_name)
        )
    """)

    # Persistent 1-min bar storage for charting
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_bars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL DEFAULT 0,
            cvd REAL DEFAULT 0,
            cvd_delta REAL DEFAULT 0,
            UNIQUE(instrument, timestamp)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ai_bars_inst_ts ON ai_bars(instrument, timestamp)")

    # Persistent 5-min bar storage for longer-term indicators
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_bars_5m (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL DEFAULT 0,
            cvd REAL DEFAULT 0,
            cvd_delta REAL DEFAULT 0,
            UNIQUE(instrument, timestamp)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ai_bars_5m_inst_ts ON ai_bars_5m(instrument, timestamp)")

    conn.commit()
    conn.close()
    _migrated = True

    # Seed default indicators
    seed_precision_scalp_indicators()


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENT BAR STORAGE
# ══════════════════════════════════════════════════════════════════════════════

def save_bar(instrument: str, timestamp: str, o: float, h: float, l: float, c: float,
             volume: float = 0, cvd: float = 0, cvd_delta: float = 0):
    """Save a completed 1-min bar to the database."""
    init_db()
    conn = db.get_connection()
    conn.execute("""
        INSERT INTO ai_bars (instrument, timestamp, open, high, low, close, volume, cvd, cvd_delta)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(instrument, timestamp) DO UPDATE SET
            high = MAX(ai_bars.high, excluded.high),
            low = MIN(ai_bars.low, excluded.low),
            close = excluded.close,
            volume = excluded.volume,
            cvd = excluded.cvd,
            cvd_delta = excluded.cvd_delta
    """, (instrument, timestamp, o, h, l, c, volume, cvd, cvd_delta))
    conn.commit()
    conn.close()


def get_bars(instrument: str, limit: int = 480) -> list[dict]:
    """Get recent bars for an instrument, ordered by timestamp ascending."""
    init_db()
    conn = db.get_connection()
    rows = conn.execute("""
        SELECT * FROM ai_bars WHERE instrument = ?
        ORDER BY timestamp DESC LIMIT ?
    """, (instrument, limit)).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


def purge_old_bars(days_1m: int = 14, days_5m: int = 60):
    """Delete old bars — 1-min after 14 days, 5-min after 60 days."""
    init_db()
    conn = db.get_connection()
    conn.execute("""
        DELETE FROM ai_bars WHERE timestamp < datetime('now', ?)
    """, (f'-{days_1m} days',))
    conn.execute("""
        DELETE FROM ai_bars_5m WHERE timestamp < datetime('now', ?)
    """, (f'-{days_5m} days',))
    conn.commit()
    conn.close()


def save_bar_5m(instrument: str, timestamp: str, o: float, h: float, l: float, c: float,
                volume: float = 0, cvd: float = 0, cvd_delta: float = 0):
    """Save a completed 5-min bar to the database."""
    init_db()
    conn = db.get_connection()
    conn.execute("""
        INSERT INTO ai_bars_5m (instrument, timestamp, open, high, low, close, volume, cvd, cvd_delta)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(instrument, timestamp) DO UPDATE SET
            high = MAX(ai_bars_5m.high, excluded.high),
            low = MIN(ai_bars_5m.low, excluded.low),
            close = excluded.close,
            volume = excluded.volume,
            cvd = excluded.cvd,
            cvd_delta = excluded.cvd_delta
    """, (instrument, timestamp, o, h, l, c, volume, cvd, cvd_delta))
    conn.commit()
    conn.close()


def get_bars_5m(instrument: str, limit: int = 480) -> list[dict]:
    """Get recent 5-min bars for an instrument."""
    init_db()
    conn = db.get_connection()
    rows = conn.execute("""
        SELECT * FROM ai_bars_5m WHERE instrument = ?
        ORDER BY timestamp DESC LIMIT ?
    """, (instrument, limit)).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


def aggregate_5m_bar(instrument: str):
    """Aggregate the last 5 completed 1-min bars into a 5-min bar.

    Called every time a 1-min bar completes. Only writes a 5-min bar
    when the minute is on a 5-min boundary (xx:00, xx:05, xx:10, etc).
    """
    init_db()
    conn = db.get_connection()

    # Get the latest 1-min bar timestamp
    latest = conn.execute("""
        SELECT timestamp FROM ai_bars WHERE instrument = ?
        ORDER BY timestamp DESC LIMIT 1
    """, (instrument,)).fetchone()

    if not latest:
        conn.close()
        return

    from datetime import datetime as dt
    ts_str = latest["timestamp"]
    try:
        ts = dt.fromisoformat(ts_str.replace("+00:00", "").replace("Z", ""))
    except:
        conn.close()
        return

    # Only aggregate on 5-min boundaries
    if ts.minute % 5 != 4:  # minute 4, 9, 14, 19, etc = end of 5-min period
        conn.close()
        return

    # Get the 5 most recent 1-min bars for this instrument
    rows = conn.execute("""
        SELECT * FROM ai_bars WHERE instrument = ?
        ORDER BY timestamp DESC LIMIT 5
    """, (instrument,)).fetchall()

    if len(rows) < 5:
        conn.close()
        return

    bars = [dict(r) for r in reversed(rows)]

    # Aggregate OHLCV
    bar_5m_ts = bars[0]["timestamp"]  # start of the 5-min period
    o = bars[0]["open"]
    h = max(b["high"] for b in bars)
    l = min(b["low"] for b in bars)
    c = bars[-1]["close"]
    vol = bars[-1].get("volume", 0)  # cumulative volume from last bar
    cvd = bars[-1].get("cvd", 0)
    cvd_delta = sum(b.get("cvd_delta", 0) for b in bars)

    conn.close()

    save_bar_5m(instrument, bar_5m_ts, o, h, l, c, vol, cvd, cvd_delta)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY INDICATOR OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def add_strategy_indicators(strategy_id: str, indicators: list[dict]):
    """Bulk insert indicators for a strategy."""
    init_db()
    conn = db.get_connection()
    for ind in indicators:
        conn.execute("""
            INSERT INTO ai_strategy_indicators
                (strategy_id, field_name, field_type, category, description)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(strategy_id, field_name) DO UPDATE SET
                field_type = excluded.field_type,
                category = excluded.category,
                description = excluded.description
        """, (strategy_id, ind["field_name"], ind.get("field_type", "number"),
              ind["category"], ind.get("description", "")))
    conn.commit()
    conn.close()


def get_strategy_indicators(strategy_id: str) -> list[dict]:
    """Get all indicators for a strategy, grouped by category."""
    init_db()
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT * FROM ai_strategy_indicators WHERE strategy_id = ? ORDER BY category, field_name",
        (strategy_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_bot_indicators(bot_id: str, indicators: dict[str, bool]):
    """Set which indicators a bot uses (dict of field_name -> enabled)."""
    init_db()
    conn = db.get_connection()
    for field_name, enabled in indicators.items():
        conn.execute("""
            INSERT INTO ai_bot_indicators (bot_id, field_name, enabled)
            VALUES (?, ?, ?)
            ON CONFLICT(bot_id, field_name) DO UPDATE SET enabled = excluded.enabled
        """, (bot_id, field_name, 1 if enabled else 0))
    conn.commit()
    conn.close()


def get_bot_indicators(bot_id: str) -> dict[str, bool]:
    """Get enabled/disabled state for a bot's indicators."""
    init_db()
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT field_name, enabled FROM ai_bot_indicators WHERE bot_id = ?",
        (bot_id,)
    ).fetchall()
    conn.close()
    return {r["field_name"]: bool(r["enabled"]) for r in rows}


def get_bot_enabled_fields(bot_id: str) -> set[str]:
    """Get just the field names that are enabled (used for payload filtering)."""
    init_db()
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT field_name FROM ai_bot_indicators WHERE bot_id = ? AND enabled = 1",
        (bot_id,)
    ).fetchall()
    conn.close()
    return {r["field_name"] for r in rows}


def _filter_payload_for_bot(payload: dict, bot_id: str) -> dict:
    """Filter payload to only include indicators the bot has enabled.
    Always keeps ALWAYS_INCLUDE_FIELDS. If no indicator config exists, returns all fields.
    """
    bot_indicators = get_bot_indicators(bot_id)
    if not bot_indicators:
        # No indicator config — backwards compatible, send everything
        return payload

    enabled_fields = {f for f, en in bot_indicators.items() if en}
    filtered = {}
    for k, v in payload.items():
        if k in ALWAYS_INCLUDE_FIELDS or k in enabled_fields:
            filtered[k] = v
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# SEED: PrecisionScalp INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def seed_precision_scalp_indicators():
    """Register all PrecisionScalp indicators. Only inserts if not already present."""
    conn = db.get_connection()
    existing = conn.execute(
        "SELECT COUNT(*) as cnt FROM ai_strategy_indicators WHERE strategy_id = 'PrecisionScalp'"
    ).fetchone()
    conn.close()
    if existing and existing["cnt"] > 0:
        return

    indicators = [
        # signal
        {"field_name": "bull_confluence", "field_type": "number", "category": "signal", "description": "Bull filter count (0-4)"},
        {"field_name": "bear_confluence", "field_type": "number", "category": "signal", "description": "Bear filter count (0-4)"},
        {"field_name": "long_exit_warn", "field_type": "boolean", "category": "signal", "description": "WaveTrend overbought warning"},
        {"field_name": "short_exit_warn", "field_type": "boolean", "category": "signal", "description": "WaveTrend oversold warning"},
        # layer
        {"field_name": "f_zlema_trend_bull", "field_type": "boolean", "category": "layer", "description": "ZLEMA micro-trend bullish"},
        {"field_name": "f_zlema_trend_bear", "field_type": "boolean", "category": "layer", "description": "ZLEMA micro-trend bearish"},
        {"field_name": "f_wt_mom_bull", "field_type": "boolean", "category": "layer", "description": "WaveTrend + momentum bullish"},
        {"field_name": "f_wt_mom_bear", "field_type": "boolean", "category": "layer", "description": "WaveTrend + momentum bearish"},
        {"field_name": "f_squeeze_off", "field_type": "boolean", "category": "layer", "description": "Squeeze momentum released"},
        {"field_name": "f_trending", "field_type": "boolean", "category": "layer", "description": "Efficiency ratio above threshold"},
        # oscillator
        {"field_name": "rsi14", "field_type": "number", "category": "oscillator", "description": "RSI 14-period"},
        {"field_name": "cci20", "field_type": "number", "category": "oscillator", "description": "CCI 20-period"},
        {"field_name": "stoch_rsi_k", "field_type": "number", "category": "oscillator", "description": "Stoch RSI K value"},
        {"field_name": "stoch_rsi_d", "field_type": "number", "category": "oscillator", "description": "Stoch RSI D value"},
        {"field_name": "stoch_in_ob", "field_type": "boolean", "category": "oscillator", "description": "Stoch RSI overbought"},
        {"field_name": "stoch_in_os", "field_type": "boolean", "category": "oscillator", "description": "Stoch RSI oversold"},
        {"field_name": "stoch_bull_cross", "field_type": "boolean", "category": "oscillator", "description": "Stoch RSI bullish crossover"},
        {"field_name": "stoch_bear_cross", "field_type": "boolean", "category": "oscillator", "description": "Stoch RSI bearish crossover"},
        {"field_name": "rsi_bull_div", "field_type": "boolean", "category": "oscillator", "description": "RSI bullish divergence"},
        {"field_name": "rsi_bear_div", "field_type": "boolean", "category": "oscillator", "description": "RSI bearish divergence"},
        {"field_name": "wt1", "field_type": "number", "category": "oscillator", "description": "WaveTrend line 1"},
        {"field_name": "wt2", "field_type": "number", "category": "oscillator", "description": "WaveTrend line 2"},
        {"field_name": "wt_bull_cross", "field_type": "boolean", "category": "oscillator", "description": "WaveTrend bullish crossover"},
        {"field_name": "wt_bear_cross", "field_type": "boolean", "category": "oscillator", "description": "WaveTrend bearish crossover"},
        # trend
        {"field_name": "zlema", "field_type": "number", "category": "trend", "description": "ZLEMA value"},
        {"field_name": "zl_slope", "field_type": "number", "category": "trend", "description": "ZLEMA slope"},
        {"field_name": "zlema_flipped_bull", "field_type": "boolean", "category": "trend", "description": "ZLEMA flipped bullish this bar"},
        {"field_name": "zlema_flipped_bear", "field_type": "boolean", "category": "trend", "description": "ZLEMA flipped bearish this bar"},
        {"field_name": "zlema_went_neutral", "field_type": "boolean", "category": "trend", "description": "ZLEMA went neutral this bar"},
        {"field_name": "ut_trail", "field_type": "number", "category": "trend", "description": "UT Bot trail level"},
        {"field_name": "ut_flipped_bull", "field_type": "boolean", "category": "trend", "description": "UT Bot flipped bullish"},
        {"field_name": "ut_flipped_bear", "field_type": "boolean", "category": "trend", "description": "UT Bot flipped bearish"},
        {"field_name": "ema20", "field_type": "number", "category": "trend", "description": "EMA 20"},
        {"field_name": "ema50", "field_type": "number", "category": "trend", "description": "EMA 50"},
        {"field_name": "ema20_slope5", "field_type": "number", "category": "trend", "description": "EMA 20 slope over 5 bars"},
        {"field_name": "price_above_ema20", "field_type": "boolean", "category": "trend", "description": "Price above EMA 20"},
        {"field_name": "price_above_ema50", "field_type": "boolean", "category": "trend", "description": "Price above EMA 50"},
        {"field_name": "htf_bias", "field_type": "string", "category": "trend", "description": "Higher timeframe EMA bias"},
        {"field_name": "sq_mom", "field_type": "number", "category": "trend", "description": "Squeeze momentum value"},
        {"field_name": "squeeze_on", "field_type": "boolean", "category": "trend", "description": "Squeeze active (BB inside KC)"},
        {"field_name": "mom_flipped_bull", "field_type": "boolean", "category": "trend", "description": "Momentum flipped bullish"},
        {"field_name": "mom_flipped_bear", "field_type": "boolean", "category": "trend", "description": "Momentum flipped bearish"},
        {"field_name": "er_val", "field_type": "number", "category": "trend", "description": "Efficiency ratio value"},
        # volatility
        {"field_name": "atr14", "field_type": "number", "category": "volatility", "description": "ATR 14-period"},
        {"field_name": "atr_consumed_pct", "field_type": "number", "category": "volatility", "description": "ATR consumed percentage"},
        {"field_name": "bb_pctb", "field_type": "number", "category": "volatility", "description": "Bollinger Band %B"},
        {"field_name": "bar_range_vs_atr", "field_type": "number", "category": "volatility", "description": "Bar range vs ATR ratio"},
        {"field_name": "vol_ratio", "field_type": "number", "category": "volatility", "description": "Volume ratio vs 20-bar avg"},
        # candle
        {"field_name": "bar_body_pct", "field_type": "number", "category": "candle", "description": "Bar body as % of range"},
        {"field_name": "bar_upper_wick_pct", "field_type": "number", "category": "candle", "description": "Upper wick as % of range"},
        {"field_name": "bar_lower_wick_pct", "field_type": "number", "category": "candle", "description": "Lower wick as % of range"},
        {"field_name": "bar_is_bullish", "field_type": "boolean", "category": "candle", "description": "Bar closed bullish"},
        {"field_name": "prev_3_pattern", "field_type": "string", "category": "candle", "description": "Last 3 bars pattern (H/L)"},
        # session
        {"field_name": "vwap_dist_ticks", "field_type": "number", "category": "session", "description": "Distance from VWAP in ticks"},
        {"field_name": "dist_sess_high_ticks", "field_type": "number", "category": "session", "description": "Distance from session high"},
        {"field_name": "dist_sess_low_ticks", "field_type": "number", "category": "session", "description": "Distance from session low"},
        {"field_name": "bars_since_swing_high", "field_type": "number", "category": "session", "description": "Bars since last swing high"},
        {"field_name": "bars_since_swing_low", "field_type": "number", "category": "session", "description": "Bars since last swing low"},
        {"field_name": "cons_range_ticks", "field_type": "number", "category": "session", "description": "Consolidation range in ticks"},
        {"field_name": "cons_range_vs_atr", "field_type": "number", "category": "session", "description": "Consolidation range vs ATR"},
        {"field_name": "session_bucket", "field_type": "string", "category": "session", "description": "Session time bucket"},
        {"field_name": "day_of_week", "field_type": "string", "category": "session", "description": "Day of week"},
        {"field_name": "mins_to_maintenance", "field_type": "number", "category": "session", "description": "Minutes until CME maintenance"},
        {"field_name": "bar_time_et", "field_type": "string", "category": "time", "description": "Bar time in Eastern (e.g. 09:45 ET)"},
        {"field_name": "bar_time_hhmm", "field_type": "number", "category": "time", "description": "Bar time as HHMM integer (e.g. 945)"},
        {"field_name": "bar_time_unix", "field_type": "number", "category": "time", "description": "Bar time as Unix timestamp"},
        {"field_name": "last_5_closes", "field_type": "array", "category": "session", "description": "Last 5 bar close prices"},
        # cvd
        {"field_name": "cvd", "field_type": "number", "category": "cvd", "description": "Cumulative volume delta"},
        {"field_name": "cvd_1m_delta", "field_type": "number", "category": "cvd", "description": "CVD change last 1 minute"},
        {"field_name": "cvd_3m_delta", "field_type": "number", "category": "cvd", "description": "CVD change last 3 minutes"},
        {"field_name": "cvd_5m_delta", "field_type": "number", "category": "cvd", "description": "CVD change last 5 minutes"},
        {"field_name": "cvd_trend", "field_type": "string", "category": "cvd", "description": "CVD trend direction"},
        {"field_name": "cvd_divergence", "field_type": "string", "category": "cvd", "description": "CVD vs price divergence"},
    ]
    add_strategy_indicators("PrecisionScalp", indicators)
    logger.info("Seeded PrecisionScalp indicators (%d)", len(indicators))


# ══════════════════════════════════════════════════════════════════════════════
# BOT CONFIG OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def add_bot(bot_id: str, mode: str, account: str, strategy_tag: str,
            source_bot: str = None, relay_id: str = None,
            entry_prompt: str = None, manage_prompt: str = None,
            strategy_name: str = None, config_json: str = None,
            relay_user: str = "titon", ai_model: str = None):
    init_db()
    if ai_model is None:
        ai_model = DEFAULT_AI_MODEL
    conn = db.get_connection()
    conn.execute("""
        INSERT INTO ai_bots (bot_id, mode, source_bot, relay_id, account,
                             strategy_tag, entry_prompt, manage_prompt,
                             strategy_name, config_json, relay_user, ai_model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(bot_id) DO UPDATE SET
            mode = excluded.mode, source_bot = excluded.source_bot,
            relay_id = excluded.relay_id, account = excluded.account,
            strategy_tag = excluded.strategy_tag,
            entry_prompt = excluded.entry_prompt,
            manage_prompt = excluded.manage_prompt,
            strategy_name = excluded.strategy_name,
            config_json = excluded.config_json,
            relay_user = excluded.relay_user,
            ai_model = excluded.ai_model
    """, (bot_id, mode, source_bot, relay_id, account, strategy_tag,
          entry_prompt, manage_prompt, strategy_name, config_json, relay_user, ai_model))
    conn.commit()
    conn.close()


def get_bot(bot_id: str) -> dict | None:
    init_db()
    conn = db.get_connection()
    row = conn.execute("SELECT * FROM ai_bots WHERE bot_id = ?", (bot_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_bots() -> list[dict]:
    init_db()
    conn = db.get_connection()
    rows = conn.execute("SELECT * FROM ai_bots ORDER BY created_at").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_bot_enabled(bot_id: str, enabled: bool):
    init_db()
    conn = db.get_connection()
    conn.execute("UPDATE ai_bots SET enabled = ? WHERE bot_id = ?", (1 if enabled else 0, bot_id))
    conn.commit()
    conn.close()


def update_bot(bot_id: str, **kwargs):
    """Update specific fields on a bot config."""
    init_db()
    allowed = {'mode', 'source_bot', 'relay_id', 'account', 'strategy_tag',
               'entry_prompt', 'manage_prompt', 'enabled', 'ai_model'}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [bot_id]
    conn = db.get_connection()
    conn.execute(f"UPDATE ai_bots SET {set_clause} WHERE bot_id = ?", values)
    conn.commit()
    conn.close()


def delete_bot(bot_id: str):
    """Delete a bot, its indicator selections, and all associated trade data."""
    init_db()
    conn = db.get_connection()
    conn.execute("DELETE FROM ai_bots WHERE bot_id = ?", (bot_id,))
    conn.execute("DELETE FROM ai_bot_indicators WHERE bot_id = ?", (bot_id,))
    conn.execute("DELETE FROM ai_gate_logs WHERE relay_id = ?", (bot_id,))
    conn.execute("DELETE FROM ai_trades WHERE relay_id = ?", (bot_id,))
    conn.execute("DELETE FROM ai_positions WHERE relay_id = ?", (bot_id,))
    conn.commit()
    conn.close()


def _get_copy_bots_for_source(source_relay_id: str) -> list[dict]:
    """Get all enabled copy bots that mirror a given source relay_id."""
    init_db()
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT * FROM ai_bots WHERE mode = 'copy' AND source_bot = ? AND enabled = 1",
        (source_relay_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


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


def get_trades(relay_user: str = None, relay_id: str = None, limit: int = 50) -> list[dict]:
    init_db()
    conn = db.get_connection()
    where, params = [], []
    if relay_user:
        where.append("relay_user = ?"); params.append(relay_user)
    if relay_id:
        where.append("relay_id = ?"); params.append(relay_id)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    params.append(limit)
    rows = conn.execute(
        f"SELECT * FROM ai_trades {clause} ORDER BY id DESC LIMIT ?", params
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_trade_stats(relay_user: str = None, relay_id: str = None) -> dict:
    init_db()
    conn = db.get_connection()
    conditions = ["status = 'closed'"]
    params = []
    if relay_user:
        conditions.append("relay_user = ?"); params.append(relay_user)
    if relay_id:
        conditions.append("relay_id = ?"); params.append(relay_id)
    where = "WHERE " + " AND ".join(conditions)
    params = tuple(params)

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


def get_logs(relay_user: str = None, relay_id: str = None, limit: int = 50) -> list[dict]:
    init_db()
    conn = db.get_connection()
    where, params = [], []
    if relay_user:
        where.append("relay_user = ?"); params.append(relay_user)
    if relay_id:
        where.append("relay_id = ?"); params.append(relay_id)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    params.append(limit)
    rows = conn.execute(
        f"SELECT * FROM ai_gate_logs {clause} ORDER BY id DESC LIMIT ?", params
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats(relay_user: str = None, relay_id: str = None) -> dict:
    init_db()
    conn = db.get_connection()
    conditions, params = [], []
    if relay_user:
        conditions.append("relay_user = ?"); params.append(relay_user)
    if relay_id:
        conditions.append("relay_id = ?"); params.append(relay_id)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params = tuple(params)

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

async def _call_anthropic(user_msg: str, system_prompt: str,
                         model_id: str = None) -> tuple[str, str, int, str]:
    """Call AI API. Returns: (decision, reason, latency_ms, raw_response)"""
    mcfg = _get_model_config(model_id or DEFAULT_AI_MODEL)
    if not mcfg["api_key"]:
        return "ERROR", f"API key not set for {mcfg['model']}", 0, ""

    # MiniMax uses thinking tokens — needs higher max_tokens for reasoning
    max_tokens = 8000 if "minimax" in mcfg["model"].lower() else 150

    request_body = {
        "model": mcfg["model"],
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_msg}]
    }

    start = datetime.now(timezone.utc)

    try:
        async with httpx.AsyncClient(timeout=ANTHROPIC_TIMEOUT) as client:
            resp = await client.post(
                mcfg["base_url"],
                json=request_body,
                headers={
                    "x-api-key": mcfg["api_key"],
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
                elif "DISAGREE" in val or "REJECT" in val or "DENY" in val or "NO" == val:
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

    After processing the source bot, fans out to any enabled copy bots.
    """
    relay_user = payload.get("relay_user", "")
    relay_id = payload.get("relay_id", "")

    if not relay_user or not relay_id:
        logger.warning(f"Bar missing required fields: {payload.keys()}")
        return

    # Process the source bot (normal mode)
    await _process_bar_for_bot(payload, body_text)

    # Fan out to copy bots that mirror this relay_id
    copy_bots = _get_copy_bots_for_source(relay_id)
    for bot in copy_bots:
        # Build modified payload with copy bot's overrides
        bot_payload = {**payload}
        bot_payload["relay_id"] = bot["bot_id"]
        bot_payload["account"] = bot["account"]
        bot_payload["strategy_tag"] = bot["strategy_tag"]
        bot_body = json.dumps(bot_payload)

        logger.info(f"[{relay_user}/{bot['bot_id']}] Copy bot processing (source: {relay_id})")
        await _process_bar_for_bot(
            bot_payload, bot_body,
            entry_prompt=bot.get("entry_prompt"),
            manage_prompt=bot.get("manage_prompt"),
            ai_model=bot.get("ai_model")
        )


async def _process_bar_for_bot(payload: dict, body_text: str,
                                entry_prompt: str = None,
                                manage_prompt: str = None,
                                ai_model: str = None):
    """Process a single bar for one bot (source or copy)."""
    relay_user = payload.get("relay_user", "")
    relay_id = payload.get("relay_id", "")
    account = payload.get("account", "")
    instrument = payload.get("instrument", "")

    if not relay_user or not account or not instrument:
        logger.warning(f"Bar missing required fields: {payload.keys()}")
        return

    # Auto-register source bots that don't have a config yet
    bot = get_bot(relay_id)
    if not bot:
        add_bot(bot_id=relay_id, mode='normal', account=account,
                strategy_tag=payload.get('strategy_tag', relay_id),
                relay_id=relay_id)
        logger.info(f"[{relay_user}/{relay_id}] Auto-created bot config (mode=normal)")
        bot = get_bot(relay_id)

    # Resolve model: explicit param > bot config > default
    model = ai_model or (bot.get("ai_model") if bot else None) or DEFAULT_AI_MODEL

    # Serialize per bot+instrument to prevent race conditions
    lock_key = f"{relay_user}:{relay_id}:{account}:{instrument}"
    async with _get_lock(lock_key):
        position = _get_position(relay_user, account, instrument)

        # Enrich payload with server-computed indicators (fills missing fields)
        enriched_payload = indicator_engine.enrich_payload(payload, instrument)

        # Filter payload to only include bot's enabled indicators
        filtered_payload = _filter_payload_for_bot(enriched_payload, relay_id)

        if position is None:
            # === FLAT — check for entry signals ===
            # Support both formats:
            #   Pine Script: long_signal=true / short_signal=true
            #   TV Strategy: action=buy / action=sell
            long_sig = filtered_payload.get("long_signal", False)
            short_sig = filtered_payload.get("short_signal", False)

            # Translate action=buy/sell to signal flags
            action = str(filtered_payload.get("action", "")).lower()
            if not long_sig and not short_sig and action:
                if action == "buy":
                    long_sig = True
                    filtered_payload["long_signal"] = True
                elif action == "sell":
                    short_sig = True
                    filtered_payload["short_signal"] = True

            if long_sig or short_sig:
                direction = "long" if long_sig else "short"
                await _handle_entry(filtered_payload, body_text, direction,
                                    entry_prompt=entry_prompt, ai_model=model)
            # else: flat, no signal — nothing to do
        else:
            # === IN POSITION — manage ===
            await _handle_manage(filtered_payload, body_text, position,
                                 manage_prompt=manage_prompt, ai_model=model)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def _handle_entry(payload: dict, body_text: str, direction: str,
                        entry_prompt: str = None, ai_model: str = None):
    """Flat + signal detected. Ask AI whether to enter."""
    relay_user = payload["relay_user"]
    relay_id = payload["relay_id"]
    account = payload["account"]
    instrument = payload["instrument"]
    strategy = payload.get("strategy", "unknown")
    confluence = payload.get("bull_confluence" if direction == "long" else "bear_confluence", 0)

    logger.info(f"[{relay_user}/{relay_id}] Signal: {direction.upper()} {instrument} (confluence {confluence}) [model={ai_model}]")

    # Call AI for entry decision (use custom prompt if provided)
    user_msg = _build_entry_message(payload)
    prompt = entry_prompt or AI_ENTRY_PROMPT
    ai_decision, ai_reason, ai_latency_ms, ai_raw = await _call_anthropic(
        user_msg, prompt, model_id=ai_model
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

async def _handle_manage(payload: dict, body_text: str, position: dict,
                         manage_prompt: str = None, ai_model: str = None):
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
    prompt = manage_prompt or AI_MANAGE_PROMPT
    ai_decision, ai_reason, ai_latency_ms, ai_raw = await _call_anthropic(
        user_msg, prompt, model_id=ai_model
    )

    logger.info(
        f"[{relay_user}/{relay_id}] AI manage: {ai_decision} ({ai_latency_ms}ms) [{ai_model}] — {ai_reason}"
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
    _watchdog_iter = 0
    while True:
        await asyncio.sleep(60)
        _watchdog_iter += 1
        # Purge old bars every ~60 iterations (hourly)
        if _watchdog_iter % 60 == 0:
            try:
                purge_old_bars(14, 60)
                logger.info("Purged old bars (1m >14d, 5m >60d)")
            except Exception as e:
                logger.error(f"Bar purge error: {e}")
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