import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_relay.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            relay_user TEXT UNIQUE NOT NULL,
            crosstrade_key TEXT NOT NULL,
            ct_webhook_url TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            relay_user TEXT NOT NULL,
            account TEXT NOT NULL,
            instrument TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            direction TEXT NOT NULL,
            opened_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(relay_user, account, instrument)
        );

        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            relay_user TEXT,
            relay_id TEXT,
            account TEXT,
            instrument TEXT,
            action TEXT,
            market_position TEXT,
            prev_market_position TEXT,
            signal_type TEXT,
            result TEXT NOT NULL,
            details TEXT
        );
    """)

    conn.commit()
    conn.close()


# --- User operations ---

def get_user(relay_user: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE relay_user = ?", (relay_user,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_user(relay_user: str, crosstrade_key: str, ct_webhook_url: str):
    conn = get_connection()
    conn.execute("""
        INSERT INTO users (relay_user, crosstrade_key, ct_webhook_url)
        VALUES (?, ?, ?)
        ON CONFLICT(relay_user) DO UPDATE SET
            crosstrade_key = excluded.crosstrade_key,
            ct_webhook_url = excluded.ct_webhook_url
    """, (relay_user, crosstrade_key, ct_webhook_url))
    conn.commit()
    conn.close()


def delete_user(relay_user: str):
    conn = get_connection()
    conn.execute("DELETE FROM users WHERE relay_user = ?", (relay_user,))
    conn.commit()
    conn.close()


def list_users() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM users").fetchall()
    conn.close()
    return [dict(row) for row in rows]


# --- Position operations ---

def get_position(relay_user: str, account: str, instrument: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM positions WHERE relay_user = ? AND account = ? AND instrument = ?",
        (relay_user, account, instrument)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def set_position(relay_user: str, account: str, instrument: str, owner_id: str, direction: str):
    conn = get_connection()
    conn.execute("""
        INSERT INTO positions (relay_user, account, instrument, owner_id, direction, opened_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(relay_user, account, instrument) DO UPDATE SET
            owner_id = excluded.owner_id,
            direction = excluded.direction,
            opened_at = excluded.opened_at
    """, (relay_user, account, instrument, owner_id, direction,
          datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()


def clear_position(relay_user: str, account: str, instrument: str):
    conn = get_connection()
    conn.execute(
        "DELETE FROM positions WHERE relay_user = ? AND account = ? AND instrument = ?",
        (relay_user, account, instrument)
    )
    conn.commit()
    conn.close()


def list_positions(relay_user: str = None) -> list[dict]:
    conn = get_connection()
    if relay_user:
        rows = conn.execute(
            "SELECT * FROM positions WHERE relay_user = ?", (relay_user,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM positions").fetchall()
    conn.close()
    return [dict(row) for row in rows]


# --- Log operations ---

def add_log(relay_user: str, relay_id: str, account: str, instrument: str,
            action: str, market_position: str, prev_market_position: str,
            signal_type: str, result: str, details: str = None):
    conn = get_connection()
    conn.execute("""
        INSERT INTO logs (timestamp, relay_user, relay_id, account, instrument,
                         action, market_position, prev_market_position,
                         signal_type, result, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now(timezone.utc).isoformat(), relay_user, relay_id,
          account, instrument, action, market_position, prev_market_position,
          signal_type, result, details))
    conn.commit()
    conn.close()


def get_logs(relay_user: str = None, limit: int = 100) -> list[dict]:
    conn = get_connection()
    if relay_user:
        rows = conn.execute(
            "SELECT * FROM logs WHERE relay_user = ? ORDER BY id DESC LIMIT ?",
            (relay_user, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
