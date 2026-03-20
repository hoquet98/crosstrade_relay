import logging
import os
import json
import re
import sqlite3

import httpx

from datetime import datetime, timezone
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager

import database as db

# Grace period (seconds) -- if a position was opened within this window,
# skip the CT API check and trust local ownership (prevents race conditions
# when two strategies fire simultaneously).
OWNERSHIP_GRACE_PERIOD = 30

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trade_relay.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trade_relay")

# --- App lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    ai_gate_init_db()
    logger.info(f"Trade Relay started -- database initialized (AI Gate: {'DRY RUN' if AI_DRY_RUN else 'LIVE'})")
    yield
    logger.info("Trade Relay shutting down")

app = FastAPI(title="Trade Relay", lifespan=lifespan)

# ══════════════════════════════════════════════════════════════════════════════
# PAYLOAD PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_payload(body: str) -> dict:
    """Parse semicolon-delimited key=value webhook payload into a dict."""
    fields = {}
    for part in body.replace("\n", ";").split(";"):
        part = part.strip()
        if not part or part.startswith("//"):
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key.strip().lower()] = value.strip()
    return fields

def build_payload(fields: dict, crosstrade_key: str) -> str:
    """Rebuild the payload string for CrossTrade, injecting the key
    and removing relay-specific fields."""
    exclude = {"relay_user", "relay_id"}
    parts = [f"key={crosstrade_key}"]
    for k, v in fields.items():
        if k in exclude or k == "key":
            continue
        parts.append(f"{k}={v}")
    return "\n".join(f"{p};" for p in parts)

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_signal(fields: dict) -> str:
    """Determine signal type from market_position fields.
    Returns: 'entry', 'exit', 'reversal', or 'unknown'
    """
    mp = fields.get("market_position", "").lower()
    pmp = fields.get("prev_market_position", "").lower()

    if not mp or not pmp:
        return "unknown"

    if pmp == "flat" and mp in ("long", "short"):
        return "entry"
    elif pmp in ("long", "short") and mp == "flat":
        return "exit"
    elif pmp in ("long", "short") and mp in ("long", "short") and pmp != mp:
        return "reversal"
    else:
        return "unknown"

def get_direction(fields: dict) -> str:
    """Get the position direction from market_position."""
    mp = fields.get("market_position", "").lower()
    if mp in ("long", "short"):
        return mp
    return "unknown"

# ══════════════════════════════════════════════════════════════════════════════
# CROSSTRADE API POSITION CHECK
# ══════════════════════════════════════════════════════════════════════════════

CT_API_BASE = "https://app.crosstrade.io/v1/api"

async def check_ct_position(api_key: str, account: str, instrument: str) -> str | None:
    """Check the actual position state in NT8 via CrossTrade API.

    Returns: 'long', 'short', 'flat', or None if the API call fails.

    The instrument from TradingView (e.g., RTYH2026) may differ from NT8 format,
    so we check all positions for the account and match loosely.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{CT_API_BASE}/accounts/{account}/positions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
            )

        if response.status_code != 200:
            logger.warning(f"CT API returned {response.status_code}: {response.text[:200]}")
            return None

        data = response.json()

        # API returns a list of positions
        positions = data if isinstance(data, list) else data.get("positions", [])

        if not positions:
            return "flat"

        # Look for a position matching our instrument
        # NT8 instrument format may differ (e.g., "RTY MAR26" vs "RTYH2026")
        # so we check the root symbol
        instrument_root = ""
        for ch in instrument:
            if ch.isalpha():
                instrument_root += ch
            else:
                break
        instrument_root = instrument_root.upper()

        for pos in positions:
            pos_instrument = pos.get("instrument", "")
            pos_root = ""
            for ch in pos_instrument:
                if ch.isalpha():
                    pos_root += ch
                else:
                    break
            pos_root = pos_root.upper()

            if pos_root == instrument_root:
                mp = pos.get("marketPosition", "").lower()
                if mp in ("long", "short"):
                    return mp
                return "flat"

        # No matching instrument found -- account is flat for this instrument
        return "flat"

    except Exception as e:
        logger.warning(f"CT API check failed: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# CORE RELAY LOGIC
# ══════════════════════════════════════════════════════════════════════════════

async def process_signal(fields: dict, raw_payload: str = None) -> dict:
    """Process an incoming signal and decide whether to forward, block, or drop.

    Returns dict with 'result' (forwarded/blocked/dropped/error) and 'details'.
    """
    relay_user = fields.get("relay_user")
    relay_id = fields.get("relay_id")
    account = fields.get("account", "")
    instrument = fields.get("instrument", "")
    action = fields.get("action", "")
    market_position = fields.get("market_position", "")
    prev_market_position = fields.get("prev_market_position", "")

    # Validate required relay fields
    if not relay_user:
        return {"result": "error", "details": "Missing relay_user field"}
    if not relay_id:
        return {"result": "error", "details": "Missing relay_id field"}

    # Look up user config
    user = db.get_user(relay_user)
    if not user:
        return {"result": "error", "details": f"Unknown relay_user: {relay_user}"}

    signal_type = classify_signal(fields)
    position = db.get_position(relay_user, account, instrument)

    result = None
    details = None

    if signal_type == "entry":
        if position is None:
            # No position -- this strategy wins, forward it
            result = "forwarded"
            details = f"Entry {get_direction(fields)} -- {relay_id} takes ownership"
            db.set_position(relay_user, account, instrument, relay_id, get_direction(fields))

        elif position["owner_id"] == relay_id:
            # Same strategy re-entering? Forward it, update direction
            result = "forwarded"
            details = f"Re-entry {get_direction(fields)} -- {relay_id} already owns position"
            db.set_position(relay_user, account, instrument, relay_id, get_direction(fields))

        else:
            # Different strategy trying to enter -- check if within grace period
            opened_at = datetime.fromisoformat(position["opened_at"])
            age_seconds = (datetime.now(timezone.utc) - opened_at).total_seconds()

            if age_seconds < OWNERSHIP_GRACE_PERIOD:
                # Position was taken very recently -- trust local DB, block without CT API
                result = "blocked"
                details = (f"Entry blocked -- position owned by {position['owner_id']} "
                           f"({age_seconds:.0f}s ago, within {OWNERSHIP_GRACE_PERIOD}s grace period)")
                logger.info(f"[{relay_user}/{relay_id}] Blocked within grace period -- "
                            f"{position['owner_id']} took ownership {age_seconds:.0f}s ago")
            else:
                # Position is old enough -- check CT API for stale ownership
                ct_state = await check_ct_position(user["crosstrade_key"], account, instrument)

                if ct_state == "flat":
                    # Position was closed externally -- clear stale ownership, let this entry through
                    result = "forwarded"
                    details = (f"Entry {get_direction(fields)} -- stale ownership cleared "
                               f"(CT API shows flat, was owned by {position['owner_id']}), "
                               f"{relay_id} takes ownership")
                    db.set_position(relay_user, account, instrument, relay_id, get_direction(fields))
                    logger.info(f"[{relay_user}] Stale position detected via CT API -- "
                                f"cleared {position['owner_id']}, {relay_id} entering")
                elif ct_state is None:
                    # API call failed -- block to be safe
                    result = "blocked"
                    details = (f"Entry blocked -- position owned by {position['owner_id']} "
                               f"(CT API check failed, blocking conservatively)")
                else:
                    # Position is still open -- block
                    result = "blocked"
                    details = (f"Entry blocked -- position owned by {position['owner_id']} "
                               f"(confirmed {ct_state} via CT API)")

    elif signal_type == "exit":
        if position is None:
            # No tracked position -- forward anyway (might be manual or out of sync)
            result = "forwarded"
            details = f"Exit with no tracked position -- forwarding to be safe"

        elif position["owner_id"] == relay_id:
            # Owner is closing -- forward and clear ownership
            result = "forwarded"
            details = f"Exit -- {relay_id} closing position"
            db.clear_position(relay_user, account, instrument)

        else:
            # Phantom exit from blocked strategy -- drop it
            result = "dropped"
            details = f"Phantom exit dropped -- position owned by {position['owner_id']}, not {relay_id}"

    elif signal_type == "reversal":
        if position is None:
            # No tracked position -- forward, set new owner
            result = "forwarded"
            details = f"Reversal to {get_direction(fields)} -- {relay_id} takes ownership"
            db.set_position(relay_user, account, instrument, relay_id, get_direction(fields))

        elif position["owner_id"] == relay_id:
            # Owner reversing -- forward and update direction
            result = "forwarded"
            details = f"Reversal to {get_direction(fields)} -- {relay_id} reversing"
            db.set_position(relay_user, account, instrument, relay_id, get_direction(fields))

        else:
            # Different strategy trying to reverse -- check if within grace period
            opened_at = datetime.fromisoformat(position["opened_at"])
            age_seconds = (datetime.now(timezone.utc) - opened_at).total_seconds()

            if age_seconds < OWNERSHIP_GRACE_PERIOD:
                result = "blocked"
                details = (f"Reversal blocked -- position owned by {position['owner_id']} "
                           f"({age_seconds:.0f}s ago, within {OWNERSHIP_GRACE_PERIOD}s grace period)")
                logger.info(f"[{relay_user}/{relay_id}] Reversal blocked within grace period -- "
                            f"{position['owner_id']} took ownership {age_seconds:.0f}s ago")
            else:
                # Position is old enough -- check CT API for stale ownership
                ct_state = await check_ct_position(user["crosstrade_key"], account, instrument)

                if ct_state == "flat":
                    result = "forwarded"
                    details = (f"Reversal to {get_direction(fields)} -- stale ownership cleared "
                               f"(CT API shows flat), {relay_id} takes ownership")
                    db.set_position(relay_user, account, instrument, relay_id, get_direction(fields))
                elif ct_state is None:
                    result = "blocked"
                    details = (f"Reversal blocked -- position owned by {position['owner_id']} "
                               f"(CT API check failed, blocking conservatively)")
                else:
                    result = "blocked"
                    details = (f"Reversal blocked -- position owned by {position['owner_id']} "
                               f"(confirmed {ct_state} via CT API)")
    else:
        # Unknown signal type -- no sync fields, forward as-is (no position tracking)
        result = "forwarded"
        details = "No market_position fields -- forwarding without relay logic"

    # Forward to CrossTrade if not blocked/dropped
    forwarded_payload = None
    if result == "forwarded":
        forwarded_payload = build_payload(fields, user["crosstrade_key"])
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    user["ct_webhook_url"],
                    content=forwarded_payload,
                    headers={"Content-Type": "text/plain"}
                )
            logger.info(f"[{relay_user}/{relay_id}] {signal_type.upper()} -> CrossTrade {response.status_code}")

            # Log the signal with both payloads
            db.add_log(
                relay_user=relay_user, relay_id=relay_id,
                account=account, instrument=instrument,
                action=action, market_position=market_position,
                prev_market_position=prev_market_position,
                signal_type=signal_type, result=result, details=details,
                raw_payload=raw_payload, forwarded_payload=forwarded_payload
            )
            return {"result": result, "details": details, "ct_status": response.status_code}

        except Exception as e:
            error_msg = f"CrossTrade forward failed: {str(e)}"
            logger.error(f"[{relay_user}/{relay_id}] {error_msg}")
            db.add_log(
                relay_user=relay_user, relay_id=relay_id,
                account=account, instrument=instrument,
                action=action, market_position=market_position,
                prev_market_position=prev_market_position,
                signal_type=signal_type, result="error", details=error_msg,
                raw_payload=raw_payload, forwarded_payload=forwarded_payload
            )
            return {"result": "error", "details": error_msg}

    else:
        logger.info(f"[{relay_user}/{relay_id}] {signal_type.upper()} -> {result.upper()}: {details}")
        db.add_log(
            relay_user=relay_user, relay_id=relay_id,
            account=account, instrument=instrument,
            action=action, market_position=market_position,
            prev_market_position=prev_market_position,
            signal_type=signal_type, result=result, details=details,
            raw_payload=raw_payload
        )
        return {"result": result, "details": details}

# ══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════

bearer_scheme = HTTPBearer()

async def verify_bearer(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    """Validate Bearer token against registered users' CrossTrade keys."""
    token = credentials.credentials
    user = db.get_user_by_key(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES — EXISTING RELAY
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/webhook")
async def webhook(request: Request):
    """Main webhook endpoint -- receives TradingView alerts."""
    body = await request.body()
    body_text = body.decode("utf-8").strip()

    if not body_text:
        raise HTTPException(status_code=400, detail="Empty payload")

    fields = parse_payload(body_text)
    result = await process_signal(fields, raw_payload=body_text)
    return JSONResponse(content=result)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ai_gate": "dry_run" if AI_DRY_RUN else "live",
        "ai_model": ANTHROPIC_MODEL,
        "api_key_set": bool(ANTHROPIC_API_KEY)
    }

@app.get("/positions")
async def positions(relay_user: str = None, _user: dict = Depends(verify_bearer)):
    """View current position ownership."""
    return db.list_positions(relay_user)

@app.get("/logs")
async def logs(relay_user: str = None, limit: int = 100, _user: dict = Depends(verify_bearer)):
    """View signal logs."""
    return db.get_logs(relay_user, limit)

@app.get("/users")
async def users(_user: dict = Depends(verify_bearer)):
    """List configured users (keys are masked)."""
    all_users = db.list_users()
    for u in all_users:
        u["crosstrade_key"] = u["crosstrade_key"][:8] + "..."
    return all_users

# --- User management routes ---

@app.post("/users")
async def create_user(request: Request, _user: dict = Depends(verify_bearer)):
    """Add or update a user."""
    data = await request.json()
    required = ["relay_user", "crosstrade_key", "ct_webhook_url"]
    for field in required:
        if field not in data:
            raise HTTPException(status_code=400, detail=f"Missing field: {field}")

    db.upsert_user(data["relay_user"], data["crosstrade_key"], data["ct_webhook_url"])
    logger.info(f"User created/updated: {data['relay_user']}")
    return {"status": "ok", "relay_user": data["relay_user"]}

@app.delete("/users/{relay_user}")
async def remove_user(relay_user: str, _user: dict = Depends(verify_bearer)):
    """Remove a user."""
    db.delete_user(relay_user)
    logger.info(f"User deleted: {relay_user}")
    return {"status": "ok", "relay_user": relay_user}

@app.post("/positions/clear")
async def clear_position(request: Request, _user: dict = Depends(verify_bearer)):
    """Manually clear position ownership (emergency reset)."""
    data = await request.json()
    required = ["relay_user", "account", "instrument"]
    for field in required:
        if field not in data:
            raise HTTPException(status_code=400, detail=f"Missing field: {field}")

    db.clear_position(data["relay_user"], data["account"], data["instrument"])
    logger.info(f"Position cleared: {data['relay_user']}/{data['account']}/{data['instrument']}")
    return {"status": "ok", "details": "Position ownership cleared"}

# ══════════════════════════════════════════════════════════════════════════════
# NINJATRADER DB QUERY ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

NT_DB_DIR = r"C:\Users\Administrator\Documents\NinjaTrader 8\db"
NT_DB_DEFAULT = "NinjaTrader.sqlite"

# SQL keywords that modify data -- block these
_WRITE_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE",
    "ATTACH", "DETACH", "REINDEX", "VACUUM", "PRAGMA",
}

def _is_read_only(sql: str) -> bool:
    """Check that a SQL statement is read-only (SELECT/WITH only)."""
    stripped = sql.strip().rstrip(";").strip()
    # Remove leading comments
    while stripped.startswith("--"):
        stripped = stripped.split("\n", 1)[-1].strip()

    # Must start with SELECT or WITH
    first_word = stripped.split()[0].upper() if stripped.split() else ""
    if first_word not in ("SELECT", "WITH"):
        return False

    # Scan for write keywords
    upper = stripped.upper()
    for kw in _WRITE_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return False

    return True

async def verify_nt_access(request: Request) -> dict:
    """Dual auth: Bearer token (CT key) + X-NT-Token header."""
    # Check Bearer token first
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    ct_key = auth_header[7:]
    user = db.get_user_by_key(ct_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Check NT-specific token
    nt_token = request.headers.get("X-NT-Token", "")
    if not nt_token:
        raise HTTPException(status_code=401, detail="Missing X-NT-Token header")

    if not user.get("nt_query_token"):
        raise HTTPException(status_code=403, detail="NT query access not configured for this user")

    if nt_token != user["nt_query_token"]:
        raise HTTPException(status_code=401, detail="Invalid NT token")

    return user

@app.post("/nt/query")
async def nt_query(request: Request):
    """Execute a read-only SQL query against the NinjaTrader SQLite database.

    Requires dual auth: Bearer token + X-NT-Token header.
    Body: {"sql": "SELECT ...", "limit": 1000}
    """
    user = await verify_nt_access(request)
    data = await request.json()
    sql = data.get("sql", "").strip()

    if not sql:
        raise HTTPException(status_code=400, detail="Missing 'sql' field")

    if not _is_read_only(sql):
        raise HTTPException(status_code=403, detail="Only SELECT queries are allowed")

    limit = min(data.get("limit", 1000), 5000)

    # Resolve database path — accepts relative path within NT_DB_DIR
    db_name = data.get("db", NT_DB_DEFAULT)

    # Prevent path traversal
    if ".." in db_name or db_name.startswith("/") or db_name.startswith("\\") or ":" in db_name:
        raise HTTPException(status_code=400, detail="Invalid db path")

    db_path = os.path.join(NT_DB_DIR, db_name)

    if not os.path.exists(db_path):
        raise HTTPException(status_code=503, detail=f"Database not found: {db_name}")

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(limit)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        result = [dict(row) for row in rows]
        conn.close()

        logger.info(f"[{user['relay_user']}] NT query: {sql[:100]}... -> {len(result)} rows")
        return {"columns": columns, "rows": result, "count": len(result)}

    except sqlite3.Error as e:
        raise HTTPException(status_code=400, detail=f"SQL error: {str(e)}")

@app.get("/nt/tables")
async def nt_tables(request: Request, db_param: str = None):
    """List all tables in the NinjaTrader database. Requires dual auth."""
    user = await verify_nt_access(request)

    db_name = db_param or NT_DB_DEFAULT
    if ".." in db_name or db_name.startswith("/") or db_name.startswith("\\") or ":" in db_name:
        raise HTTPException(status_code=400, detail="Invalid db path")

    db_path = os.path.join(NT_DB_DIR, db_name)

    if not os.path.exists(db_path):
        raise HTTPException(status_code=503, detail=f"Database not found: {db_name}")

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    except sqlite3.Error as e:
        raise HTTPException(status_code=400, detail=f"SQL error: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════════
# AI GATE — LLM CONVICTION FILTER
# ══════════════════════════════════════════════════════════════════════════════

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_TIMEOUT = 15.0  # seconds

# Dry run mode: AI evaluates and logs, but never forwards to CrossTrade.
# Set to False (or set env AI_DRY_RUN=0) when ready to go live.
AI_DRY_RUN = os.environ.get("AI_DRY_RUN", "1").lower() not in ("0", "false", "no")

# Fields the Pine Script JSON must include for CrossTrade routing
REQUIRED_ROUTING_FIELDS = {"relay_user", "relay_id", "account", "instrument", "signal", "qty"}

# --- AI Gate database table ---

_ai_migrated = False

def ai_gate_init_db():
    """Create AI gate tables if they don't exist (idempotent)."""
    global _ai_migrated
    if _ai_migrated:
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
            alert_type      TEXT DEFAULT 'entry',
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
            bars_managed    INTEGER DEFAULT 0,
            last_update     TEXT,
            UNIQUE(relay_user, account, instrument)
        )
    """)
    # Migrate: add alert_type if missing
    cols = [r[1] for r in conn.execute("PRAGMA table_info(ai_gate_logs)").fetchall()]
    if "alert_type" not in cols:
        conn.execute("ALTER TABLE ai_gate_logs ADD COLUMN alert_type TEXT DEFAULT 'entry'")
    conn.commit()
    conn.close()
    _ai_migrated = True


def ai_log_decision(
    relay_user: str, relay_id: str, account: str, instrument: str,
    signal: str, strategy: str, confluence_score: int,
    ai_decision: str, ai_reason: str, ai_latency_ms: int,
    relay_result: str = None, relay_details: str = None,
    payload_json: str = None, ai_raw_response: str = None,
    alert_type: str = "entry"
):
    ai_gate_init_db()
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


# --- AI Position Tracking ---

def ai_set_position(relay_user: str, relay_id: str, account: str,
                    instrument: str, direction: str, entry_price: float,
                    tick_value: float, strategy: str):
    ai_gate_init_db()
    conn = db.get_connection()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO ai_positions
            (relay_user, relay_id, account, instrument, direction,
             entry_price, tick_value, strategy, opened_at, bars_managed, last_update)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        ON CONFLICT(relay_user, account, instrument) DO UPDATE SET
            relay_id = excluded.relay_id,
            direction = excluded.direction,
            entry_price = excluded.entry_price,
            tick_value = excluded.tick_value,
            strategy = excluded.strategy,
            opened_at = excluded.opened_at,
            bars_managed = 0,
            last_update = excluded.last_update
    """, (relay_user, relay_id, account, instrument, direction,
          entry_price, tick_value, strategy, now, now))
    conn.commit()
    conn.close()


def ai_get_position(relay_user: str, account: str, instrument: str) -> dict | None:
    ai_gate_init_db()
    conn = db.get_connection()
    row = conn.execute(
        "SELECT * FROM ai_positions WHERE relay_user = ? AND account = ? AND instrument = ?",
        (relay_user, account, instrument)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def ai_clear_position(relay_user: str, account: str, instrument: str):
    ai_gate_init_db()
    conn = db.get_connection()
    conn.execute(
        "DELETE FROM ai_positions WHERE relay_user = ? AND account = ? AND instrument = ?",
        (relay_user, account, instrument)
    )
    conn.commit()
    conn.close()


def ai_increment_bars(relay_user: str, account: str, instrument: str):
    ai_gate_init_db()
    conn = db.get_connection()
    conn.execute("""
        UPDATE ai_positions SET bars_managed = bars_managed + 1, last_update = ?
        WHERE relay_user = ? AND account = ? AND instrument = ?
    """, (datetime.now(timezone.utc).isoformat(), relay_user, account, instrument))
    conn.commit()
    conn.close()


def ai_list_positions(relay_user: str = None) -> list[dict]:
    ai_gate_init_db()
    conn = db.get_connection()
    if relay_user:
        rows = conn.execute(
            "SELECT * FROM ai_positions WHERE relay_user = ?", (relay_user,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM ai_positions").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def ai_get_logs(relay_user: str = None, limit: int = 50) -> list[dict]:
    ai_gate_init_db()
    conn = db.get_connection()
    if relay_user:
        rows = conn.execute(
            "SELECT * FROM ai_gate_logs WHERE relay_user = ? ORDER BY id DESC LIMIT ?",
            (relay_user, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM ai_gate_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Anthropic API ---

AI_SYSTEM_PROMPT = """You are a trade conviction filter for algorithmic futures trading. You receive a JSON payload of technical indicator values computed at the moment a trading strategy fires a signal. Your job is to decide whether the signal has enough edge to execute.

You are NOT predicting whether price will go up or down. You are evaluating whether the CURRENT INDICATOR CONTEXT supports this specific strategy's signal type and trade profile.

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


AI_MANAGE_PROMPT = """You are a trade exit manager for algorithmic futures trading. You receive a JSON payload every bar while a trade is open. Your job is to decide whether to EXIT the trade now or HOLD.

The payload includes the current indicator state, position P&L, and a pre-computed exit_score based on these conditions:

## Exit Score Components
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

## Safety Nets (handled server-side, but note them)
- Hard stop: hit max adverse ticks → instant exit (server handles this)
- Time stop: max bars exceeded → exit (server handles this)
- Early exit: losing 15+ ticks AND score >= 3 → exit

## Your Decision
Use the exit_score AND the indicator context together. The exit_score gives you a structured read, but you can override:
- HOLD even with a high score if indicators show the adverse move is exhausting (e.g., oversold bounce forming in your favor)
- EXIT even with a low score if you see a clear structural breakdown that the score doesn't capture

## Response Format
Respond with EXACTLY this format — no extra text, no markdown:

DECISION: EXIT
REASON: [one concise sentence]

or

DECISION: HOLD
REASON: [one concise sentence]"""


def _build_ai_user_message(payload: dict) -> str:
    """Build the user message for the AI from the indicator payload."""
    routing_keys = {"relay_user", "relay_id", "account", "qty",
                    "order_type", "tif", "out_of_sync", "sync_strategy",
                    "strategy_tag", "alert_type"}
    indicator_data = {k: v for k, v in payload.items() if k not in routing_keys}
    return f"Signal payload:\n{json.dumps(indicator_data, indent=2)}"


async def call_anthropic(payload: dict, system_prompt: str = None) -> tuple[str, str, int, str]:
    """Call Anthropic API with indicator payload.

    Returns: (decision, reason, latency_ms, raw_response)
    """
    if not ANTHROPIC_API_KEY:
        return "ERROR", "ANTHROPIC_API_KEY not set", 0, ""

    user_msg = _build_ai_user_message(payload)

    request_body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 150,
        "system": system_prompt or AI_SYSTEM_PROMPT,
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

        # Parse decision
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


def build_ct_fields(payload: dict) -> dict:
    """Convert JSON signal payload into the field dict that process_signal() expects."""
    signal = payload.get("signal", "").upper()

    if signal == "LONG":
        action = "BUY"
        market_position = "long"
        prev_market_position = "flat"
    elif signal == "SHORT":
        action = "SELL"
        market_position = "short"
        prev_market_position = "flat"
    else:
        action = signal
        market_position = signal.lower()
        prev_market_position = "flat"

    return {
        "relay_user": payload.get("relay_user", ""),
        "relay_id": payload.get("relay_id", ""),
        "command": payload.get("command", "PLACE"),
        "account": payload.get("account", ""),
        "instrument": payload.get("instrument", ""),
        "action": action,
        "qty": str(payload.get("qty", 1)),
        "order_type": payload.get("order_type", "MARKET"),
        "tif": payload.get("tif", "DAY"),
        "sync_strategy": payload.get("sync_strategy", "true"),
        "market_position": market_position,
        "prev_market_position": prev_market_position,
        "out_of_sync": payload.get("out_of_sync", "flatten"),
        "strategy_tag": payload.get("strategy_tag", payload.get("relay_id", "")),
    }


# --- AI Gate Routes ---

@app.post("/webhook/ai")
async def webhook_ai(request: Request):
    """AI Gate webhook — receives JSON from Pine Script, gates through LLM.

    Flow:
        1. Parse JSON payload from TradingView alert
        2. Validate required routing fields
        3. Call Anthropic API for conviction decision
        4. If AGREE → build CT fields → process_signal() → CrossTrade
        5. If DISAGREE/ERROR/TIMEOUT → log and reject
    """
    ai_gate_init_db()

    # --- Parse body ---
    body = await request.body()
    body_text = body.decode("utf-8").strip()

    if not body_text:
        raise HTTPException(status_code=400, detail="Empty payload")

    try:
        payload = json.loads(body_text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # --- Validate routing fields ---
    # Route manage alerts to the manage handler (both come via same TradingView webhook)
    alert_type = payload.get("alert_type", "entry")
    if alert_type == "manage":
        return await _handle_manage(payload, body_text)

    missing = REQUIRED_ROUTING_FIELDS - set(payload.keys())
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {', '.join(missing)}")

    relay_user = payload["relay_user"]
    relay_id = payload["relay_id"]
    account = payload["account"]
    instrument = payload["instrument"]
    signal = payload["signal"]
    strategy = payload.get("strategy", "unknown")
    confluence = payload.get("confluence_score", 0)

    # --- Verify user exists ---
    user = db.get_user(relay_user)
    if not user:
        raise HTTPException(status_code=400, detail=f"Unknown relay_user: {relay_user}")

    logger.info(f"[{relay_user}/{relay_id}] AI Gate received {signal} signal for {instrument}")

    # --- Call Anthropic ---
    ai_decision, ai_reason, ai_latency_ms, ai_raw = await call_anthropic(payload)

    logger.info(
        f"[{relay_user}/{relay_id}] AI decision: {ai_decision} "
        f"({ai_latency_ms}ms) — {ai_reason}"
    )

    # --- Gate logic ---
    relay_result = None
    relay_details = None

    if ai_decision == "AGREE":
        # Track AI position for manage alerts
        entry_price = payload.get("price", 0)
        tick_value = payload.get("tick_value", 10.0)
        ai_set_position(
            relay_user, relay_id, account, instrument,
            signal.lower(), entry_price, tick_value, strategy
        )

        if AI_DRY_RUN:
            relay_result = "dry_run"
            relay_details = f"AI agreed but DRY RUN — would have forwarded {signal} to CrossTrade"
            logger.info(
                f"[{relay_user}/{relay_id}] AI AGREED (DRY RUN) — not forwarding to CrossTrade"
            )
        else:
            ct_fields = build_ct_fields(payload)
            relay_response = await process_signal(ct_fields, raw_payload=body_text)
            relay_result = relay_response.get("result", "unknown")
            relay_details = relay_response.get("details", "")
            logger.info(
                f"[{relay_user}/{relay_id}] AI AGREED → relay: {relay_result} — {relay_details}"
            )

    elif ai_decision == "DISAGREE":
        relay_result = "ai_rejected"
        relay_details = f"AI disagreed: {ai_reason}"
        logger.info(f"[{relay_user}/{relay_id}] AI DISAGREED — signal rejected")

    elif ai_decision == "TIMEOUT":
        relay_result = "ai_timeout"
        relay_details = f"AI timed out after {ai_latency_ms}ms — signal rejected"
        logger.warning(f"[{relay_user}/{relay_id}] AI TIMEOUT — signal rejected")

    else:
        relay_result = "ai_error"
        relay_details = f"AI error: {ai_reason}"
        logger.error(f"[{relay_user}/{relay_id}] AI ERROR — signal rejected: {ai_reason}")

    # --- Log ---
    ai_log_decision(
        relay_user=relay_user,
        relay_id=relay_id,
        account=account,
        instrument=instrument,
        signal=signal,
        strategy=strategy,
        confluence_score=confluence,
        ai_decision=ai_decision,
        ai_reason=ai_reason,
        ai_latency_ms=ai_latency_ms,
        relay_result=relay_result,
        relay_details=relay_details,
        payload_json=body_text,
        ai_raw_response=ai_raw,
        alert_type="entry"
    )

    return JSONResponse(content={
        "ai_decision": ai_decision,
        "ai_reason": ai_reason,
        "ai_latency_ms": ai_latency_ms,
        "relay_result": relay_result,
        "relay_details": relay_details,
        "dry_run": AI_DRY_RUN
    })


# ══════════════════════════════════════════════════════════════════════════════
# AI GATE — MANAGE ENDPOINT (per-bar exit management)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/webhook/ai/manage")
async def webhook_ai_manage(request: Request):
    """AI Manage webhook — direct endpoint (also called internally from /webhook/ai)."""
    body = await request.body()
    body_text = body.decode("utf-8").strip()
    if not body_text:
        raise HTTPException(status_code=400, detail="Empty payload")
    try:
        payload = json.loads(body_text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    return await _handle_manage(payload, body_text)


async def _handle_manage(payload: dict, body_text: str):
    """Internal manage handler — per-bar exit management.

    Flow:
        1. Parse JSON payload (alert_type == "manage")
        2. Check if an AI-tracked position exists
        3. Check server-side safety nets (hard stop, time stop, early exit)
        4. If safety net triggers → EXIT immediately, no AI call
        5. Otherwise → call Anthropic with manage prompt
        6. If EXIT → send flatten to CrossTrade (or log in dry run)
        7. If HOLD → log and continue
    """
    ai_gate_init_db()

    relay_user = payload.get("relay_user", "")
    relay_id = payload.get("relay_id", "")
    account = payload.get("account", "")
    instrument = payload.get("instrument", "")
    strategy = payload.get("strategy", "unknown")

    if not relay_user or not account or not instrument:
        raise HTTPException(status_code=400, detail="Missing relay_user, account, or instrument")

    # --- Check for tracked AI position ---
    ai_pos = ai_get_position(relay_user, account, instrument)
    if not ai_pos:
        # No AI position — Pine is tracking locally but AI never agreed to enter.
        # Silently ignore.
        return JSONResponse(content={
            "ai_decision": "IGNORED",
            "ai_reason": "No AI-tracked position exists",
            "relay_result": "no_position"
        })

    # Use server-tracked entry price as source of truth, fall back to payload
    entry_price = ai_pos.get("entry_price", 0)
    tick_value = ai_pos.get("tick_value", 10.0)
    direction = ai_pos.get("direction", "")
    bars_managed = ai_pos.get("bars_managed", 0)

    # Get current price and P&L from payload
    current_price = payload.get("price", 0)
    unrealized_ticks = payload.get("unrealized_ticks", 0)
    unrealized_dollars = payload.get("unrealized_dollars", 0)
    exit_score = payload.get("exit_score", 0)
    hard_stop_ticks = payload.get("hard_stop_ticks", 25)
    max_bars_hold = payload.get("max_bars_hold", 60)
    bars_in_trade = payload.get("bars_in_trade", bars_managed)

    logger.info(
        f"[{relay_user}/{relay_id}] MANAGE bar {bars_in_trade}: "
        f"{direction} @ {entry_price} → P&L {unrealized_ticks:.1f} ticks "
        f"(${unrealized_dollars:.2f}), exit_score={exit_score}"
    )

    ai_increment_bars(relay_user, account, instrument)

    # --- Server-side safety nets (no AI call needed) ---
    safety_exit = False
    safety_reason = ""

    # Hard stop
    if unrealized_ticks <= -abs(hard_stop_ticks):
        safety_exit = True
        safety_reason = f"HARD STOP hit: {unrealized_ticks:.1f} ticks (limit: -{hard_stop_ticks})"

    # Time stop
    elif bars_in_trade >= max_bars_hold:
        safety_exit = True
        safety_reason = f"TIME STOP: {bars_in_trade} bars (limit: {max_bars_hold})"

    # Early exit: losing 15+ ticks AND score >= 3
    elif unrealized_ticks <= -15 and exit_score >= 3:
        safety_exit = True
        safety_reason = f"EARLY EXIT: losing {unrealized_ticks:.1f} ticks with exit_score={exit_score}"

    if safety_exit:
        logger.info(f"[{relay_user}/{relay_id}] SAFETY NET → {safety_reason}")

        relay_result = "safety_exit"
        if not AI_DRY_RUN:
            # Send flatten to CrossTrade
            user = db.get_user(relay_user)
            if user:
                flatten_fields = {
                    "relay_user": relay_user,
                    "relay_id": relay_id,
                    "command": "PLACE",
                    "account": account,
                    "instrument": instrument,
                    "action": "SELL" if direction == "long" else "BUY",
                    "qty": str(payload.get("qty", 1)),
                    "order_type": "MARKET",
                    "tif": "DAY",
                    "market_position": "flat",
                    "prev_market_position": direction,
                    "sync_strategy": "true",
                    "out_of_sync": "flatten",
                    "strategy_tag": payload.get("strategy_tag", relay_id),
                }
                await process_signal(flatten_fields, raw_payload=body_text)
            relay_result = "safety_exit_forwarded"
        else:
            relay_result = "safety_exit_dry_run"

        ai_clear_position(relay_user, account, instrument)

        ai_log_decision(
            relay_user=relay_user, relay_id=relay_id,
            account=account, instrument=instrument,
            signal="EXIT", strategy=strategy, confluence_score=exit_score,
            ai_decision="SAFETY_EXIT", ai_reason=safety_reason,
            ai_latency_ms=0, relay_result=relay_result,
            relay_details=safety_reason,
            payload_json=body_text, alert_type="manage"
        )

        return JSONResponse(content={
            "ai_decision": "SAFETY_EXIT",
            "ai_reason": safety_reason,
            "ai_latency_ms": 0,
            "relay_result": relay_result,
            "dry_run": AI_DRY_RUN
        })

    # --- Call Anthropic for exit decision ---
    ai_decision, ai_reason, ai_latency_ms, ai_raw = await call_anthropic(
        payload, system_prompt=AI_MANAGE_PROMPT
    )

    logger.info(
        f"[{relay_user}/{relay_id}] AI manage decision: {ai_decision} "
        f"({ai_latency_ms}ms) — {ai_reason}"
    )

    relay_result = None
    relay_details = None

    if ai_decision == "EXIT":
        if AI_DRY_RUN:
            relay_result = "exit_dry_run"
            relay_details = f"AI says EXIT but DRY RUN — P&L: {unrealized_ticks:.1f} ticks (${unrealized_dollars:.2f})"
        else:
            user = db.get_user(relay_user)
            if user:
                flatten_fields = {
                    "relay_user": relay_user,
                    "relay_id": relay_id,
                    "command": "PLACE",
                    "account": account,
                    "instrument": instrument,
                    "action": "SELL" if direction == "long" else "BUY",
                    "qty": str(payload.get("qty", 1)),
                    "order_type": "MARKET",
                    "tif": "DAY",
                    "market_position": "flat",
                    "prev_market_position": direction,
                    "sync_strategy": "true",
                    "out_of_sync": "flatten",
                    "strategy_tag": payload.get("strategy_tag", relay_id),
                }
                relay_response = await process_signal(flatten_fields, raw_payload=body_text)
                relay_result = f"exit_{relay_response.get('result', 'unknown')}"
                relay_details = relay_response.get("details", "")
            else:
                relay_result = "exit_error"
                relay_details = f"Unknown user: {relay_user}"

        ai_clear_position(relay_user, account, instrument)
        logger.info(f"[{relay_user}/{relay_id}] AI EXIT → position cleared")

    elif ai_decision == "HOLD":
        relay_result = "hold"
        relay_details = f"Holding — bar {bars_in_trade}, P&L: {unrealized_ticks:.1f} ticks"

    elif ai_decision == "TIMEOUT":
        # On timeout during manage, HOLD (conservative = don't panic-exit)
        relay_result = "hold_timeout"
        relay_details = f"AI timed out — holding by default"
        logger.warning(f"[{relay_user}/{relay_id}] AI TIMEOUT during manage — holding")

    else:
        relay_result = "hold_error"
        relay_details = f"AI error during manage — holding: {ai_reason}"
        logger.error(f"[{relay_user}/{relay_id}] AI ERROR during manage — holding")

    # --- Log ---
    ai_log_decision(
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

    return JSONResponse(content={
        "ai_decision": ai_decision,
        "ai_reason": ai_reason,
        "ai_latency_ms": ai_latency_ms,
        "relay_result": relay_result,
        "relay_details": relay_details,
        "bars_in_trade": bars_in_trade,
        "unrealized_ticks": unrealized_ticks,
        "exit_score": exit_score,
        "dry_run": AI_DRY_RUN
    })


@app.get("/webhook/ai/logs")
async def ai_logs_endpoint(relay_user: str = None, limit: int = 50):
    """View AI gate decision logs."""
    return ai_get_logs(relay_user, limit)


@app.get("/webhook/ai/stats")
async def ai_stats_endpoint(relay_user: str = None):
    """Quick stats on AI gate decisions."""
    ai_gate_init_db()
    conn = db.get_connection()

    where = "WHERE relay_user = ?" if relay_user else ""
    params = (relay_user,) if relay_user else ()

    rows = conn.execute(f"""
        SELECT
            ai_decision,
            COUNT(*) as count,
            AVG(ai_latency_ms) as avg_latency_ms
        FROM ai_gate_logs
        {where}
        GROUP BY ai_decision
    """, params).fetchall()

    total = conn.execute(
        f"SELECT COUNT(*) as total FROM ai_gate_logs {where}", params
    ).fetchone()

    conn.close()

    stats = {row["ai_decision"]: {"count": row["count"], "avg_latency_ms": round(row["avg_latency_ms"])}
             for row in rows}
    stats["total"] = total["total"] if total else 0

    return stats


@app.get("/webhook/ai/positions")
async def ai_positions_endpoint(relay_user: str = None):
    """View AI-tracked open positions."""
    return ai_list_positions(relay_user)


@app.post("/webhook/ai/positions/clear")
async def ai_positions_clear(request: Request):
    """Manually clear an AI-tracked position."""
    data = await request.json()
    relay_user = data.get("relay_user", "")
    account = data.get("account", "")
    instrument = data.get("instrument", "")
    if not relay_user or not account or not instrument:
        raise HTTPException(status_code=400, detail="Missing relay_user, account, or instrument")
    ai_clear_position(relay_user, account, instrument)
    logger.info(f"AI position cleared: {relay_user}/{account}/{instrument}")
    return {"status": "ok", "details": "AI position cleared"}


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """AI Gate live dashboard."""
    dash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    with open(dash_path, "r") as f:
        return f.read()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
