import logging
import os
import json
import re
import sqlite3
import asyncio

import httpx

from datetime import datetime, timezone
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager

import database as db
import ai_gate
import cvd

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
    ai_gate.init_db()
    ai_gate.startup_recovery()
    watchdog = asyncio.create_task(ai_gate.stale_position_watchdog())
    # Start CVD WebSocket if API key is available
    ct_key = os.environ.get("CT_API_KEY", "")
    cvd_instruments = os.environ.get("CVD_INSTRUMENTS", "MNQ JUN26").split(",")
    cvd_instruments = [i.strip() for i in cvd_instruments if i.strip()]
    if ct_key:
        cvd.start(ct_key, cvd_instruments)
    logger.info(f"Trade Relay started -- database initialized (AI Gate: {'DRY RUN' if ai_gate.AI_DRY_RUN else 'LIVE'})")
    yield
    cvd.stop()
    watchdog.cancel()
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
        "ai_gate": "dry_run" if ai_gate.AI_DRY_RUN else "live",
        "default_model": ai_gate.DEFAULT_AI_MODEL,
        "anthropic_key_set": bool(ai_gate.ANTHROPIC_API_KEY),
        "minimax_key_set": bool(ai_gate.MINIMAX_API_KEY)
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
# AI GATE ROUTES (thin wrappers — logic lives in ai_gate.py)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/webhook/ai")
async def webhook_ai(request: Request):
    """AI Gate webhook — receives bar data JSON from Pine Script v2.0.0.
    Returns 200 immediately; processing happens in the background.
    Server decides: enter, hold, exit, or skip.
    """
    body = await request.body()
    body_text = body.decode("utf-8").strip()

    if not body_text:
        raise HTTPException(status_code=400, detail="Empty payload")

    try:
        payload = json.loads(body_text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    missing = ai_gate.REQUIRED_BAR_FIELDS - set(payload.keys())
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {', '.join(missing)}")

    relay_user = payload.get("relay_user", "")
    user = db.get_user(relay_user)
    if not user:
        raise HTTPException(status_code=400, detail=f"Unknown relay_user: {relay_user}")

    asyncio.create_task(ai_gate.process_bar(payload, body_text))
    return JSONResponse(content={"status": "received"})


@app.get("/webhook/ai/logs")
async def ai_logs_endpoint(relay_user: str = None, relay_id: str = None, limit: int = 50):
    return ai_gate.get_logs(relay_user, relay_id=relay_id, limit=limit)


@app.get("/webhook/ai/stats")
async def ai_stats_endpoint(relay_user: str = None, relay_id: str = None):
    return ai_gate.get_stats(relay_user, relay_id=relay_id)


@app.get("/webhook/ai/positions")
async def ai_positions_endpoint(relay_user: str = None):
    return ai_gate.list_positions(relay_user)


@app.post("/webhook/ai/positions/clear")
async def ai_positions_clear(request: Request):
    data = await request.json()
    relay_user = data.get("relay_user", "")
    account = data.get("account", "")
    instrument = data.get("instrument", "")
    if not relay_user or not account or not instrument:
        raise HTTPException(status_code=400, detail="Missing relay_user, account, or instrument")
    ai_gate.clear_position(relay_user, account, instrument)
    logger.info(f"AI position cleared: {relay_user}/{account}/{instrument}")
    return {"status": "ok", "details": "AI position cleared"}


@app.get("/webhook/ai/trades")
async def ai_trades_endpoint(relay_user: str = None, relay_id: str = None, limit: int = 50):
    return ai_gate.get_trades(relay_user, relay_id=relay_id, limit=limit)


@app.get("/webhook/ai/trades/stats")
async def ai_trade_stats_endpoint(relay_user: str = None, relay_id: str = None):
    return ai_gate.get_trade_stats(relay_user, relay_id=relay_id)


# ══════════════════════════════════════════════════════════════════════════════
# BOT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/webhook/ai/models")
async def ai_models_list():
    """List available AI models."""
    return [{"id": k, "label": v["label"]} for k, v in ai_gate.AI_MODELS.items()]


@app.get("/webhook/ai/bots")
async def ai_bots_list():
    return ai_gate.list_bots()


@app.post("/webhook/ai/bots")
async def ai_bots_create(request: Request, _user: dict = Depends(verify_bearer)):
    data = await request.json()
    required = ["bot_id", "mode", "account", "strategy_tag"]
    for f in required:
        if f not in data:
            raise HTTPException(status_code=400, detail=f"Missing field: {f}")
    if data["mode"] == "copy" and not data.get("source_bot"):
        raise HTTPException(status_code=400, detail="Copy mode requires source_bot")
    if data["mode"] == "python" and not data.get("strategy_name"):
        raise HTTPException(status_code=400, detail="Python mode requires strategy_name")
    ai_gate.add_bot(
        bot_id=data["bot_id"], mode=data["mode"],
        account=data["account"], strategy_tag=data["strategy_tag"],
        source_bot=data.get("source_bot"), relay_id=data.get("relay_id") or data["bot_id"],
        entry_prompt=data.get("entry_prompt"), manage_prompt=data.get("manage_prompt"),
        strategy_name=data.get("strategy_name"), config_json=data.get("config_json"),
        relay_user=data.get("relay_user", "titon"),
        ai_model=data.get("ai_model"),
    )
    return {"status": "ok", "bot_id": data["bot_id"]}


@app.get("/webhook/ai/strategies")
async def ai_strategies_list(_user: dict = Depends(verify_bearer)):
    """List available Python strategies."""
    from strategy_runner import STRATEGY_REGISTRY
    return [{"name": name, "description": cls.description}
            for name, cls in STRATEGY_REGISTRY.items()]


@app.put("/webhook/ai/bots/{bot_id}")
async def ai_bot_update(bot_id: str, request: Request, _user: dict = Depends(verify_bearer)):
    data = await request.json()
    ai_gate.update_bot(bot_id, **data)
    return {"status": "ok", "bot_id": bot_id}


@app.delete("/webhook/ai/bots/{bot_id}")
async def ai_bot_delete(bot_id: str, _user: dict = Depends(verify_bearer)):
    ai_gate.delete_bot(bot_id)
    return {"status": "ok", "bot_id": bot_id}


# ── Strategy Indicator endpoints ──

@app.get("/webhook/ai/strategies/{strategy_id}/indicators")
async def ai_strategy_indicators_get(strategy_id: str):
    """Get available indicators for a strategy."""
    return ai_gate.get_strategy_indicators(strategy_id)


@app.post("/webhook/ai/strategies/{strategy_id}/indicators")
async def ai_strategy_indicators_set(strategy_id: str, request: Request,
                                      _user: dict = Depends(verify_bearer)):
    """Bulk set indicators for a strategy."""
    data = await request.json()
    indicators = data.get("indicators", [])
    if not indicators:
        raise HTTPException(status_code=400, detail="Missing 'indicators' list")
    ai_gate.add_strategy_indicators(strategy_id, indicators)
    return {"status": "ok", "strategy_id": strategy_id, "count": len(indicators)}


# ── Bot Indicator endpoints ──

@app.get("/webhook/ai/bots/{bot_id}/indicators")
async def ai_bot_indicators_get(bot_id: str):
    """Get bot's indicator selections."""
    return ai_gate.get_bot_indicators(bot_id)


@app.post("/webhook/ai/bots/{bot_id}/indicators")
async def ai_bot_indicators_set(bot_id: str, request: Request,
                                 _user: dict = Depends(verify_bearer)):
    """Set bot's indicator selections."""
    data = await request.json()
    indicators = data.get("indicators", {})
    if not isinstance(indicators, dict):
        raise HTTPException(status_code=400, detail="'indicators' must be a dict of field_name -> bool")
    ai_gate.set_bot_indicators(bot_id, indicators)
    return {"status": "ok", "bot_id": bot_id, "count": len(indicators)}


@app.put("/webhook/ai/bots/{bot_id}/enable")
async def ai_bot_enable(bot_id: str, _user: dict = Depends(verify_bearer)):
    ai_gate.set_bot_enabled(bot_id, True)
    return {"status": "ok", "bot_id": bot_id, "enabled": True}


@app.put("/webhook/ai/bots/{bot_id}/disable")
async def ai_bot_disable(bot_id: str, _user: dict = Depends(verify_bearer)):
    ai_gate.set_bot_enabled(bot_id, False)
    return {"status": "ok", "bot_id": bot_id, "enabled": False}


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/admin/delete-bars")
async def admin_delete_bars(request: Request, _user: dict = Depends(verify_bearer)):
    """Delete bad bars by ID list. Body: {"ids": [55, 56]}"""
    data = await request.json()
    ids = data.get("ids", [])
    if not ids:
        raise HTTPException(status_code=400, detail="Missing 'ids' list")
    conn = db.get_connection()
    placeholders = ",".join("?" for _ in ids)
    conn.execute(f"DELETE FROM ai_bars WHERE id IN ({placeholders})", ids)
    conn.commit()
    conn.close()
    return {"status": "ok", "deleted": len(ids)}


@app.post("/admin/fix-bars")
async def admin_fix_bars(request: Request, _user: dict = Depends(verify_bearer)):
    """Fix bar fields by ID. Body: {"fixes": [{"id": 55, "low": 24373.5}, ...]}"""
    data = await request.json()
    fixes = data.get("fixes", [])
    if not fixes:
        raise HTTPException(status_code=400, detail="Missing 'fixes' list")
    conn = db.get_connection()
    for fix in fixes:
        bar_id = fix.pop("id", None)
        if not bar_id or not fix:
            continue
        sets = ", ".join(f"{k} = ?" for k in fix.keys())
        vals = list(fix.values()) + [bar_id]
        conn.execute(f"UPDATE ai_bars SET {sets} WHERE id = ?", vals)
    conn.commit()
    conn.close()
    return {"status": "ok", "fixed": len(fixes)}


@app.get("/admin/logs")
async def admin_logs(lines: int = 100, _user: dict = Depends(verify_bearer)):
    """Return the last N lines from trade_relay.log."""
    import os
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_relay.log")
    if not os.path.exists(log_path):
        return {"lines": []}
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()
    return {"lines": [l.rstrip() for l in all_lines[-lines:]]}


# ══════════════════════════════════════════════════════════════════════════════
# DEPLOY
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/deploy")
async def deploy(_user: dict = Depends(verify_bearer)):
    """Pull latest code from GitHub and restart the service.
    Requires Bearer auth. The response may not arrive since the service restarts.
    """
    import subprocess

    # Git pull
    try:
        result = subprocess.run(
            ["git", "pull", "origin", "main"],
            cwd=r"C:\TradeRelay",
            capture_output=True, text=True, timeout=30
        )
        pull_output = result.stdout.strip() + result.stderr.strip()
        logger.info(f"Deploy: git pull -> {pull_output}")

        if result.returncode != 0:
            return {"status": "error", "details": f"git pull failed: {pull_output}"}

        if "Already up to date" in pull_output:
            return {"status": "ok", "details": "Already up to date", "restarted": False}

    except Exception as e:
        return {"status": "error", "details": f"git pull error: {e}"}

    # Schedule restart (short delay so the response can go out)
    async def restart_service():
        await asyncio.sleep(1)
        subprocess.Popen(
            ["cmd", "/c", "net stop TradeRelay && net start TradeRelay"],
            cwd=r"C:\TradeRelay"
        )

    asyncio.create_task(restart_service())
    return {"status": "ok", "details": pull_output, "restarted": True}


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """AI Gate live dashboard."""
    with open(r"C:\traderelay\dashboard.html", "r", encoding="utf-8") as f:
        return f.read()


# ══════════════════════════════════════════════════════════════════════════════
# CVD ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/cvd")
async def cvd_all():
    """Get CVD state for all tracked instruments."""
    return cvd.get_all_cvd()


@app.get("/cvd/{instrument}")
async def cvd_instrument(instrument: str):
    """Get CVD state for a specific instrument."""
    return cvd.get_cvd(instrument)


@app.get("/cvd/{instrument}/candles")
async def cvd_candles(instrument: str):
    """Get 1-min candles for charting."""
    return cvd.get_candles(instrument)


@app.get("/cvd/{instrument}/chart")
async def cvd_chart_data(instrument: str, relay_id: str = None, limit: int = 240):
    """Get chart data with candles and trade markers for overlay."""
    # Get candles from DB (persistent) with fallback to in-memory
    bars = ai_gate.get_bars(instrument, limit)

    # If no DB bars, fall back to in-memory candles
    if not bars:
        bars = cvd.get_candles(instrument)

    # Get trade markers from decision logs
    markers = []
    # Extract root symbol for matching: "MNQ JUN26" → "MNQ", "MNQ1!" → "MNQ"
    import re
    root = re.split(r'[\s0-9!]', instrument)[0]
    inst_pattern = f'%{root}%'

    conn = db.get_connection()

    if relay_id:
        # Filtered by relay_id
        rejects = conn.execute("""
            SELECT timestamp, signal FROM ai_gate_logs
            WHERE instrument LIKE ? AND ai_decision = 'DISAGREE' AND alert_type = 'entry' AND relay_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (inst_pattern, relay_id, limit)).fetchall()
        for r in rejects:
            markers.append({"time": r["timestamp"], "type": "reject",
                            "direction": r["signal"].lower() if r["signal"] else "long"})

        entries = conn.execute("""
            SELECT timestamp, signal FROM ai_gate_logs
            WHERE instrument LIKE ? AND ai_decision = 'AGREE' AND alert_type = 'entry' AND relay_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (inst_pattern, relay_id, limit)).fetchall()
        for e in entries:
            markers.append({"time": e["timestamp"], "type": "entry",
                            "direction": e["signal"].lower() if e["signal"] else "long"})

        trades = conn.execute("""
            SELECT closed_at, direction, dollar_pnl, ticks_pnl FROM ai_trades
            WHERE instrument LIKE ? AND status = 'closed' AND closed_at IS NOT NULL AND relay_id = ?
            ORDER BY closed_at DESC LIMIT ?
        """, (inst_pattern, relay_id, limit)).fetchall()
        for t in trades:
            markers.append({"time": t["closed_at"], "type": "exit", "direction": t["direction"],
                            "dollar_pnl": t["dollar_pnl"], "ticks_pnl": t["ticks_pnl"]})
    else:
        # All bots
        rejects = conn.execute("""
            SELECT timestamp, signal, ai_decision FROM ai_gate_logs
            WHERE instrument LIKE ? AND ai_decision = 'DISAGREE' AND alert_type = 'entry'
            ORDER BY timestamp DESC LIMIT ?
        """, (inst_pattern, limit)).fetchall()
        for r in rejects:
            markers.append({"time": r["timestamp"], "type": "reject",
                            "direction": r["signal"].lower() if r["signal"] else "long"})

        entries = conn.execute("""
            SELECT timestamp, signal, ai_decision FROM ai_gate_logs
            WHERE instrument LIKE ? AND ai_decision = 'AGREE' AND alert_type = 'entry'
            ORDER BY timestamp DESC LIMIT ?
        """, (inst_pattern, limit)).fetchall()
        for e in entries:
            markers.append({"time": e["timestamp"], "type": "entry",
                            "direction": e["signal"].lower() if e["signal"] else "long"})

        trades = conn.execute("""
            SELECT closed_at, direction, dollar_pnl, ticks_pnl FROM ai_trades
            WHERE instrument LIKE ? AND status = 'closed' AND closed_at IS NOT NULL
            ORDER BY closed_at DESC LIMIT ?
        """, (inst_pattern, limit)).fetchall()
        for t in trades:
            markers.append({"time": t["closed_at"], "type": "exit", "direction": t["direction"],
                            "dollar_pnl": t["dollar_pnl"], "ticks_pnl": t["ticks_pnl"]})

    conn.close()

    return {"candles": bars, "markers": markers}


@app.post("/cvd/reset")
async def cvd_reset(instrument: str = None):
    """Reset CVD to zero."""
    cvd.reset(instrument)
    return {"status": "ok"}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
