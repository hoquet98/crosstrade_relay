import logging
import httpx
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager

import database as db

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
    logger.info("Trade Relay started — database initialized")
    yield
    logger.info("Trade Relay shutting down")


app = FastAPI(title="Trade Relay", lifespan=lifespan)


# --- Payload parsing ---

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


# --- Signal classification ---

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


# --- CrossTrade API position check ---

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

        # No matching instrument found — account is flat for this instrument
        return "flat"

    except Exception as e:
        logger.warning(f"CT API check failed: {e}")
        return None


# --- Core relay logic ---

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
            # No position — this strategy wins, forward it
            result = "forwarded"
            details = f"Entry {get_direction(fields)} — {relay_id} takes ownership"
            db.set_position(relay_user, account, instrument, relay_id, get_direction(fields))
        elif position["owner_id"] == relay_id:
            # Same strategy re-entering? Forward it, update direction
            result = "forwarded"
            details = f"Re-entry {get_direction(fields)} — {relay_id} already owns position"
            db.set_position(relay_user, account, instrument, relay_id, get_direction(fields))
        else:
            # Different strategy trying to enter — check CT API before blocking
            # Maybe the position was closed externally (ATM, manual, account mgmt)
            ct_state = await check_ct_position(user["crosstrade_key"], account, instrument)

            if ct_state == "flat":
                # Position was closed externally — clear stale ownership, let this entry through
                result = "forwarded"
                details = (f"Entry {get_direction(fields)} — stale ownership cleared "
                          f"(CT API shows flat, was owned by {position['owner_id']}), "
                          f"{relay_id} takes ownership")
                db.set_position(relay_user, account, instrument, relay_id, get_direction(fields))
                logger.info(f"[{relay_user}] Stale position detected via CT API — "
                           f"cleared {position['owner_id']}, {relay_id} entering")
            elif ct_state is None:
                # API call failed — block to be safe
                result = "blocked"
                details = (f"Entry blocked — position owned by {position['owner_id']} "
                          f"(CT API check failed, blocking conservatively)")
            else:
                # Position is still open — block
                result = "blocked"
                details = (f"Entry blocked — position owned by {position['owner_id']} "
                          f"(confirmed {ct_state} via CT API)")

    elif signal_type == "exit":
        if position is None:
            # No tracked position — forward anyway (might be manual or out of sync)
            result = "forwarded"
            details = f"Exit with no tracked position — forwarding to be safe"
        elif position["owner_id"] == relay_id:
            # Owner is closing — forward and clear ownership
            result = "forwarded"
            details = f"Exit — {relay_id} closing position"
            db.clear_position(relay_user, account, instrument)
        else:
            # Phantom exit from blocked strategy — drop it
            result = "dropped"
            details = f"Phantom exit dropped — position owned by {position['owner_id']}, not {relay_id}"

    elif signal_type == "reversal":
        if position is None:
            # No tracked position — forward, set new owner
            result = "forwarded"
            details = f"Reversal to {get_direction(fields)} — {relay_id} takes ownership"
            db.set_position(relay_user, account, instrument, relay_id, get_direction(fields))
        elif position["owner_id"] == relay_id:
            # Owner reversing — forward and update direction
            result = "forwarded"
            details = f"Reversal to {get_direction(fields)} — {relay_id} reversing"
            db.set_position(relay_user, account, instrument, relay_id, get_direction(fields))
        else:
            # Different strategy trying to reverse — check CT API before blocking
            ct_state = await check_ct_position(user["crosstrade_key"], account, instrument)

            if ct_state == "flat":
                result = "forwarded"
                details = (f"Reversal to {get_direction(fields)} — stale ownership cleared "
                          f"(CT API shows flat), {relay_id} takes ownership")
                db.set_position(relay_user, account, instrument, relay_id, get_direction(fields))
            elif ct_state is None:
                result = "blocked"
                details = (f"Reversal blocked — position owned by {position['owner_id']} "
                          f"(CT API check failed, blocking conservatively)")
            else:
                result = "blocked"
                details = (f"Reversal blocked — position owned by {position['owner_id']} "
                          f"(confirmed {ct_state} via CT API)")

    else:
        # Unknown signal type — no sync fields, forward as-is (no position tracking)
        result = "forwarded"
        details = "No market_position fields — forwarding without relay logic"

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
            logger.info(f"[{relay_user}/{relay_id}] {signal_type.upper()} → CrossTrade {response.status_code}")

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
        logger.info(f"[{relay_user}/{relay_id}] {signal_type.upper()} → {result.upper()}: {details}")
        db.add_log(
            relay_user=relay_user, relay_id=relay_id,
            account=account, instrument=instrument,
            action=action, market_position=market_position,
            prev_market_position=prev_market_position,
            signal_type=signal_type, result=result, details=details,
            raw_payload=raw_payload
        )
        return {"result": result, "details": details}


# --- Authentication ---

bearer_scheme = HTTPBearer()


async def verify_bearer(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    """Validate Bearer token against registered users' CrossTrade keys."""
    token = credentials.credentials
    user = db.get_user_by_key(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user


# --- Routes ---

@app.post("/webhook")
async def webhook(request: Request):
    """Main webhook endpoint — receives TradingView alerts."""
    body = await request.body()
    body_text = body.decode("utf-8").strip()

    if not body_text:
        raise HTTPException(status_code=400, detail="Empty payload")

    fields = parse_payload(body_text)
    result = await process_signal(fields, raw_payload=body_text)
    return JSONResponse(content=result)


@app.get("/health")
async def health():
    return {"status": "ok"}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
