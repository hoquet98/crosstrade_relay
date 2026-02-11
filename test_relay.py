"""
Test script -- simulates scenarios with CT API checks and grace period logic.
"""
import asyncio
import database as db

import relay
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

_UNSET = object()  # sentinel to distinguish "not passed" from None


async def mock_forward(*args, **kwargs):
    resp = MagicMock()
    resp.status_code = 200
    return resp


async def run_signal(signal, ct_api_return=_UNSET):
    fields = relay.parse_payload(signal["payload"])

    if ct_api_return is not _UNSET:
        # Explicit CT API mock -- used for blocked entry scenarios
        with patch("relay.check_ct_position", new_callable=AsyncMock, return_value=ct_api_return), \
             patch("httpx.AsyncClient.post", side_effect=mock_forward):
            result = await relay.process_signal(fields)
    else:
        # No API check needed (entries with no owner, exits, etc.)
        with patch("relay.check_ct_position", new_callable=AsyncMock, return_value="short"), \
             patch("httpx.AsyncClient.post", side_effect=mock_forward):
            result = await relay.process_signal(fields)

    status = result["result"].upper()
    icon = {"FORWARDED": "[OK]", "BLOCKED": "[BLOCKED]", "DROPPED": "[DROPPED]", "ERROR": "[ERROR]"}.get(status, "?")
    print(f"\n{icon} {signal['name']}")
    print(f"   Result: {status} -- {result['details']}")
    return result


def age_position(relay_user, account, instrument, seconds):
    """Backdate a position's opened_at to simulate time passing."""
    old_time = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    conn = db.get_connection()
    conn.execute(
        "UPDATE positions SET opened_at = ? WHERE relay_user = ? AND account = ? AND instrument = ?",
        (old_time, relay_user, account, instrument)
    )
    conn.commit()
    conn.close()


async def run_test():
    db.init_db()
    db.upsert_user("titon", "test-key-123", "https://example.com/webhook")

    # =========================================================================
    print("=" * 80)
    print("  TEST 1: Normal scenario -- entry, blocked, phantom, exit")
    print("=" * 80)

    await run_signal({
        "name": "Toms SELL entry (open short)",
        "payload": "relay_user=titon; relay_id=toms; command=PLACE; account=ACCT1; instrument=RTYH6; action=sell; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=short; prev_market_position=flat; out_of_sync=flatten;"
    })

    # Age position past grace period so CT API check is used
    age_position("titon", "ACCT1", "RTYH6", 60)

    await run_signal({
        "name": "Snipe SELL entry (blocked -- CT API confirms short)",
        "payload": "relay_user=titon; relay_id=snipe; command=PLACE; account=ACCT1; instrument=RTYH6; action=sell; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=short; prev_market_position=flat; out_of_sync=flatten;"
    }, ct_api_return="short")

    await run_signal({
        "name": "Snipe BUY exit (phantom -- dropped)",
        "payload": "relay_user=titon; relay_id=snipe; command=PLACE; account=ACCT1; instrument=RTYH6; action=buy; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=flat; prev_market_position=short; out_of_sync=flatten;"
    })
    await run_signal({
        "name": "Toms BUY exit (close short)",
        "payload": "relay_user=titon; relay_id=toms; command=PLACE; account=ACCT1; instrument=RTYH6; action=buy; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=flat; prev_market_position=short; out_of_sync=flatten;"
    })

    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  TEST 2: Race condition -- two strategies fire within 30s")
    print("=" * 80)

    db.clear_position("titon", "ACCT1", "RTYH6")

    await run_signal({
        "name": "Snipe BUY entry (takes ownership)",
        "payload": "relay_user=titon; relay_id=snipe; command=PLACE; account=ACCT1; instrument=RTYH6; action=buy; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=long; prev_market_position=flat; out_of_sync=flatten;"
    })

    # DO NOT age position -- simulate toms firing 1 second later
    await run_signal({
        "name": "Toms BUY entry 1s later (blocked by grace period, NO CT API call)",
        "payload": "relay_user=titon; relay_id=toms; command=PLACE; account=ACCT1; instrument=RTYH6; action=buy; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=long; prev_market_position=flat; out_of_sync=flatten;"
    })

    await run_signal({
        "name": "Snipe SELL exit (close long)",
        "payload": "relay_user=titon; relay_id=snipe; command=PLACE; account=ACCT1; instrument=RTYH6; action=sell; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=flat; prev_market_position=long; out_of_sync=flatten;"
    })

    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  TEST 3: Stale position -- ATM/manual close, then new entry (after 30s)")
    print("=" * 80)

    db.clear_position("titon", "ACCT1", "RTYH6")

    await run_signal({
        "name": "Toms enters short",
        "payload": "relay_user=titon; relay_id=toms; command=PLACE; account=ACCT1; instrument=RTYH6; action=sell; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=short; prev_market_position=flat; out_of_sync=flatten;"
    })

    print("\n   [ATM stop loss hits -- position closed externally, relay doesn't know]")

    # Age past grace period so CT API is consulted
    age_position("titon", "ACCT1", "RTYH6", 60)

    await run_signal({
        "name": "Snipe enters -- CT API shows flat, stale ownership cleared",
        "payload": "relay_user=titon; relay_id=snipe; command=PLACE; account=ACCT1; instrument=RTYH6; action=sell; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=short; prev_market_position=flat; out_of_sync=flatten;"
    }, ct_api_return="flat")

    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  TEST 4: API failure -- block conservatively (after 30s)")
    print("=" * 80)

    db.clear_position("titon", "ACCT1", "RTYH6")

    await run_signal({
        "name": "Toms enters short",
        "payload": "relay_user=titon; relay_id=toms; command=PLACE; account=ACCT1; instrument=RTYH6; action=sell; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=short; prev_market_position=flat; out_of_sync=flatten;"
    })

    # Age past grace period so CT API is consulted
    age_position("titon", "ACCT1", "RTYH6", 60)

    await run_signal({
        "name": "Snipe enters -- CT API fails, conservative block",
        "payload": "relay_user=titon; relay_id=snipe; command=PLACE; account=ACCT1; instrument=RTYH6; action=sell; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=short; prev_market_position=flat; out_of_sync=flatten;"
    }, ct_api_return=None)

    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  EXPECTED:")
    print("  Test 1: Toms entry/exit [OK], Snipe entry [BLOCKED], phantom [DROPPED]")
    print("  Test 2: Snipe [OK], Toms [BLOCKED] by grace period (no CT API call)")
    print("  Test 3: Snipe [OK] (CT API detected flat, cleared stale ownership)")
    print("  Test 4: Snipe [BLOCKED] (API failed, blocked to be safe)")
    print("=" * 80)

    import os
    os.remove(db.DB_PATH)

asyncio.run(run_test())
