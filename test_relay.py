"""
Test script — simulates today's scenario plus edge cases with CT API checks.
"""
import asyncio
import database as db

import relay
from unittest.mock import AsyncMock, patch, MagicMock

_UNSET = object()  # sentinel to distinguish "not passed" from None


async def mock_forward(*args, **kwargs):
    resp = MagicMock()
    resp.status_code = 200
    return resp


async def run_signal(signal, ct_api_return=_UNSET):
    fields = relay.parse_payload(signal["payload"])

    if ct_api_return is not _UNSET:
        # Explicit CT API mock — used for blocked entry scenarios
        with patch("relay.check_ct_position", new_callable=AsyncMock, return_value=ct_api_return), \
             patch("httpx.AsyncClient.post", side_effect=mock_forward):
            result = await relay.process_signal(fields)
    else:
        # No API check needed (entries with no owner, exits, etc.)
        with patch("relay.check_ct_position", new_callable=AsyncMock, return_value="short"), \
             patch("httpx.AsyncClient.post", side_effect=mock_forward):
            result = await relay.process_signal(fields)

    status = result["result"].upper()
    icon = {"FORWARDED": "✅", "BLOCKED": "🚫", "DROPPED": "⛔", "ERROR": "❌"}.get(status, "?")
    print(f"\n{icon} {signal['name']}")
    print(f"   Result: {status} — {result['details']}")
    return result


async def run_test():
    db.init_db()
    db.upsert_user("titon", "test-key-123", "https://example.com/webhook")

    # =========================================================================
    print("=" * 80)
    print("  TEST 1: Normal scenario (today's replay)")
    print("=" * 80)

    await run_signal({
        "name": "8:39 — Toms SELL entry (open short)",
        "payload": "relay_user=titon; relay_id=toms; command=PLACE; account=ACCT1; instrument=RTYH6; action=sell; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=short; prev_market_position=flat; out_of_sync=flatten;"
    })
    await run_signal({
        "name": "8:42 — Toms BUY exit (close short)",
        "payload": "relay_user=titon; relay_id=toms; command=PLACE; account=ACCT1; instrument=RTYH6; action=buy; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=flat; prev_market_position=short; out_of_sync=flatten;"
    })
    await run_signal({
        "name": "8:50 — Toms SELL entry (new short)",
        "payload": "relay_user=titon; relay_id=toms; command=PLACE; account=ACCT1; instrument=RTYH6; action=sell; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=short; prev_market_position=flat; out_of_sync=flatten;"
    })
    await run_signal({
        "name": "8:50 — Snipe SELL entry (blocked — CT API confirms short)",
        "payload": "relay_user=titon; relay_id=snipe; command=PLACE; account=ACCT1; instrument=RTYH6; action=sell; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=short; prev_market_position=flat; out_of_sync=flatten;"
    }, ct_api_return="short")

    await run_signal({
        "name": "8:54 — Snipe BUY exit (phantom — dropped)",
        "payload": "relay_user=titon; relay_id=snipe; command=PLACE; account=ACCT1; instrument=RTYH6; action=buy; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=flat; prev_market_position=short; out_of_sync=flatten;"
    })
    await run_signal({
        "name": "8:55 — Toms BUY exit (close short)",
        "payload": "relay_user=titon; relay_id=toms; command=PLACE; account=ACCT1; instrument=RTYH6; action=buy; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=flat; prev_market_position=short; out_of_sync=flatten;"
    })

    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  TEST 2: Stale position — ATM/manual close, then new entry")
    print("=" * 80)

    db.clear_position("titon", "ACCT1", "RTYH6")

    await run_signal({
        "name": "Toms enters short",
        "payload": "relay_user=titon; relay_id=toms; command=PLACE; account=ACCT1; instrument=RTYH6; action=sell; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=short; prev_market_position=flat; out_of_sync=flatten;"
    })

    print("\n⚡ [ATM stop loss hits — position closed externally, relay doesn't know]")

    await run_signal({
        "name": "Snipe enters — CT API shows flat, stale ownership cleared",
        "payload": "relay_user=titon; relay_id=snipe; command=PLACE; account=ACCT1; instrument=RTYH6; action=sell; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=short; prev_market_position=flat; out_of_sync=flatten;"
    }, ct_api_return="flat")

    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  TEST 3: API failure — block conservatively")
    print("=" * 80)

    db.clear_position("titon", "ACCT1", "RTYH6")

    await run_signal({
        "name": "Toms enters short",
        "payload": "relay_user=titon; relay_id=toms; command=PLACE; account=ACCT1; instrument=RTYH6; action=sell; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=short; prev_market_position=flat; out_of_sync=flatten;"
    })

    await run_signal({
        "name": "Snipe enters — CT API fails, conservative block",
        "payload": "relay_user=titon; relay_id=snipe; command=PLACE; account=ACCT1; instrument=RTYH6; action=sell; qty=1; order_type=MARKET; tif=DAY; sync_strategy=true; market_position=short; prev_market_position=flat; out_of_sync=flatten;"
    }, ct_api_return=None)

    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  EXPECTED:")
    print("  Test 1: Toms entries/exits ✅, Snipe entry 🚫, phantom ⛔")
    print("  Test 2: Snipe ✅ (CT API detected flat, cleared stale toms ownership)")
    print("  Test 3: Snipe 🚫 (API failed, blocked to be safe)")
    print("=" * 80)

    import os
    os.remove(db.DB_PATH)

asyncio.run(run_test())
