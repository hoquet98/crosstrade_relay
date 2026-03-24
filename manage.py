"""
Trade Relay Management CLI

Usage:
    python manage.py add-user
    python manage.py list-users
    python manage.py remove-user <relay_user>
    python manage.py set-nt-token <relay_user>
    python manage.py positions [relay_user]
    python manage.py clear-position <relay_user> <account> <instrument>
    python manage.py logs [relay_user] [--limit N]
    python manage.py add-bot
    python manage.py list-bots
    python manage.py enable-bot <bot_id>
    python manage.py disable-bot <bot_id>
"""

import sys
import database as db
import ai_gate


def add_user():
    print("\n--- Add / Update User ---\n")
    relay_user = input("  relay_user (unique ID, e.g. 'titon'): ").strip()
    crosstrade_key = input("  CrossTrade key: ").strip()
    ct_webhook_url = input("  CrossTrade webhook URL: ").strip()

    if not all([relay_user, crosstrade_key, ct_webhook_url]):
        print("\n  [ERROR] All fields are required.")
        return

    db.upsert_user(relay_user, crosstrade_key, ct_webhook_url)
    print(f"\n  [OK] User '{relay_user}' saved.")


def list_users():
    users = db.list_users()
    if not users:
        print("\n  No users configured.")
        return
    print(f"\n  {'relay_user':<20} {'webhook_url':<50} {'created'}")
    print(f"  {'-'*90}")
    for u in users:
        key_masked = u['crosstrade_key'][:8] + "..."
        print(f"  {u['relay_user']:<20} {u['ct_webhook_url'][:50]:<50} {u['created_at']}")


def remove_user(relay_user: str):
    db.delete_user(relay_user)
    print(f"\n  [OK] User '{relay_user}' removed.")


def show_positions(relay_user: str = None):
    positions = db.list_positions(relay_user)
    if not positions:
        print("\n  No active positions tracked.")
        return
    print(f"\n  {'user':<15} {'account':<25} {'instrument':<15} {'owner':<15} {'direction':<10} {'opened_at'}")
    print(f"  {'-'*100}")
    for p in positions:
        print(f"  {p['relay_user']:<15} {p['account']:<25} {p['instrument']:<15} {p['owner_id']:<15} {p['direction']:<10} {p['opened_at']}")


def clear_position(relay_user: str, account: str, instrument: str):
    db.clear_position(relay_user, account, instrument)
    print(f"\n  [OK] Position cleared: {relay_user}/{account}/{instrument}")


def show_logs(relay_user: str = None, limit: int = 50):
    logs = db.get_logs(relay_user, limit)
    if not logs:
        print("\n  No logs found.")
        return
    print(f"\n  {'timestamp':<22} {'user':<10} {'id':<10} {'type':<10} {'action':<6} {'result':<12} {'details'}")
    print(f"  {'-'*120}")
    for log in logs:
        ts = log['timestamp'][:19] if log['timestamp'] else ""
        print(f"  {ts:<22} {(log['relay_user'] or ''):<10} {(log['relay_id'] or ''):<10} "
              f"{(log['signal_type'] or ''):<10} {(log['action'] or ''):<6} "
              f"{log['result']:<12} {(log['details'] or '')[:50]}")


def add_bot():
    print("\n--- Add / Update Bot ---\n")
    bot_id = input("  Bot ID (unique name, e.g. 'aggressive_scalp'): ").strip()
    mode = input("  Mode (normal/copy/python): ").strip().lower()

    source_bot = None
    relay_id = None
    strategy_name = None
    if mode == "copy":
        source_bot = input("  Source bot relay_id to copy from: ").strip()
    elif mode == "python":
        strategy_name = input("  Strategy name (e.g. 'ut_bot_trend'): ").strip()
        relay_id = bot_id  # python bots use bot_id as relay_id
    else:
        relay_id = input("  Relay ID (Pine Script relay_id): ").strip()

    account = input("  Target account (e.g. 'Sim101'): ").strip()
    strategy_tag = input("  Strategy tag for CrossTrade: ").strip()

    entry_prompt = input("  Custom entry prompt (press Enter for default): ").strip() or None
    manage_prompt = input("  Custom manage prompt (press Enter for default): ").strip() or None

    if not all([bot_id, mode, account, strategy_tag]):
        print("\n  [ERROR] bot_id, mode, account, and strategy_tag are required.")
        return

    ai_gate.add_bot(
        bot_id=bot_id, mode=mode, account=account,
        strategy_tag=strategy_tag, source_bot=source_bot,
        relay_id=relay_id, entry_prompt=entry_prompt,
        manage_prompt=manage_prompt
    )
    print(f"\n  [OK] Bot '{bot_id}' saved ({mode} mode).")


def show_bots():
    bots = ai_gate.list_bots()
    if not bots:
        print("\n  No bots configured.")
        return
    print(f"\n  {'bot_id':<25} {'mode':<8} {'source':<20} {'account':<15} {'tag':<15} {'enabled'}")
    print(f"  {'-'*100}")
    for b in bots:
        src = b.get('source_bot') or b.get('relay_id') or ''
        enabled = 'YES' if b['enabled'] else 'NO'
        print(f"  {b['bot_id']:<25} {b['mode']:<8} {src:<20} {b['account']:<15} {b['strategy_tag']:<15} {enabled}")


def main():
    db.init_db()
    ai_gate.init_db()

    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "add-user":
        add_user()
    elif cmd == "list-users":
        list_users()
    elif cmd == "remove-user":
        if len(sys.argv) < 3:
            print("Usage: python manage.py remove-user <relay_user>")
            return
        remove_user(sys.argv[2])
    elif cmd == "set-nt-token":
        if len(sys.argv) < 3:
            print("Usage: python manage.py set-nt-token <relay_user>")
            return
        relay_user = sys.argv[2]
        user = db.get_user(relay_user)
        if not user:
            print(f"\n  [ERROR] User '{relay_user}' not found.")
            return
        import secrets
        token = secrets.token_urlsafe(32)
        db.set_nt_query_token(relay_user, token)
        print(f"\n  [OK] NT query token set for '{relay_user}':")
        print(f"  {token}")
        print(f"\n  Use this as the X-NT-Token header when calling /nt/query")
    elif cmd == "positions":
        relay_user = sys.argv[2] if len(sys.argv) > 2 else None
        show_positions(relay_user)
    elif cmd == "clear-position":
        if len(sys.argv) < 5:
            print("Usage: python manage.py clear-position <relay_user> <account> <instrument>")
            return
        clear_position(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "logs":
        relay_user = None
        limit = 50
        args = sys.argv[2:]
        for i, arg in enumerate(args):
            if arg == "--limit" and i + 1 < len(args):
                limit = int(args[i + 1])
            elif not arg.startswith("--"):
                relay_user = arg
        show_logs(relay_user, limit)
    elif cmd == "add-bot":
        add_bot()
    elif cmd == "list-bots":
        show_bots()
    elif cmd == "enable-bot":
        if len(sys.argv) < 3:
            print("Usage: python manage.py enable-bot <bot_id>")
            return
        ai_gate.set_bot_enabled(sys.argv[2], True)
        print(f"\n  [OK] Bot '{sys.argv[2]}' enabled.")
    elif cmd == "disable-bot":
        if len(sys.argv) < 3:
            print("Usage: python manage.py disable-bot <bot_id>")
            return
        ai_gate.set_bot_enabled(sys.argv[2], False)
        print(f"\n  [OK] Bot '{sys.argv[2]}' disabled.")
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
