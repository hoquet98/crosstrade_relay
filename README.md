# Trade Relay

A lightweight webhook relay that sits between TradingView and CrossTrade, enabling multiple strategies to share a single account without interfering with each other.

## How It Works

```
TradingView Alert → Trade Relay (your VPS) → CrossTrade Servers → NT8 Plugin (your VPS)
```

**Core logic:**
- First strategy to signal an entry wins the position
- Other strategies' entries are blocked while a position is open
- Only the winning strategy's exits are forwarded
- Phantom exits from blocked strategies are silently dropped

## Requirements

- Python 3.10+
- Windows VPS (same machine as NT8/CrossTrade)

## Installation

1. Copy the `trade-relay` folder to your VPS
2. Run `install.bat` (double-click it)
3. Add your user config:

```
python manage.py add-user
```

It will prompt for:
- **relay_user** — your unique ID (e.g., `myuser`)
- **CrossTrade key** — your secret key
- **CrossTrade webhook URL** — your full webhook URL (e.g., `https://app.crosstrade.io/v1/send/...`)

4. Start the relay:

```
start_relay.bat
```

The relay runs on port 8080 by default.

## TradingView Alert Setup

Point your TradingView alert webhook URL to your VPS:

```
http://YOUR_VPS_IP:8080/webhook
```

### Alert Message Format

Add `relay_user` and `relay_id` to your existing CrossTrade payload. Remove the `key=` line (the relay injects it automatically).

**Example — Strategy A (1 min):**
```
relay_user=myuser;
relay_id=1min;
command=PLACE;
account=YOUR_ACCOUNT_NAME;
instrument=RTYH2026;
action={{strategy.order.action}};
qty=1;
order_type=MARKET;
tif=DAY;
sync_strategy=true;
market_position={{strategy.market_position}};
prev_market_position={{strategy.prev_market_position}};
out_of_sync=flatten;
```

**Example — Strategy B (5 min):**
```
relay_user=myuser;
relay_id=5min;
command=PLACE;
account=YOUR_ACCOUNT_NAME;
instrument=RTYH2026;
action={{strategy.order.action}};
qty=1;
order_type=MARKET;
tif=DAY;
sync_strategy=true;
market_position={{strategy.market_position}};
prev_market_position={{strategy.prev_market_position}};
out_of_sync=flatten;
```

The only difference between the two is `relay_id`. Everything else passes through to CrossTrade as-is.

## Signal Logic

The relay classifies each signal using the `market_position` and `prev_market_position` fields:

| prev_market_position | market_position | Type      |
|---------------------|-----------------|-----------|
| flat                | long/short      | **Entry** |
| long/short          | flat            | **Exit**  |
| long                | short (or vice versa) | **Reversal** |

**Entry rules:**
- No position tracked → forward, record owner
- Same `relay_id` owns position → forward (re-entry)
- Different `relay_id` owns position → **check CrossTrade API first:**
  - CT API shows flat (position closed externally) → clear stale owner, forward entry
  - CT API shows position still open → **BLOCKED**
  - CT API fails → **BLOCKED** (conservative)

**Exit rules:**
- `relay_id` matches owner → forward, clear ownership
- `relay_id` doesn't match → **DROPPED** (phantom exit)
- No position tracked → forward (safety fallback)

**Reversal rules:**
- Same as entry for ownership check, updates direction if forwarded

## Management CLI

```bash
# Add or update a user
python manage.py add-user

# List all users
python manage.py list-users

# Remove a user
python manage.py remove-user myuser

# View active positions
python manage.py positions
python manage.py positions myuser

# Manually clear a stuck position
python manage.py clear-position myuser YOUR_ACCOUNT_NAME RTYH2026

# View signal logs
python manage.py logs
python manage.py logs myuser --limit 20
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST   | `/webhook` | Main webhook — receives TradingView alerts |
| GET    | `/health` | Health check |
| GET    | `/positions` | View tracked positions |
| GET    | `/logs` | View signal log |
| GET    | `/users` | List users (keys masked) |
| POST   | `/users` | Add/update user (JSON body) |
| DELETE | `/users/{relay_user}` | Remove a user |
| POST   | `/positions/clear` | Manually clear position ownership |

## CrossTrade API Position Check

When the relay is about to block an entry (because another strategy owns the position), it first checks the actual NT8 position state via CrossTrade's API. This handles edge cases where a position was closed externally (ATM stop/target, manual close, account management flatten, daily P&L limit, etc.) but the relay still thinks a strategy owns it.

- **Only called on blocked entries** — no latency added to normal entries or exits
- **If API shows flat** — stale ownership is cleared, the new entry goes through
- **If API confirms position open** — entry is blocked as expected
- **If API fails** — entry is blocked conservatively (safe default)

The relay uses the same CrossTrade key for API authentication (`Bearer` token).

## Firewall

Make sure port 8080 is open on your VPS firewall for incoming connections from TradingView's servers. TradingView sends webhooks from various IPs, so you'll need to allow all inbound traffic on 8080 or whitelist their IP ranges.

## Files

```
trade-relay/
├── relay.py           # Main FastAPI application
├── database.py        # SQLite database operations
├── manage.py          # CLI management tool
├── requirements.txt   # Python dependencies
├── install.bat        # Windows installation script
├── start_relay.bat    # Windows startup script
├── trade_relay.db     # SQLite database (created on first run)
└── trade_relay.log    # Application log file
```
