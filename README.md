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
├── relay.py              # Main FastAPI application
├── database.py           # SQLite database operations
├── manage.py             # CLI management tool
├── ai_gate.py            # AI Gate — LLM-powered trade decision engine
├── strategy_runner.py    # Python strategy registry and execution
├── vbt_adapter.py        # Universal VBT strategy adapter
├── indicator_engine.py   # Server-side indicator computation
├── indicators.py         # Technical indicator library
├── cvd.py                # Cumulative Volume Delta tracking
├── requirements.txt      # Python dependencies
├── install.bat           # Windows installation script
├── start_relay.bat       # Windows startup script
├── trade_relay.db        # SQLite database (created on first run)
└── trade_relay.log       # Application log file
```

---

## AI Gate — LLM-Powered Trading

AI Gate adds an intelligent decision layer on top of the relay. Instead of blindly forwarding every signal, an LLM evaluates each trade's indicator context before executing.

### Architecture

```
Data Sources                    Decision Engine              Execution
─────────────                   ───────────────              ─────────
TradingView Pine Script  ─┐
                           ├──▶ AI Gate (process_bar)       ┌──▶ CrossTrade Webhook
CVD WebSocket (1s ticks) ─┤     │                           │    │
                           │     ├─ Signal detected?         │    ▼
Python VBT Strategies  ───┘     │   ├─ LLM: AGREE/DISAGREE ─┘   NT8 fills order
                                │   └─ DISAGREE → skip
                                │
                                ├─ In position?
                                │   ├─ Safety nets (hard SL, time stop)
                                │   ├─ Exit score (indicator flips)
                                │   └─ LLM: HOLD/EXIT
                                │
                                └─ Logs everything to SQLite
```

### State Machine

Each bot tracks a per-instrument position through these states:

| State | Condition | Action |
|-------|-----------|--------|
| **FLAT** | Signal fires | Call LLM for entry decision |
| **FLAT** | No signal | Skip (no cost) |
| **IN POSITION** | Every bar | Compute P&L, exit score, check safety nets |
| **IN POSITION** | Exit score > 0 or periodic | Call LLM for hold/exit decision |
| **IN POSITION** | Hard stop / time stop hit | Force close (no LLM call) |

### Bot Modes

| Mode | Description |
|------|-------------|
| `normal` | Source bot — receives signals from Pine Script webhooks |
| `copy` | Mirrors another bot's signals to a different account/strategy tag |
| `python` | Runs a server-side Python strategy (VBT or custom) |

---

## Python Strategies (mode = 'python')

Python strategies run server-side using stored 1-min bar data. On each bar close, the strategy computes indicators and signals, then feeds the result through AI Gate for LLM-gated execution.

### How It Works

```
1-min bar closes (CVD WebSocket)
    │
    ▼
strategy_runner.run_python_bots()
    │
    ├─ For each enabled python bot:
    │   ├─ Load 500 bars from ai_bars table → DataFrame
    │   ├─ strategy.generate_signals(df, params) → payload dict
    │   │   ├─ long_signal: True/False
    │   │   ├─ short_signal: True/False
    │   │   └─ indicator values (RSI, CCI, ATR, OMA, etc.)
    │   └─ ai_gate.process_bar(payload)
    │       ├─ LLM evaluates → AGREE → _send_to_crosstrade()
    │       └─ LLM evaluates → DISAGREE → skip, log reason
    │
    └─ Next bot...
```

### Strategy Types

There are two kinds of Python strategies:

**1. Native strategies** — Written directly against the `BaseStrategy` interface.

```python
@register_strategy("my_strategy")
class MyStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame, params: dict) -> dict:
        # Compute indicators, return signal dict
        return {"long_signal": True, "short_signal": False, "rsi14": 42.5, ...}
```

**2. VBT strategies** — Pine Script strategies converted to Python using VectorBT Pro, then wrapped by the universal adapter (`vbt_adapter.py`).

These are the strategies in `C:\Users\hoque\vectorBT Strategy\strategies\`.

---

## VBT Strategy Adapter

The VBT adapter (`vbt_adapter.py`) bridges VectorBT strategies into AI Gate. It wraps any strategy that follows the standard VBT pattern into the `BaseStrategy` interface.

### What the Adapter Does

```
VBT Strategy Module                    AI Gate Payload
───────────────────                    ───────────────
Strategy.PARAMS_CLASS    ──▶  Build typed params from config_json
Strategy.compute_indicators(df, p)  ──▶  Run all indicators (OMA, RSI, CCI, etc.)
Strategy.compute_session(df, p)     ──▶  Session/day/exclude masks
detect_signals_from_dict(ind, s, p) ──▶  Numba-compiled signal detection
                                         │
                                         ▼
                                    {
                                      "long_signal": true,
                                      "short_signal": false,
                                      "oma_trend": 1,
                                      "l_rsi": 52.3,
                                      "s_cci": 87.4,
                                      "close": 2945.20,
                                      "session_bucket": "morning",
                                      ...
                                    }
```

### Auto-Discovery

VBT strategies are **automatically discovered** on startup. Any `.py` file in the `strategies/` folder that has both a `Strategy` class and a `detect_signals_from_dict()` function is registered.

You can also register manually:

```python
from vbt_adapter import register_vbt_strategy
register_vbt_strategy("qtp201_super_rsi_scalper", "strategies.qtp201_super_rsi_scalper")
```

### VBT Strategy File Requirements

Every VBT strategy module must have:

| Component | Purpose |
|-----------|---------|
| `@dataclass` Params class | All strategy parameters with defaults |
| `compute_indicators(data, params)` | Returns dict of numpy arrays |
| `compute_session(data, params)` | Returns session mask tuple |
| `detect_signals(...)` | `@njit` Numba-compiled, returns `(long_sig, short_sig)` |
| `detect_signals_from_dict(indicators, session, params)` | Unpacks dict → calls Numba function |
| `Strategy` class | Standard interface with `PARAMS_CLASS`, `NAME` |

### Optional: INDICATOR_FIELDS

A strategy can define which indicators to expose to the AI by adding an `INDICATOR_FIELDS` list:

```python
INDICATOR_FIELDS = [
    {"name": "rsi14",     "source": "l_rsi"},
    {"name": "cci30",     "source": "s_cci"},
    {"name": "oma_trend", "source": "oma_trend"},
    {"name": "vol_ok",    "source": "vol_ok"},
]
```

If not defined, the adapter auto-exposes all numeric/boolean arrays from the indicators dict.

---

## Creating a New VBT Strategy

### Step 1: Convert Pine Script to Python

Create `strategies/my_strategy.py` following the standard pattern. Use `strategies/qtp201_super_rsi_scalper.py` or `strategies/vector_v12.py` as templates.

```python
"""my_strategy.py — Single-file strategy module."""
import numpy as np
import pandas as pd
import vectorbtpro as vbt
from dataclasses import dataclass
from numba import njit

@dataclass
class MyParams:
    rsi_period: int = 14
    # ... all parameters from Pine Script input() calls
    tick_size: float = 0.10
    point_value: float = 100.0
    commission: float = 5.0
    initial_capital: float = 50000.0

def compute_indicators(data, params):
    close = data['close'].values.astype(np.float64)
    # ... compute all indicators
    return {'close': close, 'rsi': rsi_values, 'n': len(close), ...}

def compute_session_masks(index, params):
    # ... session/day/exclude logic
    return (long_session_ok, short_session_ok)

@njit
def detect_signals(close, rsi, n, ...):
    long_sig = np.zeros(n, dtype=np.bool_)
    short_sig = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        # ... entry logic
        pass
    return long_sig, short_sig

def detect_signals_from_dict(indicators, session, params):
    """Bridge for VBT adapter — unpacks dict for Numba function."""
    l_ok, s_ok = session
    return detect_signals(
        indicators['close'], indicators['rsi'], indicators['n'],
        # ... all args the Numba function needs
    )

class Strategy:
    PARAMS_CLASS = MyParams
    NAME = 'my_strategy'

    @staticmethod
    def compute_indicators(data, params):
        return compute_indicators(data, params)

    @staticmethod
    def compute_session(data, params):
        return compute_session_masks(data.index, params)
```

### Step 2: Drop It In

Place the file in `C:\Users\hoque\vectorBT Strategy\strategies\`. It will be auto-discovered on next startup.

### Step 3: Create a Bot

```sql
INSERT INTO ai_bots
  (bot_id, mode, account, strategy_tag, strategy_name, config_json, relay_user, enabled)
VALUES
  ('my_bot', 'python', 'SIM101', 'MyStrat', 'my_strategy',
   '{"rsi_period": 14, "instrument": "GC", "tick_size": 0.10, "point_value": 100.0}',
   'titon', 1);
```

Or via the management API:

```bash
curl -X POST http://localhost:8080/ai/bots -H "Content-Type: application/json" -d '{
  "bot_id": "my_bot",
  "mode": "python",
  "account": "SIM101",
  "strategy_tag": "MyStrat",
  "strategy_name": "my_strategy",
  "config_json": "{\"rsi_period\": 14, \"instrument\": \"GC\"}",
  "relay_user": "titon"
}'
```

### Step 4: Verify

1. Check the bot is registered: `python manage.py list-bots`
2. Watch logs for signal generation: `tail -f trade_relay.log | grep my_bot`
3. Start in dry-run mode (`AI_DRY_RUN=1`) to see decisions without executing

---

## Bot Configuration Reference

### ai_bots Table Columns

| Column | Type | Description |
|--------|------|-------------|
| `bot_id` | TEXT PK | Unique bot identifier |
| `mode` | TEXT | `'normal'`, `'copy'`, or `'python'` |
| `source_bot` | TEXT | For copy mode: relay_id of the bot to mirror |
| `relay_id` | TEXT | Source bot's relay_id |
| `account` | TEXT | NT8 account name (e.g., `'SIM101'`, `'Apex-12345'`) |
| `strategy_tag` | TEXT | Label sent to CrossTrade (appears in NT8) |
| `entry_prompt` | TEXT | Custom LLM system prompt for entry decisions |
| `manage_prompt` | TEXT | Custom LLM system prompt for exit/hold decisions |
| `strategy_name` | TEXT | For python mode: strategy registry name |
| `config_json` | TEXT | JSON parameters for python strategies |
| `relay_user` | TEXT | User ownership |
| `ai_model` | TEXT | LLM model (`'claude-sonnet-4-20250514'` or `'MiniMax-M2.7'`) |
| `enabled` | INT | 1 = active, 0 = disabled |

### config_json for VBT Strategies

The `config_json` field holds all strategy parameters as JSON. These are passed directly to the strategy's Params dataclass. Include these standard fields:

| Field | Required | Description |
|-------|----------|-------------|
| `instrument` | Yes | Instrument to trade (e.g., `"GC"`, `"MNQ1!"`) |
| `tick_size` | Yes | Min price increment (GC=0.10, ES=0.25, NQ=0.25) |
| `point_value` | Yes | Dollar value per 1.0 point (GC=100, ES=50, NQ=20) |
| `qty` | No | Contracts per trade (default: 1) |
| `strategy_type` | No | `"scalp"`, `"swing"`, `"trend"` (affects AI prompt calibration) |
| `bar_limit` | No | Bars of history to load (default: 500) |
| `hard_stop_ticks` | No | Safety net stop in ticks (default: 200) |
| *...strategy params* | | Any params accepted by the strategy's Params dataclass |

### Example: QTP201 Super RSI Scalper config_json

```json
{
  "instrument": "GC",
  "tick_size": 0.10,
  "point_value": 100.0,
  "strategy_type": "scalp",
  "hard_stop_ticks": 250,
  "l_st_period": 14,
  "l_st_mult": 3.0,
  "l_rsi_period": 14,
  "l_rsi_max": 60.0,
  "l_stop_ticks": 225,
  "l_target_ticks": 150,
  "l_use_trail": true,
  "l_trail_trigger": 90,
  "l_trail_offset": 15,
  "l_use_be": true,
  "l_be_trigger": 80,
  "l_be_offset": 20,
  "s_cci_period": 30,
  "s_cci_extreme": 130.0,
  "s_cci_exit": 50.0,
  "s_rsi_period": 12,
  "s_rsi_min": 49.0,
  "s_stop_ticks": 230,
  "s_target_ticks": 130,
  "oma_enable": true,
  "oma_len": 10,
  "oma_speed": 2.5,
  "oma_adaptive": true,
  "allow_longs": true,
  "allow_shorts": true
}
```

---

## Available Strategies

### VBT Strategies (auto-discovered from vectorBT Strategy/strategies/)

| Strategy | File | Description |
|----------|------|-------------|
| `qtp201_super_rsi_scalper` | `qtp201_super_rsi_scalper.py` | Long: Supertrend flip + RSI range + OMA bullish. Short: CCI extreme pullback + RSI gate + OMA bearish. Independent ATM per side. |
| `vector_v12` | `vector_v12.py` | Vector line cross/bounce/continuation patterns with adverse regime detection. |

### Native Strategies (in strategy_runner.py)

| Strategy | Description |
|----------|-------------|
| `ut_bot_trend` | UT Bot trend-follower with RSI + ADX + EMA(200) + Choppiness filter |

---

## AI Models

| Model | Provider | Best For |
|-------|----------|----------|
| `claude-sonnet-4-20250514` | Anthropic | Higher accuracy, higher cost |
| `MiniMax-M2.7` | MiniMax | Fast, cheap, good for scalping |

Set per-bot via `ai_model` column, or globally via `DEFAULT_AI_MODEL` in `ai_gate.py`.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Anthropic API key for Claude models |
| `MINIMAX_API_KEY` | — | MiniMax API key for M2.7 |
| `AI_DRY_RUN` | `"1"` | Set to `"0"` for live execution. `"1"` = paper mode (decisions logged, no orders sent) |
| `VBT_STRATEGY_DIR` | `C:\Users\hoque\vectorBT Strategy` | Path to VBT strategy project (for imports) |

---

## Data Flow: Bar Storage

1-second tick data from CrossTrade WebSocket is aggregated into bars:

| Table | Timeframe | Retention | Used By |
|-------|-----------|-----------|---------|
| `ai_bars` | 1-min | 14 days | Python strategies, indicator_engine |
| `ai_bars_5m` | 5-min | 60 days | HTF bias indicators |

Python strategies pull from `ai_bars` via `get_bars_as_dataframe(instrument, limit)`.

---

## Troubleshooting

### Strategy not found in registry

```
Bot qtp201_gc: strategy 'qtp201_super_rsi_scalper' not found in registry
```

**Fix:** Ensure the strategy file:
1. Is in `C:\Users\hoque\vectorBT Strategy\strategies\`
2. Has both a `Strategy` class and a `detect_signals_from_dict()` function
3. Can be imported without errors: `python -c "import strategies.qtp201_super_rsi_scalper"`

### Not enough bar data

```
Bot qtp201_gc: not enough bar data yet (12 bars)
```

**Fix:** The CVD WebSocket needs time to accumulate bars. Wait for at least 50 bars (50 minutes) of data. Check: `SELECT COUNT(*) FROM ai_bars WHERE instrument = 'GC'`

### Numba compilation slow on first run

First call to `detect_signals()` takes 5-30 seconds (Numba JIT compilation). Subsequent calls are instant. This is normal.

### LLM always disagrees

Check:
- Is the strategy generating signals? Add logging in `generate_signals()`
- Are indicator values reasonable? Check the payload in `ai_gate_logs.payload_json`
- Try a more permissive `entry_prompt` on the bot
- Try switching AI models (`claude-sonnet-4-20250514` vs `MiniMax-M2.7`)
