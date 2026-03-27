"""
VBT Strategy Adapter — Universal bridge between VectorBT strategies and AI Gate.

Wraps any VBT strategy module (vector_v12, qtp201, etc.) into the BaseStrategy
interface expected by strategy_runner.py. The adapter:

1. Builds the strategy's typed Params dataclass from bot config_json
2. Runs compute_indicators() on the bar DataFrame
3. Runs compute_session() for session/day/exclude masks
4. Runs detect_signals_from_dict() to get long/short booleans
5. Packages the last bar's signals + indicator values into an AI Gate payload

Usage:
    from vbt_adapter import register_vbt_strategy
    register_vbt_strategy("qtp201_super_rsi_scalper", "strategies.qtp201_super_rsi_scalper")

Then configure an ai_bot with mode='python', strategy_name='qtp201_super_rsi_scalper'.
"""

import sys
import os
import importlib
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dataclasses import fields as dataclass_fields

# Add vectorBT Strategy project to import path so strategies/ is importable
_VBT_PROJECT_DIR = os.environ.get(
    "VBT_STRATEGY_DIR",
    r"C:\Users\hoque\vectorBT Strategy"
)
if _VBT_PROJECT_DIR not in sys.path:
    sys.path.insert(0, _VBT_PROJECT_DIR)

from strategy_runner import BaseStrategy, STRATEGY_REGISTRY

logger = logging.getLogger("vbt_adapter")


# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT SPECS (mirrors ai_gate.INSTRUMENTS for tick_value lookup)
# ═══════════════════════════════════════════════════════════════════════════════

INSTRUMENTS = {
    "GC":  {"tick_size": 0.10, "point_value": 100.0},
    "MGC": {"tick_size": 0.10, "point_value": 10.0},
    "NQ":  {"tick_size": 0.25, "point_value": 20.0},
    "MNQ": {"tick_size": 0.25, "point_value": 2.0},
    "ES":  {"tick_size": 0.25, "point_value": 50.0},
    "MES": {"tick_size": 0.25, "point_value": 5.0},
    "CL":  {"tick_size": 0.01, "point_value": 1000.0},
    "MCL": {"tick_size": 0.01, "point_value": 100.0},
    "RTY": {"tick_size": 0.10, "point_value": 50.0},
    "M2K": {"tick_size": 0.10, "point_value": 5.0},
}


def _extract_root(instrument: str) -> str:
    """Extract root symbol: 'MNQ1!' -> 'MNQ', 'GC JUN26' -> 'GC'."""
    root = ""
    for ch in instrument.upper():
        if ch.isalpha():
            root += ch
        else:
            break
    return root


# ═══════════════════════════════════════════════════════════════════════════════
# TIME / SESSION HELPERS (for AI payload context)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_time_context(bar_time=None):
    """Compute session_bucket, day_of_week, bar_time_et, mins_to_maintenance."""
    if bar_time is None:
        bar_time = datetime.now(timezone.utc)

    # ET offset: approximate DST (March–Nov)
    month = bar_time.month
    is_dst = 3 <= month <= 10
    et_offset = 4 if is_dst else 5
    et_hour = (bar_time.hour - et_offset) % 24
    et_minute = bar_time.minute
    hhmm = et_hour * 100 + et_minute

    if hhmm < 1000:
        session_bucket = "open_drive"
    elif hhmm < 1130:
        session_bucket = "morning"
    elif hhmm < 1330:
        session_bucket = "midday_chop"
    elif hhmm < 1550:
        session_bucket = "afternoon"
    else:
        session_bucket = "close"

    # Minutes to CME maintenance (5pm ET = 1700)
    maint_start_mins = 17 * 60
    current_mins = et_hour * 60 + et_minute
    mins_to_maint = maint_start_mins - current_mins if maint_start_mins > current_mins else 0

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_of_week = dow_names[bar_time.weekday()]

    return {
        "session_bucket": session_bucket,
        "day_of_week": day_of_week,
        "bar_time_et": f"{et_hour:02d}:{et_minute:02d} ET",
        "bar_time_hhmm": hhmm,
        "bar_time_unix": int(bar_time.timestamp()),
        "mins_to_maintenance": mins_to_maint,
    }


def _safe(val):
    """Convert numpy value to Python scalar; NaN → 0."""
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.floating, float)):
        return 0.0 if np.isnan(val) else round(float(val), 4)
    if isinstance(val, (np.integer, int)):
        return int(val)
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# VBT STRATEGY ADAPTER
# ═══════════════════════════════════════════════════════════════════════════════

class VBTStrategyAdapter(BaseStrategy):
    """Universal adapter: wraps any VBT strategy module for AI Gate.

    The strategy module must have:
      - Strategy class with PARAMS_CLASS, compute_indicators(), compute_session()
      - detect_signals_from_dict(indicators, session, params) function
      - Optionally: INDICATOR_FIELDS list for AI payload mapping
    """

    def __init__(self, strategy_module):
        self.mod = strategy_module
        self.strat = strategy_module.Strategy
        self.params_class = self.strat.PARAMS_CLASS
        self.name = self.strat.NAME
        self.description = f"VBT strategy: {self.name}"

        # Optional: strategy can define which indicator fields to expose to AI
        self.indicator_fields = getattr(strategy_module, 'INDICATOR_FIELDS', None)

    def generate_signals(self, data: pd.DataFrame, params: dict) -> dict:
        """Run full strategy pipeline on bar data. Returns AI Gate payload dict.

        Args:
            data: OHLCV DataFrame (columns: open, high, low, close, volume).
                  Index: datetime. At least 200 bars.
            params: Strategy parameters from bot's config_json.

        Returns:
            dict matching AI Gate payload format with long_signal, short_signal,
            indicator values, OHLCV, and time context.
        """
        # 1. Build typed params — only pass fields the dataclass accepts
        valid_fields = {f.name for f in dataclass_fields(self.params_class)}
        filtered_params = {k: v for k, v in params.items() if k in valid_fields}
        strat_params = self.params_class(**filtered_params)

        # 2. Compute indicators
        indicators = self.strat.compute_indicators(data, strat_params)

        # 3. Compute session masks
        session = self.strat.compute_session(data, strat_params)

        # 4. Detect signals via strategy's wrapper function
        if not hasattr(self.mod, 'detect_signals_from_dict'):
            logger.error(f"Strategy {self.name} missing detect_signals_from_dict()")
            return {"long_signal": False, "short_signal": False}

        long_sig, short_sig = self.mod.detect_signals_from_dict(
            indicators, session, strat_params
        )

        # 5. Build AI Gate payload from last bar
        return self._build_payload(data, indicators, long_sig, short_sig, strat_params)

    def _build_payload(self, data, indicators, long_sig, short_sig, params):
        """Package last bar's signals + indicators into AI Gate payload dict."""
        n = len(data)
        i = n - 1  # last bar index

        # ── Core signals ──
        payload = {
            "long_signal": bool(long_sig[i]) if i < len(long_sig) else False,
            "short_signal": bool(short_sig[i]) if i < len(short_sig) else False,
        }

        # ── OHLCV ──
        payload["open"] = _safe(data['open'].iloc[i])
        payload["high"] = _safe(data['high'].iloc[i])
        payload["low"] = _safe(data['low'].iloc[i])
        payload["close"] = _safe(data['close'].iloc[i])
        payload["price"] = payload["close"]
        payload["volume"] = _safe(data.get('volume', pd.Series(0, index=data.index)).iloc[i])

        # ── Last 5 closes (for AI pattern recognition) ──
        start = max(0, n - 5)
        payload["last_5_closes"] = [_safe(data['close'].iloc[j]) for j in range(start, n)]

        # ── Strategy-specific indicators ──
        if self.indicator_fields:
            # Strategy defines explicit field mapping
            for field_def in self.indicator_fields:
                key = field_def["name"]           # AI payload key
                source = field_def["source"]      # indicators dict key
                idx = field_def.get("index", i)   # bar index (default: last)
                if source in indicators:
                    arr = indicators[source]
                    if isinstance(arr, np.ndarray) and len(arr) > idx:
                        payload[key] = _safe(arr[idx])
                    else:
                        payload[key] = _safe(arr) if not isinstance(arr, np.ndarray) else 0.0
        else:
            # Auto-expose: dump all numeric arrays from indicators at last bar
            skip_keys = {'close', 'high', 'low', 'open', 'n'}
            for key, arr in indicators.items():
                if key in skip_keys:
                    continue
                if isinstance(arr, np.ndarray):
                    if arr.dtype == np.bool_:
                        payload[key] = bool(arr[i]) if i < len(arr) else False
                    elif np.issubdtype(arr.dtype, np.integer):
                        payload[key] = int(arr[i]) if i < len(arr) else 0
                    elif np.issubdtype(arr.dtype, np.floating):
                        payload[key] = _safe(arr[i]) if i < len(arr) else 0.0

        # ── Time context ──
        bar_time = data.index[i] if hasattr(data.index[i], 'timestamp') else None
        if bar_time is not None:
            try:
                bar_time = bar_time.to_pydatetime()
                if bar_time.tzinfo is None:
                    bar_time = bar_time.replace(tzinfo=timezone.utc)
            except Exception:
                bar_time = None
        payload.update(_compute_time_context(bar_time))

        # ── Candle structure (for AI context) ──
        o, h, l, c = payload["open"], payload["high"], payload["low"], payload["close"]
        bar_range = max(h - l, 0.0001)
        payload["bar_body_pct"] = _safe(abs(c - o) / bar_range)
        payload["bar_upper_wick_pct"] = _safe((h - max(c, o)) / bar_range)
        payload["bar_lower_wick_pct"] = _safe((min(c, o) - l) / bar_range)
        payload["bar_is_bullish"] = c > o

        return payload


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def register_vbt_strategy(name: str, module_path: str):
    """Register a VBT strategy module in the AI Gate strategy registry.

    Args:
        name: Strategy name (used in ai_bots.strategy_name).
        module_path: Python import path (e.g., "strategies.qtp201_super_rsi_scalper").
    """
    try:
        module = importlib.import_module(module_path)
        adapter = VBTStrategyAdapter(module)
        STRATEGY_REGISTRY[name] = adapter
        logger.info(f"Registered VBT strategy: {name} (from {module_path})")
    except Exception as e:
        logger.error(f"Failed to register VBT strategy '{name}' from {module_path}: {e}")


def auto_discover_vbt_strategies():
    """Scan the VBT strategies directory and register any that have detect_signals_from_dict.

    Looks in the vectorBT Strategy/strategies/ folder for .py files with a Strategy class.
    """
    strategies_dir = os.path.join(_VBT_PROJECT_DIR, "strategies")
    if not os.path.isdir(strategies_dir):
        logger.warning(f"VBT strategies dir not found: {strategies_dir}")
        return

    for fname in os.listdir(strategies_dir):
        if not fname.endswith(".py") or fname.startswith("_") or fname == "window_result.py":
            continue

        module_name = fname[:-3]  # strip .py
        module_path = f"strategies.{module_name}"

        try:
            mod = importlib.import_module(module_path)
            if hasattr(mod, 'Strategy') and hasattr(mod, 'detect_signals_from_dict'):
                register_vbt_strategy(module_name, module_path)
        except Exception as e:
            logger.debug(f"Skipping {module_path}: {e}")


# Auto-discover on import
auto_discover_vbt_strategies()
