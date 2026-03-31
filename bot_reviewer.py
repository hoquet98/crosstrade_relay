"""
Bot Reviewer — AI-powered bot performance analysis and optimization.

Reviews trade history, analyzes indicator patterns at entry for wins vs losses,
and generates actionable recommendations for improving bot configuration.

Runs on-demand or scheduled (every 4 hours during market hours).
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import database as db

logger = logging.getLogger("trade_relay")


# ══════════════════════════════════════════════════════════════════════════════
# REVIEW ENGINE
# ══════════════════════════════════════════════════════════════════════════════

async def review_bot(bot_id: str, max_trades: int = 50) -> dict:
    """Review a bot's performance and generate recommendations.

    Args:
        bot_id: Bot to review
        max_trades: Max closed trades to analyze

    Returns:
        dict with stats, analysis, and recommendations
    """
    import ai_gate

    # 1. Load bot config
    bot = ai_gate.get_bot(bot_id)
    if not bot:
        return {"error": f"Bot '{bot_id}' not found"}

    # 2. Load closed trades with snapshots
    trades = ai_gate.get_trades(bot.get("relay_user"), relay_id=bot_id, limit=max_trades)
    closed_trades = [t for t in trades if t.get("status") == "closed"]

    if len(closed_trades) < 5:
        return {
            "bot_id": bot_id,
            "error": "Not enough closed trades for review (need at least 5)",
            "trade_count": len(closed_trades)
        }

    # 3. Compute performance stats
    stats = _compute_stats(closed_trades)

    # 4. Analyze indicator patterns (wins vs losses)
    patterns = _analyze_patterns(closed_trades)

    # 5. Build review prompt for AI
    review_prompt = _build_review_prompt(bot, stats, patterns, closed_trades)

    # 6. Call AI for analysis
    analysis, recommendations, latency_ms = await _call_review_ai(review_prompt, bot)

    # 7. Store review
    review_id = _save_review(bot_id, stats, analysis, recommendations)

    return {
        "bot_id": bot_id,
        "review_id": review_id,
        "stats": stats,
        "patterns": patterns,
        "analysis": analysis,
        "recommendations": recommendations,
        "latency_ms": latency_ms,
        "trade_count": len(closed_trades),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STATS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _compute_stats(trades: list) -> dict:
    """Compute performance statistics from closed trades."""
    if not trades:
        return {}

    wins = [t for t in trades if (t.get("dollar_pnl") or 0) > 0]
    losses = [t for t in trades if (t.get("dollar_pnl") or 0) <= 0]

    total_pnl = sum(t.get("dollar_pnl", 0) or 0 for t in trades)
    win_pnl = sum(t.get("dollar_pnl", 0) or 0 for t in wins)
    loss_pnl = sum(t.get("dollar_pnl", 0) or 0 for t in losses)

    avg_win = win_pnl / len(wins) if wins else 0
    avg_loss = loss_pnl / len(losses) if losses else 0
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    # Average bars held
    avg_bars = sum(t.get("bars_held", 0) or 0 for t in trades) / len(trades) if trades else 0

    # Profit factor
    gross_profit = sum(t.get("dollar_pnl", 0) or 0 for t in wins)
    gross_loss = abs(sum(t.get("dollar_pnl", 0) or 0 for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Exit reason distribution
    exit_reasons = {}
    for t in trades:
        reason = t.get("exit_reason", "unknown") or "unknown"
        # Simplify reason to category
        if "HARD STOP" in reason:
            cat = "hard_stop"
        elif "TIME STOP" in reason:
            cat = "time_stop"
        elif "EARLY EXIT" in reason:
            cat = "early_exit"
        elif "AI EXIT" in reason or "Exit score" in reason:
            cat = "ai_exit"
        elif "CONDITION" in reason:
            cat = "condition_exit"
        elif "SYNC" in reason:
            cat = "sync_exit"
        else:
            cat = "other"
        exit_reasons[cat] = exit_reasons.get(cat, 0) + 1

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_bars_held": round(avg_bars, 1),
        "profit_factor": round(profit_factor, 2),
        "exit_reasons": exit_reasons,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _analyze_patterns(trades: list) -> dict:
    """Analyze indicator patterns at entry for wins vs losses."""
    wins = [t for t in trades if (t.get("dollar_pnl") or 0) > 0]
    losses = [t for t in trades if (t.get("dollar_pnl") or 0) <= 0]

    # Extract snapshots
    win_snapshots = []
    loss_snapshots = []
    for t in wins:
        snap = t.get("entry_snapshot")
        if snap:
            if isinstance(snap, str):
                try:
                    snap = json.loads(snap)
                except json.JSONDecodeError:
                    continue
            win_snapshots.append(snap)

    for t in losses:
        snap = t.get("entry_snapshot")
        if snap:
            if isinstance(snap, str):
                try:
                    snap = json.loads(snap)
                except json.JSONDecodeError:
                    continue
            loss_snapshots.append(snap)

    if not win_snapshots and not loss_snapshots:
        return {"note": "No entry snapshots available for pattern analysis"}

    # Compare key indicators between wins and losses
    key_indicators = [
        "rsi14", "stoch_rsi_k", "wt1", "cci20", "atr14", "bb_pctb",
        "er_val", "vol_ratio", "bar_body_pct", "bull_confluence",
        "bear_confluence", "exit_score", "cvd_1m_delta",
    ]

    comparisons = {}
    for ind in key_indicators:
        win_vals = [s.get(ind) for s in win_snapshots if s.get(ind) is not None and isinstance(s.get(ind), (int, float))]
        loss_vals = [s.get(ind) for s in loss_snapshots if s.get(ind) is not None and isinstance(s.get(ind), (int, float))]

        if win_vals and loss_vals:
            comparisons[ind] = {
                "win_avg": round(sum(win_vals) / len(win_vals), 2),
                "loss_avg": round(sum(loss_vals) / len(loss_vals), 2),
                "win_count": len(win_vals),
                "loss_count": len(loss_vals),
            }

    # Boolean indicator analysis
    bool_indicators = [
        "f_zlema_trend_bull", "f_zlema_trend_bear", "f_trending",
        "f_squeeze_off", "price_above_ema20", "price_above_ema50",
        "stoch_in_ob", "stoch_in_os", "bar_is_bullish",
    ]

    bool_comparisons = {}
    for ind in bool_indicators:
        win_true = sum(1 for s in win_snapshots if s.get(ind) is True)
        loss_true = sum(1 for s in loss_snapshots if s.get(ind) is True)
        win_total = len(win_snapshots)
        loss_total = len(loss_snapshots)

        if win_total > 0 and loss_total > 0:
            bool_comparisons[ind] = {
                "win_pct": round(win_true / win_total * 100, 1),
                "loss_pct": round(loss_true / loss_total * 100, 1),
            }

    # Session analysis
    session_stats = {}
    for t in trades:
        snap = t.get("entry_snapshot")
        if snap:
            if isinstance(snap, str):
                try:
                    snap = json.loads(snap)
                except json.JSONDecodeError:
                    continue
            bucket = snap.get("session_bucket", "unknown")
            if bucket not in session_stats:
                session_stats[bucket] = {"wins": 0, "losses": 0, "pnl": 0}
            if (t.get("dollar_pnl") or 0) > 0:
                session_stats[bucket]["wins"] += 1
            else:
                session_stats[bucket]["losses"] += 1
            session_stats[bucket]["pnl"] += t.get("dollar_pnl", 0) or 0

    return {
        "numeric_indicators": comparisons,
        "boolean_indicators": bool_comparisons,
        "session_stats": session_stats,
        "win_snapshots_count": len(win_snapshots),
        "loss_snapshots_count": len(loss_snapshots),
    }


# ══════════════════════════════════════════════════════════════════════════════
# AI REVIEW CALL
# ══════════════════════════════════════════════════════════════════════════════

REVIEW_SYSTEM_PROMPT = """You are a quantitative trading analyst reviewing an algorithmic trading bot's performance. You receive performance statistics, indicator patterns at entry for winning vs losing trades, and the bot's current configuration.

Your job is to analyze the data and provide actionable recommendations to improve the bot's performance.

## Analysis Guidelines
1. Look for indicator values that differ significantly between wins and losses
2. Identify session times that perform poorly
3. Check if exit reasons suggest the bot is holding too long or cutting too short
4. Evaluate if the bot's conditions/prompts could be tightened

## Recommendation Types
For each recommendation, specify:
- type: "condition_change", "prompt_change", "indicator_change", "param_change", or "strategy_change"
- description: what to change and why
- priority: "high", "medium", or "low"

## Response Format
Respond with EXACTLY this JSON format:

```json
{
  "summary": "2-3 sentence overall assessment",
  "recommendations": [
    {
      "type": "condition_change",
      "description": "Add entry condition: rsi14 < 65 — RSI above 65 at entry correlates with 70% of losses",
      "priority": "high"
    }
  ]
}
```"""


def _build_review_prompt(bot: dict, stats: dict, patterns: dict, trades: list) -> str:
    """Build the user message for the review AI call."""
    parts = []

    parts.append(f"## Bot Configuration")
    parts.append(f"- Bot ID: {bot.get('bot_id')}")
    parts.append(f"- Mode: {bot.get('mode')}")
    parts.append(f"- Gate Mode: {bot.get('gate_mode', 'ai_only')}")
    parts.append(f"- Instrument: {bot.get('account')}")
    if bot.get("entry_conditions"):
        parts.append(f"- Entry Conditions: {bot['entry_conditions']}")
    if bot.get("exit_conditions"):
        parts.append(f"- Exit Conditions: {bot['exit_conditions']}")

    parts.append(f"\n## Performance Stats")
    parts.append(json.dumps(stats, indent=2))

    parts.append(f"\n## Indicator Patterns (Wins vs Losses)")
    if patterns.get("numeric_indicators"):
        parts.append("### Numeric Indicators")
        for ind, vals in patterns["numeric_indicators"].items():
            diff = vals["win_avg"] - vals["loss_avg"]
            parts.append(f"  {ind}: win_avg={vals['win_avg']}, loss_avg={vals['loss_avg']} (diff={diff:+.2f})")

    if patterns.get("boolean_indicators"):
        parts.append("### Boolean Indicators")
        for ind, vals in patterns["boolean_indicators"].items():
            diff = vals["win_pct"] - vals["loss_pct"]
            parts.append(f"  {ind}: win_pct={vals['win_pct']}%, loss_pct={vals['loss_pct']}% (diff={diff:+.1f}%)")

    if patterns.get("session_stats"):
        parts.append("### Session Performance")
        for bucket, vals in patterns["session_stats"].items():
            total = vals["wins"] + vals["losses"]
            wr = vals["wins"] / total * 100 if total > 0 else 0
            parts.append(f"  {bucket}: {vals['wins']}W/{vals['losses']}L ({wr:.0f}% WR) PnL=${vals['pnl']:.2f}")

    # Last 5 trade summaries
    parts.append(f"\n## Recent Trades (last 5)")
    for t in trades[:5]:
        pnl = t.get("dollar_pnl", 0) or 0
        direction = t.get("direction", "?")
        bars = t.get("bars_held", 0) or 0
        reason = (t.get("exit_reason", "") or "")[:60]
        parts.append(f"  {direction.upper()} ${pnl:+.2f} ({bars} bars) — {reason}")

    return "\n".join(parts)


async def _call_review_ai(prompt: str, bot: dict) -> tuple:
    """Call AI for bot review. Returns (analysis, recommendations, latency_ms)."""
    import ai_gate

    model = bot.get("ai_model") or ai_gate.DEFAULT_AI_MODEL
    # Use a higher max_tokens for reviews
    mcfg = ai_gate._get_model_config(model)

    start = datetime.now(timezone.utc)

    try:
        import httpx
        max_tokens = 8000 if "minimax" in mcfg["model"].lower() else 1000

        request_body = {
            "model": mcfg["model"],
            "max_tokens": max_tokens,
            "system": REVIEW_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}]
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                mcfg["base_url"],
                json=request_body,
                headers={
                    "x-api-key": mcfg["api_key"],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
            )

        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)

        if resp.status_code != 200:
            return f"API error: {resp.status_code}", [], latency_ms

        data = resp.json()
        raw_text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                raw_text += block.get("text", "")

        # Parse JSON response
        try:
            # Extract JSON from response (might be wrapped in markdown code blocks)
            json_text = raw_text
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0]
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0]

            result = json.loads(json_text.strip())
            analysis = result.get("summary", raw_text)
            recommendations = result.get("recommendations", [])
        except (json.JSONDecodeError, IndexError):
            analysis = raw_text
            recommendations = []

        return analysis, recommendations, latency_ms

    except Exception as e:
        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        logger.error(f"Review AI call failed: {e}")
        return f"Error: {e}", [], latency_ms


# ══════════════════════════════════════════════════════════════════════════════
# STORAGE
# ══════════════════════════════════════════════════════════════════════════════

def _save_review(bot_id: str, stats: dict, analysis: str, recommendations: list) -> int:
    """Save review to database."""
    conn = db.get_connection()
    cursor = conn.execute("""
        INSERT INTO ai_reviews (bot_id, review_type, trade_count, win_rate, total_pnl,
                                analysis, recommendations)
        VALUES (?, 'on_demand', ?, ?, ?, ?, ?)
    """, (
        bot_id,
        stats.get("total_trades", 0),
        stats.get("win_rate", 0),
        stats.get("total_pnl", 0),
        analysis,
        json.dumps(recommendations),
    ))
    review_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return review_id


def get_reviews(bot_id: str, limit: int = 10) -> list:
    """Get past reviews for a bot."""
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT * FROM ai_reviews WHERE bot_id = ? ORDER BY id DESC LIMIT ?",
        (bot_id, limit)
    ).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("recommendations") and isinstance(d["recommendations"], str):
            try:
                d["recommendations"] = json.loads(d["recommendations"])
            except json.JSONDecodeError:
                pass
        results.append(d)
    return results
