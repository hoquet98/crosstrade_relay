"""WindowResult — standard result dataclass for all strategies."""

from dataclasses import dataclass


@dataclass
class WindowResult:
    """Result of running one combo on one time window."""
    combo_index: int
    window_start: str
    window_end: str
    window_type: str
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade: float = 0
    best_trade: float = 0
    worst_trade: float = 0
    bars: int = 0
    duration_ms: int = 0
