from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class Action(str, Enum):
    HOLD = "hold"
    WATCH = "watch"
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    REJECT = "reject"


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    timestamp: datetime
    timeframe: str
    features: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentOpinion:
    agent: str
    symbol: str
    signal: Side
    confidence: float
    score: float
    reasons: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeDecision:
    symbol: str
    action: Action
    confidence: float
    long_score: float
    short_score: float
    risk_score: float
    position_pct: float
    leverage: int
    reasons: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["action"] = self.action.value
        row["created_at"] = self.created_at.isoformat()
        return row

