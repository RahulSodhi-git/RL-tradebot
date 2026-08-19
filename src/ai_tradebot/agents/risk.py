from __future__ import annotations

from ai_tradebot.agents.base import Agent
from ai_tradebot.core.config import RiskConfig
from ai_tradebot.core.schema import AgentOpinion, MarketSnapshot, Side


class RiskAgent(Agent):
    name = "risk_agent"

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    def evaluate(self, snapshot: MarketSnapshot) -> AgentOpinion:
        f = snapshot.features
        risk = 0.0
        reasons = []

        if f["volatility"] > 0.025:
            risk += 0.35
            reasons.append("high rolling volatility")
        if f["atr_pct"] > 0.04:
            risk += 0.25
            reasons.append("wide ATR")
        if f["volume_spike"] > 4.0:
            risk += 0.20
            reasons.append("extreme volume spike")
        if 45 <= f["rsi"] <= 55:
            risk += 0.10
            reasons.append("weak directional RSI")

        risk = min(1.0, risk)
        signal = Side.FLAT
        confidence = 0.50 + risk * 0.45
        return AgentOpinion(
            agent=self.name,
            symbol=snapshot.symbol,
            signal=signal,
            confidence=round(confidence, 4),
            score=round(risk, 4),
            reasons=reasons or ["risk acceptable"],
            extras={"risk_score": risk, "reject": risk > self.config.max_risk_score},
        )

