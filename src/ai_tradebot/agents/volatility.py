from __future__ import annotations

from ai_tradebot.agents.base import Agent
from ai_tradebot.core.schema import AgentOpinion, MarketSnapshot, Side


class VolatilityAgent(Agent):
    name = "volatility_agent"

    def evaluate(self, snapshot: MarketSnapshot) -> AgentOpinion:
        vol = snapshot.features["volatility"] * 100
        atr_pct = snapshot.features["atr_pct"] * 100
        risk_score = min(1.0, (vol + atr_pct) / 8.0)
        signal = Side.FLAT
        confidence = 0.50 + min(0.40, risk_score * 0.40)
        return AgentOpinion(
            agent=self.name,
            symbol=snapshot.symbol,
            signal=signal,
            confidence=round(confidence, 4),
            score=round(risk_score, 4),
            reasons=[f"volatility risk {risk_score:.3f}"],
            extras={"risk_component": risk_score},
        )

