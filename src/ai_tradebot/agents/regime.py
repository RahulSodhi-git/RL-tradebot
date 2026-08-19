from __future__ import annotations

from ai_tradebot.agents.base import Agent
from ai_tradebot.core.schema import AgentOpinion, MarketSnapshot, Side


class RegimeAgent(Agent):
    name = "regime_agent"

    def evaluate(self, snapshot: MarketSnapshot) -> AgentOpinion:
        f = snapshot.features
        trend_strength = abs(f["ema_trend"]) * 100
        vol = f["volatility"] * 100
        score = min(1.0, trend_strength / 1.5) - min(0.6, vol / 6.0)
        signal = Side.LONG if f["ema_trend"] > 0 else Side.SHORT if f["ema_trend"] < 0 else Side.FLAT
        confidence = 0.45 + min(0.45, abs(score) * 0.35)
        regime = "trend" if trend_strength >= 0.35 else "range"
        risk = "high_volatility" if vol >= 2.0 else "normal_volatility"
        return AgentOpinion(
            agent=self.name,
            symbol=snapshot.symbol,
            signal=signal if regime == "trend" else Side.FLAT,
            confidence=round(confidence, 4),
            score=round(score, 4),
            reasons=[f"{regime}, {risk}"],
            extras={"regime": regime, "volatility_regime": risk},
        )

