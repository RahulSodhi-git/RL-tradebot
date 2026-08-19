from __future__ import annotations

from ai_tradebot.agents.base import Agent
from ai_tradebot.core.schema import AgentOpinion, MarketSnapshot, Side


class DirectionAgent(Agent):
    name = "direction_agent"

    def evaluate(self, snapshot: MarketSnapshot) -> AgentOpinion:
        f = snapshot.features
        score = (
            0.45 * _clip(f["ema_trend"] * 100, -1, 1)
            + 0.35 * _clip(f["return_3"] * 100, -1, 1)
            + 0.20 * _clip((f["rsi"] - 50) / 25, -1, 1)
        )
        signal = Side.LONG if score > 0.12 else Side.SHORT if score < -0.12 else Side.FLAT
        confidence = min(0.95, 0.50 + abs(score) * 0.45)
        return AgentOpinion(
            agent=self.name,
            symbol=snapshot.symbol,
            signal=signal,
            confidence=round(confidence, 4),
            score=round(score, 4),
            reasons=[f"direction score {score:+.3f}"],
        )


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

