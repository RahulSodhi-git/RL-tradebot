from __future__ import annotations

from ai_tradebot.agents.base import Agent
from ai_tradebot.core.schema import AgentOpinion, MarketSnapshot, Side


class FlowAgent(Agent):
    name = "flow_agent"

    def evaluate(self, snapshot: MarketSnapshot) -> AgentOpinion:
        f = snapshot.features
        volume_score = max(0.0, min(1.0, (f["volume_spike"] - 1.0) / 2.0))
        momentum = f["return_3"]
        signal = Side.LONG if momentum > 0 else Side.SHORT if momentum < 0 else Side.FLAT
        confidence = 0.45 + 0.45 * volume_score
        return AgentOpinion(
            agent=self.name,
            symbol=snapshot.symbol,
            signal=signal,
            confidence=round(confidence, 4),
            score=round(volume_score, 4),
            reasons=[f"volume spike x{f['volume_spike']:.2f}"],
        )

