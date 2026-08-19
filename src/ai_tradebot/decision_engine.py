from __future__ import annotations

from collections.abc import Iterable

from ai_tradebot.core.config import RiskConfig
from ai_tradebot.core.schema import Action, AgentOpinion, Side, TradeDecision


class DecisionEngine:
    def __init__(self, risk_config: RiskConfig | None = None):
        self.risk_config = risk_config or RiskConfig()

    def decide(self, symbol: str, opinions: Iterable[AgentOpinion]) -> TradeDecision:
        opinions = list(opinions)
        long_score = 0.0
        short_score = 0.0
        risk_score = 0.0
        reasons: list[str] = []

        for opinion in opinions:
            weighted = opinion.confidence * abs(opinion.score)
            if opinion.signal == Side.LONG:
                long_score += weighted
            elif opinion.signal == Side.SHORT:
                short_score += weighted
            if opinion.agent == "risk_agent":
                risk_score = float(opinion.extras.get("risk_score", opinion.score))
            reasons.extend(f"{opinion.agent}: {r}" for r in opinion.reasons)

        total_direction = long_score + short_score
        confidence = 0.0 if total_direction == 0 else abs(long_score - short_score) / total_direction

        if risk_score > self.risk_config.max_risk_score:
            action = Action.REJECT
            reasons.append(f"risk score {risk_score:.2f} is above limit")
        elif confidence < self.risk_config.min_confidence_to_trade:
            action = Action.WATCH
            reasons.append(f"confidence {confidence:.2f} below trade threshold")
        elif long_score > short_score:
            action = Action.OPEN_LONG
        elif short_score > long_score:
            action = Action.OPEN_SHORT
        else:
            action = Action.HOLD

        leverage = 1 if risk_score > 0.45 else min(2, self.risk_config.max_leverage)
        position_pct = self.risk_config.max_position_pct * max(0.25, 1 - risk_score)
        return TradeDecision(
            symbol=symbol,
            action=action,
            confidence=round(confidence, 4),
            long_score=round(long_score, 4),
            short_score=round(short_score, 4),
            risk_score=round(risk_score, 4),
            position_pct=round(position_pct, 4),
            leverage=leverage,
            reasons=reasons,
        )

