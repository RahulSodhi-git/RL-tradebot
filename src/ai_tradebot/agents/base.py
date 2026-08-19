from __future__ import annotations

from abc import ABC, abstractmethod

from ai_tradebot.core.schema import AgentOpinion, MarketSnapshot


class Agent(ABC):
    name: str

    @abstractmethod
    def evaluate(self, snapshot: MarketSnapshot) -> AgentOpinion:
        """Return one specialist opinion for a market snapshot."""

