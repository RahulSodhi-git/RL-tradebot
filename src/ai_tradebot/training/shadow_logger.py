from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ai_tradebot.core.schema import AgentOpinion, MarketSnapshot, TradeDecision


class ShadowLogger:
    def __init__(self, path: str | Path = "data/shadow/observations.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        snapshot: MarketSnapshot,
        opinions: Iterable[AgentOpinion],
        decision: TradeDecision,
    ) -> None:
        row = {
            "snapshot": _encode_snapshot(snapshot),
            "opinions": [_encode_opinion(opinion) for opinion in opinions],
            "decision": decision.to_dict(),
            "labels": {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _encode_snapshot(snapshot: MarketSnapshot) -> dict:
    row = asdict(snapshot)
    row["timestamp"] = snapshot.timestamp.isoformat()
    return row


def _encode_opinion(opinion: AgentOpinion) -> dict:
    row = asdict(opinion)
    row["signal"] = opinion.signal.value
    return row

