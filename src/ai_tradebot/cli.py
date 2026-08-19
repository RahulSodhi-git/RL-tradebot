from __future__ import annotations

import argparse
import json

from ai_tradebot.agents.direction import DirectionAgent
from ai_tradebot.agents.flow import FlowAgent
from ai_tradebot.agents.regime import RegimeAgent
from ai_tradebot.agents.risk import RiskAgent
from ai_tradebot.agents.volatility import VolatilityAgent
from ai_tradebot.data.demo import make_demo_ohlcv
from ai_tradebot.decision_engine import DecisionEngine
from ai_tradebot.features.builder import build_snapshot
from ai_tradebot.training.shadow_logger import ShadowLogger


def run_demo() -> dict:
    df = make_demo_ohlcv()
    snapshot = build_snapshot(df, symbol="DEMOUSDT", timeframe="1h")
    agents = [RegimeAgent(), DirectionAgent(), VolatilityAgent(), FlowAgent(), RiskAgent()]
    opinions = [agent.evaluate(snapshot) for agent in agents]
    decision = DecisionEngine().decide(snapshot.symbol, opinions)
    ShadowLogger().log(snapshot, opinions, decision)
    return {
        "snapshot": {
            "symbol": snapshot.symbol,
            "timeframe": snapshot.timeframe,
            "features": snapshot.features,
        },
        "opinions": [
            {
                "agent": opinion.agent,
                "signal": opinion.signal.value,
                "confidence": opinion.confidence,
                "score": opinion.score,
                "reasons": opinion.reasons,
            }
            for opinion in opinions
        ],
        "decision": decision.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="run a local demo decision")
    args = parser.parse_args()
    if args.demo:
        print(json.dumps(run_demo(), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

