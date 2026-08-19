from ai_tradebot.cli import run_demo
from ai_tradebot.core.schema import Action


def test_demo_decision_shape():
    result = run_demo()
    decision = result["decision"]
    assert decision["symbol"] == "DEMOUSDT"
    assert decision["action"] in {action.value for action in Action}
    assert 0 <= decision["confidence"] <= 1
    assert 0 <= decision["risk_score"] <= 1

