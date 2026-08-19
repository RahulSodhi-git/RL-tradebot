# AI Tradebot Lab

Fresh ML/RL-first trading research system.

This project starts in **research and shadow mode only**. It does not place live orders.
The goal is to build reliable market understanding before any execution code exists.

## Step 1: Foundation

The first version contains:

- Typed market/trade schemas.
- Feature engineering from OHLCV candles.
- Specialist agents:
  - regime agent
  - direction agent
  - volatility agent
  - flow agent
  - risk agent
- A decision engine that combines agent opinions.
- A shadow logger for training data.
- A simple CLI smoke run.

## Project Layout

```text
src/ai_tradebot/
  agents/           Specialist agents.
  core/             Shared schemas and config.
  data/             Data loading helpers.
  features/         Feature engineering.
  training/         Dataset and shadow logging tools.
  decision_engine.py
  cli.py
tests/
  test_smoke.py
```

The old rule-based bot has been preserved in `_legacy_rule_bot/`.

## Run

```powershell
python -m pip install -r requirements.txt
python -m ai_tradebot.cli --demo
python -m pytest
```

Run the Streamlit research dashboard:

```powershell
streamlit run app.py
```

## Deployment

Pushes to `main` deploy the ML dashboard to Oracle through GitHub Actions.
See `DEPLOY.md`.

If running directly from the repo without installing the package:

```powershell
$env:PYTHONPATH = "src"
python -m ai_tradebot.cli --demo
```

## Design Rule

Models advise. Risk controls decide what is allowed. Execution comes later.
