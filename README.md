# AI Market Prediction Lab

Fresh ML-first trading research dashboard.

This project is currently **prediction only**. It does not place live orders.
The first serious experiment is deliberately small:

```text
BTCUSDT movement -> predict ZECUSDT direction
```

## Current Experiment

The dashboard answers simple questions:

- Did BTC move first?
- Did ZEC move with BTC or against BTC?
- Did ZEC move more strongly than BTC?
- Can a simple model predict if ZEC goes UP, DOWN, or NEUTRAL after a few candles?

## Dashboard Flow

1. Check BTC and ZEC candle data from Binance.
2. Build a BTC -> ZEC relationship dataset.
3. Train simple ML models.
4. View ZEC candles with predictions and correctness.

## Features Used

The model uses ZEC candle features plus BTC relationship features:

- ZEC recent returns
- ZEC RSI, EMA distance, volatility, volume spike
- BTC recent returns
- ZEC move minus BTC move
- ZEC/BTC rolling correlation
- ZEC/BTC rolling beta
- previous BTC movement as a lead signal

## Run Locally

```powershell
python -m pip install -r requirements.txt
pip install -e .
streamlit run app.py
```

Run tests:

```powershell
pytest -q
```

## Deployment

Pushes to `main` deploy the dashboard to Oracle through GitHub Actions.

The Oracle deploy wipes the old `/home/ubuntu/rl-tradebot` app files first, then copies the current clean dashboard and rebuilds Docker.
