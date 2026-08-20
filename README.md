# AI Market Prediction Lab

Fresh start.

The first version only checks the relationship between BTC and ZEC candles.
It does not train models yet. It does not place trades.

## What It Shows

- ZEC candlestick chart
- BTC vs ZEC percentage movement
- how often BTC and ZEC move in the same direction
- simple ZEC/BTC beta
- simple ZEC/BTC correlation

## Run Locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Push to `main`. GitHub Actions deploys to Oracle on port `8502`.
