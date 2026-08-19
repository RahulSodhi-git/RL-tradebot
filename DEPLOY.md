# Deploy Pipeline

This repo now deploys only the AI/ML prediction dashboard.

Old rule-based trading code has been removed from the deploy target. The dashboard does not place orders.

## Flow

```text
git push main
  -> GitHub Actions
  -> tests + compile check
  -> rsync to Oracle VM
  -> docker compose build
  -> restart Streamlit dashboard on port 8502
```

Oracle path:

```text
/home/ubuntu/rl-tradebot
```

Dashboard:

```text
http://161.118.176.44:8502
```

## Secrets

GitHub must have this repository secret:

```text
ORACLE_SSH_KEY
```

The Binance `.env` file is not committed. It must live on Oracle at:

```text
/home/ubuntu/rl-tradebot/.env
```

Supported original key names:

```env
BINANCE_API_KEY_FUT_TEST=
BINANCE_API_SECRET_FUT_TEST=
BINANCE_API_KEY_TEST=
BINANCE_API_SECRET_TEST=
USE_TESTNET=true
```

Supported new aliases:

```env
BINANCE_FUTURES_TESTNET_KEY=
BINANCE_FUTURES_TESTNET_SECRET=
BINANCE_USE_TESTNET=true
```

