# Dataset plan for a multi-timeframe crypto PPO agent

_Checked 2026-10-26. Scope: research/training data, not an instruction to trade._

## Recommendation

Start with one canonical **5-minute UTC timeline** for USD-M perpetuals and derive 15m/1h/4h/1d bars from it. At every simulated decision, expose only bars and auxiliary intervals whose close timestamp is no later than that decision. Store raw downloads unchanged, plus normalized Parquet tables and an ingestion manifest/checksum.

PPO is on-policy; “offline PPO” here means that the agent repeatedly interacts with a historical market simulator. It is not trained directly from rows as if they were labeled examples.

## Data needed

| Priority | Dataset | Fields/use | First-party source |
|---|---|---|---|
| Required | USD-M futures 5m klines | Open/close time, OHLC, base and quote volume, trade count, taker-buy volumes. Derive every slower timeframe from these same bars. | [Binance Public Data](https://data.binance.vision/) and its [official repository/format documentation](https://github.com/binance/binance-public-data); recent gaps can be filled with [`GET /fapi/v1/klines`](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data). |
| Required | Funding rates | Funding cost must enter portfolio equity and reward; also useful as a positioning feature. | [`GET /fapi/v1/fundingRate`](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History). |
| Required | Mark-price/index data | Use mark price—not only last trade—for unrealized P&L and liquidation approximations. | [`GET /fapi/v1/markPriceKlines`](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price-Kline-Candlestick-Data) and [`GET /fapi/v1/indexPriceKlines`](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Index-Price-Kline-Candlestick-Data). |
| Required metadata | Contract/listing and filters | Avoid trading before listing/after delisting; capture tick size, step size and minimum quantity/notional. Historical snapshots should be retained going forward. | [`GET /fapi/v1/exchangeInfo`](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information). |
| High value | Open-interest history | Positioning/participation feature. Store value, provider, interval close time and fetch time. | Binance [`GET /futures/data/openInterestHist`](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics), or Coinalyze [`open-interest-history`](https://api.coinalyze.net/v1/doc/). Binance's official connector documents that this endpoint exposes only the latest 30 days, so long-history training requires a historical provider or a recorder running now ([official connector source](https://github.com/binance/binance-futures-connector-python/blob/main/binance/um_futures/market.py)). |
| High value | Long/short liquidation history | Forced-flow feature; aggregate long and short liquidations separately by closed 5m interval. Do not interpret liquidation volume alone as trade direction. | Coinalyze [`liquidation-history`](https://api.coinalyze.net/v1/doc/). Verify the account plan's retention, interval and request limits before designing the dataset. |
| Later | Trades/order-book proxies | Calibrate spread, taker imbalance and slippage. Full historical order books are large and unnecessary for the first model. | Binance Public Data provides futures trades/aggregate trades as documented in its [official repository](https://github.com/binance/binance-public-data); collect live book snapshots if finer execution modeling is later required. |

Binance says its public archive is organized into daily and monthly files, daily files arrive the next day, monthly files on the first Monday of the month, and companion checksum files are available. The same official documentation identifies USD-M futures kline files as data from `/fapi/v1/klines` and lists their columns.

## Minimum viable acquisition

1. Choose 20–50 liquid USD-M perpetuals, but retain inactive/delisted contracts where data exists to reduce survivorship bias. Include several market regimes; three months is insufficient. A practical first target is at least 2021 onward, subject to each contract's listing date.
2. Download Binance monthly 5m futures kline archives and their checksum files; patch only the recent tail/gaps through the REST API. Download funding and mark-price history for the same symbols and dates.
3. Obtain Coinalyze 5m OI and liquidation history for the same symbol/time range. Map every Coinalyze instrument explicitly to the Binance USD-M contract; never mix spot, inverse/COIN-M or another exchange into the same series accidentally.
4. If sufficient Coinalyze history is unavailable under the chosen plan, start a recorder immediately. Train the first baseline without those features and mark them missing; do **not** replace unknown OI/liquidations with zero.
5. Validate uniqueness, monotonic timestamps, expected 5m spacing, OHLC invariants, nonnegative volume, funding timestamps and provider gaps. Preserve a boolean availability mask for every auxiliary feature.
6. Freeze chronological splits before modeling—for example train, then validation, then a final untouched test period. Fit scalers only on the training split and use rolling walk-forward evaluation.

## Minimal normalized tables

- `bars_5m`: `symbol, open_time, close_time, open, high, low, close, volume, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume`
- `funding`: `symbol, funding_time, funding_rate, mark_price_if_available`
- `mark_5m`: `symbol, open_time, close_time, open, high, low, close`
- `open_interest_5m`: `symbol, close_time, oi_contracts, oi_usd, provider, available`
- `liquidations_5m`: `symbol, close_time, long_liq_usd, short_liq_usd, provider, available`
- `contracts`: `symbol, valid_from, valid_to, status, tick_size, step_size, min_qty, min_notional`

## Important limitations

- A current `exchangeInfo` response is not historical metadata. Archive snapshots from now onward and infer older listing windows conservatively from first/last valid market records.
- Public REST limits make archive downloads preferable for bulk candles. Respect response ordering, pagination, weight limits and provider quotas.
- Data availability itself can leak future information. Do not construct the historical universe from today's surviving symbols only.
- Coinalyze OI/liquidations should be model inputs, not hard requirements until coverage and timestamp semantics have been audited.
- Binance Demo/Testnet validates integration and order handling; its fills are not a replacement for historical training or realistic slippage testing.

## Primary sources

- Binance, [Public Data repository and schemas](https://github.com/binance/binance-public-data)
- Binance, [USD-M Futures market-data REST documentation](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api)
- Binance, [official futures connector implementation](https://github.com/binance/binance-futures-connector-python/blob/main/binance/um_futures/market.py)
- Coinalyze, [API documentation](https://api.coinalyze.net/v1/doc/)
