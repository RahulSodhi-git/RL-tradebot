# Laptop dashboard

Copy the `rl_lab` folder to your laptop, including `models`, `reports` and
`data/processed_1h/metadata.json`. Do not copy the Linux `.venv` to Windows.
Install Python 3.11 or 3.12, then open PowerShell in `rl_lab`:

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r dashboard-requirements.txt
$env:BINANCE_TESTNET_API_KEY = 'YOUR_TEST_KEY'
$env:BINANCE_TESTNET_SECRET = 'YOUR_TEST_SECRET'
.\.venv\Scripts\python -m streamlit run dashboard.py --server.address 127.0.0.1
```

Open http://localhost:8501. Select the environment matching your demo keys and
enable account monitoring. Environment variables last for this PowerShell session;
keep credentials private and never commit them. The dashboard works without keys
for viewing historical reports. Keep the terminal running and disable laptop sleep.
Ctrl+C stops the dashboard. Closing the browser does not close exchange positions.

This dashboard is read-only. It refreshes account positions and orders every
30–120 seconds. It displays saved model scans and simulation reports separately.
It does not start the trading engine, schedule scans, or manage exits. Automatic
execution still needs implementation and verification before launch. Do not use
the experimental `--execute` command as an unattended trading service.

Binance setup: use a Futures demo/testnet account, test USDT and matching test
API credentials with futures access. Do not use production API credentials.
If connection fails, check the selected endpoint, API permissions, IP restrictions,
system clock and network access. This cloud environment previously received
HTTP 403/451 from Binance; laptop connectivity must be checked separately.
