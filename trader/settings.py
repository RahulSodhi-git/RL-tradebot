"""Load config from the .env file. Keys never live in code."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load tradebot/.env by absolute path, so it works no matter the launch directory
# (important when an MCP host like Claude Desktop starts this server).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

# Live keys (one key = both spot + futures on real Binance)
_LIVE_KEY = os.getenv("BINANCE_API_KEY", "")
_LIVE_SECRET = os.getenv("BINANCE_API_SECRET", "")

# On the unified demo (demo.binance.com), ONE key with all permissions covers BOTH
# spot & futures — just like live. Put it in BINANCE_API_KEY_TEST / _SECRET_TEST.
# The classic split testnets (testnet.binance.vision / testnet.binancefuture.com) have
# SEPARATE keys per market; if you use those, the *_SPOT_TEST / *_FUT_TEST vars below
# override the unified key for that market.
if USE_TESTNET:
    _TEST_KEY = os.getenv("BINANCE_API_KEY_TEST", "")
    _TEST_SECRET = os.getenv("BINANCE_API_SECRET_TEST", "")
    SPOT_KEY = os.getenv("BINANCE_API_KEY_SPOT_TEST", "") or _TEST_KEY
    SPOT_SECRET = os.getenv("BINANCE_API_SECRET_SPOT_TEST", "") or _TEST_SECRET
    FUT_KEY = os.getenv("BINANCE_API_KEY_FUT_TEST", "") or _TEST_KEY
    FUT_SECRET = os.getenv("BINANCE_API_SECRET_FUT_TEST", "") or _TEST_SECRET
else:
    SPOT_KEY = FUT_KEY = _LIVE_KEY
    SPOT_SECRET = FUT_SECRET = _LIVE_SECRET
