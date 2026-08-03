# RL - Trade Bot image (used for BOTH the trading loop and the dashboard).
# ARM64-compatible (Oracle Ampere A1) and amd64. Code + runtime data are bind-mounted at /app by
# docker-compose, so this image only carries the Python environment.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl/tzdata are handy; the rest are pip deps. `docker` SDK lets the dashboard start/stop the bot
# container through the mounted Docker socket. altair powers the Analysis page.
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-trader.txt ./
RUN pip install -r requirements.txt -r requirements-trader.txt altair docker

# The actual code and runtime files come from the bind mount (./:/app) defined in docker-compose.yml.
# The command is provided per-service by docker-compose (streamlit for the dashboard, bot.py for the loop).
CMD ["python", "-c", "print('Provide a command via docker-compose (bot or dashboard).')"]
