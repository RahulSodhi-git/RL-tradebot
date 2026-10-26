FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY rl_lab/dashboard-requirements.txt /app/requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt \
    && useradd --create-home --uid 10001 tradebot
COPY --chown=tradebot:tradebot rl_lab/dashboard.py rl_lab/ppo_crypto.py /app/rl_lab/
RUN mkdir -p /app/rl_lab/reports /app/rl_lab/models /app/rl_lab/data \
    && chown -R tradebot:tradebot /app
USER tradebot
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "rl_lab/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
