FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install -r requirements.txt && pip install -e .

COPY app.py README.md pytest.ini ./
COPY tests ./tests

EXPOSE 8502

CMD ["streamlit", "run", "app.py", "--server.port", "8502", "--server.address", "0.0.0.0", "--server.headless", "true", "--browser.gatherUsageStats", "false"]

