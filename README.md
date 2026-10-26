# RL Tradebot dashboard

Streamlit monitoring dashboard and experimental PPO training pipeline for Binance
Futures demo. The dashboard is read-only. Automatic trading, continuous learning
and unattended exit management are not complete or verified.

## Run on your laptop

Install Docker Desktop and Git. In PowerShell:

```powershell
git clone https://github.com/RahulSodhi-git/RL-tradebot.git
cd RL-tradebot
Copy-Item .env.example .env
notepad .env
docker compose up -d --build
```

Enter newly generated demo keys in `.env` locally. Do not commit or share keys.
Open http://localhost:8501 and enable account monitoring. Keep Docker Desktop
running and prevent laptop sleep. Stop with `docker compose down`.

Training data, trained model binaries and reports are excluded from Git and the
container image. Copy existing artifacts into `rl_lab/data`, `rl_lab/models` and
`rl_lab/reports` to view results or use trained models. Without them the dashboard
still starts and can monitor your demo account; it shows missing reports clearly.

## CI/CD

GitHub Actions validates Python and Compose, builds the image and checks its
Streamlit health endpoint on pushes and pull requests. Successful checks for a
push to `main` publish `ghcr.io/rahulsodhi-git/rl-tradebot:latest` and a commit tag.
Publishing an image does not deploy it to your laptop or start trades.
No Binance keys are needed in GitHub Actions. Account connectivity needs demo
keys and network access on the machine running Docker.

See [manual Python setup](rl_lab/DASHBOARD.md) for running without Docker.
