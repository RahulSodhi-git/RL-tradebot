# RL Tradebot dashboard

## Download the trained project

Download the repository source and `rl-tradebot-trained-artifacts.zip` from
[GitHub Releases](https://github.com/RahulSodhi-git/RL-tradebot/releases).
Extract the artifact ZIP **inside the repository's `rl_lab` folder**. It contains
`models`, `reports` and `data` folders. The resulting paths must be
`rl_lab/models/ppo_universal_1h.zip`, `rl_lab/reports/` and `rl_lab/data/`.
No retraining is required to obtain the saved models. Release artifacts include
the downloaded training archives, processed data, normalization metadata and
evaluation reports. The cloud Python virtual environment is intentionally omitted;
Docker installs the runtime dependencies on your laptop.

Credentials are excluded. Any key previously committed to Git remains exposed in
history even after removal from the current branch: revoke it and use replacement
demo credentials only in your local `.env` file.

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
