# RL Tradebot dashboard

## Download the trained project

Download the repository with **Code → Download ZIP**, or clone it with Git.
Trained artifacts are included directly in `rl_lab/models`, `rl_lab/reports`
and `rl_lab/data`. No separate release download or retraining is required.
The included artifacts contain
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

Training data, trained model binaries and reports are included in this repository
but excluded from the container image. Docker Compose mounts these local folders
into the dashboard. The source download is large because it includes training data.

## CI/CD

GitHub Actions validates Python and Compose, builds the image and checks its
Streamlit health endpoint on pushes and pull requests. Successful checks for a
push to `main` publish `ghcr.io/rahulsodhi-git/rl-tradebot:latest` and a commit tag.
Publishing an image does not deploy it to your laptop or start trades.
No Binance keys are needed in GitHub Actions. Account connectivity needs demo
keys and network access on the machine running Docker.

See [manual Python setup](rl_lab/DASHBOARD.md) for running without Docker.
