# Deploy pipeline — GitHub → CI/CD → Docker → Oracle

Push to `main` → GitHub Actions compile-checks the code, ships it to the Oracle VM, rebuilds the Docker
image, and restarts the dashboard. The trading loop stays a container you start from the dashboard.

```
   git push  ──►  GitHub Actions (.github/workflows/deploy.yml)
                     │  1. compile-check every module + risk_rules.json
                     │  2. rsync code to the VM (never touches .env or runtime files)
                     │  3. ssh: docker compose build + restart the dashboard
                     ▼
   Oracle VM  ──►  Docker:  rl-tradebot (dashboard :8502, always up)
                            rl-bot      (trading loop — you Start/Stop it from the dashboard)
                     both bind-mount /home/ubuntu/rl-tradebot  →  share bot_state.json / logs / history
                     .env lives ONLY on the VM (never in Git, never in the image)
```

## What you do (two one-time steps)

### 1. Push this folder to GitHub with a clean history
This folder still has the leftover **Amazon-project** git history — start fresh so the repo is just the bot:
```bash
cd "RL - tradebot"
rm -rf .git                 # drop the unrelated Amazon history (the bot code is what matters)
git init
git add -A
git commit -m "RL Trade Bot + Docker / CI-CD pipeline"
git branch -M main
git remote add origin https://github.com/RahulSodhi-git/RL-tradebot.git
git push -u origin main
```
(If push asks for a password, use a **GitHub Personal Access Token**, not your account password. `.env` and
all runtime files are git-ignored, so no secrets get pushed.)

### 2. Add ONE GitHub secret
In the repo: **Settings → Secrets and variables → Actions → New repository secret**
- **Name:** `ORACLE_SSH_KEY`
- **Value:** the private deploy key printed by the setup step (starts with `-----BEGIN OPENSSH PRIVATE KEY-----`).

That's it. Every push to `main` now auto-deploys. (The host/user are set at the top of
`.github/workflows/deploy.yml` — edit them there if the VM ever changes.)

## Day-to-day
- **Ship a change:** commit + push to `main`. Watch it in the repo's **Actions** tab.
- **Start/stop trading:** the dashboard's ▶ Start / ⏹ Stop buttons (they control the `rl-bot` container).
  A deploy never auto-starts trading — you press Start after reviewing.
- **Dashboard:** `http://161.118.176.44:8502`.
- **Change a rule value:** edit `risk_rules.json`, push — no rebuild needed (the loop re-reads it live).

## On the VM (managed for you, for reference)
```bash
cd /home/ubuntu/rl-tradebot
docker compose ps                      # what's running
docker compose logs -f rl-tradebot     # dashboard logs
docker logs -f rl-bot                  # bot logs (also mirrored to bot.log for the dashboard feed)
docker compose --profile trading up -d rl-bot   # start the bot loop by hand
docker compose stop rl-bot                       # stop the bot loop
```
`.env` (your Binance testnet keys + `BOT_UI_PIN`) lives at `/home/ubuntu/rl-tradebot/.env` and is the one
thing NOT managed by Git — set it once on the VM.

## Notes
- The image is ARM64 (Oracle Ampere) and amd64 compatible.
- Code + runtime data are bind-mounted, so a code push doesn't need an image rebuild unless
  `requirements*.txt` changed; the bot picks up new code when you Start it.
- ⚠️ Known issue to fix first: the recorded P&L doesn't match the exchange's real fills (see the manual's
  Section 0). The pipeline ships code faithfully — it can't fix a logic bug.
