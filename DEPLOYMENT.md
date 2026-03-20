# Deployment

## Render Free

Use Render as a free web service with Telegram webhook mode.

This project can run in two modes:

```text
Local / fallback: polling
Render / production: webhook
```

Render uses:

```text
python football_bot_FINAL.py
```

Set these Render environment variables:

- `TELEGRAM_BOT_TOKEN`
- `WEBHOOK_URL=https://your-service-name.onrender.com`
- `ADMIN_USER_ID` (optional, for `/reload`)
- `ODDS_API_KEY` (optional)
- `OPENWEATHER_API_KEY` (optional)
- `APP_DATA_DIR` (optional)
- `PREDICTIONS_DB_PATH` (optional)

Files:

- `render.yaml`
- `requirements.txt`
- `football_bot_FINAL.py`

## Required environment variables

- `TELEGRAM_BOT_TOKEN`

If this is missing, the bot exits with a non-zero status.

## Optional environment variables

- `ODDS_API_KEY`
- `OPENWEATHER_API_KEY`
- `WEBHOOK_URL`
- `ADMIN_USER_ID`
- `APP_DATA_DIR`
- `PREDICTIONS_DB_PATH`

`ODDS_API_KEY` enables live bookmaker odds.  
`OPENWEATHER_API_KEY` enables weather enrichment.  
`WEBHOOK_URL` switches the bot into webhook mode for Render.  
`ADMIN_USER_ID` protects `/reload`.  
`APP_DATA_DIR` or `PREDICTIONS_DB_PATH` should point to a mounted Railway volume if you want prediction history to survive restarts.

## GitHub Actions Automation

Workflows included:

- `.github/workflows/refresh-data.yml`
- `.github/workflows/retrain-models.yml`

Secrets to add in GitHub:

- `ODDS_API_KEY`
- `OPENWEATHER_API_KEY`

Schedules:

- data refresh every 6 hours
- full retrain every Monday and Thursday at 02:00 UTC

The workflows commit updated artifacts back to `master`, which should trigger a new Render deploy automatically.

## Runtime behavior

- Weekly fixtures load from `weekly_fixtures.json` when present.
- If `weekly_fixtures.json` is missing or invalid, the bot falls back to the baked-in defaults.
- Result settlement now checks both the current and previous `football-data.co.uk` EPL season URLs, so deployments do not break at season rollover.
- `/pattern` supports stat-pattern questions such as shots, corners, bookings, fouls, and shots on target.

## Local smoke test

```text
set TELEGRAM_BOT_TOKEN=...
python football_bot_FINAL.py
```

## Local webhook smoke test

```text
set TELEGRAM_BOT_TOKEN=...
set WEBHOOK_URL=https://example.ngrok-free.app
set PORT=10000
python football_bot_FINAL.py
```

## Files used at runtime

- `production_artifacts.pkl`
- `lgb_1x2.txt`
- `lgb_ou25.txt`
- `lgb_ou15.txt`
- `lgb_btts.txt`
- `market_models/`
- `player_prop_models/`
- `europe_models/`
- `europe_market_models/`
- `cv_results_enriched.csv`
- `market_cv_results.csv`
- `europe_cv_results.csv`
- `weekly_fixtures.json`
