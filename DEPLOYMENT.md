# Deployment

## Railway

This project is already configured to start with:

```text
python football_bot_FINAL.py
```

`Procfile` and `railway.toml` both point at the same worker entrypoint.

## Required environment variables

- `TELEGRAM_BOT_TOKEN`

If this is missing, the worker now exits with a non-zero status so Railway shows the deploy as failed instead of idling silently.

## Optional environment variables

- `ODDS_API_KEY`
- `OPENWEATHER_API_KEY`
- `APP_DATA_DIR`
- `PREDICTIONS_DB_PATH`

`ODDS_API_KEY` enables live bookmaker odds.  
`OPENWEATHER_API_KEY` enables weather enrichment.  
`APP_DATA_DIR` or `PREDICTIONS_DB_PATH` should point to a mounted Railway volume if you want prediction history to survive restarts.

## Recommended Railway setup

1. Create a Railway service from this repo.
2. Set `TELEGRAM_BOT_TOKEN`.
3. Optionally set `ODDS_API_KEY` and `OPENWEATHER_API_KEY`.
4. Add a persistent volume and mount it, for example at `/data`.
5. Set `APP_DATA_DIR=/data`.

This makes `predictions_db.json` persistent across redeploys and restarts.

## Runtime behavior

- Weekly fixtures load from `weekly_fixtures.json` when present.
- If `weekly_fixtures.json` is missing or invalid, the bot falls back to the baked-in defaults.
- Result settlement now checks both the current and previous `football-data.co.uk` EPL season URLs, so deployments do not break at season rollover.

## Local smoke test

```text
set TELEGRAM_BOT_TOKEN=...
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
- `cv_results_enriched.csv`
- `market_cv_results.csv`
- `weekly_fixtures.json`
