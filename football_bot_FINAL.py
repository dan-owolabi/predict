"""
Football Prediction Telegram Bot v6.0 (Data Enriched)
=====================================================
Predicts 5 markets: 1X2, Over/Under 2.5, Over/Under 1.5, BTTS, Exact Score.
Uses Dixon-Coles + LightGBM ensemble, enriched with FPL + Understat + Weather data.

SETUP:
  pip install python-telegram-bot scikit-learn pandas numpy requests lightgbm scipy understatapi
  set TELEGRAM_BOT_TOKEN=your_token
  set ODDS_API_KEY=your_key  (free at https://the-odds-api.com)
  set OPENWEATHER_API_KEY=your_key  (free at https://openweathermap.org, optional)
  python football_bot_FINAL.py
"""

import os, logging, json, requests, pickle, re
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

from feature_engine import build_rolling_features, FPL_FEATURE_COLS, WEATHER_FEATURE_COLS
from data_fetchers import get_all_external_features
from results_tracker import (
    save_match_prediction, settle_predictions,
    get_results_text, get_accuracy_summary
)

BOOT_WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "").strip()
LITE_RUNTIME = os.environ.get("LITE_RUNTIME", "").strip().lower() in {"1", "true", "yes"} or bool(BOOT_WEBHOOK_URL)

try:
    if LITE_RUNTIME:
        raise ImportError("disabled in lite runtime")
    from sportybet_value import attach_market_prices
except Exception:
    attach_market_prices = None

try:
    if LITE_RUNTIME:
        raise ImportError("disabled in lite runtime")
    from player_prop_inference import rank_fixture_players
except Exception:
    rank_fixture_players = None

try:
    from build_europe_training_data import (
        canonical_name as europe_canonical_name,
        normalize_europe_team,
        load_understat_histories as load_europe_understat_histories,
        load_support_histories as load_europe_support_histories,
        recent_stats as europe_recent_stats,
    )
except Exception:
    europe_canonical_name = None
    normalize_europe_team = None
    load_europe_understat_histories = None
    load_europe_support_histories = None
    europe_recent_stats = None

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

# ============================================================
# CONFIG
# ============================================================
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
WEBHOOK_URL = BOOT_WEBHOOK_URL
PORT = int(os.environ.get("PORT", "10000"))
ADMIN_USER_ID = os.environ.get("ADMIN_USER_ID", "").strip()
BASE_DIR = Path(__file__).parent
FIXTURES_PATH = BASE_DIR / "weekly_fixtures.json"
CV_RESULTS_PATH = BASE_DIR / "cv_results_enriched.csv"
ADDITIONAL_MARKET_DIR = BASE_DIR / "market_models"
ADDITIONAL_CV_PATH = BASE_DIR / "market_cv_results.csv"
EUROPE_MODEL_DIR = BASE_DIR / "europe_models"
EUROPE_CV_PATH = BASE_DIR / "europe_cv_results.csv"
EUROPE_DATA_PATH = BASE_DIR / "data" / "europe_training_data.csv"
EUROPE_TEAM_PATH = BASE_DIR / "data" / "europe_team_universe.csv"
EUROPE_SUPPORT_STATS_PATH = BASE_DIR / "data" / "football-data-support" / "support_leagues_all.csv"
UEFA_STATS_PATH = BASE_DIR / "data" / "uefa_stats" / "uefa_match_stats.csv"

# ============================================================
# LOAD PRODUCTION MODELS
# ============================================================
print("Loading production models and artifacts...")
try:
    with open('production_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    DC_MODEL = artifacts['dixon_coles']
    IMPUTER = artifacts['imputer']
    FEATURE_COLS = artifacts['feature_cols']
    HISTORICAL_DF = artifacts['historical_df']

    LGB_OU25 = lgb.Booster(model_file='lgb_ou25.txt')
    LGB_OU15 = lgb.Booster(model_file='lgb_ou15.txt')
    LGB_BTTS = lgb.Booster(model_file='lgb_btts.txt')
    LGB_1X2 = lgb.Booster(model_file='lgb_1x2.txt')
    print(f"Loaded pipeline successfully! ({len(HISTORICAL_DF)} matches in DB)")

    TEAMS_LIST = sorted(list(set(HISTORICAL_DF['HomeTeam'].unique()) | set(HISTORICAL_DF['AwayTeam'].unique())))
    MEDIAN_ODDS = HISTORICAL_DF.median(numeric_only=True).to_dict()
    TEAM_LATEST_ELO = {}
    TEAM_LATEST_ELO_RANK = {}
    if all(c in HISTORICAL_DF.columns for c in ['home_elo', 'away_elo']):
        home_elo_df = HISTORICAL_DF[['Date', 'HomeTeam', 'home_elo']].rename(
            columns={'HomeTeam': 'Team', 'home_elo': 'elo'})
        away_elo_df = HISTORICAL_DF[['Date', 'AwayTeam', 'away_elo']].rename(
            columns={'AwayTeam': 'Team', 'away_elo': 'elo'})
        elo_df = pd.concat([home_elo_df, away_elo_df], ignore_index=True)
        elo_df = elo_df.dropna(subset=['elo']).sort_values('Date')
        TEAM_LATEST_ELO = elo_df.groupby('Team').tail(1).set_index('Team')['elo'].to_dict()
    if all(c in HISTORICAL_DF.columns for c in ['home_elo_rank', 'away_elo_rank']):
        home_rank_df = HISTORICAL_DF[['Date', 'HomeTeam', 'home_elo_rank']].rename(
            columns={'HomeTeam': 'Team', 'home_elo_rank': 'elo_rank'})
        away_rank_df = HISTORICAL_DF[['Date', 'AwayTeam', 'away_elo_rank']].rename(
            columns={'AwayTeam': 'Team', 'away_elo_rank': 'elo_rank'})
        rank_df = pd.concat([home_rank_df, away_rank_df], ignore_index=True)
        rank_df = rank_df.dropna(subset=['elo_rank']).sort_values('Date')
        TEAM_LATEST_ELO_RANK = rank_df.groupby('Team').tail(1).set_index('Team')['elo_rank'].to_dict()

except Exception as e:
    print(f"Failed to load models: {e}")
    print("Run `python train_final_models.py` first.")
    exit(1)



def _load_market_history(path):
    """Load average backtest accuracy by target from the enriched CV file."""
    if not path.exists():
        return {}
    try:
        cv_df = pd.read_csv(path)
        if cv_df.empty or "target" not in cv_df.columns or "Ensemble_acc" not in cv_df.columns:
            return {}
        grouped = cv_df.groupby("target", as_index=False)["Ensemble_acc"].mean()
        label_map = {
            "1x2": "1X2",
            "btts": "GG/NG",
            "over25": "OVER/UNDER 2.5",
        }
        out = {}
        for _, row in grouped.iterrows():
            key = label_map.get(str(row["target"]).strip().lower())
            if key:
                out[key] = float(row["Ensemble_acc"]) * 100.0
        return out
    except Exception as exc:
        logger.warning(f"Could not load market history from {path.name}: {exc}")
        return {}


MARKET_HIST_ACC = _load_market_history(CV_RESULTS_PATH)

def _load_additional_market_bundle():
    if not ADDITIONAL_MARKET_DIR.exists():
        return [], None, [], {}
    manifest_path = ADDITIONAL_MARKET_DIR / "manifest.json"
    artifacts_path = ADDITIONAL_MARKET_DIR / "artifacts.pkl"
    if not manifest_path.exists() or not artifacts_path.exists():
        return [], None, [], {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(artifacts_path, "rb") as f:
            artifacts = pickle.load(f)
        hist_map = {}
        if ADDITIONAL_CV_PATH.exists():
            cv_df = pd.read_csv(ADDITIONAL_CV_PATH)
            if not cv_df.empty and "target" in cv_df.columns and "Model_acc" in cv_df.columns:
                hist_map = (cv_df.groupby("target")["Model_acc"].mean() * 100.0).to_dict()
        models = []
        for spec in manifest:
            model_path = ADDITIONAL_MARKET_DIR / spec["model_file"]
            if model_path.exists():
                models.append((spec, lgb.Booster(model_file=str(model_path))))
        return models, artifacts.get("imputer"), artifacts.get("feature_cols", []), hist_map
    except Exception as exc:
        logger.warning(f"Could not load additional market models: {exc}")
        return [], None, [], {}


if LITE_RUNTIME:
    ADDITIONAL_MARKET_MODELS, ADDITIONAL_IMPUTER, ADDITIONAL_FEATURE_COLS, ADDITIONAL_MARKET_HIST = [], None, [], {}
else:
    ADDITIONAL_MARKET_MODELS, ADDITIONAL_IMPUTER, ADDITIONAL_FEATURE_COLS, ADDITIONAL_MARKET_HIST = _load_additional_market_bundle()


def _load_europe_bundle():
    if not EUROPE_MODEL_DIR.exists():
        return {}, None, [], {}, pd.DataFrame(), {}
    manifest_path = EUROPE_MODEL_DIR / "manifest.json"
    artifacts_path = EUROPE_MODEL_DIR / "artifacts.pkl"
    if not manifest_path.exists() or not artifacts_path.exists():
        return {}, None, [], {}, pd.DataFrame(), {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with open(artifacts_path, "rb") as f:
            artifacts = pickle.load(f)
        models = {}
        for spec in manifest:
            model_path = EUROPE_MODEL_DIR / spec["model_file"]
            if model_path.exists():
                models[spec["target"]] = lgb.Booster(model_file=str(model_path))
        hist = {}
        if EUROPE_CV_PATH.exists():
            cv_df = pd.read_csv(EUROPE_CV_PATH)
            if not cv_df.empty:
                label_map = {"1x2": "1X2", "btts": "GG/NG", "over25": "OVER/UNDER 2.5"}
                hist = {
                    label_map.get(str(t).strip().lower(), str(t)): float(acc) * 100.0
                    for t, acc in cv_df.groupby("target")["accuracy"].mean().items()
                }
        europe_df = pd.read_csv(EUROPE_DATA_PATH, low_memory=False) if EUROPE_DATA_PATH.exists() else pd.DataFrame()
        if not europe_df.empty and "Date" in europe_df.columns:
            europe_df["Date"] = pd.to_datetime(europe_df["Date"], errors="coerce")
        team_aliases = {}
        if EUROPE_TEAM_PATH.exists():
            teams_df = pd.read_csv(EUROPE_TEAM_PATH)
            for team_name in teams_df["team_name"].dropna().astype(str).unique():
                key = europe_canonical_name(team_name) if europe_canonical_name else team_name.lower()
                team_aliases[key] = team_name
                if europe_canonical_name:
                    short = re.sub(r"^(fc|fk|sk|ac|as|ssc|bsc|ogc|rsc|pfc|gnk|vfb)\s+", "", key).strip()
                    if short and short not in team_aliases:
                        team_aliases[short] = team_name
        return models, artifacts.get("imputer"), artifacts.get("feature_cols", []), hist, europe_df, team_aliases
    except Exception as exc:
        logger.warning(f"Could not load Europe models: {exc}")
        return {}, None, [], {}, pd.DataFrame(), {}


if LITE_RUNTIME:
    EUROPE_MODELS, EUROPE_IMPUTER, EUROPE_FEATURE_COLS, EUROPE_HIST_ACC, EUROPE_HISTORY_DF, EUROPE_TEAM_ALIASES = {}, None, [], {}, pd.DataFrame(), {}
else:
    EUROPE_MODELS, EUROPE_IMPUTER, EUROPE_FEATURE_COLS, EUROPE_HIST_ACC, EUROPE_HISTORY_DF, EUROPE_TEAM_ALIASES = _load_europe_bundle()
EUROPE_INPUT_ALIASES = {
    "bayern": "FC Bayern München",
    "bayern munich": "FC Bayern München",
    "psg": "Paris Saint-Germain FC",
    "paris saint germain": "Paris Saint-Germain FC",
    "inter": "FC Internazionale Milano",
    "inter milan": "FC Internazionale Milano",
    "porto": "FC Porto",
    "benfica": "SL Benfica",
    "ajax": "AFC Ajax",
    "sporting": "Sporting Clube de Portugal",
    "roma": "AS Roma",
    "lazio": "Lazio Roma",
    "olympiacos": "Olympiakos Piraeus",
    "copenhagen": "FC København",
}


def _load_europe_team_hist():
    if load_europe_understat_histories is None or load_europe_support_histories is None:
        return pd.DataFrame()
    try:
        team_hist = pd.concat(
            [load_europe_understat_histories(), load_europe_support_histories()],
            ignore_index=True,
        )
        return team_hist.sort_values(["team_key", "Date"]).reset_index(drop=True)
    except Exception as exc:
        logger.warning(f"Could not load Europe team history: {exc}")
        return pd.DataFrame()


EUROPE_TEAM_HIST = pd.DataFrame() if LITE_RUNTIME else _load_europe_team_hist()


PATTERN_SPECS = {
    "shots": {
        "label": "Shots",
        "home_col": "HS",
        "away_col": "AS",
        "total_col": "TOTAL_SHOTS",
        "home_line": 11.5,
        "away_line": 9.5,
        "total_line": 22.5,
    },
    "sot": {
        "label": "Shots On Target",
        "home_col": "HST",
        "away_col": "AST",
        "total_col": "TOTAL_SOT",
        "home_line": 3.5,
        "away_line": 2.5,
        "total_line": 7.5,
    },
    "corners": {
        "label": "Corners",
        "home_col": "HC",
        "away_col": "AC",
        "total_col": "TOTAL_CORNERS",
        "home_line": 4.5,
        "away_line": 3.5,
        "total_line": 8.5,
    },
    "bookings": {
        "label": "Bookings",
        "home_col": "HY",
        "away_col": "AY",
        "total_col": "TOTAL_BOOKINGS",
        "home_line": 1.5,
        "away_line": 1.5,
        "total_line": 3.5,
    },
    "fouls": {
        "label": "Fouls",
        "home_col": "HF",
        "away_col": "AF",
        "total_col": "TOTAL_FOULS",
        "home_line": 9.5,
        "away_line": 9.5,
        "total_line": 20.5,
    },
}


def _pattern_market_from_text(text):
    lower = str(text or "").lower()
    if "shot on target" in lower or "shots on target" in lower or "sot" in lower:
        return "sot"
    if "corner" in lower:
        return "corners"
    if "booking" in lower or "card" in lower:
        return "bookings"
    if "foul" in lower:
        return "fouls"
    if "shot" in lower:
        return "shots"
    return None


def _normalize_pattern_df(df, home_team_col, away_team_col, date_col, comp_label, col_map):
    out = df.copy()
    out["Date"] = pd.to_datetime(out[date_col], errors="coerce")
    out["HomeTeam"] = out[home_team_col].astype(str)
    out["AwayTeam"] = out[away_team_col].astype(str)
    out["competition"] = comp_label
    for src, dst in col_map.items():
        if src in out.columns:
            out[dst] = pd.to_numeric(out[src], errors="coerce")
        else:
            out[dst] = np.nan
    out["TOTAL_SHOTS"] = out["HS"] + out["AS"]
    out["TOTAL_SOT"] = out["HST"] + out["AST"]
    out["TOTAL_CORNERS"] = out["HC"] + out["AC"]
    out["TOTAL_BOOKINGS"] = out["HY"] + out["AY"]
    out["TOTAL_FOULS"] = out["HF"] + out["AF"]
    return out[[
        "Date", "HomeTeam", "AwayTeam", "competition",
        "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HF", "AF",
        "TOTAL_SHOTS", "TOTAL_SOT", "TOTAL_CORNERS", "TOTAL_BOOKINGS", "TOTAL_FOULS",
    ]]


def _load_pattern_history():
    frames = []
    hist_cols = {"HS": "HS", "AS": "AS", "HST": "HST", "AST": "AST", "HC": "HC", "AC": "AC", "HY": "HY", "AY": "AY", "HF": "HF", "AF": "AF"}
    frames.append(_normalize_pattern_df(HISTORICAL_DF, "HomeTeam", "AwayTeam", "Date", "EPL", hist_cols))

    if EUROPE_SUPPORT_STATS_PATH.exists():
        support_df = pd.read_csv(EUROPE_SUPPORT_STATS_PATH, low_memory=False)
        frames.append(_normalize_pattern_df(support_df, "HomeTeam", "AwayTeam", "Date", "EU_SUPPORT", hist_cols))

    if UEFA_STATS_PATH.exists():
        uefa_df = pd.read_csv(UEFA_STATS_PATH, low_memory=False)
        frames.append(_normalize_pattern_df(
            uefa_df, "HomeTeam", "AwayTeam", "Date", "UEFA",
            {
                "home_shots": "HS", "away_shots": "AS",
                "home_sot": "HST", "away_sot": "AST",
                "home_corners": "HC", "away_corners": "AC",
                "home_bookings": "HY", "away_bookings": "AY",
                "home_fouls": "HF", "away_fouls": "AF",
            },
        ))

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).sort_values("Date").reset_index(drop=True)
    out = out.drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam", "HS", "AS", "HC", "AC", "HY", "AY", "HF", "AF"]
    ).reset_index(drop=True)
    out["home_key"] = out["HomeTeam"].map(lambda x: europe_canonical_name(x) if europe_canonical_name else str(x).lower())
    out["away_key"] = out["AwayTeam"].map(lambda x: europe_canonical_name(x) if europe_canonical_name else str(x).lower())
    return out


PATTERN_HISTORY_DF = None
PATTERN_PROFILE_CACHE = {}


def _safe_mean(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.mean()) if not series.empty else np.nan


def _safe_hit_rate(series, line):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return None
    return float((series > line).mean()) * 100.0


def _team_pattern_view(df, team_key, spec):
    rows = df[(df["home_key"] == team_key) | (df["away_key"] == team_key)].copy()
    if rows.empty:
        return rows
    is_home = rows["home_key"] == team_key
    rows["team"] = np.where(is_home, rows["HomeTeam"], rows["AwayTeam"])
    rows["opponent"] = np.where(is_home, rows["AwayTeam"], rows["HomeTeam"])
    rows["venue"] = np.where(is_home, "home", "away")
    rows["team_for"] = np.where(is_home, rows[spec["home_col"]], rows[spec["away_col"]])
    rows["team_against"] = np.where(is_home, rows[spec["away_col"]], rows[spec["home_col"]])
    rows["team_line"] = np.where(is_home, spec["home_line"], spec["away_line"])
    return rows.sort_values("Date").reset_index(drop=True)


def _summarize_pattern_rows(rows, total_col, line, sample_size=8):
    if rows.empty:
        return {"count": 0, "for": np.nan, "against": np.nan, "total": np.nan, "hit": None}
    sample = rows.tail(sample_size)
    return {
        "count": int(len(sample)),
        "for": _safe_mean(sample["team_for"]),
        "against": _safe_mean(sample["team_against"]),
        "total": _safe_mean(sample[total_col]),
        "hit": _safe_hit_rate(sample["team_for"], line),
    }


def _team_profile_map(df, spec):
    frames = []
    for side, team_col, opp_col, stat_for_col, stat_against_col in [
        ("home", "home_key", "away_key", spec["home_col"], spec["away_col"]),
        ("away", "away_key", "home_key", spec["away_col"], spec["home_col"]),
    ]:
        part = df[[team_col, opp_col, stat_for_col, stat_against_col, spec["total_col"]]].copy()
        part.columns = ["team_key", "opponent_key", "team_for", "team_against", "total"]
        frames.append(part)
    all_rows = pd.concat(frames, ignore_index=True)
    grouped = all_rows.groupby("team_key").agg(
        for_avg=("team_for", "mean"),
        against_avg=("team_against", "mean"),
        total_avg=("total", "mean"),
        matches=("team_for", "count"),
    )
    return grouped.to_dict("index")


def _similar_opponent_rows(team_rows, target_opponent_key, profile_map, sample_size=8):
    target_profile = profile_map.get(target_opponent_key)
    if target_profile is None or team_rows.empty:
        return team_rows.iloc[0:0].copy()
    candidate = team_rows.copy()
    candidate["opponent_key"] = candidate["opponent"].map(
        lambda x: europe_canonical_name(x) if europe_canonical_name else str(x).lower()
    )
    candidate["similarity"] = candidate["opponent_key"].map(
        lambda key: (
            abs(profile_map.get(key, {}).get("for_avg", 999.0) - target_profile["for_avg"]) +
            abs(profile_map.get(key, {}).get("against_avg", 999.0) - target_profile["against_avg"]) +
            abs(profile_map.get(key, {}).get("total_avg", 999.0) - target_profile["total_avg"])
        )
    )
    candidate = candidate[candidate["opponent_key"] != target_opponent_key]
    candidate = candidate.sort_values(["similarity", "Date"])
    return candidate.tail(sample_size).sort_values("Date")


def _fmt_pattern_line(prefix, stats, line):
    if stats["count"] == 0:
        return f"{prefix}: n/a"
    hit_txt = "n/a" if stats["hit"] is None else f"{stats['hit']:.0f}%"
    return (
        f"{prefix}: for {stats['for']:.1f} | allowed {stats['against']:.1f} | "
        f"total {stats['total']:.1f} | over {line} hit {hit_txt}"
    )


def _pattern_takeaway(home, away, spec, h2h_stats, home_recent, away_recent, home_similar, away_similar):
    scores = []
    if h2h_stats["count"]:
        scores.append(h2h_stats["total"])
    if home_recent["count"] and away_recent["count"]:
        scores.append(np.nanmean([home_recent["for"], away_recent["for"], home_recent["against"], away_recent["against"]]))
    if home_similar["count"]:
        scores.append(home_similar["for"] + home_similar["against"])
    if away_similar["count"]:
        scores.append(away_similar["for"] + away_similar["against"])
    if not scores:
        return "Takeaway: not enough history for a confident pattern call."
    projected_total = float(np.nanmean(scores))
    lean = "over" if projected_total >= spec["total_line"] else "under"
    return (
        f"Takeaway: the broader {spec['label'].lower()} pattern leans {lean} {spec['total_line']} "
        f"with an implied total around {projected_total:.1f}."
    )


def _pattern_assets(market_key):
    global PATTERN_HISTORY_DF, PATTERN_PROFILE_CACHE
    if PATTERN_HISTORY_DF is None:
        PATTERN_HISTORY_DF = _load_pattern_history()
    if market_key not in PATTERN_PROFILE_CACHE:
        PATTERN_PROFILE_CACHE[market_key] = _team_profile_map(PATTERN_HISTORY_DF, PATTERN_SPECS[market_key])
    return PATTERN_HISTORY_DF, PATTERN_PROFILE_CACHE[market_key]


def generate_pattern_report(home, away, market_key):
    spec = PATTERN_SPECS[market_key]
    home_key = europe_canonical_name(home) if europe_canonical_name else str(home).lower()
    away_key = europe_canonical_name(away) if europe_canonical_name else str(away).lower()
    df, profile_map = _pattern_assets(market_key)
    home_rows = _team_pattern_view(df, home_key, spec)
    away_rows = _team_pattern_view(df, away_key, spec)
    h2h = df[
        ((df["home_key"] == home_key) & (df["away_key"] == away_key)) |
        ((df["home_key"] == away_key) & (df["away_key"] == home_key))
    ].tail(5)
    home_recent = _summarize_pattern_rows(home_rows, spec["total_col"], spec["home_line"])
    away_recent = _summarize_pattern_rows(away_rows, spec["total_col"], spec["away_line"])
    h2h_rows = _team_pattern_view(h2h, home_key, spec)
    h2h_stats = _summarize_pattern_rows(h2h_rows, spec["total_col"], spec["home_line"], sample_size=5)
    h2h_total_hit = _safe_hit_rate(h2h[spec["total_col"]], spec["total_line"])
    home_similar_rows = _similar_opponent_rows(home_rows, away_key, profile_map)
    away_similar_rows = _similar_opponent_rows(away_rows, home_key, profile_map)
    home_similar = _summarize_pattern_rows(home_similar_rows, spec["total_col"], spec["home_line"])
    away_similar = _summarize_pattern_rows(away_similar_rows, spec["total_col"], spec["away_line"])
    projection_home = np.nanmean([home_recent["for"], away_recent["against"], home_similar["for"]])
    projection_away = np.nanmean([away_recent["for"], home_recent["against"], away_similar["for"]])
    projection_total = projection_home + projection_away if not np.isnan(projection_home) and not np.isnan(projection_away) else np.nan

    lines = [
        f"Pattern: {spec['label']}",
        f"{home} vs {away}",
        "",
        _fmt_pattern_line(f"{home} recent overall", home_recent, spec["home_line"]),
        _fmt_pattern_line(f"{away} recent overall", away_recent, spec["away_line"]),
        f"Projected pattern: {projection_home:.1f} - {projection_away:.1f} | total {projection_total:.1f}" if not np.isnan(projection_total) else "Projected pattern: n/a",
        "",
        "Direct H2H:",
        _fmt_pattern_line("H2H last 5", h2h_stats, spec["home_line"]),
        f"H2H total over {spec['total_line']}: {h2h_total_hit:.0f}% hit rate" if h2h_total_hit is not None else f"H2H total over {spec['total_line']}: n/a",
        "",
        "With other teams like this opponent:",
        _fmt_pattern_line(f"{home} vs teams like {away}", home_similar, spec["home_line"]),
        _fmt_pattern_line(f"{away} vs teams like {home}", away_similar, spec["away_line"]),
        "",
        _pattern_takeaway(home, away, spec, h2h_stats, home_recent, away_recent, home_similar, away_similar),
    ]

    sample_lines = []
    if not h2h.empty:
        sample_lines.append("")
        sample_lines.append("Recent H2H sample:")
        for _, row in h2h.tail(3).iterrows():
            sample_lines.append(
                f"- {pd.to_datetime(row['Date']).date()}: {row['HomeTeam']} vs {row['AwayTeam']} | total {row[spec['total_col']]:.0f}"
            )
    if not home_similar_rows.empty:
        sample_lines.append("")
        sample_lines.append(f"{home} vs teams like {away}:")
        for _, row in home_similar_rows.tail(3).iterrows():
            sample_lines.append(
                f"- {pd.to_datetime(row['Date']).date()}: {row['team']} vs {row['opponent']} ({row['venue']}) | for {row['team_for']:.0f} | total {row[spec['total_col']]:.0f}"
            )
    if not away_similar_rows.empty:
        sample_lines.append("")
        sample_lines.append(f"{away} vs teams like {home}:")
        for _, row in away_similar_rows.tail(3).iterrows():
            sample_lines.append(
                f"- {pd.to_datetime(row['Date']).date()}: {row['team']} vs {row['opponent']} ({row['venue']}) | for {row['team_for']:.0f} | total {row[spec['total_col']]:.0f}"
            )
    return "\n".join(lines + sample_lines)


# ============================================================
# LIVE ODDS API
# ============================================================
_odds_cache = {}
_odds_cache_time = None

TEAM_NAME_MAP = {
    'Man United': 'Manchester United', 'Man City': 'Manchester City',
    "Nott'm Forest": 'Nottingham Forest', 'Wolves': 'Wolverhampton Wanderers',
    'West Ham': 'West Ham United', 'Brighton': 'Brighton and Hove Albion',
    'Newcastle': 'Newcastle United', 'Leicester': 'Leicester City',
    'Tottenham': 'Tottenham Hotspur', 'Bournemouth': 'AFC Bournemouth',
    'Sheffield United': 'Sheffield Utd',
}

def fetch_live_odds():
    global _odds_cache, _odds_cache_time
    if _odds_cache_time and (datetime.now() - _odds_cache_time).seconds < 1800:
        return _odds_cache
    if not ODDS_API_KEY:
        return {}
    try:
        resp = requests.get("https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
                            params={'apiKey': ODDS_API_KEY, 'regions': 'uk',
                                    'markets': 'h2h,totals', 'oddsFormat': 'decimal'},
                            timeout=10)
        if resp.status_code != 200: return {}
        odds_map = {}
        for match in resp.json():
            home, away = match.get('home_team',''), match.get('away_team','')
            mo = {'home': home, 'away': away}
            for bk in match.get('bookmakers', []):
                for mkt in bk.get('markets', []):
                    if mkt['key'] == 'h2h' and 'b365h' not in mo:
                        oc = {o['name']: o['price'] for o in mkt['outcomes']}
                        mo['b365h'] = oc.get(home, 2.5)
                        mo['b365d'] = oc.get('Draw', 3.4)
                        mo['b365a'] = oc.get(away, 3.0)
                    elif mkt['key'] == 'totals' and 'b365_over' not in mo:
                        for o in mkt['outcomes']:
                            if o['name'] == 'Over' and o.get('point',0) == 2.5:
                                mo['b365_over'] = o['price']
                            elif o['name'] == 'Under' and o.get('point',0) == 2.5:
                                mo['b365_under'] = o['price']
                if 'b365h' in mo and 'b365_over' in mo: break
            if 'b365h' in mo:
                odds_map[f"{home}|{away}"] = mo
        _odds_cache = odds_map
        _odds_cache_time = datetime.now()
        logger.info(f"Fetched live odds for {len(odds_map)} matches")
        return odds_map
    except Exception as e:
        logger.warning(f"Odds API: {e}")
        return {}

def find_live_odds(home, away):
    om = fetch_live_odds()
    if not om: return None
    ha = TEAM_NAME_MAP.get(home, home)
    aa = TEAM_NAME_MAP.get(away, away)
    for h in [home, ha]:
        for a in [away, aa]:
            if f"{h}|{a}" in om: return om[f"{h}|{a}"]
    for k, v in om.items():
        ah, ap = k.split('|')
        if (home.lower() in ah.lower() or ha.lower() in ah.lower()) and \
           (away.lower() in ap.lower() or aa.lower() in ap.lower()):
            return v
    return None


# ============================================================
# WEEKLY FIXTURES (GW 28 - 38)
# ============================================================
DEFAULT_FIXTURES = [
    {"week_label": "GW28 -- Mar 2026",
     "matches": [["Wolves","Aston Villa"],["Bournemouth","Sunderland"],["Burnley","Brentford"],
                 ["Liverpool","West Ham"],["Newcastle","Everton"],["Leeds United","Man City"],
                 ["Brighton","Nott'm Forest"],["Fulham","Tottenham"],["Man United","Crystal Palace"],
                 ["Arsenal","Chelsea"]]},
    {"week_label": "GW29 -- Mar 2026",
     "matches": [["Newcastle","Man United"],["West Ham","Arsenal"],["Aston Villa","Fulham"],
                 ["Brentford","Liverpool"],["Burnley","Wolves"],["Chelsea","Bournemouth"],
                 ["Crystal Palace","Brighton"],["Everton","Leeds United"],["Man City","Sunderland"],
                 ["Nott'm Forest","Tottenham"]]},
    {"week_label": "GW30 -- Mar 14-16, 2026",
     "matches": [["West Ham","Man City"],["Burnley","Bournemouth"],["Crystal Palace","Leeds United"],
                 ["Man United","Aston Villa"],["Nott'm Forest","Fulham"],["Sunderland","Brighton"],
                 ["Chelsea","Newcastle"],["Arsenal","Everton"],["Liverpool","Tottenham"],["Brentford","Wolves"]]},
    {"week_label": "GW31 -- Mar 20-22, 2026",
     "matches": [["Bournemouth","Man United"],["Brighton","Liverpool"],["Aston Villa","West Ham"],
                 ["Fulham","Burnley"],["Man City","Crystal Palace"],["Everton","Chelsea"],
                 ["Leeds United","Brentford"],["Newcastle","Sunderland"],["Tottenham","Nott'm Forest"],
                 ["Wolves","Arsenal"]]},
    {"week_label": "GW32 -- Apr 11, 2026",
     "matches": [["Arsenal","Bournemouth"],["Brentford","Everton"],["Burnley","Brighton"],
                 ["Chelsea","Man City"],["Crystal Palace","Newcastle"],["Liverpool","Fulham"],
                 ["Man United","Leeds United"],["Nott'm Forest","Aston Villa"],["Sunderland","Tottenham"],
                 ["West Ham","Wolves"]]},
    {"week_label": "GW33 -- Apr 18, 2026",
     "matches": [["Aston Villa","Sunderland"],["Brentford","Fulham"],["Chelsea","Man United"],
                 ["Crystal Palace","West Ham"],["Everton","Liverpool"],["Leeds United","Wolves"],
                 ["Man City","Arsenal"],["Newcastle","Bournemouth"],["Nott'm Forest","Burnley"],
                 ["Tottenham","Brighton"]]},
    {"week_label": "GW34 -- Apr 25, 2026",
     "matches": [["Bournemouth","Leeds United"],["Arsenal","Newcastle"],["Brighton","Chelsea"],
                 ["Burnley","Man City"],["Fulham","Aston Villa"],["Liverpool","Crystal Palace"],
                 ["Man United","Brentford"],["Sunderland","Nott'm Forest"],["West Ham","Everton"],
                 ["Wolves","Tottenham"]]},
    {"week_label": "GW35 -- May 2, 2026",
     "matches": [["Bournemouth","Crystal Palace"],["Arsenal","Fulham"],["Aston Villa","Tottenham"],
                 ["Brentford","West Ham"],["Chelsea","Nott'm Forest"],["Everton","Man City"],
                 ["Leeds United","Burnley"],["Man United","Liverpool"],["Newcastle","Brighton"],
                 ["Wolves","Sunderland"]]},
    {"week_label": "GW36 -- May 9, 2026",
     "matches": [["Brighton","Wolves"],["Burnley","Aston Villa"],["Crystal Palace","Everton"],
                 ["Fulham","Bournemouth"],["Liverpool","Chelsea"],["Man City","Brentford"],
                 ["Nott'm Forest","Newcastle"],["Sunderland","Man United"],["Tottenham","Leeds United"],
                 ["West Ham","Arsenal"]]},
    {"week_label": "GW37 -- May 17, 2026",
     "matches": [["Bournemouth","Man City"],["Arsenal","Burnley"],["Aston Villa","Liverpool"],
                 ["Brentford","Crystal Palace"],["Chelsea","Tottenham"],["Everton","Sunderland"],
                 ["Leeds United","Brighton"],["Man United","Nott'm Forest"],["Newcastle","West Ham"],
                 ["Wolves","Fulham"]]},
    {"week_label": "GW38 -- May 24, 2026",
     "matches": [["Brighton","Man United"],["Burnley","Wolves"],["Crystal Palace","Arsenal"],
                 ["Fulham","Newcastle"],["Liverpool","Brentford"],["Man City","Aston Villa"],
                 ["Nott'm Forest","Bournemouth"],["Sunderland","Chelsea"],["Tottenham","Everton"],
                 ["West Ham","Leeds United"]]},
]

def load_fixtures():
    if FIXTURES_PATH.exists():
        try:
            with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list) and payload:
                return payload
            logger.warning(f"{FIXTURES_PATH.name} is empty or invalid, using default fixtures")
        except Exception as exc:
            logger.warning(f"Could not load {FIXTURES_PATH.name}: {exc}")
    return DEFAULT_FIXTURES


# ============================================================
# PREDICTION ENGINE
# ============================================================
def generate_match_features(home, away, b365h=None, b365d=None, b365a=None, b365_over=None, b365_under=None):
    """Build features for a single match prediction."""
    # 1. Gather Odds
    odds_src = "median"
    if not all([b365h, b365d, b365a, b365_over, b365_under]):
        live = find_live_odds(home, away)
        if live:
            b365h = b365h or live.get('b365h')
            b365d = b365d or live.get('b365d')
            b365a = b365a or live.get('b365a')
            b365_over = b365_over or live.get('b365_over')
            b365_under = b365_under or live.get('b365_under')
            odds_src = "live"

    bh = b365h or MEDIAN_ODDS.get('B365H', 2.5)
    bd = b365d or MEDIAN_ODDS.get('B365D', 3.4)
    ba = b365a or MEDIAN_ODDS.get('B365A', 3.2)
    bo = b365_over or MEDIAN_ODDS.get('B365>2.5', 1.9)
    bu = b365_under or MEDIAN_ODDS.get('B365<2.5', 2.0)

    # 2. Extract relevant recent matches
    mask = (HISTORICAL_DF['HomeTeam'].isin([home, away])) | (HISTORICAL_DF['AwayTeam'].isin([home, away]))
    mini_df = HISTORICAL_DF[mask].copy().sort_values('Date')

    # Create the new match row
    new_match = pd.DataFrame([{
        'HomeTeam': home, 'AwayTeam': away,
        'Date': pd.to_datetime(datetime.now()),
        'B365H': bh, 'B365D': bd, 'B365A': ba,
        'B365>2.5': bo, 'B365<2.5': bu,
        'FTHG': np.nan, 'FTAG': np.nan,
        'FTR': np.nan,
    }])

    # 3. Append and Build
    context_df = pd.concat([mini_df, new_match], ignore_index=True)
    df_built = build_rolling_features(context_df, windows=(3, 5, 10))

    # Extract the last row (our new match)
    last_row = df_built.iloc[-1:]

    # 4. Inject external data (FPL + Weather)
    ext_feats = get_all_external_features(home, away)
    for col in FPL_FEATURE_COLS + WEATHER_FEATURE_COLS:
        if col in FEATURE_COLS and col in ext_feats:
            last_row[col] = ext_feats[col]

    # 4b. Inject latest ClubElo snapshot proxies if model expects them
    if 'home_elo' in FEATURE_COLS:
        last_row['home_elo'] = TEAM_LATEST_ELO.get(home, np.nan)
    if 'away_elo' in FEATURE_COLS:
        last_row['away_elo'] = TEAM_LATEST_ELO.get(away, np.nan)
    if 'elo_diff' in FEATURE_COLS:
        last_row['elo_diff'] = (TEAM_LATEST_ELO.get(home, np.nan) - TEAM_LATEST_ELO.get(away, np.nan))
    if 'home_elo_rank' in FEATURE_COLS:
        last_row['home_elo_rank'] = TEAM_LATEST_ELO_RANK.get(home, np.nan)
    if 'away_elo_rank' in FEATURE_COLS:
        last_row['away_elo_rank'] = TEAM_LATEST_ELO_RANK.get(away, np.nan)
    if 'elo_rank_diff' in FEATURE_COLS:
        last_row['elo_rank_diff'] = (TEAM_LATEST_ELO_RANK.get(away, np.nan) -
                                     TEAM_LATEST_ELO_RANK.get(home, np.nan))

    # Ensure all expected model columns exist for inference
    for col in FEATURE_COLS:
        if col not in last_row.columns:
            last_row[col] = np.nan

    # 5. Impute
    raw_feats = last_row[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
    X_imputed = IMPUTER.transform(raw_feats)
    X = np.nan_to_num(X_imputed, nan=0.0, posinf=0.0, neginf=0.0)

    # 6. Dixon-Coles features (all 6 for cross-market signal)
    dc_ou25 = DC_MODEL.predict_ou25(home, away)
    dc_btts = DC_MODEL.predict_btts(home, away)
    dc_home, dc_draw, dc_away = DC_MODEL.predict_match_result(home, away)
    score_probs = DC_MODEL.predict_score_probs(home, away)
    dc_ou15 = 1.0 - (score_probs[0, 0] + score_probs[0, 1] + score_probs[1, 0])
    dc_stack = np.array([[dc_ou25, dc_btts, dc_home, dc_draw, dc_away, dc_ou15]])

    # 7. Score matrix for exact score
    top_scores = DC_MODEL.predict_top_scores(home, away, n=3)

    # Track which external data sources were used
    data_sources = [odds_src]
    if any(ext_feats.get(c) is not None and not np.isnan(ext_feats.get(c, float('nan')))
           for c in FPL_FEATURE_COLS[:1]):
        data_sources.append('fpl')
    if any(ext_feats.get(c) is not None and not np.isnan(ext_feats.get(c, float('nan')))
           for c in WEATHER_FEATURE_COLS[:1]):
        data_sources.append('weather')

    return X, dc_stack, dc_ou25, dc_btts, dc_ou15, dc_home, dc_draw, dc_away, \
        score_probs, top_scores, data_sources, last_row


def _reweight_scores(score_probs, ens_1x2, ens_ou25, ens_ou15, ens_btts):
    """Reweight DC score matrix so exact scores are fully consistent with all markets.

    Every scoreline is weighted by how well it matches ALL ensemble predictions:
    1X2 result, OU1.5, OU2.5, and BTTS. This ensures no contradictions like
    predicting 1-0 when OU1.5 is 83% or 2-0 when OU2.5 is 63%.
    """
    n = score_probs.shape[0]
    adj = score_probs.copy()

    p_home, p_draw, p_away = ens_1x2

    for i in range(n):
        for j in range(n):
            total_goals = i + j
            both_scored = (i > 0 and j > 0)
            w = 1.0

            # --- 1X2 result consistency ---
            if i > j:
                w *= (0.8 + 1.2 * p_home)     # home win score
            elif i == j:
                w *= (0.4 + 1.0 * p_draw)     # draw score (dampen, EPL ~25%)
            else:
                w *= (0.8 + 1.2 * p_away)     # away win score

            # --- OU1.5: hard gate on <=1 goal scores ---
            if total_goals <= 1:
                # OU15=0.83 → 0.17x (massive penalty), OU15=0.3 → 0.70x (mild)
                w *= (1.0 - ens_ou15)
            else:
                w *= (0.4 + 0.8 * ens_ou15)

            # --- OU2.5: strong push toward 3+ or <=2 goals ---
            if total_goals >= 3:
                # OU25=0.63 → 1.26x boost, OU25=0.3 → 0.60x penalty
                w *= (2.0 * ens_ou25)
            elif total_goals == 2:
                # Neutral-ish: slight lean toward under side
                w *= (0.6 + 0.6 * (1 - ens_ou25))
            else:
                # OU25=0.63 → 0.37x penalty, OU25=0.3 → 0.70x mild
                w *= (1.0 - ens_ou25)

            # --- BTTS consistency ---
            if both_scored:
                w *= (0.5 + 1.0 * ens_btts)
            else:
                w *= (0.5 + 1.0 * (1 - ens_btts))

            adj[i, j] *= max(w, 1e-6)

    # --- EPL base-rate anchoring ---
    epl_prior = {
        (1, 0): 1.15, (0, 1): 1.10, (2, 1): 1.15, (1, 2): 1.10,
        (2, 0): 1.10, (0, 2): 1.05, (1, 1): 0.85, (0, 0): 0.90,
        (2, 2): 0.95, (3, 1): 1.05, (1, 3): 1.00, (3, 2): 1.00,
        (2, 3): 0.95, (3, 0): 1.00, (0, 3): 0.95,
    }
    for (i, j), weight in epl_prior.items():
        if i < n and j < n:
            adj[i, j] *= weight

    # Normalize
    total = adj.sum()
    if total > 0:
        adj /= total

    return adj


def run_predictions(home, away, b365h=None, b365d=None, b365a=None, b365_over=None, b365_under=None):
    """Run all models for a given matchup."""
    X, dc_stack, dc_ou25, dc_btts, dc_ou15, dc_home, dc_draw, dc_away, \
        score_probs, top_scores, data_sources, row_df = \
        generate_match_features(home, away, b365h, b365d, b365a, b365_over, b365_under)

    # Stack features with all DC predictions
    X_full = np.nan_to_num(np.column_stack([X, dc_stack]), nan=0.0, posinf=0.0, neginf=0.0)

    # Over/Under 2.5
    lgb_p_ou25 = LGB_OU25.predict(X_full)[0]
    ens_ou25 = 0.2 * dc_ou25 + 0.8 * lgb_p_ou25

    # Over/Under 1.5
    lgb_p_ou15 = LGB_OU15.predict(X_full)[0]
    ens_ou15 = 0.2 * dc_ou15 + 0.8 * lgb_p_ou15

    # BTTS
    lgb_p_btts = LGB_BTTS.predict(X_full)[0]
    ens_btts = 0.2 * dc_btts + 0.8 * lgb_p_btts

    # 1X2
    lgb_p_1x2 = LGB_1X2.predict(X_full)[0]  # [p_H, p_D, p_A]
    dc_1x2 = np.array([dc_home, dc_draw, dc_away])
    ens_1x2 = 0.3 * dc_1x2 + 0.7 * lgb_p_1x2
    ens_1x2 = ens_1x2 / ens_1x2.sum()

    # Exact Score — reweight DC score matrix using ensemble 1X2 probabilities
    # so the predicted scoreline is consistent with the predicted result
    adj_probs = _reweight_scores(score_probs, ens_1x2, ens_ou25, ens_ou15, ens_btts)
    best_i, best_j = np.unravel_index(adj_probs.argmax(), adj_probs.shape)
    exact_score = f"{best_i}-{best_j}"
    exact_score_prob = adj_probs[best_i, best_j]

    # Top 3 from reweighted matrix
    adj_scores = []
    for i in range(adj_probs.shape[0]):
        for j in range(adj_probs.shape[1]):
            if adj_probs[i, j] > 0.001:
                adj_scores.append((f"{i}-{j}", float(adj_probs[i, j])))
    adj_scores.sort(key=lambda x: -x[1])
    top_scores = adj_scores[:3]

    extra_markets = []
    if ADDITIONAL_MARKET_MODELS and ADDITIONAL_IMPUTER is not None and ADDITIONAL_FEATURE_COLS:
        extra_row = row_df.copy()
        for col in ADDITIONAL_FEATURE_COLS:
            if col not in extra_row.columns:
                extra_row[col] = np.nan
        extra_raw = extra_row[ADDITIONAL_FEATURE_COLS].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
        extra_X = np.nan_to_num(ADDITIONAL_IMPUTER.transform(extra_raw), nan=0.0, posinf=0.0, neginf=0.0)
        extra_dc_stack = np.array([[dc_ou25, dc_btts, dc_home, dc_draw, dc_away]])
        extra_full = np.nan_to_num(np.column_stack([extra_X, extra_dc_stack]), nan=0.0, posinf=0.0, neginf=0.0)
        for spec, model in ADDITIONAL_MARKET_MODELS:
            prob_over = float(model.predict(extra_full)[0])
            positive_pick = spec.get("pick", "OVER")
            if "GG" in positive_pick:
                pick = positive_pick if prob_over >= 0.5 else "NG"
                market_name = spec["market"]
            else:
                pick = "OVER" if prob_over >= 0.5 else "UNDER"
                market_name = f"{spec['market']} {spec['line']:.1f}"
            extra_markets.append({
                "target": spec["target"],
                "market": market_name,
                "pick": pick,
                "est_accuracy": max(prob_over, 1.0 - prob_over) * 100.0,
                "hist_accuracy": ADDITIONAL_MARKET_HIST.get(spec["target"]),
                "baseline": 50.0,
                "base_rate": float(spec.get("base_rate", 0.5)) * 100.0,
            })

    return {
        'ou25': float(ens_ou25),
        'ou15': float(ens_ou15),
        'btts': float(ens_btts),
        '1x2': [float(x) for x in ens_1x2],
        'dc_ou25': float(dc_ou25),
        'dc_ou15': float(dc_ou15),
        'dc_btts': float(dc_btts),
        'dc_1x2': [float(dc_home), float(dc_draw), float(dc_away)],
        'exact_score': exact_score,
        'exact_score_prob': float(exact_score_prob),
        'top_scores': top_scores,
        'data_sources': data_sources,
        'row': row_df,
        'extra_markets': extra_markets,
    }


def _bar(pct, width=10):
    """Create a visual bar: ▓▓▓▓▓▓░░░░"""
    filled = round(pct / 100 * width)
    return "▓" * filled + "░" * (width - filled)


def _conf_icon(conf, baseline=50):
    """Confidence indicator relative to baseline."""
    edge = conf - baseline
    if edge >= 8: return "🟢"
    if edge >= 3: return "🟡"
    return "🔴"



def _market_rankings(res):
    """Return ranked fixture-level picks across modeled markets."""
    hist_map = res.get("market_hist_acc", MARKET_HIST_ACC)
    p_1x2 = res["1x2"]
    winner_idx = max(range(3), key=lambda i: p_1x2[i])
    winner_pick = ["HOME WIN", "DRAW", "AWAY WIN"][winner_idx]
    conf_1x2 = float(p_1x2[winner_idx] * 100.0)

    po25 = float(res["ou25"])
    conf_25 = max(po25, 1.0 - po25) * 100.0
    pick_25 = "OVER 2.5" if po25 > 0.5 else "UNDER 2.5"

    po15 = float(res["ou15"])
    conf_15 = max(po15, 1.0 - po15) * 100.0
    pick_15 = "OVER 1.5" if po15 > 0.5 else "UNDER 1.5"

    pbtts = float(res["btts"])
    conf_btts = max(pbtts, 1.0 - pbtts) * 100.0
    pick_btts = "GG" if pbtts > 0.5 else "NG"

    exact_pick = str(res["exact_score"])
    exact_conf = float(res.get("exact_score_prob", 0.0) * 100.0)

    rows = [
        {"market": "1X2", "pick": winner_pick, "est_accuracy": conf_1x2,
         "hist_accuracy": hist_map.get("1X2"), "baseline": 33.0, "base_rate": None},
        {"market": "OVER/UNDER 2.5", "pick": pick_25, "est_accuracy": conf_25,
         "hist_accuracy": hist_map.get("OVER/UNDER 2.5"), "baseline": 50.0, "base_rate": None},
        {"market": "OVER/UNDER 1.5", "pick": pick_15, "est_accuracy": conf_15,
         "hist_accuracy": None, "baseline": 50.0, "base_rate": None},
        {"market": "GG/NG", "pick": pick_btts, "est_accuracy": conf_btts,
         "hist_accuracy": hist_map.get("GG/NG"), "baseline": 50.0, "base_rate": None},
    ]
    if not res.get("is_europe_beta"):
        rows.append({"market": "EXACT SCORE", "pick": exact_pick, "est_accuracy": exact_conf,
                     "hist_accuracy": None, "baseline": 0.0, "base_rate": None})
    rows.extend(res.get("extra_markets", []))
    for row in rows:
        row["edge"] = row["est_accuracy"] - row["baseline"]
        hist = row.get("hist_accuracy")
        hist_bonus = 0.0 if hist is None or pd.isna(hist) else max(0.0, hist - max(row["baseline"], 50.0)) * 0.15
        base_rate = row.get("base_rate")
        majority_penalty = 0.0
        if base_rate is not None and not pd.isna(base_rate):
            majority_penalty = max(0.0, abs(base_rate - 50.0) - 12.0) * 0.10
        row["builder_score"] = row["edge"] + hist_bonus - majority_penalty
        row["book_odds"] = row.get("book_odds")
        row["implied_prob"] = row.get("implied_prob")
        row["value_edge"] = row.get("value_edge")
    rows.sort(key=lambda row: (-row["est_accuracy"], row["market"]))
    return rows


def _market_family(row):
    market = row["market"]
    if market.startswith("HOME SHOTS"):
        return "home_attack_volume"
    if market.startswith("AWAY SHOTS"):
        return "away_attack_volume"
    if market.startswith("HOME SOT"):
        return "home_attack_quality"
    if market.startswith("AWAY SOT"):
        return "away_attack_quality"
    if market.startswith("SHOTS ON TARGET"):
        return "match_sot"
    if market.startswith("SHOTS"):
        return "match_shots"
    if market.startswith("HOME CORNERS"):
        return "home_corners"
    if market.startswith("AWAY CORNERS"):
        return "away_corners"
    if market.startswith("CORNERS"):
        return "match_corners"
    if market.startswith("HOME BOOKINGS"):
        return "home_bookings"
    if market.startswith("AWAY BOOKINGS"):
        return "away_bookings"
    if market.startswith("BOOKINGS"):
        return "match_bookings"
    if market.startswith("HOME FOULS"):
        return "home_fouls"
    if market.startswith("AWAY FOULS"):
        return "away_fouls"
    if market.startswith("FOULS"):
        return "match_fouls"
    if market.startswith("1ST HALF GOALS"):
        return "first_half_goals"
    if market.startswith("1ST HALF GG/NG"):
        return "first_half_btts"
    if market.startswith("OVER/UNDER"):
        return "match_goals"
    if market == "GG/NG":
        return "match_btts"
    if market == "1X2":
        return "match_result"
    if market == "EXACT SCORE":
        return "match_score"
    return market.lower()


def _family_conflicts(family):
    groups = {
        "home_attack_volume": {"home_attack_volume", "home_attack_quality", "match_shots", "match_sot", "match_goals", "match_btts"},
        "away_attack_volume": {"away_attack_volume", "away_attack_quality", "match_shots", "match_sot", "match_goals", "match_btts"},
        "home_attack_quality": {"home_attack_volume", "home_attack_quality", "match_shots", "match_sot", "match_goals", "match_btts"},
        "away_attack_quality": {"away_attack_volume", "away_attack_quality", "match_shots", "match_sot", "match_goals", "match_btts"},
        "match_shots": {"match_shots", "home_attack_volume", "away_attack_volume", "match_sot"},
        "match_sot": {"match_sot", "home_attack_quality", "away_attack_quality", "match_shots"},
        "match_goals": {"match_goals", "match_btts", "first_half_goals", "first_half_btts"},
        "match_btts": {"match_btts", "match_goals", "first_half_btts"},
        "first_half_goals": {"first_half_goals", "first_half_btts", "match_goals"},
        "first_half_btts": {"first_half_btts", "first_half_goals", "match_btts", "match_goals"},
        "match_bookings": {"match_bookings", "home_bookings", "away_bookings", "match_fouls"},
        "home_bookings": {"home_bookings", "match_bookings"},
        "away_bookings": {"away_bookings", "match_bookings"},
        "match_fouls": {"match_fouls", "home_fouls", "away_fouls", "match_bookings"},
        "home_fouls": {"home_fouls", "match_fouls"},
        "away_fouls": {"away_fouls", "match_fouls"},
        "match_corners": {"match_corners", "home_corners", "away_corners"},
        "home_corners": {"home_corners", "match_corners"},
        "away_corners": {"away_corners", "match_corners"},
    }
    return groups.get(family, {family})


def _sorted_market_rankings(home, away, rankings):
    if attach_market_prices is None:
        return rankings, None, rankings[0]
    priced = attach_market_prices(home, away, rankings)
    value_rows = [row for row in priced if row.get("value_edge") is not None]
    top_value = max(value_rows, key=lambda x: (x["value_edge"], x["edge"], x["est_accuracy"])) if value_rows else None
    sorted_rows = sorted(
        priced,
        key=lambda row: (
            row.get("value_edge") is None,
            -(row.get("value_edge") if row.get("value_edge") is not None else row.get("est_accuracy", 0.0)),
            -row.get("est_accuracy", 0.0),
            row.get("market", ""),
        )
    )
    return sorted_rows, top_value, rankings[0]


def _is_player_market(row):
    market = str(row.get("market", "")).upper()
    return any(token in market for token in ("PLAYER", "SCORER", "ANYTIME"))


def _recommendation_thresholds(row):
    market = str(row.get("market", ""))
    family = _market_family(row)
    if market == "1X2":
        return 60.0, 3.0
    if market in {"OVER/UNDER 2.5", "OVER/UNDER 1.5", "GG/NG"}:
        return 64.0, 4.0
    if family in {
        "match_corners", "home_corners", "away_corners",
        "match_bookings", "home_bookings", "away_bookings",
        "match_shots", "home_attack_volume", "away_attack_volume",
        "match_sot", "home_attack_quality", "away_attack_quality",
        "match_fouls", "home_fouls", "away_fouls",
        "first_half_goals", "first_half_btts",
    }:
        return 68.0, 4.0
    if _is_player_market(row):
        return 76.0, 6.0
    return 70.0, 5.0


def _is_trustworthy_recommendation(row):
    market = str(row.get("market", ""))
    if row.get("beta"):
        return False
    if market == "EXACT SCORE":
        return False
    if _is_player_market(row):
        return False
    est_threshold, edge_threshold = _recommendation_thresholds(row)
    if float(row.get("est_accuracy", 0.0)) < est_threshold:
        return False
    if float(row.get("edge", 0.0)) < edge_threshold:
        return False
    hist = row.get("hist_accuracy")
    if hist is not None and not pd.isna(hist):
        min_hist = 58.0 if market == "1X2" else 60.0
        if float(hist) < min_hist:
            return False
    value_edge = row.get("value_edge")
    if value_edge is not None and float(value_edge) < 2.0:
        return False
    return True


def _select_top_recommendation(display_rankings, top_value_market, top_hit_market):
    trusted = [row for row in display_rankings if _is_trustworthy_recommendation(row)]
    if top_value_market is not None and _is_trustworthy_recommendation(top_value_market):
        return top_value_market, "value"
    if trusted:
        trusted_sorted = sorted(
            trusted,
            key=lambda row: (
                -(row.get("value_edge") if row.get("value_edge") is not None else -999.0),
                -row.get("builder_score", 0.0),
                -row.get("est_accuracy", 0.0),
                row.get("market", ""),
            ),
        )
        return trusted_sorted[0], "trust"
    return top_hit_market, "fallback"


def _build_bet_builder(rankings, limit=3):
    selected = []
    blocked = set()
    for row in sorted(rankings, key=lambda x: (-(x.get("value_edge") if x.get("value_edge") is not None else -999), -x.get("builder_score", 0.0), -x["est_accuracy"], x["market"])):
        family = _market_family(row)
        if family in {"match_score", "match_result"}:
            continue
        if row["edge"] < 6:
            continue
        hist = row.get("hist_accuracy")
        if hist is not None and not pd.isna(hist) and hist < 60:
            continue
        if family in blocked:
            continue
        selected.append(row)
        blocked.update(_family_conflicts(family))
        if len(selected) >= limit:
            break
    return selected

def _fmt_hist_accuracy(value):
    return "n/a" if value is None or pd.isna(value) else f"{float(value):.1f}%"


def _recent_europe_form(team_key, before_date, window):
    if EUROPE_HISTORY_DF.empty:
        return None
    mask = (
        ((EUROPE_HISTORY_DF.get("home_key") == team_key) | (EUROPE_HISTORY_DF.get("away_key") == team_key))
        & (EUROPE_HISTORY_DF["Date"] < before_date)
    )
    rows = EUROPE_HISTORY_DF[mask].tail(window)
    if rows.empty:
        return None
    records = []
    for _, row in rows.iterrows():
        if row["home_key"] == team_key:
            gf, ga = row["FTHG"], row["FTAG"]
        else:
            gf, ga = row["FTAG"], row["FTHG"]
        pts = 3 if gf > ga else (1 if gf == ga else 0)
        records.append({"gf": gf, "ga": ga, "points": pts})
    return {
        "gf": float(np.mean([r["gf"] for r in records])),
        "ga": float(np.mean([r["ga"] for r in records])),
        "ppg": float(np.mean([r["points"] for r in records])),
    }


def _resolve_europe_team_name(name):
    if not EUROPE_TEAM_ALIASES or europe_canonical_name is None:
        return None
    key = europe_canonical_name(name)
    if key in EUROPE_TEAM_ALIASES:
        return EUROPE_TEAM_ALIASES[key]
    manual = {
        "psg": "Paris Saint-Germain FC",
        "inter": "FC Internazionale Milano",
        "inter milan": "FC Internazionale Milano",
        "porto": "FC Porto",
        "benfica": "SL Benfica",
        "ajax": "AFC Ajax",
        "bayern": "FC Bayern München",
        "bayern munich": "FC Bayern München",
        "sporting": "Sporting Clube de Portugal",
        "roma": "AS Roma",
        "lazio": "Lazio Roma",
        "olympiacos": "Olympiakos Piraeus",
        "copenhagen": "FC København",
    }
    mapped = manual.get(key)
    if mapped:
        return mapped
    return None


def _europe_competition_from_text(text):
    lower = text.lower()
    if any(token in lower for token in ["europa", "uel"]):
        return "EL"
    return "CL"


def _detect_teams_from_text(raw_text):
    raw_text = str(raw_text or "").strip().lower()
    normalized_text = europe_canonical_name(raw_text) if europe_canonical_name else raw_text
    hits = []
    for team in TEAMS_LIST:
        target = team.lower()
        idx = raw_text.find(target)
        if idx != -1:
            hits.append((idx, -len(target), team))
        if 'utd' in raw_text and 'united' in target:
            hits.append((raw_text.find('utd'), -len(target), team))
        if 'spurs' in raw_text and 'tottenham' in target:
            hits.append((raw_text.find('spurs'), -len(target), team))
        if 'forest' in raw_text and "nott" in target:
            hits.append((raw_text.find('forest'), -len(target), team))
    for key, team in EUROPE_TEAM_ALIASES.items():
        if key:
            idx = normalized_text.find(key)
            if idx != -1:
                hits.append((idx, -len(key), team))
    for key, team in EUROPE_INPUT_ALIASES.items():
        idx = normalized_text.find(key)
        if idx != -1:
            hits.append((idx, -len(key), team))

    if not hits:
        return []

    canonical_seen = set()
    ordered = []
    for _, _, team in sorted(hits):
        canon = europe_canonical_name(team) if europe_canonical_name else team.lower()
        if canon in canonical_seen:
            continue
        canonical_seen.add(canon)
        ordered.append(team)
    return ordered


def generate_europe_match_features(home, away, competition="CL"):
    if EUROPE_IMPUTER is None or not EUROPE_FEATURE_COLS or EUROPE_TEAM_HIST.empty or europe_recent_stats is None or normalize_europe_team is None:
        raise RuntimeError("Europe inference assets unavailable.")

    now = pd.Timestamp(datetime.now())
    home_key = normalize_europe_team(home)
    away_key = normalize_europe_team(away)
    row = {
        "Date": now,
        "competition": competition,
        "HomeTeam": home,
        "AwayTeam": away,
        "home_key": home_key,
        "away_key": away_key,
        "competition_flag": 1.0 if competition == "CL" else 0.0,
    }

    for window in (3, 5, 10):
        home_stats = europe_recent_stats(EUROPE_TEAM_HIST, home_key, now, window)
        away_stats = europe_recent_stats(EUROPE_TEAM_HIST, away_key, now, window)
        for key, value in home_stats.items():
            row[f"home_{key}"] = value
        for key, value in away_stats.items():
            row[f"away_{key}"] = value
        row[f"xg_atk_def_diff_r{window}"] = row.get(f"home_xgf_r{window}", np.nan) - row.get(f"away_xga_r{window}", np.nan)
        row[f"ppg_proxy_diff_r{window}"] = (
            (row.get(f"home_gf_r{window}", np.nan) - row.get(f"home_ga_r{window}", np.nan)) -
            (row.get(f"away_gf_r{window}", np.nan) - row.get(f"away_ga_r{window}", np.nan))
        )
        row[f"atk_def_diff_r{window}"] = row.get(f"home_gf_r{window}", np.nan) - row.get(f"away_ga_r{window}", np.nan)
        row[f"def_atk_diff_r{window}"] = row.get(f"away_gf_r{window}", np.nan) - row.get(f"home_ga_r{window}", np.nan)

    for window in (3, 5):
        home_eu = _recent_europe_form(home_key, now, window)
        away_eu = _recent_europe_form(away_key, now, window)
        if home_eu:
            row[f"home_eu_gf_r{window}"] = home_eu["gf"]
            row[f"home_eu_ga_r{window}"] = home_eu["ga"]
            row[f"home_eu_ppg_r{window}"] = home_eu["ppg"]
        if away_eu:
            row[f"away_eu_gf_r{window}"] = away_eu["gf"]
            row[f"away_eu_ga_r{window}"] = away_eu["ga"]
            row[f"away_eu_ppg_r{window}"] = away_eu["ppg"]
        row[f"eu_gd_diff_r{window}"] = (
            (row.get(f"home_eu_gf_r{window}", np.nan) - row.get(f"home_eu_ga_r{window}", np.nan)) -
            (row.get(f"away_eu_gf_r{window}", np.nan) - row.get(f"away_eu_ga_r{window}", np.nan))
        )
        row[f"eu_ppg_diff_r{window}"] = row.get(f"home_eu_ppg_r{window}", np.nan) - row.get(f"away_eu_ppg_r{window}", np.nan)

    frame = pd.DataFrame([row])
    for col in EUROPE_FEATURE_COLS:
        if col not in frame.columns:
            frame[col] = np.nan
    raw_feats = frame[EUROPE_FEATURE_COLS].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
    X = np.nan_to_num(EUROPE_IMPUTER.transform(raw_feats), nan=0.0, posinf=0.0, neginf=0.0)
    return X, frame


def run_europe_predictions(home, away, competition="CL"):
    X, row_df = generate_europe_match_features(home, away, competition=competition)
    p_ou25 = float(np.clip(EUROPE_MODELS["over25"].predict(X)[0], 0.01, 0.99))
    p_btts = float(np.clip(EUROPE_MODELS["btts"].predict(X)[0], 0.01, 0.99))
    p_1x2 = np.clip(EUROPE_MODELS["1x2"].predict(X)[0], 0.01, 0.99)
    p_1x2 = p_1x2 / p_1x2.sum()
    p_ou15 = float(np.clip(0.35 + 0.45 * p_ou25 + 0.20 * p_btts, 0.05, 0.95))

    extra_markets = []
    if ADDITIONAL_MARKET_MODELS and ADDITIONAL_IMPUTER is not None and ADDITIONAL_FEATURE_COLS:
        extra_row = row_df.copy()
        for col in ADDITIONAL_FEATURE_COLS:
            if col not in extra_row.columns:
                extra_row[col] = np.nan
        extra_raw = extra_row[ADDITIONAL_FEATURE_COLS].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
        extra_X = np.nan_to_num(ADDITIONAL_IMPUTER.transform(extra_raw), nan=0.0, posinf=0.0, neginf=0.0)
        extra_dc_stack = np.array([[p_ou25, p_btts, p_1x2[0], p_1x2[1], p_1x2[2]]])
        extra_full = np.nan_to_num(np.column_stack([extra_X, extra_dc_stack]), nan=0.0, posinf=0.0, neginf=0.0)
        for spec, model in ADDITIONAL_MARKET_MODELS:
            prob_over = float(np.clip(model.predict(extra_full)[0], 0.01, 0.99))
            positive_pick = spec.get("pick", "OVER")
            if "GG" in positive_pick:
                pick = positive_pick if prob_over >= 0.5 else "NG"
                market_name = spec["market"]
            else:
                pick = "OVER" if prob_over >= 0.5 else "UNDER"
                market_name = f"{spec['market']} {spec['line']:.1f}"
            extra_markets.append({
                "target": spec["target"],
                "market": market_name,
                "pick": pick,
                "est_accuracy": max(prob_over, 1.0 - prob_over) * 100.0,
                "hist_accuracy": None,
                "baseline": 50.0,
                "base_rate": float(spec.get("base_rate", 0.5)) * 100.0,
                "beta": True,
            })

    return {
        "ou25": p_ou25,
        "ou15": p_ou15,
        "btts": p_btts,
        "1x2": [float(x) for x in p_1x2],
        "exact_score": "beta-only",
        "exact_score_prob": 0.0,
        "top_scores": [],
        "data_sources": ["Europe competitions", "Domestic support", "Understat support"],
        "row": row_df,
        "extra_markets": extra_markets,
        "competition": competition,
        "is_europe_beta": True,
        "market_hist_acc": EUROPE_HIST_ACC,
    }

def format_unified_prediction(home, away, res):
    """Format 5-market prediction card for Telegram."""
    if res.get("is_europe_beta"):
        rankings = _market_rankings(res)
        display_rankings, top_value_market, top_hit_market = _sorted_market_rankings(home, away, rankings)
        top_market, recommendation_mode = _select_top_recommendation(display_rankings, top_value_market, top_hit_market)
        builder_legs = _build_bet_builder(display_rankings)
        competition = res.get("competition", "CL")
        lines = [
            f"{home} vs {away}",
            f"Competition: {'Champions League' if competition == 'CL' else 'Europa League'}",
            "Mode: Europe beta",
            "",
            f"Top recommendation: {top_market['market']} -> {top_market['pick']} ({top_market['est_accuracy']:.1f}% est)",
            f"Recommendation mode: {recommendation_mode}",
            f"Historical accuracy: {_fmt_hist_accuracy(top_market.get('hist_accuracy'))}",
            "Value edge: n/a",
            f"Highest hit-rate market: {top_hit_market['market']} -> {top_hit_market['pick']} ({top_hit_market['est_accuracy']:.1f}% est)",
            "",
            "All market accuracy:",
        ]
        for row in display_rankings:
            beta_tag = " | beta" if row.get("beta") else ""
            lines.append(
                f"- {row['market']}: {row['pick']} | est {row['est_accuracy']:.1f}% | hist {_fmt_hist_accuracy(row.get('hist_accuracy'))}{beta_tag}"
            )
        if builder_legs:
            lines += ["", "Bet builder shortlist:"]
            for idx, row in enumerate(builder_legs, 1):
                beta_tag = " beta" if row.get("beta") else ""
                lines.append(
                    f"- Leg {idx}: {row['market']} -> {row['pick']} | est {row['est_accuracy']:.1f}% | edge {row['edge']:.1f}%{beta_tag}"
                )
        lines += [
            "",
            "1X2 split:",
            f"- Home {res['1x2'][0] * 100:.1f}% | Draw {res['1x2'][1] * 100:.1f}% | Away {res['1x2'][2] * 100:.1f}%",
            "",
            "Expanded Europe markets use beta scoring until Europe-specific settled labels are built.",
        ]
        return "\n".join(lines)

    p_1x2 = res["1x2"]
    winner_idx = max(range(3), key=lambda i: p_1x2[i])
    winner_labels = ["HOME WIN", "DRAW", "AWAY WIN"]
    winner_str = winner_labels[winner_idx]
    conf_1x2 = p_1x2[winner_idx] * 100

    po25 = res["ou25"]
    str_o25 = "OVER 2.5" if po25 > 0.5 else "UNDER 2.5"
    conf_25 = max(po25, 1-po25) * 100

    po15 = res["ou15"]
    str_o15 = "OVER 1.5" if po15 > 0.5 else "UNDER 1.5"
    conf_15 = max(po15, 1-po15) * 100

    pbtts = res["btts"]
    str_btts = "GG" if pbtts > 0.5 else "NG"
    conf_btts = max(pbtts, 1-pbtts) * 100

    top = res["top_scores"]
    rankings = _market_rankings(res)
    display_rankings, top_value_market, top_hit_market = _sorted_market_rankings(home, away, rankings)
    top_market, recommendation_mode = _select_top_recommendation(display_rankings, top_value_market, top_hit_market)
    builder_legs = _build_bet_builder(display_rankings)
    top_edge = top_market["edge"]
    if top_edge >= 8:
        adv = "Strong edge"
    elif top_edge >= 3:
        adv = "Moderate edge"
    else:
        adv = "Low edge"

    date_str = datetime.now().strftime("%Y-%m-%d")
    save_match_prediction(home, away, date_str, res)

    player_props = rank_fixture_players(home, away, top_n=5) if rank_fixture_players is not None else pd.DataFrame()

    sources = res.get("data_sources", ["median"])
    src_labels = []
    if "live" in sources:
        src_labels.append("Live odds")
    else:
        src_labels.append("Median odds")
    if "fpl" in sources:
        src_labels.append("FPL")
    if "weather" in sources:
        src_labels.append("Weather")
    src_str = " | ".join(src_labels)

    lines = [
        f"{home} vs {away}",
        "",
        f"Top recommendation: {top_market['market']} -> {top_market['pick']} ({top_market['est_accuracy']:.1f}% est)",
        f"Recommendation mode: {recommendation_mode}",
        f"Historical accuracy: {_fmt_hist_accuracy(top_market['hist_accuracy'])}",
        f"Value edge: {top_market['value_edge']:+.1f}%" if top_market.get('value_edge') is not None else "Value edge: n/a",
        f"Highest hit-rate market: {top_hit_market['market']} -> {top_hit_market['pick']} ({top_hit_market['est_accuracy']:.1f}% est)",
        f"Edge grade: {adv}",
        "",
        "All market accuracy:",
    ]
    for row in display_rankings:
        extra = ""
        if row.get("book_odds") is not None and row.get("value_edge") is not None:
            extra = f" | odds {row['book_odds']:.2f} | value {row['value_edge']:+.1f}%"
        lines.append(
            f"- {row['market']}: {row['pick']} | est {row['est_accuracy']:.1f}% | hist {_fmt_hist_accuracy(row.get('hist_accuracy'))}{extra}"
        )
    if builder_legs:
        lines += ["", "Bet builder shortlist:"]
        for idx, row in enumerate(builder_legs, 1):
            lines.append(
                f"- Leg {idx}: {row['market']} -> {row['pick']} | est {row['est_accuracy']:.1f}% | edge {row['edge']:.1f}%"
            )
    if not player_props.empty:
        lines += ["", "Top player props:"]
        for _, prow in player_props.head(5).iterrows():
            lines.append(
                f"- {prow['player']} ({prow['team']}): {prow['label']} YES | prob {prow['prob_yes']:.1f}% | edge {prow['edge']:.1f}%"
            )
    lines += [
        "",
        "1X2 split:",
        f"- Home {p_1x2[0] * 100:.1f}% | Draw {p_1x2[1] * 100:.1f}% | Away {p_1x2[2] * 100:.1f}%",
        "",
        "Top exact scores:",
    ]
    for s, p in top[:3]:
        lines.append(f"- {s}: {p * 100:.1f}%")
    lines += [
        "",
        f"Data sources: {src_str}",
    ]
    return "\n".join(lines)


# ============================================================
# WEEKLY PREDICTIONS
# ============================================================
def get_weekly_predictions(week_idx=0, high_only=False):
    fx = load_fixtures()
    if not fx: return "No fixtures.", 0, []
    idx = max(0, min(week_idx, len(fx) - 1))
    week = fx[idx]
    label = week.get('week_label', f'Week {idx+1}')

    lines = [
        f"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓",
        f"┃  📅 {label}",
        f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫",
    ]
    hi_n = 0

    fx_btns = []
    for home, away in week.get('matches', []):
        if home not in TEAMS_LIST or away not in TEAMS_LIST:
            lines.append(f"┃  ⚠️ {home} vs {away} — not in DB")
            continue

        res = run_predictions(home, away)

        rankings = _market_rankings(res)
        display_rankings, top_value_market, top_hit_market = _sorted_market_rankings(home, away, rankings)
        top_market, recommendation_mode = _select_top_recommendation(display_rankings, top_value_market, top_hit_market)
        builder_legs = _build_bet_builder(display_rankings)
        one_x_two = next(row for row in display_rankings if row["market"] == "1X2")
        ou25 = next(row for row in display_rankings if row["market"] == "OVER/UNDER 2.5")
        ou15 = next(row for row in display_rankings if row["market"] == "OVER/UNDER 1.5")
        ggn = next(row for row in display_rankings if row["market"] == "GG/NG")
        exact = next(row for row in display_rankings if row["market"] == "EXACT SCORE")

        is_hi = (top_market["est_accuracy"] - top_market["baseline"]) >= 7
        if high_only and not is_hi: continue
        if is_hi: hi_n += 1
        flag = "HIGH" if is_hi else "    "

        lines.append(f"Match {flag} {home} vs {away}")
        lines.append(f"  Top: {top_market['market']} -> {top_market['pick']} {top_market['est_accuracy']:.0f}% ({recommendation_mode})")
        lines.append(f"  1X2 {one_x_two['pick']} {one_x_two['est_accuracy']:.0f}% | O/U2.5 {ou25['pick']} {ou25['est_accuracy']:.0f}%")
        lines.append(f"  O/U1.5 {ou15['pick']} {ou15['est_accuracy']:.0f}% | GG/NG {ggn['pick']} {ggn['est_accuracy']:.0f}%")
        lines.append(f"  Exact {exact['pick']} {exact['est_accuracy']:.0f}%")
        if builder_legs:
            lines.append("  Builder: " + " | ".join(f"{row['market']} {row['pick']}" for row in builder_legs[:2]))

        if home in TEAMS_LIST and away in TEAMS_LIST:
            h_idx = TEAMS_LIST.index(home)
            a_idx = TEAMS_LIST.index(away)
            fx_btns.append([InlineKeyboardButton(f"🔮 {home} vs {away}", callback_data=f'fx_{h_idx}_{a_idx}')])

    if high_only and hi_n == 0:
        lines.append("┃  No high confidence picks this week.")
    lines += [
        f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫",
        f"┃  🔥 = High conf  │  Page {idx+1}/{len(fx)}",
        f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛",
    ]

    return "\n".join(lines), idx, fx_btns


# ============================================================
# TELEGRAM HANDLERS
# ============================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("🔮 Predict Match", callback_data='pred_home'),
         InlineKeyboardButton("❓ Ask Pattern", callback_data='ask_help')],
        [InlineKeyboardButton("📅 Gameweeks", callback_data='wk_0'),
         InlineKeyboardButton("📊 Results", callback_data='results')],
        [InlineKeyboardButton("ℹ️ How It Works", callback_data='how')],
    ]
    odds_icon = "📡" if ODDS_API_KEY else "📴"
    odds_s = "ON" if ODDS_API_KEY else "OFF"
    await update.message.reply_text(
        f"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        f"┃  ⚽ Football Prediction Bot\n"
        f"┃  v6.0 — EPL Specialist\n"
        f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
        f"┃  📚 {len(HISTORICAL_DF)} matches trained\n"
        f"┃  🧠 Dixon-Coles + LightGBM\n"
        f"┃  {odds_icon} Live odds: {odds_s}\n"
        f"┃  👥 FPL squad data\n"
        f"┃  📈 Understat xG\n"
        f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
        f"┃  Markets:\n"
        f"┃  🏆 1X2    │  📈 O/U 2.5\n"
        f"┃  📊 O/U 1.5│  🤝 BTTS\n"
        f"┃  🎯 Exact Score\n"
        f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
        f"┃  🟢 = Strong  🟡 = Moderate\n"
        f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛",
        reply_markup=InlineKeyboardMarkup(kb))


def menu_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔮 Predict Match", callback_data='pred_home'),
         InlineKeyboardButton("❓ Ask Pattern", callback_data='ask_help')],
        [InlineKeyboardButton("📅 Gameweeks", callback_data='wk_0'),
         InlineKeyboardButton("📊 Results", callback_data='results')],
        [InlineKeyboardButton("ℹ️ How It Works", callback_data='how')],
    ])


def team_kb(prefix, exclude=None):
    btns = []; row = []
    for t in TEAMS_LIST:
        if t == exclude: continue
        row.append(InlineKeyboardButton(t, callback_data=f'{prefix}{t}'))
        if len(row) == 2: btns.append(row); row = []
    if row: btns.append(row)
    btns.append([InlineKeyboardButton("🏠 Menu", callback_data='menu')])
    return InlineKeyboardMarkup(btns)


def week_nav(idx, hi=False, fx_btns=None):
    fx = load_fixtures()
    mx = len(fx) - 1
    p = 'wkh_' if hi else 'wk_'
    btns = []
    if fx_btns:
        btns.extend(fx_btns)
    nav = []
    if idx < mx: nav.append(InlineKeyboardButton("⬅️ Earlier", callback_data=f'{p}{idx+1}'))
    if idx > 0:  nav.append(InlineKeyboardButton("➡️ Later", callback_data=f'{p}{idx-1}'))
    if nav: btns.append(nav)
    btns.append([InlineKeyboardButton("🔥 High Only" if not hi else "📋 Show All",
                                       callback_data=f'{"wkh_" if not hi else "wk_"}{idx}')])
    btns.append([InlineKeyboardButton("🏠 Menu", callback_data='menu')])
    return InlineKeyboardMarkup(btns)


async def btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    d = q.data

    if d == 'menu':
        await q.edit_message_text(
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃  ⚽ Football Prediction Bot\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
            "\nChoose an option below:",
            reply_markup=menu_kb())

    elif d == 'pred_home':
        await q.edit_message_text(
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃  🏠 Select HOME team\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛",
            reply_markup=team_kb('predh_'))

    elif d.startswith('predh_'):
        home = d[6:]
        context.user_data['pred_home'] = home
        await q.edit_message_text(
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            f"┃  🏠 Home: {home}\n"
            "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
            "┃  ✈️ Select AWAY team\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛",
            reply_markup=team_kb('preda_', exclude=home))

    elif d.startswith('preda_'):
        away = d[6:]
        home = context.user_data.get('pred_home')
        await q.edit_message_text(f"⏳ Analyzing {home} vs {away}...")
        res = run_predictions(home, away)
        result_txt = format_unified_prediction(home, away, res)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔮 New Prediction", callback_data='pred_home')],
            [InlineKeyboardButton("📅 Gameweeks", callback_data='wk_0'),
             InlineKeyboardButton("🏠 Menu", callback_data='menu')],
        ])
        await q.edit_message_text(result_txt, reply_markup=kb)

    elif d == 'ask_help':
        await q.edit_message_text(
            "Ask me about patterns between any two teams.\n\n"
            "Examples:\n"
            "- Arsenal vs Chelsea shots\n"
            "- find the pattern in corners between Barcelona and Bayern Munich\n"
            "- show Liverpool fouls pattern with teams like Atletico Madrid\n"
            "- /ask Arsenal vs Chelsea shots\n\n"
            "Markets: shots, shots on target, corners, bookings/cards, fouls",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔮 Predict Match", callback_data='pred_home')],
                [InlineKeyboardButton("🏠 Menu", callback_data='menu')],
            ]))

    elif d.startswith('wkh_'):
        idx = int(d[4:])
        r, ai, fx_btns = get_weekly_predictions(idx, True)
        await q.edit_message_text(r, reply_markup=week_nav(ai, True, fx_btns))

    elif d.startswith('wk_'):
        idx = int(d[3:])
        r, ai, fx_btns = get_weekly_predictions(idx, False)
        await q.edit_message_text(r, reply_markup=week_nav(ai, False, fx_btns))

    elif d.startswith('fx_'):
        parts = d.split('_')
        home = TEAMS_LIST[int(parts[1])]
        away = TEAMS_LIST[int(parts[2])]
        await q.edit_message_text(f"⏳ Analyzing {home} vs {away}...")
        res = run_predictions(home, away)
        result_txt = format_unified_prediction(home, away, res)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📅 Back to Gameweek", callback_data='wk_0')],
            [InlineKeyboardButton("🔮 New Prediction", callback_data='pred_home'),
             InlineKeyboardButton("🏠 Menu", callback_data='menu')],
        ])
        await q.edit_message_text(result_txt, reply_markup=kb)

    elif d == 'results':
        settle_predictions()
        text = get_results_text()
        await q.edit_message_text(text,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Refresh", callback_data='results')],
                [InlineKeyboardButton("🏠 Menu", callback_data='menu')]
            ]))

    elif d == 'hist':
        text = get_results_text(limit=20)
        await q.edit_message_text(text,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Menu", callback_data='menu')]]))

    elif d == 'how':
        await q.edit_message_text(
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃  ℹ️ How It Works (v6.0)\n"
            "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
            "┃  🧠 MODEL PIPELINE\n"
            "┃\n"
            "┃  1. Dixon-Coles Goal Model\n"
            "┃     Attack/defense ratings (MLE)\n"
            "┃     Time-decay weighting\n"
            "┃     Low-score correction (rho)\n"
            "┃\n"
            "┃  2. Rolling Stats Engine\n"
            "┃     Form, xG, shots, corners\n"
            "┃     H2H history + clean sheets\n"
            "┃     3 / 5 / 10 match windows\n"
            "┃\n"
            "┃  3. LightGBM Ensemble\n"
            "┃     4 models: 1X2, O/U 2.5,\n"
            "┃     O/U 1.5, BTTS\n"
            "┃     Cross-market DC features\n"
            "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
            "┃  📡 DATA SOURCES\n"
            "┃  👥 FPL: injuries & form\n"
            "┃  📈 Understat: live xG\n"
            "┃  📡 Live odds (The Odds API)\n"
            "┃  🌤️ Weather (OpenWeatherMap)\n"
            "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
            "┃  📊 STATS\n"
            f"┃  📚 {len(HISTORICAL_DF)} EPL matches\n"
            f"┃  📐 {len(FEATURE_COLS)} + 6 DC features\n"
            "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
            "┃  🎯 MARKETS\n"
            "┃  🏆 1X2    │  📈 O/U 2.5\n"
            "┃  📊 O/U 1.5│  🤝 BTTS\n"
            "┃  🎯 Exact Score\n"
            "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
            "┃  🟢 Strong  🟡 Moderate  🔴 Low\n"
            "┃\n"
            "┃  ⚠️ Gamble responsibly.\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Menu", callback_data='menu')]]))


# Smart text search handler
async def msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().lower()
    market_key = _pattern_market_from_text(text)
    detected_teams = _detect_teams_from_text(text)

    if market_key is not None and ("pattern" in text or "trend" in text):
        if len(detected_teams) >= 2:
            report = generate_pattern_report(detected_teams[0], detected_teams[1], market_key)
            await update.message.reply_text(report)
            return
        await update.message.reply_text("I found the market, but I still need two clubs. Example: find the pattern in shots between Arsenal and Chelsea")
        return

    if len(detected_teams) >= 2:
        home, away = detected_teams[0], detected_teams[1]

        # Determine order based on text position
        idx_t1 = text.find(home.lower()[:5])
        idx_t2 = text.find(away.lower()[:5])
        if idx_t1 > idx_t2 and idx_t2 != -1:
            home, away = away, home

        await update.message.reply_text(f"⏳ Analyzing {home} vs {away}...")
        if home in TEAMS_LIST and away in TEAMS_LIST:
            res = run_predictions(home, away)
        else:
            comp = _europe_competition_from_text(text)
            res = run_europe_predictions(home, away, competition=comp)
        r = format_unified_prediction(home, away, res)
        await update.message.reply_text(r)
    elif len(detected_teams) == 1:
        await update.message.reply_text(f"Found {detected_teams[0]} — need two teams!\nTry: {detected_teams[0]} vs Chelsea")
    else:
        await update.message.reply_text("⚽ Type a match to predict\nExample: Arsenal vs Chelsea\n\nOr use /start for the menu")


async def predict_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("Usage: /predict Arsenal vs Chelsea  |  /predict Arsenal FC vs Paris Saint-Germain FC CL")
        return
    raw = " ".join(args).strip()
    comp = "EL" if any(token in raw.lower().split() for token in {"el", "uel", "europa"}) else "CL"
    raw = re.sub(r"\b(CL|EL|UCL|UEL|champions|europa)\b", "", raw, flags=re.IGNORECASE).strip()

    home = away = None
    if re.search(r"\s+vs\s+|\s+v\s+", raw, flags=re.IGNORECASE):
        parts = re.split(r"\s+vs\s+|\s+v\s+", raw, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            left = parts[0].strip()
            right = parts[1].strip()
            left_detected = _detect_teams_from_text(left)
            right_detected = _detect_teams_from_text(right)
            home = left_detected[0] if left_detected else None
            away = right_detected[0] if right_detected else None
    else:
        detected = _detect_teams_from_text(raw)
        if len(detected) >= 2:
            home, away = detected[0], detected[1]

    if not home or not away:
        await update.message.reply_text("Could not parse both teams. Use: /predict Arsenal FC vs Paris Saint-Germain FC CL")
        return
    await update.message.reply_text(f"⏳ Analyzing {home} vs {away}...")
    if home in TEAMS_LIST and away in TEAMS_LIST:
        res = run_predictions(home, away)
    else:
        res = run_europe_predictions(home, away, competition=comp)
    r = format_unified_prediction(home, away, res)
    await update.message.reply_text(r)


async def pattern_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text("Usage: /pattern Arsenal vs Chelsea shots")
        return
    market_key = _pattern_market_from_text(raw)
    teams = _detect_teams_from_text(raw)
    if market_key is None:
        await update.message.reply_text("Ask for a stat market like shots, corners, bookings, fouls, or shots on target.")
        return
    if len(teams) < 2:
        await update.message.reply_text("I need two clubs. Example: /pattern Arsenal vs Chelsea shots")
        return
    await update.message.reply_text(f"Analyzing pattern for {teams[0]} vs {teams[1]}...")
    report = generate_pattern_report(teams[0], teams[1], market_key)
    await update.message.reply_text(report)


async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text(
            "Usage: /ask Arsenal vs Chelsea shots\n"
            "You can also ask naturally, for example:\n"
            "find the pattern in corners between Arsenal and Chelsea"
        )
        return
    market_key = _pattern_market_from_text(raw)
    teams = _detect_teams_from_text(raw)
    if market_key is None:
        await update.message.reply_text("Ask for shots, shots on target, corners, bookings/cards, or fouls.")
        return
    if len(teams) < 2:
        await update.message.reply_text("I need two teams. Example: /ask Arsenal vs Chelsea shots")
        return
    await update.message.reply_text(f"Analyzing pattern for {teams[0]} vs {teams[1]}...")
    report = generate_pattern_report(teams[0], teams[1], market_key)
    await update.message.reply_text(report)


async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(getattr(update.effective_user, "id", ""))
    if not ADMIN_USER_ID or user_id != ADMIN_USER_ID:
        await update.message.reply_text("Unauthorized.")
        return
    await update.message.reply_text("Reload is not hot-swapped in-process. Deploy the latest commit to reload models safely.")


# ============================================================
# MAIN
# ============================================================
def main():
    if not BOT_TOKEN:
        print("Missing required environment variable: TELEGRAM_BOT_TOKEN")
        raise SystemExit(1)
    print("\nStarting bot...")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("pattern", pattern_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CallbackQueryHandler(btn))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, msg))
    print("Bot live! Send /start in Telegram.\n")
    if WEBHOOK_URL:
        print(f"Webhook mode on port {PORT}")
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=BOT_TOKEN,
            webhook_url=f"{WEBHOOK_URL.rstrip('/')}/{BOT_TOKEN}",
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
    else:
        app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == '__main__':
    main()
