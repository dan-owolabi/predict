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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
BASE_DIR = Path(__file__).parent
FIXTURES_PATH = BASE_DIR / "weekly_fixtures.json"

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

except Exception as e:
    print(f"Failed to load models: {e}")
    print("Run `python train_final_models.py` first.")
    exit(1)


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
        'row': row_df
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


def format_unified_prediction(home, away, res):
    """Format 5-market prediction card for Telegram."""
    # 1X2
    p_1x2 = res['1x2']
    winner_idx = max(range(3), key=lambda i: p_1x2[i])
    winner_labels = ['🏠 HOME WIN', '🤝 DRAW', '✈️ AWAY WIN']
    winner_str = winner_labels[winner_idx]
    conf_1x2 = p_1x2[winner_idx] * 100

    # OU2.5
    po25 = res['ou25']
    str_o25 = "⬆️ OVER 2.5" if po25 > 0.5 else "⬇️ UNDER 2.5"
    conf_25 = max(po25, 1-po25) * 100

    # OU1.5
    po15 = res['ou15']
    str_o15 = "⬆️ OVER 1.5" if po15 > 0.5 else "⬇️ UNDER 1.5"
    conf_15 = max(po15, 1-po15) * 100

    # BTTS
    pbtts = res['btts']
    str_btts = "YES" if pbtts > 0.5 else "NO"
    conf_btts = max(pbtts, 1-pbtts) * 100

    # Exact Score
    top = res['top_scores']

    # Best advice
    best_c = max(conf_1x2 - 33, conf_25 - 50, conf_15 - 50, conf_btts - 50)
    if best_c >= 8:     adv = "💎 Strong edge detected"
    elif best_c >= 3:   adv = "⚡ Moderate edge"
    else:               adv = "💤 Low edge — consider skipping"

    # Save prediction for tracking
    date_str = datetime.now().strftime('%Y-%m-%d')
    save_match_prediction(home, away, date_str, res)

    # Data source indicators
    sources = res.get('data_sources', ['median'])
    src_icons = []
    if 'live' in sources: src_icons.append("📡 Odds")
    else: src_icons.append("📊 Odds")
    if 'fpl' in sources: src_icons.append("👥 FPL")
    if 'weather' in sources: src_icons.append("🌤️ Wx")
    src_str = " │ ".join(src_icons)

    # Build the card
    L = "┃"
    lines = [
        f"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓",
        f"{L}  ⚽ {home} vs {away}",
        f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫",
        f"{L}  📊 MATCH RESULT",
        f"{L}  {_conf_icon(conf_1x2, 33)} {winner_str}  ({conf_1x2:.0f}%)",
        f"{L}  {_bar(conf_1x2)}",
        f"{L}  H {p_1x2[0]*100:.0f}%  │  D {p_1x2[1]*100:.0f}%  │  A {p_1x2[2]*100:.0f}%",
        f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫",
        f"{L}  📈 GOALS",
        f"{L}  {_conf_icon(conf_25, 50)} {str_o25}  ({conf_25:.1f}%)",
        f"{L}  {_bar(conf_25)}",
        f"{L}  {_conf_icon(conf_15, 50)} {str_o15}  ({conf_15:.1f}%)",
        f"{L}  {_bar(conf_15)}",
        f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫",
        f"{L}  🤝 BOTH TEAMS TO SCORE",
        f"{L}  {_conf_icon(conf_btts, 50)} GG: {str_btts}  ({conf_btts:.1f}%)",
        f"{L}  {_bar(conf_btts)}",
        f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫",
        f"{L}  🎯 EXACT SCORE",
    ]
    for s, p in top[:3]:
        pct = p * 100
        lines.append(f"{L}    {s}  ({pct:.0f}%)  {_bar(pct, 6)}")
    lines += [
        f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫",
        f"{L}  {adv}",
        f"{L}  {src_str}",
        f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛",
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

        # 1X2
        p_1x2 = res['1x2']
        pred_1x2 = ['H', 'D', 'A'][max(range(3), key=lambda i: p_1x2[i])]
        c1x2 = max(p_1x2) * 100

        # OU2.5
        o25 = "O" if res['ou25'] > 0.5 else "U"
        c25 = max(res['ou25'], 1-res['ou25']) * 100

        # OU1.5
        o15 = "O" if res['ou15'] > 0.5 else "U"
        c15 = max(res['ou15'], 1-res['ou15']) * 100

        # BTTS
        btts = "GG" if res['btts'] > 0.5 else "NG"
        cb = max(res['btts'], 1-res['btts']) * 100

        # Exact score
        score = res['exact_score']

        # High confidence check
        is_hi = (c1x2 - 33 >= 8) or (c25 - 50 >= 7) or (cb - 50 >= 7)
        if high_only and not is_hi: continue
        if is_hi: hi_n += 1
        flag = "🔥" if is_hi else "  "

        lines.append(f"┃ {flag} {home} vs {away}")
        lines.append(f"┃    {pred_1x2} {c1x2:.0f}% │ {o25}2.5 {c25:.0f}% │ {o15}1.5 {c15:.0f}%")
        lines.append(f"┃    {btts} {cb:.0f}% │ {score}")

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
        [InlineKeyboardButton("🔮 Predict Match", callback_data='pred_home')],
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
        [InlineKeyboardButton("🔮 Predict Match", callback_data='pred_home')],
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

    detected_teams = []
    for team in TEAMS_LIST:
        target = team.lower()
        if target in text:
            detected_teams.append(team)

        # Common abbreviations
        if 'utd' in text and 'united' in target:
            detected_teams.append(team)
        if 'spurs' in text and 'tottenham' in target:
            detected_teams.append(team)
        if 'forest' in text and "nott" in target:
            detected_teams.append(team)

    detected_teams = list(dict.fromkeys(detected_teams))  # deduplicate preserving order

    if len(detected_teams) >= 2:
        home, away = detected_teams[0], detected_teams[1]

        # Determine order based on text position
        idx_t1 = text.find(home.lower()[:5])
        idx_t2 = text.find(away.lower()[:5])
        if idx_t1 > idx_t2 and idx_t2 != -1:
            home, away = away, home

        await update.message.reply_text(f"⏳ Analyzing {home} vs {away}...")
        res = run_predictions(home, away)
        r = format_unified_prediction(home, away, res)
        await update.message.reply_text(r)
    elif len(detected_teams) == 1:
        await update.message.reply_text(f"Found {detected_teams[0]} — need two teams!\nTry: {detected_teams[0]} vs Chelsea")
    else:
        await update.message.reply_text("⚽ Type a match to predict\nExample: Arsenal vs Chelsea\n\nOr use /start for the menu")


async def predict_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("Usage: /predict Liverpool Arsenal")
        return
    home = next((t for t in TEAMS_LIST if t.lower() == args[0].lower()), None)
    away = next((t for t in TEAMS_LIST if t.lower() == args[1].lower()), None)
    if not home: await update.message.reply_text(f"'{args[0]}' not found."); return
    if not away: await update.message.reply_text(f"'{args[1]}' not found."); return
    await update.message.reply_text(f"⏳ Analyzing {home} vs {away}...")
    res = run_predictions(home, away)
    r = format_unified_prediction(home, away, res)
    await update.message.reply_text(r)


# ============================================================
# MAIN
# ============================================================
def main():
    if not BOT_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN environment variable")
        return
    print("\nStarting bot...")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict_cmd))
    app.add_handler(CallbackQueryHandler(btn))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, msg))
    print("Bot live! Send /start in Telegram.\n")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
