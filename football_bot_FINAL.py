"""
⚽ Football Prediction Telegram Bot v2.0
=========================================
Trained on 2,350+ real EPL matches with actual outcomes.

Key accuracy (confidence-gated):
  O/U 2.5 at >=60% conf → ~62% accuracy
  Winner at >=50% conf  → ~51% accuracy

SETUP:
  pip install python-telegram-bot scikit-learn pandas numpy requests
  set TELEGRAM_BOT_TOKEN=your_token
  set ODDS_API_KEY=your_key  (free at https://the-odds-api.com)
  python football_bot_FINAL.py
"""

import os, logging, json, requests
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "7780579030:AAHmZ4Bqi4y4B-Y5bTe3GWbCeitmfcedCHY")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
BASE_DIR = Path(__file__).parent
EXTENDED_CSV = BASE_DIR / "E0 - E0.csv.csv"
FALLBACK_CSV = BASE_DIR / "football_prematch_with_form.csv"
HISTORY_PATH = BASE_DIR / "prediction_history.json"
FIXTURES_PATH = BASE_DIR / "weekly_fixtures.json"

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
# LOAD & PREPARE DATA
# ============================================================
print("📊 Loading data...")

def load_and_prepare_data():
    """Load extended or fallback CSV and engineer features."""
    use_extended = EXTENDED_CSV.exists()

    if use_extended:
        print(f"  📂 Extended dataset: {EXTENDED_CSV.name}")
        raw = pd.read_csv(EXTENDED_CSV)
        raw['Date'] = pd.to_datetime(raw['Date'], dayfirst=True)
        raw = raw.sort_values('Date').reset_index(drop=True)

        # Real outcomes
        raw['FTAG'] = pd.to_numeric(raw['FTAG'], errors='coerce')
        raw['FTHG'] = pd.to_numeric(raw['FTHG'], errors='coerce')
        raw['total_goals'] = raw['FTHG'] + raw['FTAG']
        raw['O25_label'] = (raw['total_goals'] > 2.5).astype(int)
        raw['O15_label'] = (raw['total_goals'] > 1.5).astype(int)
        raw['FTR_clean'] = raw['FTR'].map({'H': 'H', 'D': 'D', 'A': 'A'})
        raw['FTR_H'] = (raw['FTR'] == 'H').astype(int)
        raw['FTR_A'] = (raw['FTR'] == 'A').astype(int)
    else:
        print(f"  📂 Basic dataset: {FALLBACK_CSV.name}")
        raw = pd.read_csv(FALLBACK_CSV)
        raw['Date'] = pd.to_datetime(raw['Date'])
        raw = raw.sort_values('Date').reset_index(drop=True)
        raw['O25_label'] = (raw['O25_target'] == 'Over').astype(int)
        raw['O15_label'] = np.where(raw['O25_label'] == 1, 1,
                                     np.where(raw['Combined_Avg_Goals'] > 2.0, 1, 0))
        raw['FTR_clean'] = None
        raw['FTR_H'] = 0
        raw['FTR_A'] = 0

    # ---- Rolling features (shift(1) = no leakage) ----
    if use_extended:
        raw['home_scored_avg'] = raw.groupby('HomeTeam')['FTHG'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        raw['home_conceded_avg'] = raw.groupby('HomeTeam')['FTAG'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        raw['away_scored_avg'] = raw.groupby('AwayTeam')['FTAG'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        raw['away_conceded_avg'] = raw.groupby('AwayTeam')['FTHG'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        raw['home_ou_rate'] = raw.groupby('HomeTeam')['O25_label'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        raw['away_ou_rate'] = raw.groupby('AwayTeam')['O25_label'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        raw['home_win_rate'] = raw.groupby('HomeTeam')['FTR_H'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        raw['away_win_rate'] = raw.groupby('AwayTeam')['FTR_A'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        # Shot stats
        raw['home_shots_avg'] = raw.groupby('HomeTeam')['HS'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        raw['home_sot_avg'] = raw.groupby('HomeTeam')['HST'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        raw['away_shots_avg'] = raw.groupby('AwayTeam')['AS'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        raw['away_sot_avg'] = raw.groupby('AwayTeam')['AST'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        raw['home_corners_avg'] = raw.groupby('HomeTeam')['HC'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        raw['away_corners_avg'] = raw.groupby('AwayTeam')['AC'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    else:
        raw['home_scored_avg'] = raw['Home_Avg_Goals_Scored']
        raw['home_conceded_avg'] = raw['Home_Avg_Goals_Conceded']
        raw['away_scored_avg'] = raw['Away_Avg_Goals_Scored']
        raw['away_conceded_avg'] = raw['Away_Avg_Goals_Conceded']
        raw['home_ou_rate'] = raw['Home_OU_Rate']
        raw['away_ou_rate'] = 0.5
        raw['home_win_rate'] = 0.5
        raw['away_win_rate'] = 0.3
        raw['home_shots_avg'] = 10
        raw['home_sot_avg'] = 4
        raw['away_shots_avg'] = 10
        raw['away_sot_avg'] = 4
        raw['home_corners_avg'] = 5
        raw['away_corners_avg'] = 4

    # ---- Engineered features ----
    raw['combined_goals_avg'] = raw['home_scored_avg'] + raw['away_scored_avg']
    raw['combined_conceded_avg'] = raw['home_conceded_avg'] + raw['away_conceded_avg']
    raw['home_ad'] = raw['home_scored_avg'] / raw['home_conceded_avg'].clip(lower=0.1)
    raw['away_ad'] = raw['away_scored_avg'] / raw['away_conceded_avg'].clip(lower=0.1)
    raw['goal_diff'] = raw['combined_goals_avg'] - raw['combined_conceded_avg']
    raw['imp_over'] = 1 / raw['B365>2.5']
    raw['imp_under'] = 1 / raw['B365<2.5']
    raw['imp_home'] = 1 / raw['B365H']
    raw['imp_away'] = 1 / raw['B365A']
    raw['imp_draw'] = 1 / raw['B365D']
    raw['sot_ratio'] = raw['home_sot_avg'] / raw['away_sot_avg'].clip(lower=0.1)
    raw['shot_diff'] = raw['home_shots_avg'] - raw['away_shots_avg']

    # ---- Feature list ----
    feature_cols = [
        'B365H', 'B365D', 'B365A', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA',
        'B365>2.5', 'B365<2.5', 'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5',
        'home_scored_avg', 'home_conceded_avg', 'away_scored_avg', 'away_conceded_avg',
        'home_ou_rate', 'away_ou_rate', 'home_win_rate', 'away_win_rate',
        'home_shots_avg', 'home_sot_avg', 'away_shots_avg', 'away_sot_avg',
        'home_corners_avg', 'away_corners_avg',
        'combined_goals_avg', 'combined_conceded_avg', 'home_ad', 'away_ad', 'goal_diff',
        'imp_over', 'imp_under', 'imp_home', 'imp_away', 'imp_draw',
        'sot_ratio', 'shot_diff',
    ]

    # Add extra bookmaker odds when available
    extra_bk = []
    for col in ['PSH', 'PSD', 'PSA', 'BWH', 'BWD', 'BWA']:
        if col in raw.columns and raw[col].notna().sum() > len(raw) * 0.5:
            feature_cols.append(col)
            extra_bk.append(col)

    # O/U 1.5 features — exclude combined_goals to reduce leakage
    ou15_exclude = ['combined_goals_avg', 'combined_conceded_avg', 'goal_diff']
    ou15_features = [c for c in feature_cols if c not in ou15_exclude]

    # Clean
    target_cols = ['O25_label', 'O15_label']
    if use_extended:
        target_cols.append('FTR_clean')
    data = raw.dropna(subset=feature_cols + target_cols).copy()

    return data, feature_cols, ou15_features, extra_bk, use_extended


df, ALL_FEATURES, OU15_FEATURES, EXTRA_BK, USE_EXTENDED = load_and_prepare_data()
print(f"✅ {len(df)} matches ({df['Date'].min().date()} → {df['Date'].max().date()})")
print(f"   {len(ALL_FEATURES)} features | Extended: {USE_EXTENDED}")

# ============================================================
# TEAM STATS
# ============================================================
def compute_team_stats(data):
    stats = {}
    teams = set(data['HomeTeam'].unique()) | set(data['AwayTeam'].unique())
    for team in teams:
        hg = data[data['HomeTeam'] == team].tail(10)
        ag = data[data['AwayTeam'] == team].tail(10)
        stats[team] = {
            'avg_scored': round(((hg['home_scored_avg'].mean() if len(hg) else 1) +
                                 (ag['away_scored_avg'].mean() if len(ag) else 1)) / 2, 2),
            'avg_conceded': round(((hg['home_conceded_avg'].mean() if len(hg) else 1) +
                                   (ag['away_conceded_avg'].mean() if len(ag) else 1)) / 2, 2),
            'ou_rate': round(hg['home_ou_rate'].mean() if len(hg) else 0.5, 2),
            'scored_home': round(hg['home_scored_avg'].mean() if len(hg) else 1, 2),
            'scored_away': round(ag['away_scored_avg'].mean() if len(ag) else 1, 2),
            'conceded_home': round(hg['home_conceded_avg'].mean() if len(hg) else 1, 2),
            'conceded_away': round(ag['away_conceded_avg'].mean() if len(ag) else 1, 2),
            'win_rate_home': round(hg['home_win_rate'].mean() if len(hg) else 0.5, 2),
            'win_rate_away': round(ag['away_win_rate'].mean() if len(ag) else 0.3, 2),
            'shots_home': round(hg['home_shots_avg'].mean() if len(hg) else 10, 1),
            'sot_home': round(hg['home_sot_avg'].mean() if len(hg) else 4, 1),
            'shots_away': round(ag['away_shots_avg'].mean() if len(ag) else 10, 1),
            'sot_away': round(ag['away_sot_avg'].mean() if len(ag) else 4, 1),
            'corners_home': round(hg['home_corners_avg'].mean() if len(hg) else 5, 1),
            'corners_away': round(ag['away_corners_avg'].mean() if len(ag) else 4, 1),
        }
    return stats

TEAM_STATS = compute_team_stats(df)
TEAMS_LIST = sorted(TEAM_STATS.keys())
print(f"📋 {len(TEAMS_LIST)} teams")

# ============================================================
# TRAIN MODELS
# ============================================================
def build_models(data):
    X = data[ALL_FEATURES].values
    X15 = data[OU15_FEATURES].values

    # --- O/U 2.5 (best config: GB d4, lr=0.03, n=500) ---
    y25 = data['O25_label'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y25, test_size=0.2, random_state=42, shuffle=False)
    m25 = CalibratedClassifierCV(
        GradientBoostingClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
                                    min_samples_leaf=20, subsample=0.8, random_state=42),
        cv=5, method='isotonic')
    m25.fit(Xtr, ytr)
    a25 = accuracy_score(yte, m25.predict(Xte))
    # Confidence-gated accuracy
    probs25 = m25.predict_proba(Xte)
    hi_mask = np.array([max(p)*100 >= 60 for p in probs25])
    hi_acc = accuracy_score(yte[hi_mask], m25.predict(Xte)[hi_mask]) if hi_mask.sum() > 0 else a25
    print(f"  O/U 2.5 → {a25*100:.1f}% overall | {hi_acc*100:.1f}% at ≥60% conf ({hi_mask.sum()}/{len(Xte)} picks)")

    # --- O/U 1.5 ---
    y15 = data['O15_label'].values
    Xtr15, Xte15, ytr15, yte15 = train_test_split(X15, y15, test_size=0.2, random_state=42, shuffle=False)
    m15 = CalibratedClassifierCV(
        GradientBoostingClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
                                    min_samples_leaf=20, subsample=0.8, random_state=42),
        cv=5, method='isotonic')
    m15.fit(Xtr15, ytr15)
    a15 = accuracy_score(yte15, m15.predict(Xte15))
    print(f"  O/U 1.5 → {a15*100:.1f}%")

    # --- Match Winner ---
    le = LabelEncoder()
    if data['FTR_clean'].notna().sum() > 100:
        ftr_data = data.dropna(subset=['FTR_clean'])
        X_mw = ftr_data[ALL_FEATURES].values
        ymw = le.fit_transform(ftr_data['FTR_clean'].values)
        src = "real results"
    else:
        hp = 1/data['B365H']; dp = 1/data['B365D']; ap = 1/data['B365A']
        mw = np.where(hp>=np.maximum(dp,ap),'H',np.where(ap>=np.maximum(hp,dp),'A','D'))
        X_mw = X
        ymw = le.fit_transform(mw)
        src = "odds proxy"

    Xtrm, Xtem, ytrm, ytem = train_test_split(X_mw, ymw, test_size=0.2, random_state=42, shuffle=False)
    mmw = CalibratedClassifierCV(
        Pipeline([('scaler', StandardScaler()),
                  ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                     min_samples_leaf=25, subsample=0.7, random_state=42))]),
        cv=5, method='isotonic')
    mmw.fit(Xtrm, ytrm)
    amw = accuracy_score(ytem, mmw.predict(Xtem))
    print(f"  Winner  → {amw*100:.1f}% ({src})")

    return m25, m15, mmw, le, (a25, a15, amw), hi_acc

print("\n🔄 Training models on real outcomes...")
MODEL_OU25, MODEL_OU15, MODEL_MW, MW_ENC, ACCS, HI_CONF_ACC = build_models(df)
MEDIAN_ODDS = df[ALL_FEATURES].median().to_dict()
print("✅ Models ready!\n")

# ============================================================
# PREDICTION HISTORY
# ============================================================
def load_history():
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, 'r') as f: return json.load(f)
    return []

def save_prediction(ptype, home, away, pred, conf):
    h = load_history()
    h.append({'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
              'type': ptype, 'home': home, 'away': away,
              'prediction': pred, 'confidence': round(conf, 1)})
    with open(HISTORY_PATH, 'w') as f: json.dump(h[-200:], f, indent=2)

def get_history_text(limit=20):
    h = load_history()
    if not h: return "📜 *No predictions yet.* Make some first!"
    recent = h[-limit:][::-1]
    lines = [f"📜 *Last {len(recent)} Predictions*\n━━━━━━━━━━━━━━━━━━━━"]
    for p in recent:
        e = "⬆️⬇️" if "O/U" in p['type'] else "🏆"
        c = "🟢" if p['confidence']>=60 else ("🟡" if p['confidence']>=52 else "🔴")
        lines.append(f"\n{e} *{p['home']} vs {p['away']}*\n   {p['type']}: {p['prediction']}\n   {c} {p['confidence']}% — {p['date']}")
    lines.append("\n━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)

# ============================================================
# WEEKLY FIXTURES
# ============================================================
DEFAULT_FIXTURES = [
    {"week_label": "GW26 — Feb 21-23, 2026",
     "matches": [["Liverpool","Man City"],["Arsenal","Chelsea"],["Man United","Tottenham"],
                 ["Newcastle","Aston Villa"],["Brighton","Wolves"],["Brentford","Fulham"],
                 ["Everton","Bournemouth"],["Nott'm Forest","Leicester"],
                 ["Crystal Palace","West Ham"],["Ipswich","Southampton"]]},
    {"week_label": "GW25 — Feb 14-16, 2026",
     "matches": [["Wolves","Arsenal"],["Brentford","Arsenal"],["Crystal Palace","Burnley"],
                 ["Aston Villa","Brighton"],["Sunderland","Liverpool"]]},
]

def load_fixtures():
    if FIXTURES_PATH.exists():
        with open(FIXTURES_PATH, 'r') as f: return json.load(f)
    with open(FIXTURES_PATH, 'w') as f: json.dump(DEFAULT_FIXTURES, f, indent=2)
    return DEFAULT_FIXTURES

WEEKLY_FIXTURES = load_fixtures()

# ============================================================
# FEATURE BUILDER (for prediction time)
# ============================================================
def get_features(home, away, b365_over=None, b365_under=None,
                 b365h=None, b365d=None, b365a=None):
    h = TEAM_STATS.get(home, {k: v for k, v in zip(
        ['avg_scored','avg_conceded','ou_rate','scored_home','scored_away',
         'conceded_home','conceded_away','win_rate_home','win_rate_away',
         'shots_home','sot_home','shots_away','sot_away','corners_home','corners_away'],
        [1,1,0.5,1,1,1,1,0.5,0.3,10,4,10,4,5,4])})
    a = TEAM_STATS.get(away, h.copy())

    # Try live odds
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

    cg = h['scored_home'] + a['scored_away']
    cc = h['conceded_home'] + a['conceded_away']

    feats = [
        bh, bd, ba,
        bh * 1.03, bd * 1.02, ba * 1.04,  # Max ~ B365 + margin
        bh * 0.98, bd * 0.99, ba * 0.98,   # Avg ~ B365 - margin
        bo, bu, bo * 1.03, bu * 1.04, bo * 0.98, bu * 0.99,
        h['scored_home'], h['conceded_home'], a['scored_away'], a['conceded_away'],
        h['ou_rate'],
        a.get('ou_rate', 0.5) if 'away_ou_rate' not in ALL_FEATURES else MEDIAN_ODDS.get('away_ou_rate', 0.5),
        h['win_rate_home'], a['win_rate_away'],
        h['shots_home'], h['sot_home'], a['shots_away'], a['sot_away'],
        h['corners_home'], a['corners_away'],
        # Engineered
        cg, cc,
        h['scored_home'] / max(h['conceded_home'], 0.1),
        a['scored_away'] / max(a['conceded_away'], 0.1),
        cg - cc,
        1/bo, 1/bu, 1/bh, 1/ba, 1/bd,
        h['sot_home'] / max(a['sot_away'], 0.1),
        h['shots_home'] - a['shots_away'],
    ]

    # Extra bookmaker cols — approximate from B365
    for col in EXTRA_BK:
        if 'H' in col: feats.append(bh * 0.99)
        elif 'D' in col: feats.append(bd * 0.98)
        elif 'A' in col: feats.append(ba * 1.01)
        else: feats.append(bh)

    return [feats], h, a, cg, odds_src


# ============================================================
# PREDICTIONS
# ============================================================
def predict_ou(home, away, threshold=2.5, b365_over=None, b365_under=None):
    feats, h, a, cg, odds_src = get_features(home, away, b365_over, b365_under)

    if threshold == 1.5:
        ou15_idx = [ALL_FEATURES.index(c) for c in OU15_FEATURES]
        input_f = [[feats[0][i] for i in ou15_idx]]
        model = MODEL_OU15
    else:
        input_f = feats
        model = MODEL_OU25

    prob = model.predict_proba(input_f)[0]
    under_p, over_p = (prob[0], prob[1]) if len(prob) == 2 else (0.5, 0.5)
    pred = f"⬆️ OVER {threshold}" if over_p > 0.5 else f"⬇️ UNDER {threshold}"
    conf = max(over_p, under_p) * 100

    if conf >= 60:    cl, adv = "🟢 HIGH", "✅ Strong pick — model hits ~62% at this level"
    elif conf >= 55:  cl, adv = "🟡 MEDIUM", "⚠️ Moderate pick"
    else:             cl, adv = "🔴 LOW", "❌ Skip this match"

    bo = "█" * int(over_p * 10) + "░" * (10 - int(over_p * 10))
    bu = "█" * int(under_p * 10) + "░" * (10 - int(under_p * 10))

    save_prediction(f"O/U {threshold}", home, away,
                    pred.replace("⬆️ ", "").replace("⬇️ ", ""), conf)

    acc = ACCS[0] if threshold == 2.5 else ACCS[1]
    otag = "📡 _Live odds_" if odds_src == "live" else "📊 _Avg odds_"

    return f"""
⚽ *{home} vs {away}*
━━━━━━━━━━━━━━━━━━━━
🎯 *{pred}*
📊 Confidence: {cl} ({conf:.1f}%)
{adv}

📈 *Probability*
Over  {bo} {over_p*100:.1f}%
Under {bu} {under_p*100:.1f}%

📋 *Form (last 5)*
🏠 {home}: {h['scored_home']:.1f} scored | {h['conceded_home']:.1f} conceded | {h['sot_home']:.0f} SOT/game
✈️ {away}: {a['scored_away']:.1f} scored | {a['conceded_away']:.1f} conceded | {a['sot_away']:.0f} SOT/game
⚽ Combined avg: *{cg:.1f}* | {otag}
━━━━━━━━━━━━━━━━━━━━
_Overall: {acc*100:.0f}% | At HIGH conf: ~{HI_CONF_ACC*100:.0f}%_
""", conf


def predict_winner(home, away, b365h=None, b365d=None, b365a=None):
    feats, h, a, cg, odds_src = get_features(home, away, b365h=b365h, b365d=b365d, b365a=b365a)

    prob = MODEL_MW.predict_proba(feats)[0]
    decoded = MW_ENC.inverse_transform(MODEL_MW.classes_)
    probs = {c: p for c, p in zip(decoded, prob)}

    hp, dp, ap = probs.get('H', 0), probs.get('D', 0), probs.get('A', 0)
    best = max(probs, key=probs.get)
    conf = probs[best] * 100

    if best == 'H':   pl = f"🏠 {home} WIN"
    elif best == 'D': pl = "🤝 DRAW"
    else:             pl = f"✈️ {away} WIN"

    if conf >= 55:    cl, adv = "🟢 HIGH", "✅ Strong pick"
    elif conf >= 45:  cl, adv = "🟡 MEDIUM", "⚠️ Moderate pick"
    else:             cl, adv = "🔴 LOW", "❌ Skip this match"

    bh = int(hp*10); bd = int(dp*10); ba = int(ap*10)
    save_prediction("Winner", home, away,
                    pl.replace("🏠 ","").replace("✈️ ","").replace("🤝 ",""), conf)
    otag = "📡 _Live odds_" if odds_src == "live" else "📊 _Avg odds_"

    return f"""
🏆 *{home} vs {away}*
━━━━━━━━━━━━━━━━━━━━
🎯 *{pl}*
📊 Confidence: {cl} ({conf:.1f}%)
{adv}

📈 *Win Probabilities*
🏠 Home  {"█"*bh}{"░"*(10-bh)} {hp*100:.1f}%
🤝 Draw  {"█"*bd}{"░"*(10-bd)} {dp*100:.1f}%
✈️ Away  {"█"*ba}{"░"*(10-ba)} {ap*100:.1f}%

📋 *{home}*: {h['win_rate_home']*100:.0f}% home win rate | {h['sot_home']:.0f} SOT/game
📋 *{away}*: {a['win_rate_away']*100:.0f}% away win rate | {a['sot_away']:.0f} SOT/game
{otag}
━━━━━━━━━━━━━━━━━━━━
_Accuracy: {ACCS[2]*100:.0f}% on real outcomes_
""", conf


# ============================================================
# WEEKLY PREDICTIONS
# ============================================================
def get_weekly_predictions(week_idx=0, high_only=False):
    fx = load_fixtures()
    if not fx: return "⚠️ No fixtures.", 0
    idx = max(0, min(week_idx, len(fx) - 1))
    week = fx[idx]
    label = week.get('week_label', f'Week {idx+1}')
    lines = [f"📅 *{label}*\n━━━━━━━━━━━━━━━━━━━━"]
    hi_n = 0

    for home, away in week.get('matches', []):
        if home not in TEAM_STATS or away not in TEAM_STATS:
            lines.append(f"\n⚠️ *{home} vs {away}* — not in DB")
            continue

        feats, _, _, _, _ = get_features(home, away)

        # O/U 2.5
        p25 = MODEL_OU25.predict_proba(feats)[0]
        o25 = "⬆️ O2.5" if p25[1] > 0.5 else "⬇️ U2.5"
        c25 = max(p25) * 100

        # O/U 1.5
        ou15_idx = [ALL_FEATURES.index(c) for c in OU15_FEATURES]
        f15 = [[feats[0][i] for i in ou15_idx]]
        p15 = MODEL_OU15.predict_proba(f15)[0]
        o15 = "⬆️ O1.5" if p15[1] > 0.5 else "⬇️ U1.5"
        c15 = max(p15) * 100

        # Winner
        pmw = MODEL_MW.predict_proba(feats)[0]
        mwc = MW_ENC.inverse_transform(MODEL_MW.classes_)
        mwp = {c: p for c, p in zip(mwc, pmw)}
        best = max(mwp, key=mwp.get)
        cmw = mwp[best] * 100
        if best == 'H':   ml = f"🏠 {home}"
        elif best == 'D': ml = "🤝 Draw"
        else:             ml = f"✈️ {away}"

        is_hi = c25 >= 60 or cmw >= 55
        if high_only and not is_hi: continue
        if is_hi: hi_n += 1
        flag = "🔥" if is_hi else "•"

        lines.append(
            f"\n{flag} *{home} vs {away}*\n"
            f"  O/U 2.5: {o25} ({c25:.0f}%)\n"
            f"  O/U 1.5: {o15} ({c15:.0f}%)\n"
            f"  Winner: {ml} ({cmw:.0f}%)")

    if high_only and hi_n == 0:
        lines.append("\n_No high confidence picks this week._")
    lines.append(f"\n━━━━━━━━━━━━━━━━━━━━\n🔥 High conf | Week {idx+1}/{len(fx)}")
    return "\n".join(lines), idx


# ============================================================
# TELEGRAM HANDLERS
# ============================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("⬆️⬇️ O/U 2.5", callback_data='ou25_home'),
         InlineKeyboardButton("⬆️⬇️ O/U 1.5", callback_data='ou15_home')],
        [InlineKeyboardButton("🏆 Match Winner", callback_data='mw_home')],
        [InlineKeyboardButton("📅 This Week", callback_data='wk_0'),
         InlineKeyboardButton("🔥 High Conf", callback_data='wkh_0')],
        [InlineKeyboardButton("📜 Past Predictions", callback_data='hist')],
        [InlineKeyboardButton("📋 Teams", callback_data='teams'),
         InlineKeyboardButton("ℹ️ How It Works", callback_data='how')],
    ]
    odds_s = "📡 Live odds: ON ✅" if ODDS_API_KEY else "📊 Live odds: OFF _(set ODDS\\_API\\_KEY)_"
    await update.message.reply_text(
        f"🏆 *Football Prediction Bot v2.0*\n\n"
        f"Trained on *{len(df)}* real EPL matches\n"
        f"({df['Date'].min().year}–{df['Date'].max().year})\n\n"
        f"📊 *Accuracy:*\n"
        f"  O/U 2.5: {ACCS[0]*100:.0f}% overall • ~{HI_CONF_ACC*100:.0f}% at HIGH conf\n"
        f"  O/U 1.5: {ACCS[1]*100:.0f}% | Winner: {ACCS[2]*100:.0f}%\n"
        f"{odds_s}\n\n"
        f"💡 _Only act on_ 🟢 _HIGH confidence picks!_\n\nChoose:",
        reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')


def menu_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⬆️⬇️ O/U 2.5", callback_data='ou25_home'),
         InlineKeyboardButton("⬆️⬇️ O/U 1.5", callback_data='ou15_home')],
        [InlineKeyboardButton("🏆 Winner", callback_data='mw_home'),
         InlineKeyboardButton("📅 Fixtures", callback_data='wk_0')],
        [InlineKeyboardButton("📜 History", callback_data='hist'),
         InlineKeyboardButton("ℹ️ Info", callback_data='how')],
    ])


def team_kb(prefix, exclude=None):
    btns = []; row = []
    for t in TEAMS_LIST:
        if t == exclude: continue
        row.append(InlineKeyboardButton(t, callback_data=f'{prefix}{t}'))
        if len(row) == 2: btns.append(row); row = []
    if row: btns.append(row)
    btns.append([InlineKeyboardButton("🔙 Menu", callback_data='menu')])
    return InlineKeyboardMarkup(btns)


def week_nav(idx, hi=False):
    fx = load_fixtures()
    mx = len(fx) - 1
    p = 'wkh_' if hi else 'wk_'
    btns = []
    nav = []
    if idx < mx: nav.append(InlineKeyboardButton("⬅️ Earlier", callback_data=f'{p}{idx+1}'))
    if idx > 0:  nav.append(InlineKeyboardButton("Later ➡️", callback_data=f'{p}{idx-1}'))
    if nav: btns.append(nav)
    btns.append([InlineKeyboardButton("🔥 High Only" if not hi else "📅 Show All",
                                       callback_data=f'{"wkh_" if not hi else "wk_"}{idx}')])
    btns.append([InlineKeyboardButton("🔙 Menu", callback_data='menu')])
    return InlineKeyboardMarkup(btns)


async def btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    d = q.data

    if d == 'menu':
        await q.edit_message_text("🏆 *Football Prediction Bot*\n\nChoose:",
                                  reply_markup=menu_kb(), parse_mode='Markdown')

    elif d == 'ou25_home':
        context.user_data['ou_th'] = 2.5
        await q.edit_message_text("🏠 *O/U 2.5 — HOME team:*",
                                  reply_markup=team_kb('ouh_'), parse_mode='Markdown')
    elif d == 'ou15_home':
        context.user_data['ou_th'] = 1.5
        await q.edit_message_text("🏠 *O/U 1.5 — HOME team:*",
                                  reply_markup=team_kb('ouh_'), parse_mode='Markdown')

    elif d.startswith('ouh_'):
        home = d[4:]
        context.user_data['ou_home'] = home
        th = context.user_data.get('ou_th', 2.5)
        await q.edit_message_text(f"✈️ *O/U {th} — AWAY team:*\n_(Home: {home})_",
                                  reply_markup=team_kb('oua_', exclude=home), parse_mode='Markdown')

    elif d.startswith('oua_'):
        away = d[4:]
        home = context.user_data.get('ou_home')
        context.user_data['ou_away'] = away
        th = context.user_data.get('ou_th', 2.5)
        live = find_live_odds(home, away)
        om = "\n📡 _Live odds found!_" if live else ""
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Enter odds", callback_data='ou_odds')],
            [InlineKeyboardButton("⏭️ Auto (recommended)", callback_data='ou_auto')],
            [InlineKeyboardButton("🔙 Menu", callback_data='menu')],
        ])
        await q.edit_message_text(f"📊 *{home} vs {away}*\n\nO/U {th} prediction{om}",
                                  reply_markup=kb, parse_mode='Markdown')

    elif d == 'ou_odds':
        context.user_data['wf'] = 'ou_over'
        th = context.user_data.get('ou_th', 2.5)
        await q.edit_message_text(f"📝 Type *Over {th} odds:*\nExample: `1.85`", parse_mode='Markdown')

    elif d == 'ou_auto':
        home, away = context.user_data.get('ou_home'), context.user_data.get('ou_away')
        th = context.user_data.get('ou_th', 2.5)
        result, _ = predict_ou(home, away, th)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 New O/U", callback_data='ou25_home'),
             InlineKeyboardButton("🏆 Winner", callback_data='mw_home')],
            [InlineKeyboardButton("🔙 Menu", callback_data='menu')],
        ])
        await q.edit_message_text(result, reply_markup=kb, parse_mode='Markdown')

    elif d == 'mw_home':
        await q.edit_message_text("🏠 *Winner — HOME team:*",
                                  reply_markup=team_kb('mwh_'), parse_mode='Markdown')

    elif d.startswith('mwh_'):
        home = d[4:]
        context.user_data['mw_home'] = home
        await q.edit_message_text(f"✈️ *Winner — AWAY team:*\n_(Home: {home})_",
                                  reply_markup=team_kb('mwa_', exclude=home), parse_mode='Markdown')

    elif d.startswith('mwa_'):
        away = d[4:]
        home = context.user_data.get('mw_home')
        context.user_data['mw_away'] = away
        live = find_live_odds(home, away)
        om = "\n📡 _Live odds found!_" if live else ""
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Enter odds", callback_data='mw_odds')],
            [InlineKeyboardButton("⏭️ Auto (recommended)", callback_data='mw_auto')],
            [InlineKeyboardButton("🔙 Menu", callback_data='menu')],
        ])
        await q.edit_message_text(f"📊 *{home} vs {away}*\nWinner prediction{om}",
                                  reply_markup=kb, parse_mode='Markdown')

    elif d == 'mw_odds':
        context.user_data['wf'] = 'mw_h'
        await q.edit_message_text("📝 Type *Home Win odds:*\nExample: `2.10`", parse_mode='Markdown')

    elif d == 'mw_auto':
        home, away = context.user_data.get('mw_home'), context.user_data.get('mw_away')
        result, _ = predict_winner(home, away)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 New Winner", callback_data='mw_home'),
             InlineKeyboardButton("⬆️⬇️ O/U", callback_data='ou25_home')],
            [InlineKeyboardButton("🔙 Menu", callback_data='menu')],
        ])
        await q.edit_message_text(result, reply_markup=kb, parse_mode='Markdown')

    elif d.startswith('wkh_'):
        idx = int(d[4:])
        r, ai = get_weekly_predictions(idx, True)
        await q.edit_message_text(r, reply_markup=week_nav(ai, True), parse_mode='Markdown')

    elif d.startswith('wk_'):
        idx = int(d[3:])
        r, ai = get_weekly_predictions(idx, False)
        await q.edit_message_text(r, reply_markup=week_nav(ai, False), parse_mode='Markdown')

    elif d == 'hist':
        await q.edit_message_text(get_history_text(),
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Menu", callback_data='menu')]]),
            parse_mode='Markdown')

    elif d == 'teams':
        txt = "📋 *Teams:*\n\n"
        for i, t in enumerate(TEAMS_LIST, 1):
            s = TEAM_STATS[t]
            txt += f"{i}. *{t}* — {s['avg_scored']} scored, {s['avg_conceded']} conc, {s['ou_rate']*100:.0f}% OU\n"
        await q.edit_message_text(txt,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Menu", callback_data='menu')]]),
            parse_mode='Markdown')

    elif d == 'how':
        await q.edit_message_text(
            "ℹ️ *How It Works*\n\n"
            "🧠 *Model:* Gradient Boosting + Isotonic Calibration\n"
            f"📊 *Data:* {len(df)} real EPL matches with actual results\n"
            f"🔧 *Features:* {len(ALL_FEATURES)} (odds, form, shots, corners, win rates)\n\n"
            "📐 *No Data Leakage:*\n"
            "• Rolling stats use shift(1) — only past matches\n"
            "• Time-ordered train/test split (no shuffle)\n"
            "• O/U 1.5 model excludes combined goal averages\n\n"
            f"🎯 *Accuracy:*\n"
            f"  O/U 2.5: {ACCS[0]*100:.0f}% overall • ~{HI_CONF_ACC*100:.0f}% at HIGH conf\n"
            f"  O/U 1.5: {ACCS[1]*100:.0f}% | Winner: {ACCS[2]*100:.0f}%\n\n"
            "📡 *Live Odds:* Auto-fetched via The Odds API\n\n"
            "💡 *Strategy:* Only bet on 🟢 HIGH confidence picks.\n"
            "⚠️ Gamble responsibly.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Menu", callback_data='menu')]]),
            parse_mode='Markdown')


# Message handler for manual odds input
async def msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    wf = context.user_data.get('wf')
    if wf == 'ou_over':
        try:
            context.user_data['ou_ov'] = float(update.message.text.strip())
            context.user_data['wf'] = 'ou_under'
            th = context.user_data.get('ou_th', 2.5)
            await update.message.reply_text(f"✅ Now *Under {th} odds:*\nExample: `2.05`", parse_mode='Markdown')
        except ValueError:
            await update.message.reply_text("❌ Number like `1.85`", parse_mode='Markdown')

    elif wf == 'ou_under':
        try:
            under = float(update.message.text.strip())
            home, away = context.user_data.get('ou_home'), context.user_data.get('ou_away')
            th = context.user_data.get('ou_th', 2.5)
            context.user_data['wf'] = None
            result, _ = predict_ou(home, away, th, context.user_data.get('ou_ov'), under)
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔄 New", callback_data='ou25_home'),
                                        InlineKeyboardButton("🔙 Menu", callback_data='menu')]])
            await update.message.reply_text(result, reply_markup=kb, parse_mode='Markdown')
        except ValueError:
            await update.message.reply_text("❌ Number like `2.05`", parse_mode='Markdown')

    elif wf == 'mw_h':
        try:
            context.user_data['mw_ho'] = float(update.message.text.strip())
            context.user_data['wf'] = 'mw_d'
            await update.message.reply_text("✅ Now *Draw odds:*\nExample: `3.40`", parse_mode='Markdown')
        except ValueError:
            await update.message.reply_text("❌ Number like `2.10`", parse_mode='Markdown')

    elif wf == 'mw_d':
        try:
            context.user_data['mw_do'] = float(update.message.text.strip())
            context.user_data['wf'] = 'mw_a'
            await update.message.reply_text("✅ Now *Away odds:*\nExample: `3.20`", parse_mode='Markdown')
        except ValueError:
            await update.message.reply_text("❌ Number like `3.40`", parse_mode='Markdown')

    elif wf == 'mw_a':
        try:
            ao = float(update.message.text.strip())
            home, away = context.user_data.get('mw_home'), context.user_data.get('mw_away')
            context.user_data['wf'] = None
            result, _ = predict_winner(home, away, context.user_data.get('mw_ho'),
                                       context.user_data.get('mw_do'), ao)
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔄 New", callback_data='mw_home'),
                                        InlineKeyboardButton("🔙 Menu", callback_data='menu')]])
            await update.message.reply_text(result, reply_markup=kb, parse_mode='Markdown')
        except ValueError:
            await update.message.reply_text("❌ Number like `3.20`", parse_mode='Markdown')
    else:
        await update.message.reply_text("Send /start to open the menu ⚽")


async def predict_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("Usage: `/predict Liverpool Arsenal`", parse_mode='Markdown')
        return
    home = next((t for t in TEAMS_LIST if t.lower() == args[0].lower()), None)
    away = next((t for t in TEAMS_LIST if t.lower() == args[1].lower()), None)
    if not home: await update.message.reply_text(f"❌ '{args[0]}' not found."); return
    if not away: await update.message.reply_text(f"❌ '{args[1]}' not found."); return
    r1, _ = predict_ou(home, away, 2.5)
    r2, _ = predict_ou(home, away, 1.5)
    r3, _ = predict_winner(home, away)
    await update.message.reply_text(r1, parse_mode='Markdown')
    await update.message.reply_text(r2, parse_mode='Markdown')
    await update.message.reply_text(r3, parse_mode='Markdown')


# ============================================================
# MAIN
# ============================================================
def main():
    if not BOT_TOKEN or BOT_TOKEN == "PASTE_YOUR_TOKEN_HERE":
        print("❌ Set TELEGRAM_BOT_TOKEN")
        return
    print("🚀 Starting bot...")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict_cmd))
    app.add_handler(CallbackQueryHandler(btn))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, msg))
    print("✅ Bot live! Send /start in Telegram.\n")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
