"""
⚽ Football Prediction Telegram Bot v3.0 (Advanced Pipeline)
===========================================================
Uses Dixon-Coles goal modeling + LightGBM stacked ensemble.
Predicts Over/Under 2.5 and BTTS (Both Teams To Score).

SETUP:
  pip install python-telegram-bot scikit-learn pandas numpy requests lightgbm scipy
  set TELEGRAM_BOT_TOKEN=your_token
  set ODDS_API_KEY=your_key  (free at https://the-odds-api.com)
  python football_bot_FINAL.py
"""

import os, logging, json, requests, pickle
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

from feature_engine import build_rolling_features

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "7780579030:AAHmZ4Bqi4y4B-Y5bTe3GWbCeitmfcedCHY") # Replace with env logic if preferred
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
BASE_DIR = Path(__file__).parent
HISTORY_PATH = BASE_DIR / "prediction_history.json"
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
    LGB_BTTS = lgb.Booster(model_file='lgb_btts.txt')
    print(f"✅ Loaded pipeline successfully! ({len(HISTORICAL_DF)} matches in DB)")
    
    # Calculate some summary stats for the UI
    TEAMS_LIST = sorted(list(set(HISTORICAL_DF['HomeTeam'].unique()) | set(HISTORICAL_DF['AwayTeam'].unique())))
    MEDIAN_ODDS = HISTORICAL_DF.median(numeric_only=True).to_dict()
    
    # Validation scores (from CV)
    ACC_OU25 = 0.546
    ACC_BTTS = 0.551
except Exception as e:
    print(f"❌ Failed to load models: {e}")
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
        e = "⬆️⬇️" if "O/U" in p['type'] else "⚽"
        c = "🟢" if p['confidence']>=55 else ("🟡" if p['confidence']>=52 else "🔴")
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
# PREDICTION ENGINE (Live feature generation & inference)
# ============================================================
def generate_match_features(home, away, b365h=None, b365d=None, b365a=None, b365_over=None, b365_under=None):
    """
    Appends a dummy row to historical_df, computes rolling features using the feature_engine,
    and returns the fully processed features for the new match.
    """
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

    # 2. Extract relevant recent matches (last 20 for speed) to avoid building full history
    # Or simply run the builder on the last N matches per team.
    # To keep code simple, we'll run it on the subset of data involving the two teams.
    mask = (HISTORICAL_DF['HomeTeam'].isin([home, away])) | (HISTORICAL_DF['AwayTeam'].isin([home, away]))
    mini_df = HISTORICAL_DF[mask].copy().sort_values('Date')
    
    # Create the new match row
    new_match = pd.DataFrame([{
        'HomeTeam': home, 'AwayTeam': away, 
        'Date': pd.to_datetime(datetime.now()), # Predict today
        'B365H': bh, 'B365D': bd, 'B365A': ba,
        'B365>2.5': bo, 'B365<2.5': bu,
        'FTHG': np.nan, 'FTAG': np.nan # Unknown
    }])
    
    # 3. Append and Build
    context_df = pd.concat([mini_df, new_match], ignore_index=True)
    df_built = build_rolling_features(context_df, windows=(3, 5, 10))
    
    # Extract the last row (our new match)
    last_row = df_built.iloc[-1:]
    
    # 4. Impute
    raw_feats = last_row[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
    X_imputed = IMPUTER.transform(raw_feats)
    X = np.nan_to_num(X_imputed, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 5. Dixon-Coles features
    dc_ou25 = DC_MODEL.predict_ou25(home, away)
    dc_btts = DC_MODEL.predict_btts(home, away)
    
    return X, dc_ou25, dc_btts, odds_src, last_row

def run_predictions(home, away, b365h=None, b365d=None, b365a=None, b365_over=None, b365_under=None):
    """Run all models for a given matchup."""
    X, dc_ou25, dc_btts, odds_src, row_df = generate_match_features(home, away, b365h, b365d, b365a, b365_over, b365_under)
    
    # Run O/U 2.5
    X_ou25 = np.nan_to_num(np.column_stack([X, [dc_ou25]]), nan=0.0, posinf=0.0, neginf=0.0)
    lgb_p_ou25 = LGB_OU25.predict(X_ou25)[0]
    ens_ou25 = 0.3 * dc_ou25 + 0.7 * lgb_p_ou25
    
    # Run BTTS
    X_btts = np.nan_to_num(np.column_stack([X, [dc_btts]]), nan=0.0, posinf=0.0, neginf=0.0)
    lgb_p_btts = LGB_BTTS.predict(X_btts)[0]
    ens_btts = 0.3 * dc_btts + 0.7 * lgb_p_btts
    
    return {
        'ou25': ens_ou25,
        'btts': ens_btts,
        'dc_ou25': dc_ou25,
        'dc_btts': dc_btts,
        'odds_src': odds_src,
        'row': row_df
    }


def format_prediction_ou(home, away, result_dict):
    p_over = result_dict['ou25']
    p_under = 1.0 - p_over
    
    pred = "⬆️ OVER 2.5" if p_over > 0.5 else "⬇️ UNDER 2.5"
    conf = max(p_over, p_under) * 100
    
    if conf >= 55:    cl, adv = "🟢 HIGH", "✅ Strong edge detected"
    elif conf >= 52:  cl, adv = "🟡 MEDIUM", "⚠️ Moderate edge"
    else:             cl, adv = "🔴 LOW", "❌ Skip this match (No edge)"

    bo = "█" * int(p_over * 10) + "░" * (10 - int(p_over * 10))
    bu = "█" * int(p_under * 10) + "░" * (10 - int(p_under * 10))

    save_prediction("O/U 2.5", home, away, pred.replace("⬆️ ", "").replace("⬇️ ", ""), conf)
    otag = "📡 _Live odds_" if result_dict['odds_src'] == "live" else "📊 _Avg odds_"

    return f"""
⚽ *{home} vs {away}*
━━━━━━━━━━━━━━━━━━━━
🎯 *{pred}*
📊 Confidence: {cl} ({conf:.1f}%)
{adv}

📈 *Probability Engine*
Over 2.5  {bo} {p_over*100:.1f}%
Under 2.5 {bu} {p_under*100:.1f}%

📋 *Model Inputs*
Dixon-Coles estimate: O2.5 @ {result_dict['dc_ou25']*100:.1f}%
{otag}
━━━━━━━━━━━━━━━━━━━━
_Bot Accuracy (Stack CV): {ACC_OU25*100:.1f}%_
"""

def format_prediction_btts(home, away, result_dict):
    p_yes = result_dict['btts']
    p_no = 1.0 - p_yes
    
    pred = "⚽ BTTS: YES" if p_yes > 0.5 else "🛑 BTTS: NO"
    conf = max(p_yes, p_no) * 100
    
    if conf >= 55:    cl, adv = "🟢 HIGH", "✅ Strong edge detected"
    elif conf >= 52:  cl, adv = "🟡 MEDIUM", "⚠️ Moderate edge"
    else:             cl, adv = "🔴 LOW", "❌ Skip this match (No edge)"

    by = "█" * int(p_yes * 10) + "░" * (10 - int(p_yes * 10))
    bn = "█" * int(p_no * 10) + "░" * (10 - int(p_no * 10))

    save_prediction("BTTS", home, away, pred.replace("⚽ BTTS: ", "").replace("🛑 BTTS: ", ""), conf)
    otag = "📡 _Live odds_" if result_dict['odds_src'] == "live" else "📊 _Avg odds_"

    return f"""
🏆 *{home} vs {away}*
━━━━━━━━━━━━━━━━━━━━
🎯 *{pred}*
📊 Confidence: {cl} ({conf:.1f}%)
{adv}

📈 *Probability Engine*
BTTS: Yes {by} {p_yes*100:.1f}%
BTTS: No  {bn} {p_no*100:.1f}%

📋 *Model Inputs*
Dixon-Coles estimate: BTTS-Y @ {result_dict['dc_btts']*100:.1f}%
{otag}
━━━━━━━━━━━━━━━━━━━━
_Bot Accuracy (Stack CV): {ACC_BTTS*100:.1f}%_
"""


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
        if home not in TEAMS_LIST or away not in TEAMS_LIST:
            lines.append(f"\n⚠️ *{home} vs {away}* — not in DB")
            continue

        res = run_predictions(home, away)
        
        o25 = "⬆️ O2.5" if res['ou25'] > 0.5 else "⬇️ U2.5"
        c25 = max(res['ou25'], 1-res['ou25']) * 100
        
        btts = "⚽ BTTS-Y" if res['btts'] > 0.5 else "🛑 BTTS-N"
        cb = max(res['btts'], 1-res['btts']) * 100

        is_hi = c25 >= 55 or cb >= 55
        if high_only and not is_hi: continue
        if is_hi: hi_n += 1
        flag = "🔥" if is_hi else "•"

        lines.append(
            f"\n{flag} *{home} vs {away}*\n"
            f"  O/U 2.5: {o25} ({c25:.1f}%)\n"
            f"  BTTS   : {btts} ({cb:.1f}%)")

    if high_only and hi_n == 0:
        lines.append("\n_No high confidence picks this week._")
    lines.append(f"\n━━━━━━━━━━━━━━━━━━━━\n🔥 High conf (>= 55%) | Week {idx+1}/{len(fx)}")
    
    # Generate fixture buttons
    fx_btns = []
    for home, away in week.get('matches', []):
        if home in TEAMS_LIST and away in TEAMS_LIST:
            # We use indices to save callback data space (Telegram limit is 64 bytes)
            h_idx = TEAMS_LIST.index(home)
            a_idx = TEAMS_LIST.index(away)
            fx_btns.append([InlineKeyboardButton(f"🔮 Predict: {home} vs {away}", callback_data=f'fx_{h_idx}_{a_idx}')])
            
    return "\n".join(lines), idx, fx_btns


# ============================================================
# TELEGRAM HANDLERS
# ============================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("⬆️⬇️ O/U 2.5", callback_data='ou25_home'),
         InlineKeyboardButton("⚽ BTTS", callback_data='btts_home')],
        [InlineKeyboardButton("📅 This Week", callback_data='wk_0'),
         InlineKeyboardButton("🔥 High Conf", callback_data='wkh_0')],
        [InlineKeyboardButton("📜 Past Predictions", callback_data='hist')],
        [InlineKeyboardButton("ℹ️ How it works", callback_data='how')],
    ]
    odds_s = "📡 Live odds: ON ✅" if ODDS_API_KEY else "📊 Live odds: OFF _(set ODDS_API_KEY)_"
    await update.message.reply_text(
        f"🏆 *Football Prediction Bot v3.0*\n\n"
        f"Trained on *{len(HISTORICAL_DF)}* real EPL matches\n"
        f"Powered by Dixon-Coles Model + LightGBM 🧠\n\n"
        f"📊 *Baseline Edge (Stack CV Accuracy):*\n"
        f"  O/U 2.5: {ACC_OU25*100:.1f}% vs Bookmaker ~50%\n"
        f"  BTTS   : {ACC_BTTS*100:.1f}% vs Bookmaker ~50%\n"
        f"{odds_s}\n\n"
        f"💡 _Only act on_ 🟢 _HIGH confidence picks!_ (>= 55%)\n\nChoose:",
        reply_markup=InlineKeyboardMarkup(kb))


def menu_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⬆️⬇️ O/U 2.5", callback_data='ou25_home'),
         InlineKeyboardButton("⚽ BTTS", callback_data='btts_home')],
        [InlineKeyboardButton("📅 Fixtures", callback_data='wk_0'),
         InlineKeyboardButton("📜 History", callback_data='hist')],
        [InlineKeyboardButton("ℹ️ Info", callback_data='how')],
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


def week_nav(idx, hi=False, fx_btns=None):
    fx = load_fixtures()
    mx = len(fx) - 1
    p = 'wkh_' if hi else 'wk_'
    btns = []
    if fx_btns:
        btns.extend(fx_btns)
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
                                  reply_markup=menu_kb())

    elif d == 'ou25_home':
        await q.edit_message_text("🏠 *O/U 2.5 — HOME team:*",
                                  reply_markup=team_kb('ouh_'))
    elif d == 'btts_home':
        await q.edit_message_text("🏠 *BTTS — HOME team:*",
                                  reply_markup=team_kb('btth_'))

    elif d.startswith('ouh_'):
        home = d[4:]
        context.user_data['ou_home'] = home
        await q.edit_message_text(f"✈️ *O/U 2.5 — AWAY team:*\n_(Home: {home})_",
                                  reply_markup=team_kb('oua_', exclude=home))

    elif d.startswith('oua_'):
        away = d[4:]
        home = context.user_data.get('ou_home')
        context.user_data['ou_away'] = away
        # Instantly run prediction
        await q.edit_message_text(f"⏳ _Analyzing {home} vs {away}..._")
        res = run_predictions(home, away)
        result_txt = format_prediction_ou(home, away, res)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 New O/U", callback_data='ou25_home'),
             InlineKeyboardButton("⚽ BTTS", callback_data='btts_home')],
            [InlineKeyboardButton("🔙 Menu", callback_data='menu')],
        ])
        await q.edit_message_text(result_txt, reply_markup=kb)

    elif d.startswith('btth_'):
        home = d[5:]
        context.user_data['btts_home'] = home
        await q.edit_message_text(f"✈️ *BTTS — AWAY team:*\n_(Home: {home})_",
                                  reply_markup=team_kb('btta_', exclude=home))

    elif d.startswith('btta_'):
        away = d[5:]
        home = context.user_data.get('btts_home')
        context.user_data['btts_away'] = away
        # Instantly run prediction
        await q.edit_message_text(f"⏳ _Analyzing {home} vs {away}..._")
        res = run_predictions(home, away)
        result_txt = format_prediction_btts(home, away, res)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 New BTTS", callback_data='btts_home'),
             InlineKeyboardButton("⬆️⬇️ O/U", callback_data='ou25_home')],
            [InlineKeyboardButton("🔙 Menu", callback_data='menu')],
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
        await q.edit_message_text(f"⏳ _Analyzing {home} vs {away}..._")
        res = run_predictions(home, away)
        r1 = format_prediction_ou(home, away, res)
        r2 = format_prediction_btts(home, away, res)
        # Send a new message since there are two long texts
        await q.message.reply_text(r1)
        await q.message.reply_text(r2)
        # Reset the original message to menu
        await q.edit_message_text("🏆 *Football Prediction Bot*\n\nChoose:", reply_markup=menu_kb())

    elif d == 'hist':
        await q.edit_message_text(get_history_text(),
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Menu", callback_data='menu')]]))

    elif d == 'how':
        await q.edit_message_text(
            "ℹ️ *How It Works v3.0*\n\n"
            "🧠 *Model Pipeline:*\n"
            "1️⃣ Dixon-Coles statistical goal model\n"
            "2️⃣ Rolling stats engineer (xG, differential, form)\n"
            "3️⃣ LightGBM classifier ensemble\n\n"
            f"📊 *Data:* {len(HISTORICAL_DF)} real EPL matches with actual results\n"
            f"🔧 *Features:* {len(FEATURE_COLS)} dynamically generated at inference\n\n"
            "📐 *No Data Leakage:*\n"
            "• Rolling stats use shift(1) — only past matches\n"
            "• Time-ordered expanding window Cross-Validation\n\n"
            f"🎯 *Edge (Log Loss optimized):*\n"
            f"  O/U 2.5: {ACC_OU25*100:.1f}% vs Bookmaker ~50%\n"
            f"  BTTS: {ACC_BTTS*100:.1f}% vs Bookmaker ~50%\n\n"
            "📡 *Live Odds:* Auto-fetched via The Odds API\n\n"
            "💡 *Strategy:* Only bet on 🟢 HIGH confidence picks (>=55%).\n"
            "⚠️ Gamble responsibly. Past performance is no guarantee of future returns.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Menu", callback_data='menu')]]))


import re

# Smart text search handler
async def msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().lower()
    
    # Check for team names in the message
    detected_teams = []
    for team in TEAMS_LIST:
        # Check simple variations
        target = team.lower()
        if target in text or target.replace(' man', 'mancester') in text or target.replace('man ', 'manchester ') in text:
            detected_teams.append(team)
            
        # Common abbreviations
        if 'utd' in text and 'united' in target: detected_teams.append(team)
        if 'spurs' in text and team == 'Tottenham Hotspur': detected_teams.append(team)
        if 'forest' in text and team == 'Nottingham Forest': detected_teams.append(team)
        
    detected_teams = list(set(detected_teams)) # remove duplicates
    
    if len(detected_teams) == 2:
        home, away = detected_teams[0], detected_teams[1]
        
        # Determine strict home/away based on exact order in text if possible
        idx_t1 = text.find(home.lower()[:5])
        idx_t2 = text.find(away.lower()[:5])
        if idx_t1 > idx_t2 and idx_t2 != -1:  # swapped order
            home, away = away, home
            
        await update.message.reply_text(f"🔍 Found: *{home} vs {away}*\n⏳ _Analyzing..._")
        res = run_predictions(home, away)
        r1 = format_prediction_ou(home, away, res)
        r2 = format_prediction_btts(home, away, res)
        await update.message.reply_text(r1)
        await update.message.reply_text(r2)
    elif len(detected_teams) == 1:
        await update.message.reply_text(f"🔍 Found {detected_teams[0]}, but I need two teams! Try: `{detected_teams[0]} vs Chelsea`")
    else:
        # Fallback to normal behavior
        await update.message.reply_text("⚽ Just type a match (e.g. `Arsenal vs Chelsea`) or use /start to open the menu!")


async def predict_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("Usage: `/predict Liverpool Arsenal`")
        return
    home = next((t for t in TEAMS_LIST if t.lower() == args[0].lower()), None)
    away = next((t for t in TEAMS_LIST if t.lower() == args[1].lower()), None)
    if not home: await update.message.reply_text(f"❌ '{args[0]}' not found."); return
    if not away: await update.message.reply_text(f"❌ '{args[1]}' not found."); return
    res = run_predictions(home, away)
    r1 = format_prediction_ou(home, away, res)
    r2 = format_prediction_btts(home, away, res)
    await update.message.reply_text(r1)
    await update.message.reply_text(r2)


# ============================================================
# MAIN
# ============================================================
def main():
    if not BOT_TOKEN or BOT_TOKEN == "PASTE_YOUR_TOKEN_HERE":
        print("❌ Set TELEGRAM_BOT_TOKEN")
        return
    print("\n🚀 Starting bot...")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict_cmd))
    app.add_handler(CallbackQueryHandler(btn))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, msg))
    print("✅ Bot live! Send /start in Telegram.\n")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
