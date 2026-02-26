"""
Results Tracking System: stores predictions, fetches actual results, computes accuracy.
"""
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from io import StringIO

DB_PATH = Path(__file__).parent / "predictions_db.json"
RESULTS_CSV_URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"


def _load_db():
    if DB_PATH.exists():
        with open(DB_PATH, 'r') as f:
            return json.load(f)
    return {"predictions": [], "last_settled": None}


def _save_db(db):
    with open(DB_PATH, 'w') as f:
        json.dump(db, f, indent=2)


def save_match_prediction(home, away, date_str, res):
    """Store a full prediction for all 4 markets."""
    db = _load_db()

    pred_id = f"{date_str}_{home}_{away}"

    # Check for duplicate
    for p in db['predictions']:
        if p['id'] == pred_id:
            return  # already saved

    # 1X2
    p_1x2 = res['1x2']
    winner_idx = int(max(range(3), key=lambda i: p_1x2[i]))
    pred_1x2 = ['H', 'D', 'A'][winner_idx]
    conf_1x2 = float(p_1x2[winner_idx] * 100)

    # OU 2.5
    pred_ou25 = "Over" if res['ou25'] > 0.5 else "Under"
    conf_ou25 = float(max(res['ou25'], 1 - res['ou25']) * 100)

    # BTTS
    pred_btts = "Yes" if res['btts'] > 0.5 else "No"
    conf_btts = float(max(res['btts'], 1 - res['btts']) * 100)

    # Exact score
    pred_score = res['exact_score']
    score_prob = float(res['exact_score_prob'] * 100)

    record = {
        "id": pred_id,
        "date": date_str,
        "home": home,
        "away": away,
        "pred_1x2": pred_1x2,
        "conf_1x2": round(conf_1x2, 1),
        "pred_ou25": pred_ou25,
        "conf_ou25": round(conf_ou25, 1),
        "pred_btts": pred_btts,
        "conf_btts": round(conf_btts, 1),
        "pred_score": pred_score,
        "score_prob": round(score_prob, 1),
        "actual_ftr": None,
        "actual_score": None,
        "actual_goals": None,
        "settled": False,
    }

    db['predictions'].append(record)
    # Keep last 500 predictions
    db['predictions'] = db['predictions'][-500:]
    _save_db(db)


def fetch_actual_results():
    """Download current season results from football-data.co.uk."""
    try:
        resp = requests.get(RESULTS_CSV_URL, timeout=15)
        if resp.status_code != 200:
            return None
        csv_df = pd.read_csv(StringIO(resp.text))
        csv_df['Date'] = pd.to_datetime(csv_df['Date'], dayfirst=True, errors='coerce')
        return csv_df
    except Exception:
        return None


def settle_predictions():
    """Compare stored predictions against actual results."""
    db = _load_db()
    unsettled = [p for p in db['predictions'] if not p['settled']]
    if not unsettled:
        return 0

    results_df = fetch_actual_results()
    if results_df is None or results_df.empty:
        return 0

    settled_count = 0
    for pred in unsettled:
        home, away = pred['home'], pred['away']

        # Find matching result
        mask = (results_df['HomeTeam'] == home) & (results_df['AwayTeam'] == away)
        matches = results_df[mask]
        if matches.empty:
            continue

        # Take the most recent match
        match = matches.iloc[-1]
        fthg = match.get('FTHG')
        ftag = match.get('FTAG')
        ftr = match.get('FTR')

        if pd.isna(fthg) or pd.isna(ftag) or pd.isna(ftr):
            continue

        fthg, ftag = int(fthg), int(ftag)
        total_goals = fthg + ftag

        pred['actual_ftr'] = ftr
        pred['actual_score'] = f"{fthg}-{ftag}"
        pred['actual_goals'] = total_goals

        # Check each market
        pred['correct_1x2'] = pred['pred_1x2'] == ftr
        pred['correct_ou25'] = (pred['pred_ou25'] == "Over" and total_goals > 2.5) or \
                                (pred['pred_ou25'] == "Under" and total_goals <= 2.5)
        pred['correct_btts'] = (pred['pred_btts'] == "Yes" and fthg > 0 and ftag > 0) or \
                                (pred['pred_btts'] == "No" and (fthg == 0 or ftag == 0))
        pred['correct_score'] = pred['pred_score'] == f"{fthg}-{ftag}"

        pred['settled'] = True
        settled_count += 1

    db['last_settled'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    _save_db(db)
    return settled_count


def get_accuracy_summary():
    """Return accuracy stats per market across settled predictions."""
    db = _load_db()
    settled = [p for p in db['predictions'] if p.get('settled')]

    if not settled:
        return None

    # Use last 50 settled predictions
    recent = settled[-50:]
    n = len(recent)

    stats = {
        '1x2': sum(1 for p in recent if p.get('correct_1x2', False)),
        'ou25': sum(1 for p in recent if p.get('correct_ou25', False)),
        'btts': sum(1 for p in recent if p.get('correct_btts', False)),
        'score': sum(1 for p in recent if p.get('correct_score', False)),
        'total': n,
    }
    return stats


def _bar(pct, width=8):
    """Create a visual bar."""
    filled = round(pct / 100 * width)
    return "▓" * filled + "░" * (width - filled)


def get_results_text(limit=15):
    """Format recent results for Telegram display."""
    db = _load_db()
    settled = [p for p in db['predictions'] if p.get('settled')]

    if not settled:
        return (
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃  📊 PREDICTION RESULTS\n"
            "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n"
            "┃  No settled predictions yet.\n"
            "┃  Results appear after matches.\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
        )

    stats = get_accuracy_summary()
    n = stats['total']

    pct_1x2 = stats['1x2'] / n * 100
    pct_ou25 = stats['ou25'] / n * 100
    pct_btts = stats['btts'] / n * 100
    pct_score = stats['score'] / n * 100

    L = "┃"
    lines = [
        "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓",
        f"{L}  📊 PREDICTION RESULTS",
        f"{L}  Last {n} settled predictions",
        "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫",
        f"{L}  🏆 1X2:   {stats['1x2']}/{n}  ({pct_1x2:.0f}%)  {_bar(pct_1x2)}",
        f"{L}  📈 OU2.5: {stats['ou25']}/{n}  ({pct_ou25:.0f}%)  {_bar(pct_ou25)}",
        f"{L}  🤝 BTTS:  {stats['btts']}/{n}  ({pct_btts:.0f}%)  {_bar(pct_btts)}",
        f"{L}  🎯 Score: {stats['score']}/{n}  ({pct_score:.0f}%)  {_bar(pct_score)}",
        "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┫",
        f"{L}  RECENT MATCHES",
    ]

    recent = settled[-limit:][::-1]
    for p in recent:
        c1 = "✅" if p.get('correct_1x2') else "❌"
        c2 = "✅" if p.get('correct_ou25') else "❌"
        c3 = "✅" if p.get('correct_btts') else "❌"
        c4 = "✅" if p.get('correct_score') else "❌"
        lines.append(f"{L}")
        lines.append(f"{L}  ⚽ {p['home']} {p['actual_score']} {p['away']}")
        lines.append(f"{L}  {c1} {p['pred_1x2']}  {c2} {p['pred_ou25']}  {c3} {p['pred_btts']}  {c4} {p['pred_score']}")

    lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

    return "\n".join(lines)
