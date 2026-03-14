"""Results tracking and settlement for core and expanded EPL markets."""
import json
import os
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).parent
APP_DATA_DIR = Path(os.environ.get("APP_DATA_DIR", BASE_DIR))
DB_PATH = Path(os.environ.get("PREDICTIONS_DB_PATH", APP_DATA_DIR / "predictions_db.json"))
MARKET_MANIFEST_PATH = Path(__file__).parent / "market_models" / "manifest.json"


def _season_codes(now=None):
    now = now or datetime.now()
    start_year = now.year if now.month >= 7 else now.year - 1
    current = f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"
    previous = f"{(start_year - 1) % 100:02d}{start_year % 100:02d}"
    return [current, previous]

def _load_db():
    if DB_PATH.exists():
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"predictions": [], "last_settled": None}

def _save_db(db):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

def _load_manifest_map():
    if not MARKET_MANIFEST_PATH.exists():
        return {}
    with open(MARKET_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return {row["target"]: row for row in manifest}

def save_match_prediction(home, away, date_str, res):
    """Store a prediction snapshot for settlement and live accuracy tracking."""
    db = _load_db()
    pred_id = f"{date_str}_{home}_{away}"
    for p in db["predictions"]:
        if p["id"] == pred_id:
            return

    p_1x2 = res["1x2"]
    winner_idx = int(max(range(3), key=lambda i: p_1x2[i]))
    record = {
        "id": pred_id,
        "date": date_str,
        "home": home,
        "away": away,
        "pred_1x2": ["H", "D", "A"][winner_idx],
        "conf_1x2": round(float(p_1x2[winner_idx] * 100), 1),
        "pred_ou25": "Over" if res["ou25"] > 0.5 else "Under",
        "conf_ou25": round(float(max(res["ou25"], 1 - res["ou25"]) * 100), 1),
        "pred_ou15": "Over" if res.get("ou15", 0.5) > 0.5 else "Under",
        "conf_ou15": round(float(max(res.get("ou15", 0.5), 1 - res.get("ou15", 0.5)) * 100), 1),
        "pred_btts": "Yes" if res["btts"] > 0.5 else "No",
        "conf_btts": round(float(max(res["btts"], 1 - res["btts"]) * 100), 1),
        "pred_score": res["exact_score"],
        "score_prob": round(float(res["exact_score_prob"] * 100), 1),
        "extra_markets": [
            {
                "target": row.get("target"),
                "market": row.get("market"),
                "pick": row.get("pick"),
                "est_accuracy": round(float(row.get("est_accuracy", 0.0)), 1),
            }
            for row in res.get("extra_markets", [])
        ],
        "actual_ftr": None,
        "actual_score": None,
        "actual_goals": None,
        "settled": False,
    }

    db["predictions"].append(record)
    db["predictions"] = db["predictions"][-500:]
    _save_db(db)

def fetch_actual_results():
    for season_code in _season_codes():
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            csv_df = pd.read_csv(StringIO(resp.text))
            csv_df["Date"] = pd.to_datetime(csv_df["Date"], dayfirst=True, errors="coerce")
            if not csv_df.empty:
                return csv_df
        except Exception:
            continue
    return None

def _actual_values(match):
    fthg = int(match.get("FTHG", 0))
    ftag = int(match.get("FTAG", 0))
    hthg = int(match.get("HTHG", 0)) if not pd.isna(match.get("HTHG")) else 0
    htag = int(match.get("HTAG", 0)) if not pd.isna(match.get("HTAG")) else 0
    values = {
        "total_goals": fthg + ftag,
        "btts": int(fthg > 0 and ftag > 0),
        "total_corners": float(match.get("HC", 0) or 0) + float(match.get("AC", 0) or 0),
        "home_corners": float(match.get("HC", 0) or 0),
        "away_corners": float(match.get("AC", 0) or 0),
        "total_bookings": float(match.get("HY", 0) or 0) + float(match.get("AY", 0) or 0),
        "home_bookings": float(match.get("HY", 0) or 0),
        "away_bookings": float(match.get("AY", 0) or 0),
        "total_shots": float(match.get("HS", 0) or 0) + float(match.get("AS", 0) or 0),
        "home_shots": float(match.get("HS", 0) or 0),
        "away_shots": float(match.get("AS", 0) or 0),
        "total_sot": float(match.get("HST", 0) or 0) + float(match.get("AST", 0) or 0),
        "home_sot": float(match.get("HST", 0) or 0),
        "away_sot": float(match.get("AST", 0) or 0),
        "total_fouls": float(match.get("HF", 0) or 0) + float(match.get("AF", 0) or 0),
        "home_fouls": float(match.get("HF", 0) or 0),
        "away_fouls": float(match.get("AF", 0) or 0),
        "first_half_goals": hthg + htag,
        "first_half_btts_source": int(hthg > 0 and htag > 0),
    }
    return values

def _settle_extra_market(extra_pred, values, manifest_map):
    target = extra_pred.get("target")
    spec = manifest_map.get(target)
    if not spec:
        return False, None
    source = spec.get("source")
    actual_value = values.get(source)
    if actual_value is None:
        return False, None

    positive_pick = spec.get("pick", "OVER")
    if "GG" in positive_pick:
        correct = (extra_pred.get("pick") == "GG" and bool(actual_value)) or (
            extra_pred.get("pick") == "NG" and not bool(actual_value)
        )
    else:
        line = float(spec.get("line", 0.0))
        correct = (extra_pred.get("pick") == "OVER" and actual_value > line) or (
            extra_pred.get("pick") == "UNDER" and actual_value <= line
        )
    return bool(correct), actual_value

def settle_predictions():
    db = _load_db()
    unsettled = [p for p in db["predictions"] if not p.get("settled")]
    if not unsettled:
        return 0

    results_df = fetch_actual_results()
    if results_df is None or results_df.empty:
        return 0

    manifest_map = _load_manifest_map()
    settled_count = 0
    for pred in unsettled:
        mask = (results_df["HomeTeam"] == pred["home"]) & (results_df["AwayTeam"] == pred["away"])
        matches = results_df[mask]
        if matches.empty:
            continue

        match = matches.iloc[-1]
        fthg = match.get("FTHG")
        ftag = match.get("FTAG")
        ftr = match.get("FTR")
        if pd.isna(fthg) or pd.isna(ftag) or pd.isna(ftr):
            continue

        fthg, ftag = int(fthg), int(ftag)
        values = _actual_values(match)
        total_goals = values["total_goals"]

        pred["actual_ftr"] = ftr
        pred["actual_score"] = f"{fthg}-{ftag}"
        pred["actual_goals"] = total_goals
        pred["correct_1x2"] = pred["pred_1x2"] == ftr
        pred["correct_ou25"] = (pred["pred_ou25"] == "Over" and total_goals > 2.5) or (
            pred["pred_ou25"] == "Under" and total_goals <= 2.5
        )
        pred["correct_ou15"] = (pred.get("pred_ou15", "Over") == "Over" and total_goals > 1.5) or (
            pred.get("pred_ou15", "Over") == "Under" and total_goals <= 1.5
        )
        pred["correct_btts"] = (pred["pred_btts"] == "Yes" and values["btts"]) or (
            pred["pred_btts"] == "No" and not values["btts"]
        )
        pred["correct_score"] = pred["pred_score"] == f"{fthg}-{ftag}"

        for extra_pred in pred.get("extra_markets", []):
            correct, actual_value = _settle_extra_market(extra_pred, values, manifest_map)
            extra_pred["correct"] = correct
            extra_pred["actual_value"] = actual_value

        pred["settled"] = True
        settled_count += 1

    db["last_settled"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    _save_db(db)
    return settled_count

def get_accuracy_summary():
    db = _load_db()
    settled = [p for p in db["predictions"] if p.get("settled")]
    if not settled:
        return None

    recent = settled[-50:]
    stats = {
        "1x2": sum(1 for p in recent if p.get("correct_1x2", False)),
        "ou25": sum(1 for p in recent if p.get("correct_ou25", False)),
        "ou15": sum(1 for p in recent if p.get("correct_ou15", False)),
        "btts": sum(1 for p in recent if p.get("correct_btts", False)),
        "score": sum(1 for p in recent if p.get("correct_score", False)),
        "total": len(recent),
        "extra": {},
    }

    for p in recent:
        for extra_pred in p.get("extra_markets", []):
            key = extra_pred.get("market")
            if not key:
                continue
            stats["extra"].setdefault(key, {"correct": 0, "total": 0})
            stats["extra"][key]["total"] += 1
            if extra_pred.get("correct"):
                stats["extra"][key]["correct"] += 1
    return stats

def _bar(pct, width=8):
    filled = round(pct / 100 * width)
    return "#" * filled + "." * (width - filled)

def get_results_text(limit=15):
    db = _load_db()
    settled = [p for p in db["predictions"] if p.get("settled")]
    if not settled:
        return "Prediction results\n\nNo settled predictions yet."

    stats = get_accuracy_summary()
    n = stats["total"]
    lines = [
        "Prediction results",
        f"Last {n} settled predictions",
        "",
        f"1X2: {stats['1x2']}/{n} ({stats['1x2'] / n * 100:.0f}%) {_bar(stats['1x2'] / n * 100)}",
        f"OU2.5: {stats['ou25']}/{n} ({stats['ou25'] / n * 100:.0f}%) {_bar(stats['ou25'] / n * 100)}",
        f"OU1.5: {stats['ou15']}/{n} ({stats['ou15'] / n * 100:.0f}%) {_bar(stats['ou15'] / n * 100)}",
        f"BTTS: {stats['btts']}/{n} ({stats['btts'] / n * 100:.0f}%) {_bar(stats['btts'] / n * 100)}",
        f"Score: {stats['score']}/{n} ({stats['score'] / n * 100:.0f}%) {_bar(stats['score'] / n * 100)}",
    ]

    extra_items = []
    for market, row in stats.get("extra", {}).items():
        if row["total"] > 0:
            extra_items.append((market, row["correct"] / row["total"] * 100, row["correct"], row["total"]))
    extra_items.sort(key=lambda x: (-x[1], x[0]))
    if extra_items:
        lines += ["", "Extra market accuracy:"]
        for market, pct, correct, total in extra_items[:12]:
            lines.append(f"- {market}: {correct}/{total} ({pct:.0f}%) {_bar(pct)}")

    lines += ["", "Recent matches:"]
    for p in settled[-limit:][::-1]:
        lines.append(f"- {p['home']} {p['actual_score']} {p['away']} | 1X2 {p['pred_1x2']} | OU2.5 {p['pred_ou25']} | BTTS {p['pred_btts']}")
    return "\n".join(lines)

