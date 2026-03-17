import json
import pickle
import re
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA_PATH = Path("data") / "player_props" / "player_match_props_starters.csv"
MODEL_DIR = Path("player_prop_models")
MANIFEST_PATH = MODEL_DIR / "manifest.json"
ARTIFACTS_PATH = MODEL_DIR / "artifacts.pkl"
FPL_PLAYERS_PATH = Path("data") / "fpl" / "players.csv"
FPL_TEAMS_PATH = Path("data") / "fpl" / "teams.csv"

TEAM_NAME_MAP = {
    "Tottenham": "Spurs",
    "Tottenham Hotspur": "Spurs",
    "Man United": "Man Utd",
    "Manchester United": "Man Utd",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "AFC Bournemouth": "Bournemouth",
    "Brighton and Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham",
}


def canonical_name(s):
    s = str(s or "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def team_alias(name):
    return TEAM_NAME_MAP.get(name, name)


@lru_cache(maxsize=1)
def load_assets():
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["player_key"] = data["player_key"].map(canonical_name)
    data["team_name"] = data["team_name"].map(team_alias)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    models = {row["target"]: lgb.Booster(model_file=str(MODEL_DIR / row["model_file"])) for row in manifest}
    label_map = {row["target"]: row for row in manifest}
    return data, artifacts, models, label_map


@lru_cache(maxsize=1)
def load_fpl_current_squads():
    players = pd.read_csv(FPL_PLAYERS_PATH)
    teams = pd.read_csv(FPL_TEAMS_PATH)
    team_map = teams.set_index("id")["name"].to_dict()
    players["team_name"] = players["team"].map(team_map).map(team_alias)
    players["player_name"] = (players["first_name"].fillna("") + " " + players["second_name"].fillna("")).str.strip()
    players["player_key_full"] = players["player_name"].map(canonical_name)
    players["player_key_web"] = players["web_name"].map(canonical_name)
    players = players[(players["team_name"].notna()) & (~players["status"].isin(["u"]))].copy()
    return players


def _name_score(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    a_parts = a.split()
    b_parts = b.split()
    bonus = 0.0
    last_match = a_parts and b_parts and a_parts[-1] == b_parts[-1]
    if last_match:
        bonus += 0.20
    if a_parts and b_parts and a_parts[0][0] == b_parts[0][0]:
        bonus += 0.05
    return min(1.0, SequenceMatcher(None, a, b).ratio() + bonus)


def _match_current_squad(team, latest_team_rows):
    fpl = load_fpl_current_squads()
    squad = fpl[fpl["team_name"] == team].copy()
    if squad.empty or latest_team_rows.empty:
        return latest_team_rows.iloc[0:0].copy()

    understat_keys = latest_team_rows["player_key"].tolist()
    used = set()
    matched_rows = []
    for _, row in squad.sort_values(["minutes", "expected_goals"], ascending=[False, False]).iterrows():
        candidates = [row["player_key_full"], row["player_key_web"]]
        best_key, best_score = None, 0.0
        for candidate in candidates:
            for ukey in understat_keys:
                if ukey in used:
                    continue
                score = _name_score(candidate, ukey)
                if score > best_score:
                    best_key, best_score = ukey, score
        if best_key:
            candidate_last = candidate.split()[-1] if candidate else ""
            matched_last = best_key.split()[-1] if best_key else ""
            strict_ok = (candidate_last and candidate_last == matched_last and best_score >= 0.72) or best_score >= 0.90
        else:
            strict_ok = False
        if best_key and strict_ok:
            used.add(best_key)
            matched = latest_team_rows[latest_team_rows["player_key"] == best_key].tail(1).copy()
            if not matched.empty:
                matched["fpl_name"] = row["player_name"]
                matched_rows.append(matched)
    if not matched_rows:
        return latest_team_rows.iloc[0:0].copy()
    return pd.concat(matched_rows, ignore_index=True)


def candidate_rows(data, team, is_home):
    latest = data.sort_values(["player_key", "date", "match_id"]).groupby("player_key").tail(1).copy()
    latest = latest[latest["team_name"] == team].copy()
    latest = _match_current_squad(team, latest)
    latest = latest[(latest["minutes_r5"] >= 35) & (latest["apps_r10"] >= 3)].copy()
    latest["is_home"] = 1 if is_home else 0
    latest["h_a"] = "h" if is_home else "a"
    latest["days_since_last"] = (pd.Timestamp(datetime.now().date()) - latest["date"]).dt.days.clip(lower=3).fillna(7)
    return latest


def feature_matrix(df, feature_cols):
    return df.reindex(columns=feature_cols, fill_value=np.nan).apply(pd.to_numeric, errors="coerce")


def _player_prop_thresholds(target):
    if target == "anytime_scorer":
        return 58.0, 6.0, 65.0
    if target in {"shots_over_2_5", "sot_over_1_5"}:
        return 62.0, 5.0, 70.0
    if target in {"shots_over_1_5", "sot_over_0_5", "booked_yes"}:
        return 60.0, 4.0, 65.0
    return 57.0, 3.0, 60.0


def _is_trustworthy_player_prop(row):
    prob_yes_min, edge_min, minutes_min = _player_prop_thresholds(row["target"])
    if float(row.get("minutes_r5", 0.0)) < minutes_min:
        return False
    if float(row.get("prob_yes", 0.0)) < prob_yes_min:
        return False
    if float(row.get("edge", 0.0)) < edge_min:
        return False
    if row.get("position") == "GK":
        return False
    return True


def rank_fixture_players(home, away, top_n=10):
    data, artifacts, models, label_map = load_assets()
    feature_cols = artifacts["feature_cols"]
    imputer = artifacts["imputer"]

    candidates = pd.concat([
        candidate_rows(data, team_alias(home), True),
        candidate_rows(data, team_alias(away), False),
    ], ignore_index=True)
    if candidates.empty:
        return pd.DataFrame()

    X_raw = feature_matrix(candidates, feature_cols).values.astype(float)
    X = np.nan_to_num(imputer.transform(X_raw), nan=0.0, posinf=0.0, neginf=0.0)

    rows = []
    for target, model in models.items():
        probs = model.predict(X)
        meta = label_map[target]
        for idx, prob in enumerate(probs):
            player_row = candidates.iloc[idx]
            positive_prob = float(prob)
            pick = "YES" if positive_prob >= 0.5 else "NO"
            position = str(player_row.get("position", ""))
            if target in {"shots_over_0_5", "shots_over_1_5", "shots_over_2_5", "sot_over_0_5", "sot_over_1_5", "anytime_scorer"} and position == "GK":
                continue
            rows.append({
                "team": player_row["team_name"],
                "player": player_row.get("fpl_name") or player_row["player"],
                "position": position,
                "target": target,
                "label": meta["label"],
                "prob_yes": positive_prob * 100.0,
                "est_accuracy": max(positive_prob, 1.0 - positive_prob) * 100.0,
                "pick": pick,
                "base_rate": float(meta.get("base_rate", 0.0)) * 100.0,
                "minutes_r5": float(player_row.get("minutes_r5", 0.0)),
                "shots_r5": float(player_row.get("shots_r5", 0.0)),
                "xG_r5": float(player_row.get("xG_r5", 0.0)),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[(out["pick"] == "YES") & (out["minutes_r5"] >= 55)].copy()
    out["edge"] = out["prob_yes"] - out["base_rate"]
    out = out[out.apply(_is_trustworthy_player_prop, axis=1)].copy()
    if out.empty:
        return out
    out = out.sort_values(["edge", "prob_yes", "team", "player"], ascending=[False, False, True, True]).reset_index(drop=True)
    return out.head(top_n)


if __name__ == "__main__":
    import sys
    home = sys.argv[1] if len(sys.argv) > 1 else "Arsenal"
    away = sys.argv[2] if len(sys.argv) > 2 else "Chelsea"
    res = rank_fixture_players(home, away, top_n=20)
    if res.empty:
        print("No player candidates found.")
    else:
        print(res.to_string(index=False))
