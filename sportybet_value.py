from functools import lru_cache
from pathlib import Path
import re

import pandas as pd

SPORTY_DIR = Path("data") / "sportybet"

TEAM_NAME_MAP = {
    "Manchester United": "Man United",
    "Man Utd": "Man United",
    "Manchester City": "Man City",
    "Tottenham": "Spurs",
    "Tottenham Hotspur": "Spurs",
    "Nottingham Forest": "Nott'm Forest",
    "Brighton and Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham",
    "AFC Bournemouth": "Bournemouth",
    "Newcastle United": "Newcastle",
}


def team_alias(name):
    return TEAM_NAME_MAP.get(name, name)


def _norm(s):
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


@lru_cache(maxsize=1)
def _all_event_files():
    return sorted(SPORTY_DIR.glob("sr_match_*.csv"))


def load_match_market_df(home, away):
    home_n = _norm(team_alias(home))
    away_n = _norm(team_alias(away))
    matched = []
    for path in _all_event_files():
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        h = _norm(df.iloc[0].get("home_team"))
        a = _norm(df.iloc[0].get("away_team"))
        if h == home_n and a == away_n:
            matched.append((path, df))
    if not matched:
        return None
    matched.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
    return matched[0][1]


def _selection_value_prob(odds):
    odds = float(odds)
    if odds <= 1.0:
        return None
    return 1.0 / odds


def _pick_maps(row, home, away):
    market = str(row.get("market", ""))
    pick = str(row.get("pick", ""))
    if market == "1X2":
        sel = {"HOME WIN": "Home", "DRAW": "Draw", "AWAY WIN": "Away"}.get(pick)
        return [("1X2", sel)] if sel else []
    if market == "GG/NG":
        return [("GG/NG", "GG" if pick == "GG" else "NG")]
    if market.startswith("OVER/UNDER "):
        line = market.split()[-1]
        return [("Over/Under", f"{'Over' if pick.startswith('OVER') else 'Under'} {line}")]
    if market.startswith("CORNERS "):
        line = market.split()[-1]
        return [("Corners - Over/Under", f"{pick.title()} {line}")]
    if market.startswith("BOOKINGS "):
        line = market.split()[-1]
        return [("Bookings - Over/Under", f"{pick.title()} {line}")]
    if market.startswith("FOULS "):
        line = market.split()[-1]
        return [("Fouls - Over/Under", f"{pick.title()} {line}")]
    if market.startswith("SHOTS ON TARGET "):
        line = market.split()[-1]
        return [("Shots On Target - Over/Under", f"{pick.title()} {line}")]
    if market.startswith("SHOTS "):
        line = market.split()[-1]
        return [("Shots - Over/Under", f"{pick.title()} {line}")]
    if market.startswith("1ST HALF GOALS "):
        line = market.split()[-1]
        return [("1st Half - Over/Under", f"{pick.title()} {line}")]
    if market == "1ST HALF GG/NG":
        return [("1st Half - GG/NG", "GG" if pick == "GG" else "NG")]
    if market.startswith("HOME SHOTS "):
        line = market.split()[-1]
        return [(f"{home} Total Shots - Over/Under", f"{pick.title()} {line}"), (f"Home Team Shots - Over/Under", f"{pick.title()} {line}")]
    if market.startswith("AWAY SHOTS "):
        line = market.split()[-1]
        return [(f"{away} Total Shots - Over/Under", f"{pick.title()} {line}"), (f"Away Team Shots - Over/Under", f"{pick.title()} {line}")]
    if market.startswith("HOME SOT "):
        line = market.split()[-1]
        return [(f"{home} Shots On Target - Over/Under", f"{pick.title()} {line}"), (f"Home Team Shots On Target - Over/Under", f"{pick.title()} {line}")]
    if market.startswith("AWAY SOT "):
        line = market.split()[-1]
        return [(f"{away} Shots On Target - Over/Under", f"{pick.title()} {line}"), (f"Away Team Shots On Target - Over/Under", f"{pick.title()} {line}")]
    if market.startswith("HOME BOOKINGS "):
        line = market.split()[-1]
        return [(f"{home} Bookings - Over/Under", f"{pick.title()} {line}"), (f"Home Team Bookings - Over/Under", f"{pick.title()} {line}")]
    if market.startswith("AWAY BOOKINGS "):
        line = market.split()[-1]
        return [(f"{away} Bookings - Over/Under", f"{pick.title()} {line}"), (f"Away Team Bookings - Over/Under", f"{pick.title()} {line}")]
    if market.startswith("HOME FOULS "):
        line = market.split()[-1]
        return [(f"{home} Fouls - Over/Under", f"{pick.title()} {line}"), (f"Home Team Fouls - Over/Under", f"{pick.title()} {line}")]
    if market.startswith("AWAY FOULS "):
        line = market.split()[-1]
        return [(f"{away} Fouls - Over/Under", f"{pick.title()} {line}"), (f"Away Team Fouls - Over/Under", f"{pick.title()} {line}")]
    if market.startswith("HOME CORNERS "):
        line = market.split()[-1]
        return [(f"{home} Corners - Over/Under", f"{pick.title()} {line}"), (f"Home Team Corners - Over/Under", f"{pick.title()} {line}")]
    if market.startswith("AWAY CORNERS "):
        line = market.split()[-1]
        return [(f"{away} Corners - Over/Under", f"{pick.title()} {line}"), (f"Away Team Corners - Over/Under", f"{pick.title()} {line}")]
    return []


def attach_market_prices(home, away, rankings):
    df = load_match_market_df(home, away)
    if df is None or df.empty:
        return rankings
    work = df.copy()
    work["market_title_norm"] = work["market_title"].map(_norm)
    work["selection_norm"] = work["selection"].map(_norm)

    out = []
    for row in rankings:
        row = dict(row)
        row["book_odds"] = None
        row["implied_prob"] = None
        row["value_edge"] = None
        for title, selection in _pick_maps(row, home, away):
            mask = (work["market_title_norm"] == _norm(title)) & (work["selection_norm"] == _norm(selection))
            hit = work[mask]
            if not hit.empty:
                odds = float(hit.iloc[0]["odds"])
                implied = _selection_value_prob(odds)
                row["book_odds"] = odds
                row["implied_prob"] = implied * 100.0 if implied is not None else None
                row["value_edge"] = row["est_accuracy"] - row["implied_prob"] if row["implied_prob"] is not None else None
                break
        out.append(row)
    return out
