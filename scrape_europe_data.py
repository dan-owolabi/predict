import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


DATA_DIR = Path("data")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}

EURO_COMPETITIONS = {
    "CL": "Champions League",
    "EL": "Europa League",
}

UNDERSTAT_LEAGUES = {
    "EPL": "epl",
    "La_Liga": "la_liga",
    "Bundesliga": "bundesliga",
    "Serie_A": "serie_a",
    "Ligue_1": "ligue_1",
}

OPENFOOTBALL_FILES = {
    "CL": "cl.txt",
    "EL": "el.txt",
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def season_code(start_year: int) -> str:
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def current_season_start() -> int:
    now = datetime.now()
    return now.year if now.month >= 7 else now.year - 1


def download_bytes(url: str, timeout: int = 30):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.RequestException:
        return None, "request_error"
    if resp.status_code != 200:
        return None, resp.status_code
    return resp.content, resp.status_code


def download_text(url: str, timeout: int = 30):
    payload, status = download_bytes(url, timeout=timeout)
    if payload is None:
        return None, status
    return payload.decode("utf-8"), status


def fetch_football_data_europe(start_year=2010, end_year=None, competitions=None):
    if end_year is None:
        end_year = current_season_start()
    competitions = competitions or list(EURO_COMPETITIONS.keys())

    out_dir = DATA_DIR / "football-data-europe"
    ensure_dir(out_dir)

    summary = {}
    for comp_code in competitions:
        comp_rows = []
        success_count = 0
        for year in range(start_year, end_year + 1):
            season = season_code(year)
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{comp_code}.csv"
            out_path = out_dir / f"{comp_code}_{year}_{year + 1}.csv"
            payload, status = download_bytes(url)
            if payload is None:
                if year >= end_year - 1:
                    # Stop after failing at the tail; older seasons may still exist,
                    # but the recent-season probe tells us the code is likely wrong.
                    break
                continue
            out_path.write_bytes(payload)
            try:
                df = pd.read_csv(out_path, low_memory=False)
                df["season_start"] = year
                df["competition_code"] = comp_code
                df["competition_label"] = EURO_COMPETITIONS.get(comp_code, comp_code)
                comp_rows.append(df)
                success_count += 1
            except Exception:
                pass
            time.sleep(0.7)

        if comp_rows:
            combined = pd.concat(comp_rows, ignore_index=True)
            combined.to_csv(out_dir / f"{comp_code}_all.csv", index=False)
        summary[comp_code] = {
            "label": EURO_COMPETITIONS.get(comp_code, comp_code),
            "seasons_downloaded": success_count,
        }

    return summary


def _parse_openfootball_date(date_text: str, season_start: int):
    text = date_text.strip().strip("[]")
    m = re.search(r"([A-Za-z]{3})\s+([A-Za-z]{3})/(\d{1,2})(?:\s+(\d{4}))?$", text)
    if not m:
        return None
    month_name = m.group(2)
    day = int(m.group(3))
    year = int(m.group(4)) if m.group(4) else None
    month_num = datetime.strptime(month_name, "%b").month
    if year is None:
        year = season_start if month_num >= 7 else season_start + 1
    return datetime(year, month_num, day).date()


def _clean_europe_team_name(name: str):
    name = str(name or "").strip()
    name = re.sub(r"\s+\([A-Z]{2,3}\)$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _parse_openfootball_matches(text: str, competition_code: str, season_start: int):
    rows = []
    current_stage = None
    current_date = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("="):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_date = _parse_openfootball_date(line, season_start)
            continue
        if re.match(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+[A-Za-z]{3}/\d{1,2}(?:\s+\d{4})?$", line):
            current_date = _parse_openfootball_date(line, season_start)
            continue
        if line.startswith("Group ") or line.startswith("League") or line.startswith("Playoffs") or line.startswith("Finals") or line.startswith("Qualifying") or line.startswith("Preliminary"):
            current_stage = line
            continue
        if line.startswith("»"):
            current_stage = line.lstrip("»").strip()
            continue
        if " v " not in line:
            continue

        match = re.match(
            r"^(?:(?P<time>\d{1,2}[.:]\d{2})\s+)?(?P<home>.+?)\s+v\s+(?P<away>.+?)\s+(?P<score>\d+-\d+(?:\s+pen\.)?)(?:\s+(?P<detail>.+))?$",
            line,
        )
        if not match:
            continue

        score_token = match.group("score")
        score_numbers = re.match(r"(?P<h>\d+)-(?P<a>\d+)", score_token)
        if not score_numbers:
            continue

        rows.append(
            {
                "date": current_date.isoformat() if current_date else None,
                "time": match.group("time"),
                "home_team": _clean_europe_team_name(match.group("home")),
                "away_team": _clean_europe_team_name(match.group("away")),
                "home_goals": int(score_numbers.group("h")),
                "away_goals": int(score_numbers.group("a")),
                "stage": current_stage,
                "score_token": score_token,
                "detail": match.group("detail"),
                "competition_code": competition_code,
                "competition_label": EURO_COMPETITIONS.get(competition_code, competition_code),
                "season_start": season_start,
            }
        )

    return rows


def fetch_openfootball_europe(start_year=2011, end_year=None, competitions=None):
    if end_year is None:
        end_year = current_season_start()
    competitions = competitions or list(OPENFOOTBALL_FILES.keys())

    out_dir = DATA_DIR / "openfootball-europe"
    ensure_dir(out_dir)

    summary = {}
    for comp_code in competitions:
        file_name = OPENFOOTBALL_FILES.get(comp_code)
        if not file_name:
            continue
        all_rows = []
        seasons_downloaded = 0
        for year in range(start_year, end_year + 1):
            season_label = f"{year}-{str(year + 1)[-2:]}"
            url = f"https://raw.githubusercontent.com/openfootball/champions-league/master/{season_label}/{file_name}"
            text, status = download_text(url)
            if text is None:
                continue
            raw_path = out_dir / f"{comp_code}_{season_label}.txt"
            raw_path.write_text(text, encoding="utf-8")
            parsed = _parse_openfootball_matches(text, comp_code, year)
            if parsed:
                pd.DataFrame(parsed).to_csv(out_dir / f"{comp_code}_{year}_{year + 1}.csv", index=False)
                all_rows.extend(parsed)
            seasons_downloaded += 1
            time.sleep(0.4)
        if all_rows:
            pd.DataFrame(all_rows).to_csv(out_dir / f"{comp_code}_all.csv", index=False)
        summary[comp_code] = {
            "label": EURO_COMPETITIONS.get(comp_code, comp_code),
            "seasons_downloaded": seasons_downloaded,
            "matches": len(all_rows),
        }
    return summary


def fetch_understat_multi(start_year=2014, end_year=None, leagues=None):
    if end_year is None:
        end_year = current_season_start()
    leagues = leagues or list(UNDERSTAT_LEAGUES.keys())

    out_dir = DATA_DIR / "understat_multi"
    ensure_dir(out_dir)

    try:
        from understatapi import UnderstatClient
    except Exception as exc:
        raise RuntimeError("understatapi not installed") from exc

    match_rows = []
    player_rows = []
    team_rows = []

    with UnderstatClient() as client:
        for league in leagues:
            league_slug = UNDERSTAT_LEAGUES.get(league, league.lower().replace(" ", "_"))
            league_dir = out_dir / league_slug
            ensure_dir(league_dir)
            for year in range(start_year, end_year + 1):
                season_dir = league_dir / f"{year}_{year + 1}"
                ensure_dir(season_dir)
                matches_path = season_dir / "matches.json"
                players_path = season_dir / "players.json"
                teams_path = season_dir / "teams.json"

                if matches_path.exists() and players_path.exists() and teams_path.exists():
                    matches = json.loads(matches_path.read_text(encoding="utf-8"))
                    players = json.loads(players_path.read_text(encoding="utf-8"))
                    teams = json.loads(teams_path.read_text(encoding="utf-8"))
                else:
                    try:
                        matches = client.league(league=league).get_match_data(season=str(year))
                        players = client.league(league=league).get_player_data(season=str(year))
                        teams = client.league(league=league).get_team_data(season=str(year))
                    except requests.RequestException:
                        continue
                    except Exception:
                        continue

                    matches_path.write_text(json.dumps(matches, indent=2), encoding="utf-8")
                    players_path.write_text(json.dumps(players, indent=2), encoding="utf-8")
                    teams_path.write_text(json.dumps(teams, indent=2), encoding="utf-8")

                for match in matches:
                    match_rows.append(
                        {
                            "league": league,
                            "league_slug": league_slug,
                            "season_start": year,
                            "match_id": match.get("id"),
                            "date": match.get("datetime"),
                            "home_team": match.get("h", {}).get("title"),
                            "away_team": match.get("a", {}).get("title"),
                            "home_goals": match.get("goals", {}).get("h"),
                            "away_goals": match.get("goals", {}).get("a"),
                            "home_xg": match.get("xG", {}).get("h"),
                            "away_xg": match.get("xG", {}).get("a"),
                            "forecast": json.dumps(match.get("forecast", {})),
                        }
                    )

                for player in players:
                    row = dict(player)
                    row["league"] = league
                    row["league_slug"] = league_slug
                    row["season_start"] = year
                    player_rows.append(row)

                for team_name, payload in teams.items():
                    row = {
                        "league": league,
                        "league_slug": league_slug,
                        "season_start": year,
                        "team_name": team_name,
                    }
                    if isinstance(payload, dict):
                        for key, value in payload.items():
                            row[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
                    team_rows.append(row)

                time.sleep(0.7)

    if match_rows:
        pd.DataFrame(match_rows).to_csv(out_dir / "matches_all.csv", index=False)
    if player_rows:
        pd.DataFrame(player_rows).to_csv(out_dir / "players_all.csv", index=False)
    if team_rows:
        pd.DataFrame(team_rows).to_csv(out_dir / "teams_all.csv", index=False)

    return {
        "matches": len(match_rows),
        "players": len(player_rows),
        "teams": len(team_rows),
    }


def summarize_europe_team_universe():
    out_dirs = [DATA_DIR / "football-data-europe", DATA_DIR / "openfootball-europe"]
    team_sets = []
    for out_dir in out_dirs:
        if not out_dir.exists():
            continue
        for comp_code in EURO_COMPETITIONS:
            comp_path = out_dir / f"{comp_code}_all.csv"
            if not comp_path.exists():
                continue
            df = pd.read_csv(comp_path, low_memory=False)
            home_col = "HomeTeam" if "HomeTeam" in df.columns else "home_team"
            away_col = "AwayTeam" if "AwayTeam" in df.columns else "away_team"
            teams = sorted(set(df.get(home_col, pd.Series(dtype=str)).dropna().astype(str)) |
                           set(df.get(away_col, pd.Series(dtype=str)).dropna().astype(str)))
            pd.DataFrame({"team_name": teams}).to_csv(out_dir / f"{comp_code}_teams.csv", index=False)
            team_sets.append(pd.DataFrame({"team_name": teams, "competition_code": comp_code}))
    if team_sets:
        combined = pd.concat(team_sets, ignore_index=True).drop_duplicates()
        combined.to_csv(DATA_DIR / "europe_team_universe.csv", index=False)
        return len(combined)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--competitions", default="CL,EL")
    parser.add_argument("--understat-start", type=int, default=2014)
    parser.add_argument("--understat-leagues", default="EPL,La_Liga,Bundesliga,Serie_A,Ligue_1")
    args = parser.parse_args()

    ensure_dir(DATA_DIR)

    competitions = [c.strip() for c in args.competitions.split(",") if c.strip()]
    leagues = [l.strip() for l in args.understat_leagues.split(",") if l.strip()]

    print("Fetching Champions League / Europa League historical data...")
    football_data_summary = fetch_football_data_europe(
        start_year=args.start_year,
        end_year=args.end_year,
        competitions=competitions,
    )
    print(f"football-data summary: {football_data_summary}")

    print("Fetching openfootball Europe competition data...")
    openfootball_summary = fetch_openfootball_europe(
        start_year=max(args.start_year, 2011),
        end_year=args.end_year,
        competitions=competitions,
    )
    print(f"openfootball summary: {openfootball_summary}")

    print("Building Europe team universe...")
    team_count = summarize_europe_team_universe()
    print(f"team universe rows: {team_count}")

    print("Fetching multi-league Understat support data...")
    understat_summary = fetch_understat_multi(
        start_year=args.understat_start,
        end_year=args.end_year,
        leagues=leagues,
    )
    print(f"understat summary: {understat_summary}")


if __name__ == "__main__":
    main()
