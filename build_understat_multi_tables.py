import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path("data") / "understat_multi"


def season_start_from_dir(name: str) -> int:
    return int(name.split("_")[0])


def build_tables(base_dir: Path = BASE_DIR):
    match_rows = []
    player_rows = []
    team_rows = []

    if not base_dir.exists():
        raise FileNotFoundError(base_dir)

    for league_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        league_slug = league_dir.name
        for season_dir in sorted(p for p in league_dir.iterdir() if p.is_dir()):
            season_start = season_start_from_dir(season_dir.name)
            matches_path = season_dir / "matches.json"
            players_path = season_dir / "players.json"
            teams_path = season_dir / "teams.json"

            if matches_path.exists():
                matches = json.loads(matches_path.read_text(encoding="utf-8"))
                for match in matches:
                    match_rows.append(
                        {
                            "league_slug": league_slug,
                            "season_start": season_start,
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

            if players_path.exists():
                players = json.loads(players_path.read_text(encoding="utf-8"))
                for player in players:
                    row = dict(player)
                    row["league_slug"] = league_slug
                    row["season_start"] = season_start
                    player_rows.append(row)

            if teams_path.exists():
                teams = json.loads(teams_path.read_text(encoding="utf-8"))
                for team_name, payload in teams.items():
                    row = {
                        "league_slug": league_slug,
                        "season_start": season_start,
                        "team_name": team_name,
                    }
                    if isinstance(payload, dict):
                        for key, value in payload.items():
                            row[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
                    team_rows.append(row)

    if match_rows:
        pd.DataFrame(match_rows).to_csv(base_dir / "matches_all.csv", index=False)
    if player_rows:
        pd.DataFrame(player_rows).to_csv(base_dir / "players_all.csv", index=False)
    if team_rows:
        pd.DataFrame(team_rows).to_csv(base_dir / "teams_all.csv", index=False)

    return {
        "matches": len(match_rows),
        "players": len(player_rows),
        "teams": len(team_rows),
    }


if __name__ == "__main__":
    print(build_tables())
