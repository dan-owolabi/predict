import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


DATA_DIR = Path("data") / "football-data-support"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

LEAGUES = {
    "E0": "England Premier League",
    "SP1": "Spain La Liga",
    "D1": "Germany Bundesliga",
    "I1": "Italy Serie A",
    "F1": "France Ligue 1",
    "SC0": "Scotland Premiership",
    "N1": "Netherlands Eredivisie",
    "B1": "Belgium Jupiler League",
    "P1": "Portugal Primeira Liga",
    "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
    "AUT": "Austria Bundesliga",
    "DNK": "Denmark Superliga",
    "RUS": "Russia Premier League",
    "SWZ": "Switzerland Super League",
    "SWE": "Sweden Allsvenskan",
    "NOR": "Norway Eliteserien",
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def season_code(start_year: int) -> str:
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def current_season_start() -> int:
    now = datetime.now()
    return now.year if now.month >= 7 else now.year - 1


def fetch_mmz_comp(comp_code: str, start_year: int, end_year: int):
    rows = []
    success = 0
    for year in range(start_year, end_year + 1):
        season = season_code(year)
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{comp_code}.csv"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException:
            continue
        if resp.status_code != 200:
            continue
        out_path = DATA_DIR / f"{comp_code}_{year}_{year + 1}.csv"
        out_path.write_bytes(resp.content)
        try:
            df = pd.read_csv(out_path, low_memory=False)
        except Exception:
            continue
        df["season_start"] = year
        df["competition_code"] = comp_code
        rows.append(df)
        success += 1
        time.sleep(0.4)
    return rows, success


def fetch_new_comp(comp_code: str):
    url = f"https://www.football-data.co.uk/new/{comp_code}.csv"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
    except requests.RequestException:
        return [], 0
    if resp.status_code != 200:
        return [], 0
    out_path = DATA_DIR / f"{comp_code}_new.csv"
    out_path.write_bytes(resp.content)
    try:
        df = pd.read_csv(out_path, low_memory=False)
    except Exception:
        return [], 0
    df["competition_code"] = comp_code
    return [df], 1


def main():
    ensure_dir(DATA_DIR)
    end_year = current_season_start()
    summaries = {}
    all_frames = []

    for comp_code, label in LEAGUES.items():
        if comp_code in {"AUT", "DNK", "RUS", "SWZ", "SWE", "NOR"}:
            frames, success = fetch_new_comp(comp_code)
        else:
            frames, success = fetch_mmz_comp(comp_code, 2011, end_year)
        for df in frames:
            df["competition_label"] = label
            all_frames.append(df)
        summaries[comp_code] = {"label": label, "files": success}

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(DATA_DIR / "support_leagues_all.csv", index=False)

    print(summaries)


if __name__ == "__main__":
    main()
