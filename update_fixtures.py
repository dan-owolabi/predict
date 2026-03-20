import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data") / "football-data" / "E0_all.csv"
LEGACY_DATA_PATH = Path("E0 - E0.csv.csv")
OUT_PATH = Path("weekly_fixtures.json")


def load_source():
    for path in [DATA_PATH, LEGACY_DATA_PATH]:
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            return df
    raise FileNotFoundError("No fixture data source found.")


def build_weekly_fixture_payload(df: pd.DataFrame, horizon_days: int = 10):
    now = pd.Timestamp(datetime.now().date())
    future = df.copy()
    future = future[future["Date"].notna()]
    if "FTHG" in future.columns and "FTAG" in future.columns:
        future["FTHG_num"] = pd.to_numeric(future["FTHG"], errors="coerce")
        future["FTAG_num"] = pd.to_numeric(future["FTAG"], errors="coerce")
        future = future[future["FTHG_num"].isna() | future["FTAG_num"].isna()]
    future = future[(future["Date"] >= now) & (future["Date"] <= now + timedelta(days=horizon_days))]
    future = future.sort_values(["Date", "HomeTeam", "AwayTeam"])

    weeks = []
    for week_start, chunk in future.groupby(future["Date"].dt.to_period("W-MON")):
        matches = [(str(row["HomeTeam"]), str(row["AwayTeam"])) for _, row in chunk.iterrows()]
        if not matches:
            continue
        week_label = f"{chunk['Date'].min().date()} to {chunk['Date'].max().date()}"
        weeks.append({"label": week_label, "matches": matches})

    if not weeks:
        weeks = [{"label": "No upcoming fixtures found", "matches": []}]
    return {"generated_at": datetime.utcnow().isoformat() + "Z", "weeks": weeks}


def main():
    df = load_source()
    payload = build_weekly_fixture_payload(df)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    total_matches = sum(len(w["matches"]) for w in payload["weeks"])
    print({"weeks": len(payload["weeks"]), "matches": total_matches, "out": str(OUT_PATH)})


if __name__ == "__main__":
    main()
