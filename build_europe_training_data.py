import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("data")
EUROPE_DIR = BASE_DIR / "openfootball-europe"
UNDERSTAT_DIR = BASE_DIR / "understat_multi"
SUPPORT_DIR = BASE_DIR / "football-data-support"
OUT_PATH = BASE_DIR / "europe_training_data.csv"


EUROPE_TO_UNDERSTAT = {
    "Bayern München": "Bayern Munich",
    "Aston Villa FC": "Aston Villa",
    "Juventus FC": "Juventus",
    "PSV": "PSV Eindhoven",
    "Sporting Clube de Portugal": "Sporting CP",
    "Sporting CP": "Sporting CP",
    "FC Bayern München": "Bayern Munich",
    "SL Benfica": "Benfica",
    "Sport Lisboa e Benfica": "Benfica",
    "Olympiakos Piraeus": "Olympiacos",
    "Olympique Marseille": "Marseille",
    "FC Porto": "Porto",
    "Real Madrid CF": "Real Madrid",
    "Real Madrid": "Real Madrid",
    "FC Barcelona": "Barcelona",
    "AC Milan": "Milan",
    "AC Milan ": "Milan",
    "Chelsea FC": "Chelsea",
    "Arsenal FC": "Arsenal",
    "Manchester City": "Manchester City",
    "Manchester United": "Manchester United",
    "Manchester United FC": "Manchester United",
    "Bor. Mönchengladbach": "Borussia M.Gladbach",
    "Borussia Dortmund": "Borussia Dortmund",
    "Lille OSC": "Lille",
    "Stade Rennais": "Rennes",
    "Paris Saint-Germain": "Paris Saint Germain",
    "Paris Saint-Germain FC": "Paris Saint Germain",
    "PSV Eindhoven": "PSV Eindhoven",
    "AFC Ajax": "Ajax",
    "Feyenoord": "Feyenoord",
    "FC København": "FC Copenhagen",
    "Crvena Zvezda": "Red Star Belgrade",
    "RB Salzburg": "Red Bull Salzburg",
    "BSC Young Boys": "Young Boys",
    "Qarabağ FK": "Qarabag",
    "Viktoria Plzeň": "Viktoria Plzen",
    "1899 Hoffenheim": "Hoffenheim",
    "West Ham United": "West Ham",
    "Leicester City": "Leicester",
    "Zenit St. Petersburg": "Zenit",
    "CSKA Moskva": "CSKA Moscow",
    "Lokomotiv Moskva": "Lokomotiv Moscow",
    "Spartak Moskva": "Spartak Moscow",
    "Dinamo Kiev": "Dynamo Kyiv",
    "Malmö FF": "Malmo",
    "Beşiktaş": "Besiktas",
    "Fenerbahçe": "Fenerbahce",
    "Liverpool FC": "Liverpool",
    "Bologna FC 1909": "Bologna",
    "FK Shakhtar Donetsk": "Shakhtar Donetsk",
    "Shakhtar Donetsk": "Shakhtar Donetsk",
    "AC Sparta Praha": "Sparta Prague",
    "FC Red Bull Salzburg": "Red Bull Salzburg",
    "FC Internazionale Milano": "Inter",
    "Girona FC": "Girona",
    "Club Brugge KV": "Club Brugge",
    "Celtic FC": "Celtic",
    "ŠK Slovan Bratislava": "Slovan Bratislava",
    "Feyenoord Rotterdam": "Feyenoord",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "FK Crvena Zvezda": "Red Star Belgrade",
    "Sport Lisboa e Benfica": "Benfica",
    "AS Monaco FC": "Monaco",
    "Stade Brestois 29": "Brest",
    "Atalanta BC": "Atalanta",
    "Arsenal FC": "Arsenal",
    "Club Atlético de Madrid": "Atletico Madrid",
    "RB Leipzig": "RasenBallsport Leipzig",
    "Manchester City FC": "Manchester City",
    "Borussia Dortmund": "Borussia Dortmund",
    "Manchester United": "Manchester United",
    "Manchester United FC": "Manchester United",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham Hotspur FC": "Tottenham",
    "AS Roma": "Roma",
    "Lazio Roma": "Lazio",
    "Athletic Club": "Athletic Club",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Olympique Lyonnais": "Lyon",
    "Olympiakos Piraeus": "Olympiacos",
    "AFC Ajax": "Ajax",
    "Rangers FC": "Rangers",
    "AZ Alkmaar": "AZ Alkmaar",
    "Real Sociedad": "Real Sociedad",
    "Galatasaray": "Galatasaray",
    "Fenerbahçe": "Fenerbahce",
    "Malmö FF": "Malmo",
    "Beşiktaş": "Besiktas",
    "Qarabağ FK": "Qarabag",
    "Viktoria Plzeň": "Viktoria Plzen",
}


def canonical_name(name: str) -> str:
    text = str(name or "").strip()
    replacements = {
        "ä": "a", "á": "a", "à": "a", "â": "a", "ã": "a",
        "ö": "o", "ó": "o", "ò": "o", "ô": "o",
        "ü": "u", "ú": "u", "ù": "u", "û": "u",
        "í": "i", "ì": "i", "î": "i",
        "é": "e", "è": "e", "ê": "e",
        "ç": "c", "ş": "s", "š": "s", "ž": "z", "ø": "o",
        "å": "a", "ý": "y", "ñ": "n",
    }
    lower = text.lower()
    for src, dst in replacements.items():
        lower = lower.replace(src, dst)
    lower = re.sub(r"\b(fc|cf|ac|as|ssc|fk|sk|bsc|afc|ogc|rsc|pfc|gnk|vfb)\b", " ", lower)
    lower = re.sub(r"[^a-z0-9 ]+", " ", lower)
    lower = re.sub(r"\s+", " ", lower).strip()
    return lower


def normalize_europe_team(name: str) -> str:
    mapped = EUROPE_TO_UNDERSTAT.get(str(name).strip(), str(name).strip())
    return canonical_name(mapped)


def load_europe_matches():
    frames = []
    for comp_code in ["CL", "EL"]:
        path = EUROPE_DIR / f"{comp_code}_all.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
        df["HomeTeam"] = df["home_team"].astype(str)
        df["AwayTeam"] = df["away_team"].astype(str)
        df["FTHG"] = pd.to_numeric(df["home_goals"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["away_goals"], errors="coerce")
        df["competition"] = comp_code
        df["home_key"] = df["HomeTeam"].map(normalize_europe_team)
        df["away_key"] = df["AwayTeam"].map(normalize_europe_team)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No Europe competition files found.")
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Date", "FTHG", "FTAG"]).sort_values("Date").reset_index(drop=True)
    out["over25"] = ((out["FTHG"] + out["FTAG"]) > 2.5).astype(int)
    out["btts"] = ((out["FTHG"] > 0) & (out["FTAG"] > 0)).astype(int)
    out["ftr_encoded"] = np.select(
        [out["FTHG"] > out["FTAG"], out["FTHG"] == out["FTAG"], out["FTHG"] < out["FTAG"]],
        [0, 1, 2],
        default=np.nan,
    )
    return out


def load_understat_histories():
    path = UNDERSTAT_DIR / "matches_all.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_key"] = df["home_team"].map(canonical_name)
    df["away_key"] = df["away_team"].map(canonical_name)
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df["home_xg"] = pd.to_numeric(df["home_xg"], errors="coerce")
    df["away_xg"] = pd.to_numeric(df["away_xg"], errors="coerce")

    home_rows = df[["Date", "home_key", "home_goals", "away_goals", "home_xg", "away_xg"]].rename(
        columns={
            "home_key": "team_key",
            "home_goals": "gf",
            "away_goals": "ga",
            "home_xg": "xgf",
            "away_xg": "xga",
        }
    )
    away_rows = df[["Date", "away_key", "away_goals", "home_goals", "away_xg", "home_xg"]].rename(
        columns={
            "away_key": "team_key",
            "away_goals": "gf",
            "home_goals": "ga",
            "away_xg": "xgf",
            "home_xg": "xga",
        }
    )
    team_hist = pd.concat([home_rows, away_rows], ignore_index=True)
    team_hist = team_hist.dropna(subset=["Date", "team_key"]).sort_values(["team_key", "Date"]).reset_index(drop=True)
    team_hist["source"] = "understat"
    return team_hist


def load_support_histories():
    path = SUPPORT_DIR / "support_leagues_all.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Date", "team_key", "gf", "ga", "shots", "sot", "corners", "fouls", "ycards", "source"])
    df = pd.read_csv(path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    for col in ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["home_key"] = df["HomeTeam"].map(canonical_name)
    df["away_key"] = df["AwayTeam"].map(canonical_name)

    home_rows = df[["Date", "home_key", "FTHG", "FTAG", "HS", "HST", "HC", "HF", "HY"]].rename(
        columns={"home_key": "team_key", "FTHG": "gf", "FTAG": "ga", "HS": "shots", "HST": "sot", "HC": "corners", "HF": "fouls", "HY": "ycards"}
    )
    away_rows = df[["Date", "away_key", "FTAG", "FTHG", "AS", "AST", "AC", "AF", "AY"]].rename(
        columns={"away_key": "team_key", "FTAG": "gf", "FTHG": "ga", "AS": "shots", "AST": "sot", "AC": "corners", "AF": "fouls", "AY": "ycards"}
    )
    team_hist = pd.concat([home_rows, away_rows], ignore_index=True)
    team_hist["xgf"] = np.nan
    team_hist["xga"] = np.nan
    team_hist = team_hist.dropna(subset=["Date", "team_key"]).sort_values(["team_key", "Date"]).reset_index(drop=True)
    team_hist["source"] = "support_league"
    return team_hist


def recent_stats(team_hist: pd.DataFrame, team_key: str, before_date: pd.Timestamp, window: int):
    rows = team_hist[(team_hist["team_key"] == team_key) & (team_hist["Date"] < before_date)].tail(window)
    if rows.empty:
        return {
            f"gf_r{window}": np.nan,
            f"ga_r{window}": np.nan,
            f"shots_r{window}": np.nan,
            f"sot_r{window}": np.nan,
            f"corners_r{window}": np.nan,
            f"fouls_r{window}": np.nan,
            f"ycards_r{window}": np.nan,
            f"xgf_r{window}": np.nan,
            f"xga_r{window}": np.nan,
            f"games_r{window}": 0.0,
        }
    return {
        f"gf_r{window}": rows["gf"].mean(),
        f"ga_r{window}": rows["ga"].mean(),
        f"shots_r{window}": rows["shots"].dropna().mean() if "shots" in rows else np.nan,
        f"sot_r{window}": rows["sot"].dropna().mean() if "sot" in rows else np.nan,
        f"corners_r{window}": rows["corners"].dropna().mean() if "corners" in rows else np.nan,
        f"fouls_r{window}": rows["fouls"].dropna().mean() if "fouls" in rows else np.nan,
        f"ycards_r{window}": rows["ycards"].dropna().mean() if "ycards" in rows else np.nan,
        f"xgf_r{window}": rows["xgf"].dropna().mean() if rows["xgf"].notna().any() else np.nan,
        f"xga_r{window}": rows["xga"].dropna().mean() if rows["xga"].notna().any() else np.nan,
        f"games_r{window}": float(len(rows)),
    }


def attach_domestic_form(euro_df: pd.DataFrame, team_hist: pd.DataFrame, windows=(3, 5, 10)):
    records = []
    for _, row in euro_df.iterrows():
        rec = {
            "Date": row["Date"],
            "competition": row["competition"],
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "FTHG": row["FTHG"],
            "FTAG": row["FTAG"],
            "over25": row["over25"],
            "btts": row["btts"],
            "ftr_encoded": row["ftr_encoded"],
            "home_key": row["home_key"],
            "away_key": row["away_key"],
        }
        for window in windows:
            home_stats = recent_stats(team_hist, row["home_key"], row["Date"], window)
            away_stats = recent_stats(team_hist, row["away_key"], row["Date"], window)
            for key, value in home_stats.items():
                rec[f"home_{key}"] = value
            for key, value in away_stats.items():
                rec[f"away_{key}"] = value
            rec[f"xg_atk_def_diff_r{window}"] = rec[f"home_xgf_r{window}"] - rec[f"away_xga_r{window}"]
            rec[f"ppg_proxy_diff_r{window}"] = (
                (rec[f"home_gf_r{window}"] - rec[f"home_ga_r{window}"]) -
                (rec[f"away_gf_r{window}"] - rec[f"away_ga_r{window}"])
            )
            rec[f"atk_def_diff_r{window}"] = rec[f"home_gf_r{window}"] - rec[f"away_ga_r{window}"]
            rec[f"def_atk_diff_r{window}"] = rec[f"away_gf_r{window}"] - rec[f"home_ga_r{window}"]
        records.append(rec)
    out = pd.DataFrame(records)
    out["competition_flag"] = out["competition"].map({"CL": 1.0, "EL": 0.0})
    out["season"] = out["Date"].apply(lambda d: f"{d.year-1}/{d.year}" if d.month < 7 else f"{d.year}/{d.year+1}")
    return out


def attach_europe_form(training_df: pd.DataFrame, windows=(3, 5)):
    out = training_df.sort_values("Date").reset_index(drop=True).copy()
    for window in windows:
        for prefix in [
            f"home_eu_gf_r{window}",
            f"home_eu_ga_r{window}",
            f"home_eu_ppg_r{window}",
            f"away_eu_gf_r{window}",
            f"away_eu_ga_r{window}",
            f"away_eu_ppg_r{window}",
            f"eu_gd_diff_r{window}",
            f"eu_ppg_diff_r{window}",
        ]:
            out[prefix] = np.nan

    history = {}
    for idx, row in out.iterrows():
        home_key = row["home_key"]
        away_key = row["away_key"]

        for window in windows:
            home_hist = history.get(home_key, [])[-window:]
            away_hist = history.get(away_key, [])[-window:]

            if home_hist:
                out.at[idx, f"home_eu_gf_r{window}"] = np.mean([r["gf"] for r in home_hist])
                out.at[idx, f"home_eu_ga_r{window}"] = np.mean([r["ga"] for r in home_hist])
                out.at[idx, f"home_eu_ppg_r{window}"] = np.mean([r["points"] for r in home_hist])
            if away_hist:
                out.at[idx, f"away_eu_gf_r{window}"] = np.mean([r["gf"] for r in away_hist])
                out.at[idx, f"away_eu_ga_r{window}"] = np.mean([r["ga"] for r in away_hist])
                out.at[idx, f"away_eu_ppg_r{window}"] = np.mean([r["points"] for r in away_hist])

            if home_hist and away_hist:
                out.at[idx, f"eu_gd_diff_r{window}"] = (
                    out.at[idx, f"home_eu_gf_r{window}"] - out.at[idx, f"home_eu_ga_r{window}"]
                ) - (
                    out.at[idx, f"away_eu_gf_r{window}"] - out.at[idx, f"away_eu_ga_r{window}"]
                )
                out.at[idx, f"eu_ppg_diff_r{window}"] = (
                    out.at[idx, f"home_eu_ppg_r{window}"] - out.at[idx, f"away_eu_ppg_r{window}"]
                )

        home_pts = 3 if row["FTHG"] > row["FTAG"] else (1 if row["FTHG"] == row["FTAG"] else 0)
        away_pts = 3 if row["FTAG"] > row["FTHG"] else (1 if row["FTHG"] == row["FTAG"] else 0)
        history.setdefault(home_key, []).append({"gf": row["FTHG"], "ga": row["FTAG"], "points": home_pts})
        history.setdefault(away_key, []).append({"gf": row["FTAG"], "ga": row["FTHG"], "points": away_pts})

    return out


def main():
    euro_df = load_europe_matches()
    team_hist = pd.concat([load_understat_histories(), load_support_histories()], ignore_index=True)
    team_hist = team_hist.sort_values(["team_key", "Date"]).reset_index(drop=True)
    training_df = attach_domestic_form(euro_df, team_hist)
    training_df = attach_europe_form(training_df)
    training_df.to_csv(OUT_PATH, index=False)
    coverage = {
        "rows": len(training_df),
        "date_min": str(training_df["Date"].min().date()) if not training_df.empty else None,
        "date_max": str(training_df["Date"].max().date()) if not training_df.empty else None,
        "home_gf_r5_cov": int(training_df["home_gf_r5"].notna().sum()) if "home_gf_r5" in training_df.columns else 0,
        "away_gf_r5_cov": int(training_df["away_gf_r5"].notna().sum()) if "away_gf_r5" in training_df.columns else 0,
        "home_xgf_r5_cov": int(training_df["home_xgf_r5"].notna().sum()) if "home_xgf_r5" in training_df.columns else 0,
        "away_xgf_r5_cov": int(training_df["away_xgf_r5"].notna().sum()) if "away_xgf_r5" in training_df.columns else 0,
    }
    print(coverage)


if __name__ == "__main__":
    main()
