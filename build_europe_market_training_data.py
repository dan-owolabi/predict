from pathlib import Path

import pandas as pd
from build_europe_training_data import canonical_name


EUROPE_PATH = Path("data") / "europe_training_data.csv"
STATS_PATH = Path("data") / "uefa_stats" / "uefa_match_stats.csv"
OUT_PATH = Path("data") / "europe_market_training_data.csv"


def add_market_targets(df: pd.DataFrame):
    df = df.copy()
    specs = [
        ("corners_over_8_5", "total_corners", 8.5),
        ("corners_over_9_5", "total_corners", 9.5),
        ("corners_over_10_5", "total_corners", 10.5),
        ("home_corners_over_4_5", "home_corners", 4.5),
        ("away_corners_over_3_5", "away_corners", 3.5),
        ("bookings_over_3_5", "total_bookings", 3.5),
        ("bookings_over_4_5", "total_bookings", 4.5),
        ("bookings_over_5_5", "total_bookings", 5.5),
        ("home_bookings_over_1_5", "home_bookings", 1.5),
        ("away_bookings_over_1_5", "away_bookings", 1.5),
        ("shots_over_22_5", "total_shots", 22.5),
        ("shots_over_24_5", "total_shots", 24.5),
        ("shots_over_26_5", "total_shots", 26.5),
        ("home_shots_over_11_5", "home_shots", 11.5),
        ("away_shots_over_9_5", "away_shots", 9.5),
        ("sot_over_7_5", "total_sot", 7.5),
        ("sot_over_8_5", "total_sot", 8.5),
        ("sot_over_9_5", "total_sot", 9.5),
        ("home_sot_over_3_5", "home_sot", 3.5),
        ("away_sot_over_2_5", "away_sot", 2.5),
    ]
    for target, source, line in specs:
        df[target] = pd.NA
        mask = pd.to_numeric(df[source], errors="coerce").notna()
        df.loc[mask, target] = (pd.to_numeric(df.loc[mask, source], errors="coerce") > line).astype(int)
    return df


def main():
    europe_df = pd.read_csv(EUROPE_PATH, low_memory=False)
    stat_df = pd.read_csv(STATS_PATH, low_memory=False)
    europe_df["Date"] = pd.to_datetime(europe_df["Date"], errors="coerce").dt.date.astype(str)
    stat_df["Date"] = pd.to_datetime(stat_df["Date"], errors="coerce").dt.date.astype(str)
    europe_df["home_key_merge"] = europe_df["HomeTeam"].map(canonical_name)
    europe_df["away_key_merge"] = europe_df["AwayTeam"].map(canonical_name)
    stat_df["home_key_merge"] = stat_df["HomeTeam"].map(canonical_name)
    stat_df["away_key_merge"] = stat_df["AwayTeam"].map(canonical_name)
    merged = europe_df.merge(
        stat_df,
        on=["Date", "FTHG", "FTAG", "home_key_merge", "away_key_merge"],
        how="left",
        suffixes=("", "_stat"),
    )
    merged = add_market_targets(merged)
    merged.to_csv(OUT_PATH, index=False)
    print({
        "rows": int(len(merged)),
        "stat_rows": int(merged["total_shots"].notna().sum()) if "total_shots" in merged.columns else 0,
    })


if __name__ == "__main__":
    main()
