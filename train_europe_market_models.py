import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss


DATA_PATH = Path("data") / "europe_market_training_data.csv"
CV_PATH = Path("europe_market_cv_results.csv")
OUT_DIR = Path("europe_market_models")

LGB_PARAMS_BINARY = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 16,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_samples": 10,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.3,
    "reg_lambda": 0.5,
    "verbose": -1,
    "n_jobs": -1,
}

MARKET_SPECS = [
    {"target": "corners_over_8_5", "label": "Corners O8.5", "market": "CORNERS", "line": 8.5, "pick": "OVER 8.5"},
    {"target": "corners_over_9_5", "label": "Corners O9.5", "market": "CORNERS", "line": 9.5, "pick": "OVER 9.5"},
    {"target": "corners_over_10_5", "label": "Corners O10.5", "market": "CORNERS", "line": 10.5, "pick": "OVER 10.5"},
    {"target": "home_corners_over_4_5", "label": "Home Corners O4.5", "market": "HOME CORNERS", "line": 4.5, "pick": "OVER 4.5"},
    {"target": "away_corners_over_3_5", "label": "Away Corners O3.5", "market": "AWAY CORNERS", "line": 3.5, "pick": "OVER 3.5"},
    {"target": "bookings_over_3_5", "label": "Bookings O3.5", "market": "BOOKINGS", "line": 3.5, "pick": "OVER 3.5"},
    {"target": "bookings_over_4_5", "label": "Bookings O4.5", "market": "BOOKINGS", "line": 4.5, "pick": "OVER 4.5"},
    {"target": "bookings_over_5_5", "label": "Bookings O5.5", "market": "BOOKINGS", "line": 5.5, "pick": "OVER 5.5"},
    {"target": "home_bookings_over_1_5", "label": "Home Bookings O1.5", "market": "HOME BOOKINGS", "line": 1.5, "pick": "OVER 1.5"},
    {"target": "away_bookings_over_1_5", "label": "Away Bookings O1.5", "market": "AWAY BOOKINGS", "line": 1.5, "pick": "OVER 1.5"},
    {"target": "shots_over_22_5", "label": "Shots O22.5", "market": "SHOTS", "line": 22.5, "pick": "OVER 22.5"},
    {"target": "shots_over_24_5", "label": "Shots O24.5", "market": "SHOTS", "line": 24.5, "pick": "OVER 24.5"},
    {"target": "shots_over_26_5", "label": "Shots O26.5", "market": "SHOTS", "line": 26.5, "pick": "OVER 26.5"},
    {"target": "home_shots_over_11_5", "label": "Home Shots O11.5", "market": "HOME SHOTS", "line": 11.5, "pick": "OVER 11.5"},
    {"target": "away_shots_over_9_5", "label": "Away Shots O9.5", "market": "AWAY SHOTS", "line": 9.5, "pick": "OVER 9.5"},
    {"target": "sot_over_7_5", "label": "SOT O7.5", "market": "SHOTS ON TARGET", "line": 7.5, "pick": "OVER 7.5"},
    {"target": "sot_over_8_5", "label": "SOT O8.5", "market": "SHOTS ON TARGET", "line": 8.5, "pick": "OVER 8.5"},
    {"target": "sot_over_9_5", "label": "SOT O9.5", "market": "SHOTS ON TARGET", "line": 9.5, "pick": "OVER 9.5"},
    {"target": "home_sot_over_3_5", "label": "Home SOT O3.5", "market": "HOME SOT", "line": 3.5, "pick": "OVER 3.5"},
    {"target": "away_sot_over_2_5", "label": "Away SOT O2.5", "market": "AWAY SOT", "line": 2.5, "pick": "OVER 2.5"},
]


def feature_columns(df: pd.DataFrame):
    return [
        col for col in df.columns
        if any(
            col.startswith(prefix) for prefix in [
                "home_gf_r", "home_ga_r", "home_shots_r", "home_sot_r", "home_corners_r", "home_fouls_r", "home_ycards_r", "home_xgf_r", "home_xga_r", "home_games_r",
                "away_gf_r", "away_ga_r", "away_shots_r", "away_sot_r", "away_corners_r", "away_fouls_r", "away_ycards_r", "away_xgf_r", "away_xga_r", "away_games_r",
                "xg_atk_def_diff_r", "ppg_proxy_diff_r", "atk_def_diff_r", "def_atk_diff_r",
                "home_eu_gf_r", "home_eu_ga_r", "home_eu_ppg_r", "away_eu_gf_r", "away_eu_ga_r", "away_eu_ppg_r",
                "eu_gd_diff_r", "eu_ppg_diff_r",
            ]
        )
    ] + ["competition_flag"]


def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def evaluate_markets(df, feats):
    seasons = sorted(df["season"].dropna().unique())
    if len(seasons) < 2:
        return pd.DataFrame()
    train_df = df[df["season"].isin(seasons[:-1])].copy()
    val_df = df[df["season"] == seasons[-1]].copy()
    X_train_full = train_df[feats].apply(pd.to_numeric, errors="coerce").values
    X_val_full = val_df[feats].apply(pd.to_numeric, errors="coerce").values
    imputer = SimpleImputer(strategy="median")
    X_train_full = imputer.fit_transform(X_train_full)
    X_val_full = imputer.transform(X_val_full)
    rows = []
    for spec in MARKET_SPECS:
        train_mask = train_df[spec["target"]].notna()
        val_mask = val_df[spec["target"]].notna()
        if train_mask.sum() < 30 or val_mask.sum() < 10:
            continue
        y_train = train_df.loc[train_mask, spec["target"]].astype(int).values
        y_val = val_df.loc[val_mask, spec["target"]].astype(int).values
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue
        booster = lgb.train(LGB_PARAMS_BINARY, lgb.Dataset(X_train_full[train_mask.values], y_train), num_boost_round=120)
        probs = np.clip(booster.predict(X_val_full[val_mask.values]), 0.01, 0.99)
        rows.append({
            "season": seasons[-1],
            "target": spec["target"],
            "label": spec["label"],
            "market": spec["market"],
            "line": spec["line"],
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "base_rate": float(y_train.mean()),
            "log_loss": float(log_loss(y_val, probs)),
            "brier": float(brier_score_loss(y_val, probs)),
            "accuracy": float(((probs >= 0.5).astype(int) == y_val).mean()),
        })
    return pd.DataFrame(rows)


def train_full_models(df, feats):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_all = df[feats].apply(pd.to_numeric, errors="coerce").values
    imputer = SimpleImputer(strategy="median")
    X_all = imputer.fit_transform(X_all)
    manifest = []
    for spec in MARKET_SPECS:
        mask = df[spec["target"]].notna()
        if mask.sum() < 40:
            continue
        y = df.loc[mask, spec["target"]].astype(int).values
        if len(np.unique(y)) < 2:
            continue
        booster = lgb.train(LGB_PARAMS_BINARY, lgb.Dataset(X_all[mask.values], y), num_boost_round=120)
        model_file = f"{spec['target']}.txt"
        booster.save_model(str(OUT_DIR / model_file))
        manifest.append({**spec, "model_file": model_file, "base_rate": float(y.mean())})
    with open(OUT_DIR / "artifacts.pkl", "wb") as f:
        pickle.dump({"imputer": imputer, "feature_cols": feats}, f)
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main():
    df = load_data()
    feats = feature_columns(df)
    cv = evaluate_markets(df, feats)
    cv.to_csv(CV_PATH, index=False)
    if not cv.empty:
        print(cv.sort_values("accuracy", ascending=False).to_string(index=False))
    train_full_models(df, feats)
    print({"rows": len(df), "features": len(feats), "cv_rows": len(cv)})


if __name__ == "__main__":
    main()
