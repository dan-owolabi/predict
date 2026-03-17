import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")


DATA_PATH = Path("data") / "europe_training_data.csv"
OUT_PATH = Path("europe_cv_results.csv")
MODEL_DIR = Path("europe_models")

LGB_PARAMS_BINARY = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 24,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "verbose": -1,
    "n_jobs": -1,
}

LGB_PARAMS_1X2 = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 24,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "verbose": -1,
    "n_jobs": -1,
}


def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "FTHG", "FTAG", "ftr_encoded"]).sort_values("Date").reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame):
    return [
        col for col in df.columns
        if any(
            col.startswith(prefix) for prefix in [
                "home_gf_r", "home_ga_r", "home_shots_r", "home_sot_r", "home_corners_r", "home_fouls_r", "home_ycards_r", "home_xgf_r", "home_xga_r", "home_games_r",
                "away_gf_r", "away_ga_r", "away_shots_r", "away_sot_r", "away_corners_r", "away_fouls_r", "away_ycards_r", "away_xgf_r", "away_xga_r", "away_games_r",
                "xg_atk_def_diff_r", "ppg_proxy_diff_r",
                "atk_def_diff_r", "def_atk_diff_r",
                "home_eu_gf_r", "home_eu_ga_r", "home_eu_ppg_r",
                "away_eu_gf_r", "away_eu_ga_r", "away_eu_ppg_r",
                "eu_gd_diff_r", "eu_ppg_diff_r",
            ]
        )
    ] + ["competition_flag"]


def expanding_cv(df: pd.DataFrame, min_train_seasons: int = 3):
    seasons = sorted(df["season"].dropna().unique())
    feats = feature_columns(df)
    results = []

    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        val_season = seasons[i]
        train_df = df[df["season"].isin(train_seasons)].copy()
        val_df = df[df["season"] == val_season].copy()

        if len(train_df) < 200 or len(val_df) < 30:
            continue

        X_train_raw = train_df[feats].apply(pd.to_numeric, errors="coerce").values
        X_val_raw = val_df[feats].apply(pd.to_numeric, errors="coerce").values
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train_raw)
        X_val = imputer.transform(X_val_raw)

        for target_name, target_col in [("over25", "over25"), ("btts", "btts")]:
            y_train = train_df[target_col].astype(int).values
            y_val = val_df[target_col].astype(int).values
            booster = lgb.train(
                LGB_PARAMS_BINARY,
                lgb.Dataset(X_train, label=y_train),
                num_boost_round=200,
            )
            probs = np.clip(booster.predict(X_val), 0.01, 0.99)
            preds = (probs >= 0.5).astype(int)
            results.append(
                {
                    "season": val_season,
                    "target": target_name,
                    "log_loss": log_loss(y_val, probs),
                    "accuracy": accuracy_score(y_val, preds),
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                }
            )

        y_train = train_df["ftr_encoded"].astype(int).values
        y_val = val_df["ftr_encoded"].astype(int).values
        booster = lgb.train(
            LGB_PARAMS_1X2,
            lgb.Dataset(X_train, label=y_train),
            num_boost_round=250,
        )
        probs = booster.predict(X_val)
        probs = np.clip(probs, 0.01, 0.99)
        probs = probs / probs.sum(axis=1, keepdims=True)
        preds = probs.argmax(axis=1)
        results.append(
            {
                "season": val_season,
                "target": "1x2",
                "log_loss": log_loss(y_val, probs),
                "accuracy": accuracy_score(y_val, preds),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
            }
        )

    out = pd.DataFrame(results)
    if not out.empty:
        out.to_csv(OUT_PATH, index=False)
    return out


def main():
    df = load_data()
    out = expanding_cv(df)
    if out.empty:
        print("No CV rows produced.")
    else:
        summary = out.groupby("target")[["log_loss", "accuracy"]].mean().reset_index()
        print(summary.to_string(index=False))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    feats = feature_columns(df)
    X_raw = df[feats].apply(pd.to_numeric, errors="coerce").values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw)

    artifacts = {
        "feature_cols": feats,
        "imputer": imputer,
    }
    pd.to_pickle(artifacts, MODEL_DIR / "artifacts.pkl")

    manifest = []
    for target_name, target_col, params, rounds in [
        ("over25", "over25", LGB_PARAMS_BINARY, 220),
        ("btts", "btts", LGB_PARAMS_BINARY, 220),
        ("1x2", "ftr_encoded", LGB_PARAMS_1X2, 260),
    ]:
        y = df[target_col].astype(int).values
        booster = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=rounds)
        model_file = f"{target_name}.txt"
        booster.save_model(str(MODEL_DIR / model_file))
        manifest.append({"target": target_name, "model_file": model_file})

    pd.DataFrame(manifest).to_json(MODEL_DIR / "manifest.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
