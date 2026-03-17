import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


EUROPE_DATA_PATH = Path("data") / "europe_training_data.csv"
MODEL_DIR = Path("market_models")
OUT_PATH = Path("data") / "europe_market_beta_scores.csv"


def load_models():
    with open(MODEL_DIR / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(MODEL_DIR / "artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    models = {}
    for row in manifest:
        models[row["target"]] = {
            "meta": row,
            "booster": lgb.Booster(model_file=str(MODEL_DIR / row["model_file"])),
        }
    return manifest, artifacts, models


def main():
    df = pd.read_csv(EUROPE_DATA_PATH, low_memory=False)
    manifest, artifacts, models = load_models()

    feature_cols = artifacts["feature_cols"]
    X_raw = df.reindex(columns=feature_cols, fill_value=np.nan).apply(pd.to_numeric, errors="coerce").values
    X = np.nan_to_num(artifacts["imputer"].transform(X_raw), nan=0.0, posinf=0.0, neginf=0.0)
    dc_stack = np.full((len(df), 5), 0.33, dtype=float)
    dc_stack[:, 0] = 0.5
    dc_stack[:, 1] = 0.5
    X_full = np.column_stack([X, dc_stack])

    out = df[["Date", "competition", "HomeTeam", "AwayTeam"]].copy()
    for item in manifest:
        target = item["target"]
        probs = np.clip(models[target]["booster"].predict(X_full), 0.01, 0.99)
        out[f"{target}_prob"] = probs
        out[f"{target}_pick"] = np.where(probs >= 0.5, item["pick"], f"UNDER {item['line']}" if "line" in item else "NO")

    out.to_csv(OUT_PATH, index=False)
    print({"rows": len(out), "markets": len(manifest), "out_path": str(OUT_PATH)})


if __name__ == "__main__":
    main()
