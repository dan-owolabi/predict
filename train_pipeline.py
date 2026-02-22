"""
Ensemble training pipeline: Dixon-Coles + LightGBM.
Uses expanding window CV on EPL seasons.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.impute import SimpleImputer
import lightgbm as lgb

from dixon_coles import DixonColesModel
from feature_engine import (parse_understat_xg, merge_xg,
                            build_rolling_features, get_feature_columns)

DATA_PATH = Path('E0 - E0.csv.csv')
XG_PATH = Path('understat_xg.csv')


def load_and_prepare():
    """Load data, merge xG, build features."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
    df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
    for c in ['HS','AS','HST','AST','HC','AC']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Date','FTHG','FTAG']).sort_values('Date').reset_index(drop=True)

    # Merge xG
    if XG_PATH.exists():
        print("Merging xG data from Understat...")
        xg_df = parse_understat_xg(XG_PATH)
        df = merge_xg(df, xg_df)
        xg_count = df['home_xg'].notna().sum()
        print(f"   Matched {xg_count}/{len(df)} matches with xG data")

    # Targets
    df['over25'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
    df['btts'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)

    # Build rolling features
    print("Building rolling features (k=3,5,10)...")
    df = build_rolling_features(df, windows=(3, 5, 10))

    # Derive season for CV
    df['season'] = df['Date'].apply(
        lambda d: f"{d.year-1}/{d.year}" if d.month < 7 else f"{d.year}/{d.year+1}")

    print(f"Done: {len(df)} matches | {df['season'].nunique()} seasons")
    print(f"   Over 2.5 rate: {df['over25'].mean()*100:.1f}%")
    print(f"   BTTS rate: {df['btts'].mean()*100:.1f}%")
    return df


def train_dixon_coles(train_df):
    """Train Dixon-Coles on training data."""
    dc = DixonColesModel(decay_rate=0.003)
    dc.fit(
        train_df['HomeTeam'].values,
        train_df['AwayTeam'].values,
        train_df['FTHG'].values,
        train_df['FTAG'].values,
        train_df['Date'].values
    )
    return dc


def get_dc_features(dc, df):
    """Get Dixon-Coles probabilities as features."""
    dc_ou25, dc_btts = [], []
    for _, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        if ht in dc.teams and at in dc.teams:
            dc_ou25.append(dc.predict_ou25(ht, at))
            dc_btts.append(dc.predict_btts(ht, at))
        else:
            dc_ou25.append(0.5)
            dc_btts.append(0.5)
    return np.array(dc_ou25), np.array(dc_btts)


def expanding_window_cv(df, min_train_seasons=3):
    """Expanding window CV: train on seasons up to k, validate on k+1."""
    seasons = sorted(df['season'].unique())
    print(f"\nSeasons: {seasons}")

    all_results = []
    feature_cols = get_feature_columns(df)
    print(f"Feature columns: {len(feature_cols)}")

    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        val_season = seasons[i]

        train_df = df[df['season'].isin(train_seasons)].copy()
        val_df = df[df['season'] == val_season].copy()

        # Drop rows with too many NaN features
        train_clean = train_df.dropna(subset=feature_cols, thresh=len(feature_cols)*0.7)
        val_clean = val_df.dropna(subset=feature_cols, thresh=len(feature_cols)*0.7)

        if len(train_clean) < 100 or len(val_clean) < 20:
            continue

        print(f"\n{'='*50}")
        print(f"Train: {train_seasons} ({len(train_clean)} matches)")
        print(f"Val:   {val_season} ({len(val_clean)} matches)")

        # Impute NaNs
        X_train_raw = train_clean[feature_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
        X_val_raw = val_clean[feature_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
        imputer = SimpleImputer(strategy='median')
        X_train = np.nan_to_num(imputer.fit_transform(X_train_raw), nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(imputer.transform(X_val_raw), nan=0.0, posinf=0.0, neginf=0.0)

        # --- Model 1: Dixon-Coles ---
        dc = train_dixon_coles(train_clean)
        dc_ou25_train, dc_btts_train = get_dc_features(dc, train_clean)
        dc_ou25_val, dc_btts_val = get_dc_features(dc, val_clean)

        for target_name, target_col in [('over25', 'over25'), ('btts', 'btts')]:
            y_train = train_clean[target_col].values
            y_val = val_clean[target_col].values

            dc_feat_train = dc_ou25_train if target_name == 'over25' else dc_btts_train
            dc_feat_val = dc_ou25_val if target_name == 'over25' else dc_btts_val

            Xt = np.column_stack([X_train, dc_feat_train])
            Xv = np.column_stack([X_val, dc_feat_val])

            # --- LightGBM ---
            lgb_train = lgb.Dataset(Xt, y_train)
            lgb_val_ds = lgb.Dataset(Xv, y_val, reference=lgb_train)

            params = {
                'objective': 'binary', 'metric': 'binary_logloss',
                'boosting_type': 'gbdt', 'num_leaves': 16, 'max_depth': 4,
                'learning_rate': 0.05, 'min_child_samples': 25,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'reg_alpha': 1.0, 'reg_lambda': 1.0,
                'verbose': -1, 'n_jobs': -1,
            }

            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            model = lgb.train(params, lgb_train, num_boost_round=500,
                              valid_sets=[lgb_val_ds], callbacks=callbacks)
            lgb_probs = model.predict(Xv)

            # --- Ensemble: weighted average DC + LGB ---
            dc_probs = np.nan_to_num(dc_feat_val, nan=0.5)
            lgb_probs = np.nan_to_num(lgb_probs, nan=0.5)
            ensemble_probs = 0.3 * dc_probs + 0.7 * lgb_probs

            # --- Evaluate ---
            base_rate = y_val.mean()
            naive_ll = log_loss(y_val, [base_rate] * len(y_val))

            results = {
                'season': val_season, 'target': target_name,
                'n_val': len(y_val), 'base_rate': base_rate,
                'naive_logloss': naive_ll,
            }

            for name, probs in [('DC', dc_probs), ('LGB', lgb_probs), ('Ensemble', ensemble_probs)]:
                probs_clipped = np.clip(probs, 0.01, 0.99)
                ll = log_loss(y_val, probs_clipped)
                bs = brier_score_loss(y_val, probs_clipped)
                acc = ((probs_clipped > 0.5).astype(int) == y_val).mean()
                results[f'{name}_logloss'] = ll
                results[f'{name}_brier'] = bs
                results[f'{name}_acc'] = acc

            all_results.append(results)

            print(f"\n  {target_name.upper()} (base rate: {base_rate*100:.1f}%)")
            print(f"     Naive LL: {naive_ll:.4f}")
            for m in ['DC', 'LGB', 'Ensemble']:
                print(f"     {m:10s}: LL={results[f'{m}_logloss']:.4f} | Brier={results[f'{m}_brier']:.4f} | Acc={results[f'{m}_acc']*100:.1f}%")

    return pd.DataFrame(all_results)


if __name__ == '__main__':
    df = load_and_prepare()
    results = expanding_window_cv(df)

    print("\n" + "="*60)
    print("SUMMARY ACROSS ALL FOLDS")
    print("="*60)
    for target in ['over25', 'btts']:
        t_res = results[results['target'] == target]
        if len(t_res) == 0:
            continue
        print(f"\n  {target.upper()}")
        print(f"  {'='*40}")
        print(f"  Naive LL:    {t_res['naive_logloss'].mean():.4f}")
        for m in ['DC', 'LGB', 'Ensemble']:
            ll = t_res[f'{m}_logloss'].mean()
            bs = t_res[f'{m}_brier'].mean()
            acc = t_res[f'{m}_acc'].mean()
            print(f"  {m:10s}:  LL={ll:.4f} | Brier={bs:.4f} | Acc={acc*100:.1f}%")

    results.to_csv('cv_results.csv', index=False)
    print("\nResults saved to cv_results.csv")
