"""
Ensemble training pipeline: Dixon-Coles + LightGBM.
Uses expanding window CV on EPL seasons.
Trains models for: 1X2 (match result), Over/Under 2.5, BTTS.
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
    df['ftr_encoded'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

    # Build rolling features
    print("Building rolling features (k=3,5,10)...")
    df = build_rolling_features(df, windows=(3, 5, 10))

    # Derive season for CV
    df['season'] = df['Date'].apply(
        lambda d: f"{d.year-1}/{d.year}" if d.month < 7 else f"{d.year}/{d.year+1}")

    print(f"Done: {len(df)} matches | {df['season'].nunique()} seasons")
    print(f"   Over 2.5 rate: {df['over25'].mean()*100:.1f}%")
    print(f"   BTTS rate: {df['btts'].mean()*100:.1f}%")
    print(f"   Home/Draw/Away: {(df['FTR']=='H').mean()*100:.1f}% / {(df['FTR']=='D').mean()*100:.1f}% / {(df['FTR']=='A').mean()*100:.1f}%")
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
    dc_home, dc_draw, dc_away = [], [], []
    for _, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        if ht in dc.teams and at in dc.teams:
            dc_ou25.append(dc.predict_ou25(ht, at))
            dc_btts.append(dc.predict_btts(ht, at))
            ph, pd_, pa = dc.predict_match_result(ht, at)
            dc_home.append(ph)
            dc_draw.append(pd_)
            dc_away.append(pa)
        else:
            dc_ou25.append(0.5)
            dc_btts.append(0.5)
            dc_home.append(0.33)
            dc_draw.append(0.34)
            dc_away.append(0.33)
    return (np.array(dc_ou25), np.array(dc_btts),
            np.array(dc_home), np.array(dc_draw), np.array(dc_away))


LGB_PARAMS_BINARY = {
    'objective': 'binary', 'metric': 'binary_logloss',
    'boosting_type': 'gbdt', 'num_leaves': 16, 'max_depth': 4,
    'learning_rate': 0.05, 'min_child_samples': 25,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 1.0, 'reg_lambda': 1.0,
    'verbose': -1, 'n_jobs': -1,
}

LGB_PARAMS_1X2 = {
    'objective': 'multiclass', 'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt', 'num_leaves': 16, 'max_depth': 4,
    'learning_rate': 0.05, 'min_child_samples': 25,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 1.0, 'reg_lambda': 1.0,
    'verbose': -1, 'n_jobs': -1,
}


def find_best_ensemble_weight(dc_probs, lgb_probs, y_true, is_multiclass=False):
    """Find optimal DC/LGB ensemble weight by grid search."""
    best_w, best_ll = 0.0, float('inf')
    for w_dc in np.arange(0.0, 0.55, 0.05):
        blend = w_dc * dc_probs + (1 - w_dc) * lgb_probs
        if is_multiclass:
            blend = np.clip(blend, 0.01, 0.99)
            blend = blend / blend.sum(axis=1, keepdims=True)
            ll = log_loss(y_true, blend)
        else:
            blend = np.clip(blend, 0.01, 0.99)
            ll = log_loss(y_true, blend)
        if ll < best_ll:
            best_ll = ll
            best_w = w_dc
    return best_w


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

        # --- Dixon-Coles ---
        dc = train_dixon_coles(train_clean)
        dc_ou25_train, dc_btts_train, dc_home_train, dc_draw_train, dc_away_train = get_dc_features(dc, train_clean)
        dc_ou25_val, dc_btts_val, dc_home_val, dc_draw_val, dc_away_val = get_dc_features(dc, val_clean)

        # All DC features stacked (cross-market signal)
        dc_stack_train = np.column_stack([dc_ou25_train, dc_btts_train, dc_home_train, dc_draw_train, dc_away_train])
        dc_stack_val = np.column_stack([dc_ou25_val, dc_btts_val, dc_home_val, dc_draw_val, dc_away_val])

        # ============================================================
        # BINARY MODELS: Over 2.5, BTTS
        # ============================================================
        for target_name, target_col, dc_feat_train, dc_feat_val in [
            ('over25', 'over25', dc_ou25_train, dc_ou25_val),
            ('btts', 'btts', dc_btts_train, dc_btts_val),
        ]:
            y_train = train_clean[target_col].values
            y_val = val_clean[target_col].values

            # Stack ALL DC features for richer signal
            Xt = np.column_stack([X_train, dc_stack_train])
            Xv = np.column_stack([X_val, dc_stack_val])

            # LightGBM
            lgb_train = lgb.Dataset(Xt, y_train)
            lgb_val_ds = lgb.Dataset(Xv, y_val, reference=lgb_train)

            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            model = lgb.train(LGB_PARAMS_BINARY, lgb_train, num_boost_round=500,
                              valid_sets=[lgb_val_ds], callbacks=callbacks)
            lgb_probs = model.predict(Xv)

            # Find optimal ensemble weight
            dc_probs = np.nan_to_num(dc_feat_val, nan=0.5)
            lgb_probs_clean = np.nan_to_num(lgb_probs, nan=0.5)
            best_w = find_best_ensemble_weight(dc_probs, lgb_probs_clean, y_val)
            ensemble_probs = best_w * dc_probs + (1 - best_w) * lgb_probs_clean

            # Evaluate
            base_rate = y_val.mean()
            naive_ll = log_loss(y_val, [base_rate] * len(y_val))

            results = {
                'season': val_season, 'target': target_name,
                'n_val': len(y_val), 'base_rate': base_rate,
                'naive_logloss': naive_ll, 'best_dc_weight': best_w,
            }

            for name, probs in [('DC', dc_probs), ('LGB', lgb_probs_clean), ('Ensemble', ensemble_probs)]:
                probs_clipped = np.clip(probs, 0.01, 0.99)
                ll = log_loss(y_val, probs_clipped)
                bs = brier_score_loss(y_val, probs_clipped)
                acc = ((probs_clipped > 0.5).astype(int) == y_val).mean()
                results[f'{name}_logloss'] = ll
                results[f'{name}_brier'] = bs
                results[f'{name}_acc'] = acc

            all_results.append(results)

            print(f"\n  {target_name.upper()} (base rate: {base_rate*100:.1f}%) [DC weight: {best_w:.2f}]")
            print(f"     Naive LL: {naive_ll:.4f}")
            for m in ['DC', 'LGB', 'Ensemble']:
                print(f"     {m:10s}: LL={results[f'{m}_logloss']:.4f} | Brier={results[f'{m}_brier']:.4f} | Acc={results[f'{m}_acc']*100:.1f}%")

        # ============================================================
        # MULTICLASS MODEL: 1X2 (Match Result)
        # ============================================================
        y_train_1x2 = train_clean['ftr_encoded'].values
        y_val_1x2 = val_clean['ftr_encoded'].values

        # Drop rows with NaN target
        mask_train = ~np.isnan(y_train_1x2)
        mask_val = ~np.isnan(y_val_1x2)

        if mask_train.sum() > 100 and mask_val.sum() > 20:
            Xt_1x2 = np.column_stack([X_train[mask_train], dc_stack_train[mask_train]])
            Xv_1x2 = np.column_stack([X_val[mask_val], dc_stack_val[mask_val]])
            yt_1x2 = y_train_1x2[mask_train].astype(int)
            yv_1x2 = y_val_1x2[mask_val].astype(int)

            lgb_train_1x2 = lgb.Dataset(Xt_1x2, yt_1x2)
            lgb_val_1x2 = lgb.Dataset(Xv_1x2, yv_1x2, reference=lgb_train_1x2)

            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            model_1x2 = lgb.train(LGB_PARAMS_1X2, lgb_train_1x2, num_boost_round=500,
                                   valid_sets=[lgb_val_1x2], callbacks=callbacks)
            lgb_1x2_probs = model_1x2.predict(Xv_1x2)  # shape: (n_val, 3)

            # DC 1X2 probs
            dc_1x2_probs = np.column_stack([dc_home_val[mask_val], dc_draw_val[mask_val], dc_away_val[mask_val]])
            dc_1x2_probs = np.nan_to_num(dc_1x2_probs, nan=0.33)

            # Find optimal ensemble weight
            best_w_1x2 = find_best_ensemble_weight(dc_1x2_probs, lgb_1x2_probs, yv_1x2, is_multiclass=True)
            ens_1x2 = best_w_1x2 * dc_1x2_probs + (1 - best_w_1x2) * lgb_1x2_probs
            ens_1x2 = ens_1x2 / ens_1x2.sum(axis=1, keepdims=True)

            # Evaluate
            base_rates_1x2 = np.array([(yv_1x2 == c).mean() for c in range(3)])
            naive_ll_1x2 = log_loss(yv_1x2, np.tile(base_rates_1x2, (len(yv_1x2), 1)))

            dc_1x2_clipped = np.clip(dc_1x2_probs, 0.01, 0.99)
            dc_1x2_clipped = dc_1x2_clipped / dc_1x2_clipped.sum(axis=1, keepdims=True)
            lgb_1x2_clipped = np.clip(lgb_1x2_probs, 0.01, 0.99)
            lgb_1x2_clipped = lgb_1x2_clipped / lgb_1x2_clipped.sum(axis=1, keepdims=True)
            ens_1x2_clipped = np.clip(ens_1x2, 0.01, 0.99)
            ens_1x2_clipped = ens_1x2_clipped / ens_1x2_clipped.sum(axis=1, keepdims=True)

            results_1x2 = {
                'season': val_season, 'target': '1x2',
                'n_val': len(yv_1x2), 'base_rate': base_rates_1x2.max(),
                'naive_logloss': naive_ll_1x2, 'best_dc_weight': best_w_1x2,
            }

            for name, probs in [('DC', dc_1x2_clipped), ('LGB', lgb_1x2_clipped), ('Ensemble', ens_1x2_clipped)]:
                ll = log_loss(yv_1x2, probs)
                acc = (np.argmax(probs, axis=1) == yv_1x2).mean()
                results_1x2[f'{name}_logloss'] = ll
                results_1x2[f'{name}_brier'] = 0.0  # not meaningful for multiclass
                results_1x2[f'{name}_acc'] = acc

            all_results.append(results_1x2)

            print(f"\n  1X2 (H:{base_rates_1x2[0]*100:.1f}% D:{base_rates_1x2[1]*100:.1f}% A:{base_rates_1x2[2]*100:.1f}%) [DC weight: {best_w_1x2:.2f}]")
            print(f"     Naive LL: {naive_ll_1x2:.4f}")
            for m in ['DC', 'LGB', 'Ensemble']:
                print(f"     {m:10s}: LL={results_1x2[f'{m}_logloss']:.4f} | Acc={results_1x2[f'{m}_acc']*100:.1f}%")

    return pd.DataFrame(all_results)


if __name__ == '__main__':
    df = load_and_prepare()
    results = expanding_window_cv(df)

    print("\n" + "="*60)
    print("SUMMARY ACROSS ALL FOLDS")
    print("="*60)
    for target in ['over25', 'btts', '1x2']:
        t_res = results[results['target'] == target]
        if len(t_res) == 0:
            continue
        print(f"\n  {target.upper()}")
        print(f"  {'='*40}")
        print(f"  Naive LL:    {t_res['naive_logloss'].mean():.4f}")
        if 'best_dc_weight' in t_res.columns:
            print(f"  Avg DC wt:   {t_res['best_dc_weight'].mean():.2f}")
        for m in ['DC', 'LGB', 'Ensemble']:
            ll = t_res[f'{m}_logloss'].mean()
            acc = t_res[f'{m}_acc'].mean()
            print(f"  {m:10s}:  LL={ll:.4f} | Acc={acc*100:.1f}%")

    results.to_csv('cv_results.csv', index=False)
    print("\nResults saved to cv_results.csv")
