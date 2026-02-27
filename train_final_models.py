"""
Train the final production models on the ENTIRE dataset.
Saves the models to disk so the bot can load them instantly.
Models: 1X2 (multiclass), Over/Under 2.5 (binary), Over/Under 1.5 (binary), BTTS (binary).
"""
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from dixon_coles import DixonColesModel
from feature_engine import get_feature_columns
from train_pipeline import load_and_prepare, get_dc_features, LGB_PARAMS_BINARY, LGB_PARAMS_1X2


def train_final_models():
    print("Loading data...")
    df = load_and_prepare()
    feature_cols = get_feature_columns(df)

    # Add OU1.5 target
    df['over15'] = ((df['FTHG'] + df['FTAG']) > 1.5).astype(int)

    print(f"\nTraining Dixon-Coles on all {len(df)} matches...")
    dc = DixonColesModel(decay_rate=0.003)
    dc.fit(df['HomeTeam'].values, df['AwayTeam'].values,
           df['FTHG'].values, df['FTAG'].values, df['Date'].values)

    # Verify DC is actually producing varied predictions
    test_teams = list(dc.teams.keys())[:5]
    print(f"\nDC sanity check (home_adv={dc.home_adv:.3f}, rho={dc.rho:.4f}):")
    for i in range(min(3, len(test_teams)-1)):
        ht, at = test_teams[i], test_teams[i+1]
        lh, la = dc.predict_goals(ht, at)
        ou25 = dc.predict_ou25(ht, at)
        ph, pd_, pa = dc.predict_match_result(ht, at)
        print(f"  {ht} vs {at}: xG={lh:.2f}-{la:.2f} | OU25={ou25:.3f} | H={ph:.3f} D={pd_:.3f} A={pa:.3f}")

    print("\nGetting Dixon-Coles features...")
    dc_ou25, dc_btts, dc_home, dc_draw, dc_away = get_dc_features(dc, df)

    # DC OU1.5 predictions
    dc_ou15 = []
    for _, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        if ht in dc.teams and at in dc.teams:
            probs = dc.predict_score_probs(ht, at)
            p_under15 = probs[0, 0] + probs[0, 1] + probs[1, 0]  # 0-0, 0-1, 1-0
            dc_ou15.append(1.0 - p_under15)
        else:
            dc_ou15.append(0.5)
    dc_ou15 = np.array(dc_ou15)

    print("Imputing remaining NaNs...")
    X_raw = df[feature_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_raw)
    X = np.nan_to_num(X_imputed, nan=0.0, posinf=0.0, neginf=0.0)

    # All DC features stacked (cross-market signal for all models)
    dc_stack = np.column_stack([dc_ou25, dc_btts, dc_home, dc_draw, dc_away, dc_ou15])

    y_ou25 = df['over25'].values
    y_ou15 = df['over15'].values
    y_btts = df['btts'].values
    y_1x2 = df['ftr_encoded'].values

    # 1. Train LightGBM Over 2.5
    print("\nTraining LightGBM Over 2.5 Model...")
    X_full = np.column_stack([X, dc_stack])
    lgb_train_ou25 = lgb.Dataset(X_full, y_ou25)
    model_ou25 = lgb.train(LGB_PARAMS_BINARY, lgb_train_ou25, num_boost_round=200)

    # 2. Train LightGBM Over 1.5
    print("Training LightGBM Over 1.5 Model...")
    lgb_train_ou15 = lgb.Dataset(X_full, y_ou15)
    model_ou15 = lgb.train(LGB_PARAMS_BINARY, lgb_train_ou15, num_boost_round=200)

    # 3. Train LightGBM BTTS
    print("Training LightGBM BTTS Model...")
    lgb_train_btts = lgb.Dataset(X_full, y_btts)
    model_btts = lgb.train(LGB_PARAMS_BINARY, lgb_train_btts, num_boost_round=200)

    # 4. Train LightGBM 1X2 (Match Result)
    print("Training LightGBM 1X2 Model...")
    mask_1x2 = ~np.isnan(y_1x2)
    X_1x2 = X_full[mask_1x2]
    y_1x2_clean = y_1x2[mask_1x2].astype(int)
    lgb_train_1x2 = lgb.Dataset(X_1x2, y_1x2_clean)
    model_1x2 = lgb.train(LGB_PARAMS_1X2, lgb_train_1x2, num_boost_round=200)

    # Save everything needed for inference
    print("\nSaving models and artifacts to disk...")
    artifacts = {
        'dixon_coles': dc,
        'imputer': imputer,
        'feature_cols': feature_cols,
        'historical_df': df,
        'dc_feature_names': ['dc_ou25', 'dc_btts', 'dc_home', 'dc_draw', 'dc_away', 'dc_ou15'],
    }

    with open('production_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)

    model_ou25.save_model('lgb_ou25.txt')
    model_ou15.save_model('lgb_ou15.txt')
    model_btts.save_model('lgb_btts.txt')
    model_1x2.save_model('lgb_1x2.txt')

    print("\nSaved:")
    print(f"  production_artifacts.pkl ({len(feature_cols)} features, {len(df)} matches)")
    print(f"  lgb_ou25.txt")
    print(f"  lgb_ou15.txt")
    print(f"  lgb_btts.txt")
    print(f"  lgb_1x2.txt")
    print(f"  OU1.5 base rate: {y_ou15.mean()*100:.1f}%")
    print("\nDone!")


if __name__ == '__main__':
    train_final_models()
