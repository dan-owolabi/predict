"""
Train the final production models on the ENTIRE dataset.
Saves the models to disk so the bot can load them instantly.
"""
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from dixon_coles import DixonColesModel
from feature_engine import get_feature_columns
from train_pipeline import load_and_prepare, get_dc_features

def train_final_models():
    print("Loading data...")
    df = load_and_prepare()
    feature_cols = get_feature_columns(df)
    
    print(f"\nTraining Dixon-Coles on all {len(df)} matches...")
    dc = DixonColesModel(decay_rate=0.003)
    dc.fit(df['HomeTeam'].values, df['AwayTeam'].values, df['FTHG'].values, df['FTAG'].values, df['Date'].values)
    
    print("Getting Dixon-Coles features...")
    dc_ou25, dc_btts = get_dc_features(dc, df)
    
    print("Imputing remaining NaNs...")
    X_raw = df[feature_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_raw)
    X = np.nan_to_num(X_imputed, nan=0.0, posinf=0.0, neginf=0.0)
    
    y_ou25 = df['over25'].values
    y_btts = df['btts'].values
    
    # 1. Train LightGBM OVER 2.5
    print("\nTraining LightGBM Over 2.5 Model...")
    X_ou25 = np.column_stack([X, dc_ou25])
    lgb_train_ou25 = lgb.Dataset(X_ou25, y_ou25)
    params = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'boosting_type': 'gbdt', 'num_leaves': 16, 'max_depth': 4,
        'learning_rate': 0.05, 'min_child_samples': 25,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 1.0, 'reg_lambda': 1.0,
        'verbose': -1, 'n_jobs': -1,
    }
    model_ou25 = lgb.train(params, lgb_train_ou25, num_boost_round=150) # use fixed rounds for full dataset
    
    # 2. Train LightGBM BTTS
    print("Training LightGBM BTTS Model...")
    X_btts = np.column_stack([X, dc_btts])
    lgb_train_btts = lgb.Dataset(X_btts, y_btts)
    model_btts = lgb.train(params, lgb_train_btts, num_boost_round=150)
    
    # Save everything needed for inference
    print("\nSaving models and artifacts to disk...")
    artifacts = {
        'dixon_coles': dc,
        'imputer': imputer,
        'feature_cols': feature_cols,
        # Save the dataset to compute rolling features on the fly
        'historical_df': df, 
    }
    
    with open('production_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
        
    model_ou25.save_model('lgb_ou25.txt')
    model_btts.save_model('lgb_btts.txt')
    
    print("✅ Successfully trained and saved all production models!")

if __name__ == '__main__':
    train_final_models()
