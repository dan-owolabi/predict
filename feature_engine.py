"""
Feature Engine: builds rolling stats, matchup diffs, rest days, xG features.
All features use strict temporal ordering (only past matches).
"""
import pandas as pd
import numpy as np
import ast
import json


def parse_understat_xg(xg_path):
    """Parse Understat xG CSV with JSON-like columns."""
    df = pd.read_csv(xg_path)
    records = []
    for _, row in df.iterrows():
        try:
            h = ast.literal_eval(row['h'])
            a = ast.literal_eval(row['a'])
            goals = ast.literal_eval(row['goals'])
            xg = ast.literal_eval(row['xG'])
            records.append({
                'home_team_xg': h.get('title', ''),
                'away_team_xg': a.get('title', ''),
                'home_goals_xg': int(goals.get('h', 0)),
                'away_goals_xg': int(goals.get('a', 0)),
                'home_xg': float(xg.get('h', 0)),
                'away_xg': float(xg.get('a', 0)),
                'date_xg': pd.to_datetime(row['datetime']),
                'season_xg': row.get('Season', ''),
            })
        except Exception:
            continue
    return pd.DataFrame(records)


# Team name mapping: Understat -> E0 dataset
XG_TO_E0 = {
    'Manchester United': 'Man United',
    'Manchester City': 'Man City',
    'Wolverhampton Wanderers': 'Wolves',
    'Newcastle United': 'Newcastle',
    'Nottingham Forest': "Nott'm Forest",
    'Sheffield United': 'Sheffield United',
    'West Ham': 'West Ham',
    'Tottenham': 'Tottenham',
    'Brighton': 'Brighton',
    'Leicester': 'Leicester',
    'Leeds': 'Leeds',
    'West Bromwich Albion': 'West Brom',
    'Norwich': 'Norwich',
    'Watford': 'Watford',
    'Luton': 'Luton',
}


def merge_xg(main_df, xg_df):
    """Merge xG data into the main dataset by date + team matching."""
    xg_df = xg_df.copy()
    xg_df['home_team_xg'] = xg_df['home_team_xg'].replace(XG_TO_E0)
    xg_df['away_team_xg'] = xg_df['away_team_xg'].replace(XG_TO_E0)
    xg_df['date_match'] = xg_df['date_xg'].dt.date

    main_df = main_df.copy()
    main_df['date_match'] = main_df['Date'].dt.date

    merged = main_df.merge(
        xg_df[['home_team_xg', 'away_team_xg', 'date_match', 'home_xg', 'away_xg']],
        left_on=['HomeTeam', 'AwayTeam', 'date_match'],
        right_on=['home_team_xg', 'away_team_xg', 'date_match'],
        how='left'
    )
    merged.drop(columns=['home_team_xg', 'away_team_xg', 'date_match'], inplace=True)
    return merged


def build_rolling_features(df, windows=(3, 5, 10)):
    """Build rolling team features with strict temporal ordering."""
    df = df.sort_values('Date').reset_index(drop=True)

    # Initialize feature columns
    feature_cols = []
    for w in windows:
        for side in ['home', 'away']:
            for stat in ['gf', 'ga', 'xgf', 'xga', 'shots', 'sot', 'corners']:
                col = f'{side}_{stat}_r{w}'
                feature_cols.append(col)
                df[col] = np.nan

    # Overall form columns (all venues)
    for w in windows:
        for side in ['home', 'away']:
            for stat in ['ppg', 'win_rate', 'draw_rate', 'gf_all', 'ga_all', 'cs_rate']:
                col = f'{side}_{stat}_r{w}'
                df[col] = np.nan

    # H2H columns
    for stat in ['h2h_home_wins', 'h2h_draws', 'h2h_total_goals']:
        df[stat] = np.nan

    # Rest days
    df['rest_days_home'] = np.nan
    df['rest_days_away'] = np.nan

    # Track per-team history
    team_home_history = {}  # team -> list of dicts (home matches)
    team_away_history = {}  # team -> list of dicts (away matches)
    team_all_history = {}   # team -> list of dicts (all matches, for overall form)
    team_last_date = {}     # team -> last match date
    h2h_history = {}        # (team1, team2) sorted -> list of dicts

    for idx, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        date = row['Date']

        # Rest days
        if ht in team_last_date:
            df.at[idx, 'rest_days_home'] = (date - team_last_date[ht]).days
        if at in team_last_date:
            df.at[idx, 'rest_days_away'] = (date - team_last_date[at]).days

        # H2H features
        h2h_key = tuple(sorted([ht, at]))
        h2h = h2h_history.get(h2h_key, [])
        if len(h2h) >= 2:
            home_wins = sum(1 for m in h2h if m['winner'] == ht)
            draws = sum(1 for m in h2h if m['winner'] == 'draw')
            total_goals = np.mean([m['total_goals'] for m in h2h])
            df.at[idx, 'h2h_home_wins'] = home_wins / len(h2h)
            df.at[idx, 'h2h_draws'] = draws / len(h2h)
            df.at[idx, 'h2h_total_goals'] = total_goals

        # Compute rolling features FROM PAST matches only
        for w in windows:
            # Home team's home record
            hh = team_home_history.get(ht, [])[-w:]
            if len(hh) >= 2:
                df.at[idx, f'home_gf_r{w}'] = np.mean([m['gf'] for m in hh])
                df.at[idx, f'home_ga_r{w}'] = np.mean([m['ga'] for m in hh])
                df.at[idx, f'home_shots_r{w}'] = np.mean([m['shots'] for m in hh])
                df.at[idx, f'home_sot_r{w}'] = np.mean([m['sot'] for m in hh])
                df.at[idx, f'home_corners_r{w}'] = np.mean([m['corners'] for m in hh])
                xgf_vals = [m['xgf'] for m in hh if m['xgf'] is not None]
                if xgf_vals:
                    df.at[idx, f'home_xgf_r{w}'] = np.mean(xgf_vals)
                    df.at[idx, f'home_xga_r{w}'] = np.mean([m['xga'] for m in hh if m['xga'] is not None])

            # Away team's away record
            ah = team_away_history.get(at, [])[-w:]
            if len(ah) >= 2:
                df.at[idx, f'away_gf_r{w}'] = np.mean([m['gf'] for m in ah])
                df.at[idx, f'away_ga_r{w}'] = np.mean([m['ga'] for m in ah])
                df.at[idx, f'away_shots_r{w}'] = np.mean([m['shots'] for m in ah])
                df.at[idx, f'away_sot_r{w}'] = np.mean([m['sot'] for m in ah])
                df.at[idx, f'away_corners_r{w}'] = np.mean([m['corners'] for m in ah])
                xgf_vals = [m['xgf'] for m in ah if m['xgf'] is not None]
                if xgf_vals:
                    df.at[idx, f'away_xgf_r{w}'] = np.mean(xgf_vals)
                    df.at[idx, f'away_xga_r{w}'] = np.mean([m['xga'] for m in ah if m['xga'] is not None])

            # Overall form (all venues) for home team
            ht_all = team_all_history.get(ht, [])[-w:]
            if len(ht_all) >= 2:
                df.at[idx, f'home_ppg_r{w}'] = np.mean([m['points'] for m in ht_all])
                df.at[idx, f'home_win_rate_r{w}'] = np.mean([1 if m['points'] == 3 else 0 for m in ht_all])
                df.at[idx, f'home_draw_rate_r{w}'] = np.mean([1 if m['points'] == 1 else 0 for m in ht_all])
                df.at[idx, f'home_gf_all_r{w}'] = np.mean([m['gf'] for m in ht_all])
                df.at[idx, f'home_ga_all_r{w}'] = np.mean([m['ga'] for m in ht_all])
                df.at[idx, f'home_cs_rate_r{w}'] = np.mean([1 if m['ga'] == 0 else 0 for m in ht_all])

            # Overall form (all venues) for away team
            at_all = team_all_history.get(at, [])[-w:]
            if len(at_all) >= 2:
                df.at[idx, f'away_ppg_r{w}'] = np.mean([m['points'] for m in at_all])
                df.at[idx, f'away_win_rate_r{w}'] = np.mean([1 if m['points'] == 3 else 0 for m in at_all])
                df.at[idx, f'away_draw_rate_r{w}'] = np.mean([1 if m['points'] == 1 else 0 for m in at_all])
                df.at[idx, f'away_gf_all_r{w}'] = np.mean([m['gf'] for m in at_all])
                df.at[idx, f'away_ga_all_r{w}'] = np.mean([m['ga'] for m in at_all])
                df.at[idx, f'away_cs_rate_r{w}'] = np.mean([1 if m['ga'] == 0 else 0 for m in at_all])

        # NOW update history with current match (after feature computation)
        fthg = row['FTHG']
        ftag = row['FTAG']

        home_record = {
            'gf': fthg, 'ga': ftag,
            'shots': row.get('HS', 0), 'sot': row.get('HST', 0),
            'corners': row.get('HC', 0),
            'xgf': row.get('home_xg'), 'xga': row.get('away_xg'),
        }
        away_record = {
            'gf': ftag, 'ga': fthg,
            'shots': row.get('AS', 0), 'sot': row.get('AST', 0),
            'corners': row.get('AC', 0),
            'xgf': row.get('away_xg'), 'xga': row.get('home_xg'),
        }
        team_home_history.setdefault(ht, []).append(home_record)
        team_away_history.setdefault(at, []).append(away_record)

        # Overall form records (need points)
        if not np.isnan(fthg) and not np.isnan(ftag):
            fthg_int, ftag_int = int(fthg), int(ftag)
            home_pts = 3 if fthg_int > ftag_int else (1 if fthg_int == ftag_int else 0)
            away_pts = 3 if ftag_int > fthg_int else (1 if fthg_int == ftag_int else 0)
            winner = ht if fthg_int > ftag_int else (at if ftag_int > fthg_int else 'draw')

            team_all_history.setdefault(ht, []).append({
                'gf': fthg_int, 'ga': ftag_int, 'points': home_pts
            })
            team_all_history.setdefault(at, []).append({
                'gf': ftag_int, 'ga': fthg_int, 'points': away_pts
            })

            # H2H
            h2h_history.setdefault(h2h_key, []).append({
                'winner': winner, 'total_goals': fthg_int + ftag_int,
                'home': ht, 'away': at
            })

        team_last_date[ht] = date
        team_last_date[at] = date

    # Matchup differences (high signal)
    for w in windows:
        df[f'atk_def_diff_r{w}'] = df[f'home_gf_r{w}'] - df[f'away_ga_r{w}']
        df[f'def_atk_diff_r{w}'] = df[f'away_gf_r{w}'] - df[f'home_ga_r{w}']
        # xG diffs
        df[f'xg_atk_def_diff_r{w}'] = df[f'home_xgf_r{w}'] - df[f'away_xga_r{w}']
        # Finishing luck
        df[f'home_luck_r{w}'] = df[f'home_gf_r{w}'] - df[f'home_xgf_r{w}']
        df[f'away_luck_r{w}'] = df[f'away_gf_r{w}'] - df[f'away_xgf_r{w}']
        # Form differential (overall)
        df[f'ppg_diff_r{w}'] = df[f'home_ppg_r{w}'] - df[f'away_ppg_r{w}']
        df[f'form_gd_diff_r{w}'] = (df[f'home_gf_all_r{w}'] - df[f'home_ga_all_r{w}']) - \
                                     (df[f'away_gf_all_r{w}'] - df[f'away_ga_all_r{w}'])

    # Congestion flag
    df['congestion_home'] = (df['rest_days_home'] <= 4).astype(float)
    df['congestion_away'] = (df['rest_days_away'] <= 4).astype(float)

    # League regime
    df['total_goals'] = df['FTHG'] + df['FTAG']
    df['league_avg_goals_r20'] = df['total_goals'].rolling(20, min_periods=10).mean().shift(1)

    # Season phase (matchweek bucket)
    df['season'] = df['Date'].apply(lambda d: f"{d.year-1}/{d.year}" if d.month < 7 else f"{d.year}/{d.year+1}")
    df['matchweek'] = df.groupby('season').cumcount() + 1
    df['season_phase'] = pd.cut(df['matchweek'], bins=[0, 12, 26, 50], labels=[0, 1, 2]).astype(float)

    # Odds implied probabilities (de-overrounded)
    for col in ['B365H', 'B365D', 'B365A']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if all(c in df.columns for c in ['B365H', 'B365D', 'B365A']):
        raw_h = 1 / df['B365H']
        raw_d = 1 / df['B365D']
        raw_a = 1 / df['B365A']
        margin = raw_h + raw_d + raw_a
        df['implied_home'] = raw_h / margin
        df['implied_draw'] = raw_d / margin
        df['implied_away'] = raw_a / margin
        df['odds_margin'] = margin

    for col in ['B365>2.5', 'B365<2.5']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'B365>2.5' in df.columns and 'B365<2.5' in df.columns:
        raw_o = 1 / df['B365>2.5']
        raw_u = 1 / df['B365<2.5']
        ou_margin = raw_o + raw_u
        df['implied_over25'] = raw_o / ou_margin
        df['implied_under25'] = raw_u / ou_margin

    return df


def get_feature_columns(df):
    """Return the list of feature columns available in the processed dataframe."""
    rolling_cols = [c for c in df.columns if any(c.startswith(p) for p in
                    ['home_gf_r', 'home_ga_r', 'home_xgf_r', 'home_xga_r',
                     'home_shots_r', 'home_sot_r', 'home_corners_r',
                     'away_gf_r', 'away_ga_r', 'away_xgf_r', 'away_xga_r',
                     'away_shots_r', 'away_sot_r', 'away_corners_r',
                     'atk_def_diff_r', 'def_atk_diff_r', 'xg_atk_def_diff_r',
                     'home_luck_r', 'away_luck_r',
                     'home_ppg_r', 'home_win_rate_r', 'home_draw_rate_r',
                     'home_gf_all_r', 'home_ga_all_r', 'home_cs_rate_r',
                     'away_ppg_r', 'away_win_rate_r', 'away_draw_rate_r',
                     'away_gf_all_r', 'away_ga_all_r', 'away_cs_rate_r',
                     'ppg_diff_r', 'form_gd_diff_r'])]

    h2h_cols = [c for c in ['h2h_home_wins', 'h2h_draws', 'h2h_total_goals']
                if c in df.columns]

    context_cols = ['rest_days_home', 'rest_days_away', 'congestion_home', 'congestion_away',
                    'league_avg_goals_r20', 'season_phase']

    odds_cols = [c for c in ['implied_home', 'implied_draw', 'implied_away',
                             'odds_margin', 'implied_over25', 'implied_under25']
                 if c in df.columns]

    return rolling_cols + h2h_cols + context_cols + odds_cols
