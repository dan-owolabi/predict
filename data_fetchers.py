"""
Data Fetchers: External API integrations for enriching predictions.
- FPL API: Player injuries, form, expected stats (no auth needed)
- Understat: Live xG data (no auth needed)
- OpenWeatherMap: Match-day weather (free API key)
"""

import os
import json
import logging
import requests
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

# ============================================================
# CACHING
# ============================================================
_cache = {}


def _get_cached(key, max_age_hours):
    """Return cached data if fresh enough."""
    if key in _cache:
        data, ts = _cache[key]
        if (datetime.now() - ts).total_seconds() < max_age_hours * 3600:
            return data
    return None


def _set_cache(key, data):
    _cache[key] = (data, datetime.now())


# ============================================================
# FPL API
# ============================================================
FPL_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

# FPL short_name -> E0 dataset name
FPL_TO_E0 = {
    'ARS': 'Arsenal',
    'AVL': 'Aston Villa',
    'BUR': 'Burnley',
    'BOU': 'Bournemouth',
    'BRE': 'Brentford',
    'BHA': 'Brighton',
    'CHE': 'Chelsea',
    'CRY': 'Crystal Palace',
    'EVE': 'Everton',
    'FUL': 'Fulham',
    'LEE': 'Leeds United',
    'LIV': 'Liverpool',
    'MCI': 'Man City',
    'MUN': 'Man United',
    'NEW': 'Newcastle',
    'NFO': "Nott'm Forest",
    'SUN': 'Sunderland',
    'TOT': 'Tottenham',
    'WHU': 'West Ham',
    'WOL': 'Wolves',
    # Championship promoted teams (may change per season)
    'LEI': 'Leicester',
    'SHU': 'Sheffield United',
    'LUT': 'Luton',
    'IPS': 'Ipswich',
    'SOT': 'Southampton',
}


def fetch_fpl_data():
    """Fetch FPL bootstrap data and compute per-team stats."""
    cached = _get_cached('fpl', max_age_hours=6)
    if cached is not None:
        return cached

    try:
        resp = requests.get(FPL_URL, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"FPL API returned {resp.status_code}")
            return None
        data = resp.json()
    except Exception as e:
        logger.warning(f"FPL API error: {e}")
        return None

    # Build team ID -> E0 name mapping
    team_id_to_e0 = {}
    for t in data.get('teams', []):
        short = t.get('short_name', '')
        e0_name = FPL_TO_E0.get(short)
        if e0_name:
            team_id_to_e0[t['id']] = e0_name

    # Aggregate player stats per team
    team_stats = {}
    for p in data.get('elements', []):
        tid = p.get('team')
        e0_name = team_id_to_e0.get(tid)
        if not e0_name:
            continue

        if e0_name not in team_stats:
            team_stats[e0_name] = {
                'players': [],
                'total_form_weighted': 0.0,
                'total_price': 0.0,
                'injury_cost': 0.0,
                'key_missing': 0,
                'xg_total': 0.0,
                'xa_total': 0.0,
                'ict_total': 0.0,
                'available_count': 0,
            }

        ts = team_stats[e0_name]
        price = (p.get('now_cost', 50) or 50) / 10.0  # price in millions
        form = float(p.get('form', 0) or 0)
        status = p.get('status', 'a')
        xg = float(p.get('expected_goals', 0) or 0)
        xa = float(p.get('expected_assists', 0) or 0)
        ict = float(p.get('ict_index', 0) or 0)

        is_available = status == 'a'

        if is_available:
            ts['total_form_weighted'] += form * price
            ts['total_price'] += price
            ts['available_count'] += 1
            ts['xg_total'] += xg
            ts['xa_total'] += xa
            ts['ict_total'] += ict
        else:
            # Injured/doubtful/suspended
            ts['injury_cost'] += price
            if price >= 8.0:
                ts['key_missing'] += 1

    # Compute final per-team metrics
    result = {}
    for team, ts in team_stats.items():
        total_p = ts['total_price']
        result[team] = {
            'team_strength': ts['total_form_weighted'] / total_p if total_p > 0 else 0.0,
            'injury_impact': ts['injury_cost'],
            'key_missing': ts['key_missing'],
            'xg_potential': ts['xg_total'],
            'xa_potential': ts['xa_total'],
            'ict_total': ts['ict_total'],
        }

    logger.info(f"FPL: loaded data for {len(result)} teams")
    _set_cache('fpl', result)
    return result


def get_fpl_features(home, away):
    """Get FPL-derived features for a specific match."""
    fpl = fetch_fpl_data()
    if not fpl:
        return {}

    h = fpl.get(home, {})
    a = fpl.get(away, {})

    return {
        'home_team_strength': h.get('team_strength', np.nan),
        'away_team_strength': a.get('team_strength', np.nan),
        'home_injury_impact': h.get('injury_impact', np.nan),
        'away_injury_impact': a.get('injury_impact', np.nan),
        'strength_diff': (h.get('team_strength', 0) - a.get('team_strength', 0))
                         if h and a else np.nan,
        'home_key_missing': h.get('key_missing', np.nan),
        'away_key_missing': a.get('key_missing', np.nan),
        'home_xg_potential': h.get('xg_potential', np.nan),
        'away_xg_potential': a.get('xg_potential', np.nan),
    }


# ============================================================
# UNDERSTAT LIVE xG (via understatapi library)
# ============================================================

# Understat team name -> E0 dataset name
UNDERSTAT_TO_E0 = {
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
    'Leeds': 'Leeds United',
    'West Bromwich Albion': 'West Brom',
    'Norwich': 'Norwich',
    'Watford': 'Watford',
    'Luton': 'Luton',
}


def fetch_understat_season(season_year=None):
    """Fetch xG data for a season from Understat using understatapi.

    season_year: the starting year, e.g. 2025 for 2025/26 season.
    """
    if season_year is None:
        now = datetime.now()
        season_year = now.year if now.month >= 7 else now.year - 1

    cache_key = f'understat_{season_year}'
    cached = _get_cached(cache_key, max_age_hours=24)
    if cached is not None:
        return cached

    try:
        from understatapi import UnderstatClient

        with UnderstatClient() as client:
            matches_data = client.league(league='EPL').get_match_data(season=str(season_year))

        records = []
        for m in matches_data:
            if not m.get('isResult', False):
                continue
            try:
                h_name = m['h']['title']
                a_name = m['a']['title']
                records.append({
                    'home_team_xg': UNDERSTAT_TO_E0.get(h_name, h_name),
                    'away_team_xg': UNDERSTAT_TO_E0.get(a_name, a_name),
                    'home_xg': float(m['xG']['h']),
                    'away_xg': float(m['xG']['a']),
                    'home_goals': int(m['goals']['h']),
                    'away_goals': int(m['goals']['a']),
                    'date_xg': pd.to_datetime(m['datetime']),
                    'season': f"{season_year}/{season_year + 1}",
                })
            except (KeyError, ValueError):
                continue

        df = pd.DataFrame(records)
        logger.info(f"Understat: fetched {len(df)} matches for {season_year}/{season_year + 1}")
        _set_cache(cache_key, df)
        return df

    except ImportError:
        logger.warning("understatapi not installed — run: pip install understatapi")
        return None
    except Exception as e:
        logger.warning(f"Understat error: {e}")
        return None


def fetch_understat_multi_season(start_year=2018):
    """Fetch xG data for multiple seasons and return combined DataFrame."""
    now = datetime.now()
    current_year = now.year if now.month >= 7 else now.year - 1

    all_dfs = []
    for year in range(start_year, current_year + 1):
        df = fetch_understat_season(year)
        if df is not None and len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['home_team_xg', 'away_team_xg', 'date_xg'])
    logger.info(f"Understat: {len(combined)} total matches across {len(all_dfs)} seasons")
    return combined


# ============================================================
# OPENWEATHERMAP
# ============================================================
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

# EPL stadium coordinates (lat, lon)
STADIUM_COORDS = {
    'Arsenal': (51.5549, -0.1084),           # Emirates
    'Aston Villa': (52.5092, -1.8847),       # Villa Park
    'Bournemouth': (50.7352, -1.8383),       # Vitality
    'Brentford': (51.4907, -0.2887),         # Gtech Community
    'Brighton': (50.8616, -0.0837),          # Amex
    'Burnley': (53.7890, -2.2302),           # Turf Moor
    'Chelsea': (51.4817, -0.1910),           # Stamford Bridge
    'Crystal Palace': (51.3983, -0.0855),    # Selhurst Park
    'Everton': (53.4388, -2.9664),           # Goodison Park
    'Fulham': (51.4750, -0.2217),            # Craven Cottage
    'Leeds United': (53.7779, -1.5720),      # Elland Road
    'Leicester': (52.6204, -1.1422),         # King Power
    'Liverpool': (53.4308, -2.9609),         # Anfield
    'Man City': (53.4831, -2.2004),          # Etihad
    'Man United': (53.4631, -2.2913),        # Old Trafford
    'Newcastle': (54.9756, -1.6217),         # St James' Park
    "Nott'm Forest": (52.9400, -1.1326),     # City Ground
    'Sheffield United': (53.3703, -1.4710),  # Bramall Lane
    'Southampton': (50.9058, -1.3911),       # St Mary's
    'Sunderland': (54.9146, -1.3882),        # Stadium of Light
    'Tottenham': (51.6043, -0.0661),         # Tottenham Hotspur Stadium
    'West Ham': (51.5387, -0.0166),          # London Stadium
    'Wolves': (52.5901, -2.1306),            # Molineux
    'Ipswich': (52.0545, 1.1449),            # Portman Road
    'Luton': (51.8843, -0.4316),             # Kenilworth Road
}


def fetch_weather(home_team):
    """Fetch current weather for the home team's stadium."""
    if not OPENWEATHER_API_KEY:
        return None

    cache_key = f'weather_{home_team}'
    cached = _get_cached(cache_key, max_age_hours=3)
    if cached is not None:
        return cached

    coords = STADIUM_COORDS.get(home_team)
    if not coords:
        logger.warning(f"No stadium coords for {home_team}")
        return None

    lat, lon = coords
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                'lat': lat, 'lon': lon,
                'appid': OPENWEATHER_API_KEY,
                'units': 'metric',
            },
            timeout=10
        )
        if resp.status_code != 200:
            logger.warning(f"Weather API returned {resp.status_code}")
            return None

        w = resp.json()
        main = w.get('main', {})
        wind = w.get('wind', {})
        rain = w.get('rain', {})

        result = {
            'temperature': main.get('temp', np.nan),
            'humidity': main.get('humidity', np.nan),
            'wind_speed': wind.get('speed', np.nan),
            'rain_mm': rain.get('1h', 0.0),
            'is_rainy': 1.0 if rain.get('1h', 0) > 0 else 0.0,
            'is_windy': 1.0 if wind.get('speed', 0) > 8 else 0.0,
        }

        logger.info(f"Weather for {home_team}: {result['temperature']:.0f}°C, "
                     f"wind={result['wind_speed']:.1f}m/s, rain={result['rain_mm']:.1f}mm")
        _set_cache(cache_key, result)
        return result

    except Exception as e:
        logger.warning(f"Weather API error: {e}")
        return None


def get_weather_features(home_team):
    """Get weather features dict for a match."""
    w = fetch_weather(home_team)
    if not w:
        return {
            'temperature': np.nan,
            'humidity': np.nan,
            'wind_speed': np.nan,
            'rain_mm': np.nan,
            'is_rainy': np.nan,
            'is_windy': np.nan,
        }
    return w


# ============================================================
# COMBINED FEATURES
# ============================================================
def get_all_external_features(home, away):
    """Get all external API features for a match.

    Returns a dict of feature_name -> value.
    """
    features = {}

    # FPL
    fpl_feats = get_fpl_features(home, away)
    features.update(fpl_feats)

    # Weather
    weather_feats = get_weather_features(home)
    features.update(weather_feats)

    return features


# ============================================================
# STANDALONE TEST
# ============================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("=" * 50)
    print("Testing FPL API...")
    print("=" * 50)
    fpl = fetch_fpl_data()
    if fpl:
        print(f"\nLoaded {len(fpl)} teams:")
        for team, stats in sorted(fpl.items()):
            print(f"  {team:20s} | strength={stats['team_strength']:.2f} | "
                  f"injury={stats['injury_impact']:.1f} | "
                  f"key_missing={stats['key_missing']} | "
                  f"xG={stats['xg_potential']:.1f} | "
                  f"xA={stats['xa_potential']:.1f} | "
                  f"ICT={stats['ict_total']:.0f}")
    else:
        print("FPL API failed")

    print("\n" + "=" * 50)
    print("Testing Understat xG...")
    print("=" * 50)
    xg = fetch_understat_season()
    if xg is not None and len(xg) > 0:
        print(f"\nFetched {len(xg)} matches")
        print(xg[['home_team_xg', 'away_team_xg', 'home_xg', 'away_xg']].tail(5).to_string())
    else:
        print("Understat fetch failed")

    print("\n" + "=" * 50)
    print("Testing Weather API...")
    print("=" * 50)
    if OPENWEATHER_API_KEY:
        w = fetch_weather('Arsenal')
        if w:
            print(f"\nArsenal (Emirates): {w}")
        else:
            print("Weather fetch failed")
    else:
        print("OPENWEATHER_API_KEY not set — skipping")

    print("\n" + "=" * 50)
    print("Testing combined features...")
    print("=" * 50)
    feats = get_all_external_features('Arsenal', 'Chelsea')
    for k, v in sorted(feats.items()):
        print(f"  {k:25s} = {v}")
