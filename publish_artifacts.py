import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).parent
STATUS_PATH = ROOT / "automation_status.json"

REQUIRED = [
    "production_artifacts.pkl",
    "lgb_1x2.txt",
    "lgb_ou25.txt",
    "lgb_ou15.txt",
    "lgb_btts.txt",
    "market_models/manifest.json",
    "player_prop_models/manifest.json",
    "cv_results_enriched.csv",
    "weekly_fixtures.json",
]

OPTIONAL = [
    "europe_models/manifest.json",
    "europe_cv_results.csv",
    "europe_market_models/manifest.json",
]


def stat_for(path_str):
    path = ROOT / path_str
    if not path.exists():
        return None
    return {
        "path": path_str,
        "size": path.stat().st_size,
        "modified": datetime.utcfromtimestamp(path.stat().st_mtime).isoformat() + "Z",
    }


def main():
    missing = [path for path in REQUIRED if stat_for(path) is None]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")
    payload = {
        "published_at": datetime.utcnow().isoformat() + "Z",
        "required": [stat_for(path) for path in REQUIRED],
        "optional": [stat_for(path) for path in OPTIONAL if stat_for(path) is not None],
    }
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
