import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent


def run_step(command, allow_fail=False):
    print(f"\n==> {' '.join(command)}")
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0 and not allow_fail:
        raise SystemExit(result.returncode)


def main():
    steps = [
        ([sys.executable, "scrape_europe_data.py"], True),
        ([sys.executable, "build_understat_multi_tables.py"], False),
        ([sys.executable, "fetch_europe_support_leagues.py"], False),
        ([sys.executable, "build_europe_training_data.py"], False),
        ([sys.executable, "train_europe_models.py"], False),
        ([sys.executable, "scrape_uefa_ts_stats.py"], True),
        ([sys.executable, "build_europe_market_training_data.py"], True),
        ([sys.executable, "train_europe_market_models.py"], True),
    ]
    for command, allow_fail in steps:
        run_step(command, allow_fail=allow_fail)


if __name__ == "__main__":
    main()
