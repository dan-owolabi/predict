import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent


def run_step(command):
    print(f"\n==> {' '.join(command)}")
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    steps = [
        [sys.executable, "scrape_epl_data.py", "--no-fpl-history"],
        [sys.executable, "fpl_fetch_missing.py"],
        [sys.executable, "understat_deep_fetch_missing.py"],
        [sys.executable, "understat_build_deep_tables.py"],
        [sys.executable, "fetch_clubelo.py"],
        [sys.executable, "fetch_europe_support_leagues.py"],
        [sys.executable, "build_understat_multi_tables.py"],
        [sys.executable, "build_europe_training_data.py"],
        [sys.executable, "update_fixtures.py"],
    ]
    for step in steps:
        run_step(step)


if __name__ == "__main__":
    main()
