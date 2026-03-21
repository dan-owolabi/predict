import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent


def run_step(command, allow_missing=False):
    script_path = ROOT / command[1]
    if allow_missing and not script_path.exists():
        print(f"\n==> skipping missing script: {script_path.name}")
        return
    print(f"\n==> {' '.join(command)}")
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    steps = [
        ([sys.executable, "scrape_epl_data.py", "--no-fpl-history"], True),
        ([sys.executable, "fpl_fetch_missing.py"], True),
        ([sys.executable, "understat_deep_fetch_missing.py"], True),
        ([sys.executable, "understat_build_deep_tables.py"], True),
        ([sys.executable, "fetch_clubelo.py"], True),
        ([sys.executable, "fetch_europe_support_leagues.py"], False),
        ([sys.executable, "build_understat_multi_tables.py"], False),
        ([sys.executable, "build_europe_training_data.py"], False),
        ([sys.executable, "update_fixtures.py"], False),
    ]
    for step, allow_missing in steps:
        run_step(step, allow_missing=allow_missing)


if __name__ == "__main__":
    main()
