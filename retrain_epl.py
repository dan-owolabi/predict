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
    for step, allow_missing in [
        ([sys.executable, "train_pipeline.py"], False),
        ([sys.executable, "train_final_models.py"], False),
        ([sys.executable, "train_market_expansion.py"], True),
        ([sys.executable, "train_player_prop_models.py"], True),
        ([sys.executable, "evaluate.py"], False),
    ]:
        run_step(step, allow_missing=allow_missing)


if __name__ == "__main__":
    main()
