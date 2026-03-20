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
    for step in [
        [sys.executable, "train_pipeline.py"],
        [sys.executable, "train_final_models.py"],
        [sys.executable, "train_market_expansion.py"],
        [sys.executable, "train_player_prop_models.py"],
        [sys.executable, "evaluate.py"],
    ]:
        run_step(step)


if __name__ == "__main__":
    main()
