"""Run every tests/test_*.py script; exit non-zero if any fails.

Run:  .venv/Scripts/python.exe tests/run_all.py
"""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    failed = []
    for path in sorted(here.glob("test_*.py")):
        print(f"=== {path.name}", flush=True)
        result = subprocess.run([sys.executable, str(path)], cwd=here.parent)
        if result.returncode != 0:
            failed.append(path.name)
    print("\nFAILED: " + ", ".join(failed) if failed else "\nALL TEST FILES PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
