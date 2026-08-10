from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main() -> int:
    scripts = [
        ("Phase tests", [sys.executable, str(ROOT / "scripts" / "verify_phases.py")]),
        ("Accuracy benchmark", [sys.executable, str(ROOT / "scripts" / "benchmark_accuracy.py")]),
        ("App import", [sys.executable, "-c", "from app.main import main; from app.ui.dashboard_tab import DashboardTab; print('OK')"]),
    ]
    failed = 0
    for name, cmd in scripts:
        print(f"\n=== {name} ===")
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
        if r.returncode != 0:
            print(f"FAILED: {name}")
            failed += 1
        else:
            print(f"PASSED: {name}")
    print(f"\n{len(scripts) - failed}/{len(scripts)} checks passed")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
