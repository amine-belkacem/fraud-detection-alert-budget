"""Rebuild every result and figure of the project, in order, with one command.

Usage:
    python scripts/run_all.py

Each step runs as its own process (same interpreter), so a failure stops the
chain with the step's own error message. The wall-clock time of each step is
printed and written to results/run_all_timing.csv.
"""

from __future__ import annotations

import csv
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = (
    ("download_data.py", "download and verify the dataset"),
    ("run_baseline.py", "J1 baselines and the accuracy trap"),
    ("tune.py", "J2 hyperparameter search on validation"),
    ("run_models.py", "J2 models, seeds and bootstrap"),
    ("missed_frauds.py", "J2 frauds missed by every model"),
    ("run_thresholds.py", "J3 thresholds, cost and calibration"),
    ("run_stress.py", "J4 prevalence stress test"),
    ("run_drift.py", "J4 split comparison and drift"),
)


def main() -> int:
    timings = []
    start_all = time.perf_counter()
    for script, what in STEPS:
        print(f"\n=== {script}: {what}", flush=True)
        start = time.perf_counter()
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT)
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            print(f"ECHEC : {script} returned {result.returncode} after {elapsed:.0f} s", file=sys.stderr)
            return result.returncode
        timings.append({"step": script, "seconds": round(elapsed, 1)})
        print(f"=== {script} done in {elapsed:.0f} s", flush=True)

    total = time.perf_counter() - start_all
    timings.append({"step": "total", "seconds": round(total, 1)})
    with open(ROOT / "results" / "run_all_timing.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "seconds"])
        writer.writeheader()
        writer.writerows(timings)
    print(f"\nall steps done in {total / 60:.1f} min "
          f"(Python {platform.python_version()}, {platform.processor() or platform.machine()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
