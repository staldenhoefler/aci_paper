"""Run all four experiments in sequence. Stops on the first failure."""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

EXPERIMENTS = [
    "experiments/e1_ga_sensitivity.py",
    "experiments/e2_neuroevo_stability.py",
    "experiments/e3_scaling.py",
    "experiments/e4_head_to_head.py",
]

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ["ACI_RUN_ID"] = RUN_ID
print(f"Run ID: {RUN_ID}\n")

for script in EXPERIMENTS:
    print(f"\n{'='*60}")
    print(f"  Running {script}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, script], cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"\nFailed: {script}  (exit code {result.returncode})")
        sys.exit(result.returncode)

print("\nAll experiments completed successfully.")
print(f"Results and figures are in results/{RUN_ID}/ and figures/{RUN_ID}/")
