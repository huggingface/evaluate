import statistics
import subprocess
import sys
import time

STATEMENTS = [
    "import evaluate",
    "from evaluate import load",
    "from evaluate import EvaluationSuite",
    "from evaluate import evaluator",
]

for statement in STATEMENTS:
    measurements = []

    for _ in range(7):
        start = time.perf_counter()
        subprocess.run(
            [sys.executable, "-c", statement],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        measurements.append(time.perf_counter() - start)

    print(
        f"{statement:40} "
        f"median={statistics.median(measurements):.3f}s "
        f"min={min(measurements):.3f}s "
        f"max={max(measurements):.3f}s"
    )
