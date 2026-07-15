from __future__ import annotations

import argparse
from pathlib import Path

from flood_system.candidate_evaluation import run_candidate_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate warning-to-object candidate ranking and calibration.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/floodagent_bench/candidate_evaluation_input.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("output/candidate_evaluation/report.json"))
    args = parser.parse_args()
    report = run_candidate_evaluation(args.input, args.output)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
