from __future__ import annotations

import argparse
from pathlib import Path

from flood_system.scenario_acceptance import write_scenario_acceptance_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the 24-scenario automated acceptance traceability report.")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("output/floodagent_bench/scenario_acceptance_report.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("output/floodagent_bench/scenario_acceptance_report.md"),
    )
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    json_target, markdown_target = write_scenario_acceptance_report(
        repo_root,
        (repo_root / args.json_output).resolve(),
        (repo_root / args.markdown_output).resolve(),
        seed=args.seed,
    )
    print(json_target)
    print(markdown_target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
