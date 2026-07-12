from __future__ import annotations

import argparse
from pathlib import Path

from flood_system.simulation_dataset import DEFAULT_SIMULATION_SEED, write_floodagent_bench


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the deterministic FloodAgent-Bench simulation dataset.")
    parser.add_argument("--output", type=Path, default=Path("output/floodagent_bench/floodagent_bench.json"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SIMULATION_SEED)
    args = parser.parse_args()
    target = write_floodagent_bench(args.output, seed=args.seed)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
