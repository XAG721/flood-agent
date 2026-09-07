from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.qasc_confirmation import (
    EXPECTED_SOURCE_SHA256,
    SOURCE_FILENAME,
    SOURCE_REVISION,
    SOURCE_URL,
    sha256,
)


def _validate_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "frc-qasc-confirmation-protocol-v1":
        raise ValueError("unsupported QASC confirmation protocol")
    source = protocol["frozen_source"]
    if source["revision"] != SOURCE_REVISION:
        raise ValueError("QASC protocol revision does not match code")
    if source["path_label"] != SOURCE_FILENAME:
        raise ValueError("QASC protocol path label does not match code")
    if source["sha256"] != EXPECTED_SOURCE_SHA256:
        raise ValueError("QASC protocol source hash does not match code")
    if source["url"] != SOURCE_URL:
        raise ValueError("QASC protocol URL does not match code")
    for key in ("adapter", "download_runner", "score_runner"):
        registered_path = (
            REPOSITORY_ROOT / protocol["frozen_implementation"][f"{key}_path"]
        )
        if (
            sha256(registered_path)
            != protocol["frozen_implementation"][f"{key}_sha256"]
        ):
            raise ValueError(f"QASC frozen implementation hash mismatch: {key}")
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the pinned QASC validation parquet after protocol freeze."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/conformal_qasc_confirmation_protocol.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/benchmarks/qasc/validation.parquet"),
    )
    args = parser.parse_args()
    if not args.protocol.is_file():
        parser.error(f"frozen protocol does not exist: {args.protocol}")
    _validate_protocol(args.protocol)
    if args.output.is_file():
        if sha256(args.output) != EXPECTED_SOURCE_SHA256:
            parser.error("existing QASC output has the wrong hash")
        print(args.output)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "flood-agent-qasc-confirmation/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with partial.open("wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
    actual = sha256(partial)
    if actual != EXPECTED_SOURCE_SHA256:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"downloaded QASC hash mismatch: {actual} != {EXPECTED_SOURCE_SHA256}"
        )
    partial.replace(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
