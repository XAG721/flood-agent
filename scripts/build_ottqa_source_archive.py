from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path


PINNED_REVISION = "b289bcca691db21a5259c3420e6b9819a9be9ba3"
EXPECTED_TABLE_FILES = 8891
EXPECTED_PASSAGE_FILES = 8891


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def selected(relative: str) -> bool:
    return (
        relative == "LICENSE"
        or relative == "preprocessed_data/dev_linked.json"
        or relative.startswith("data/traindev_tables_tok/")
        or relative.startswith("data/traindev_request_tok/")
    )


def group(relative: str) -> str:
    if relative.startswith("data/traindev_tables_tok/"):
        return "tables"
    if relative.startswith("data/traindev_request_tok/"):
        return "passages"
    return relative


def build(source_path: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite source archive: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    byte_counts: Counter[str] = Counter()
    member_hashes: list[tuple[str, str, int]] = []
    with tarfile.open(source_path, mode="r:gz") as source:
        with tarfile.open(output_path, mode="w", format=tarfile.PAX_FORMAT) as target:
            for member in source:
                if not member.isfile() or "/" not in member.name:
                    continue
                relative = member.name.split("/", 1)[1]
                if not selected(relative):
                    continue
                handle = source.extractfile(member)
                if handle is None:
                    raise ValueError(f"codeload member is not readable: {member.name}")
                payload = handle.read()
                if len(payload) != member.size:
                    raise ValueError(f"codeload member is truncated: {member.name}")
                info = tarfile.TarInfo(name=relative)
                info.size = len(payload)
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                target.addfile(info, io.BytesIO(payload))
                category = group(relative)
                counts[category] += 1
                byte_counts[category] += len(payload)
                member_hashes.append(
                    (relative, hashlib.sha256(payload).hexdigest(), len(payload))
                )
    expected = {
        "LICENSE": 1,
        "preprocessed_data/dev_linked.json": 1,
        "tables": EXPECTED_TABLE_FILES,
        "passages": EXPECTED_PASSAGE_FILES,
    }
    if dict(counts) != expected:
        output_path.unlink(missing_ok=True)
        raise ValueError(
            f"registered member coverage mismatch: actual={dict(counts)!r}, "
            f"expected={expected!r}"
        )
    manifest_payload = "\n".join(
        f"{name} {digest} {size}" for name, digest, size in sorted(member_hashes)
    ).encode("utf-8")
    return {
        "schema_version": "frc-ottqa-v44-source-archive-build-v1",
        "pinned_revision": PINNED_REVISION,
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256(source_path),
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "output_bytes": output_path.stat().st_size,
        "member_counts": dict(sorted(counts.items())),
        "member_bytes": dict(sorted(byte_counts.items())),
        "selected_member_content_manifest_sha256": hashlib.sha256(
            manifest_payload
        ).hexdigest(),
        "json_content_parsed": False,
        "members_extracted_to_filesystem": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream registered OTT-QA members into a Windows-safe tar."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = build(args.source, args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
