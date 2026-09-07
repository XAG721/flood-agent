from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = (
    ROOT
    / "docs/progressive_upgrade/"
    "twowiki_cascaded_style_residual_implementation_v82.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_v82_locked_implementation_files_are_unchanged() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["experiment_id"] == (
        "FRC-2WIKI-CASCADED-STYLE-RESIDUAL-PROSPECTIVE-V82"
    )
    assert lock["invariants"]["target_ids_selected_before_lock"] == 0
    assert lock["invariants"]["target_rows_read_before_lock"] == 0
    assert lock["invariants"]["target_scores_or_gold_metrics_seen_before_lock"] is False
    assert lock["invariants"]["gate_2"] == "NO-GO/SHADOW"
    assert len(lock["files"]) == 9
    for contract in lock["files"].values():
        path = ROOT / contract["path"]
        assert path.stat().st_size == contract["bytes"]
        assert _sha256(path) == contract["sha256"]
