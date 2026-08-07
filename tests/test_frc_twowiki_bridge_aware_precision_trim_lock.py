from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = (
    ROOT
    / "docs/progressive_upgrade/"
    "twowiki_bridge_aware_precision_trim_implementation_v83.json"
)


def test_v83_target_precedes_no_locked_file_drift() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["invariants"]["target_ids_selected_before_lock"] == 0
    assert lock["invariants"]["target_rows_read_before_lock"] == 0
    assert lock["invariants"]["target_scores_or_gold_metrics_seen_before_lock"] is False
    assert len(lock["files"]) == 9
    for contract in lock["files"].values():
        path = ROOT / contract["path"]
        payload = path.read_bytes()
        assert len(payload) == contract["bytes"]
        assert hashlib.sha256(payload).hexdigest() == contract["sha256"]
