from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = (
    ROOT / "docs/progressive_upgrade/hotpot_question_type_cardinality_implementation_v84.json"
)


def test_v84_target_precedes_no_locked_file_drift() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["frozen_before_v84_target_id_selection"] is True
    assert len(lock["files"]) == 9
    for contract in lock["files"].values():
        path = ROOT / contract["path"]
        assert path.stat().st_size == contract["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == contract["sha256"]
