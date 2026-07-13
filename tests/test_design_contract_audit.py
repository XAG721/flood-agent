from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from flood_system.design_contract_audit import (
    DesignContractAuditError,
    build_design_contract_audit,
    extract_design_contract_items,
    validate_legacy_baseline_manifest,
    write_design_contract_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "洪水预警响应系统_渐进式迭代开发与升级设计.md"
POLICY_PATH = (
    REPO_ROOT / "docs/progressive_upgrade/design_contract_evidence_policy.json"
)
BASELINE_PATH = REPO_ROOT / "output/acceptance/legacy_baseline_manifest.json"


def test_design_contract_extracts_all_89_explicit_requirements() -> None:
    items = extract_design_contract_items(SOURCE_PATH)

    assert len(items) == 89
    assert {item["item_id"] for item in items} == {
        *(f"18.3-{index:02d}" for index in range(1, 11)),
        *(f"18.4-{index:02d}" for index in range(1, 11)),
        *(f"18.5-{index:02d}" for index in range(1, 11)),
        *(f"18.6-{index:02d}" for index in range(1, 9)),
        *(f"22.1-{index:02d}" for index in range(1, 6)),
        *(f"22.2-{index:02d}" for index in range(1, 8)),
        *(f"22.3-{index:02d}" for index in range(1, 11)),
        *(f"22.4-{index:02d}" for index in range(1, 18)),
        *(f"23-{index:02d}" for index in range(1, 13)),
    }
    assert all(item["source_line"] > 0 for item in items)


def test_current_design_contract_is_accounted_without_overstating_production() -> None:
    report = build_design_contract_audit(REPO_ROOT)

    assert report["summary"] == {
        "item_count": 89,
        "status_counts": {
            "PASS_LOCAL": 7,
            "PASS_CONTROLLED": 80,
            "NO_GO_GATE": 0,
            "NO_GO_EXTERNAL": 2,
        },
        "controlled_or_local_pass_count": 87,
        "external_no_go_count": 2,
        "external_no_go_ids": ["18.4-10", "22.4-10"],
        "all_items_accounted": True,
        "controlled_scope_complete": True,
        "full_production_complete": False,
        "overall": "CONTROLLED_CONTRACT_ACCOUNTED_EXTERNAL_NO_GO",
    }
    assert report["legacy_baseline"]["offline_reproducible"] is True


def test_design_contract_rejects_a_missing_policy_assignment(tmp_path: Path) -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["assignments"]["migration_controlled"].remove("18.3-01")
    tampered_policy = tmp_path / "policy.json"
    tampered_policy.write_text(json.dumps(policy, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(DesignContractAuditError, match="not one-to-one"):
        build_design_contract_audit(REPO_ROOT, policy_path=tampered_policy)


def test_design_contract_rejects_evidence_paths_outside_repository(
    tmp_path: Path,
) -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["evidence_groups"]["legacy_baseline"]["evidence"][0]["path"] = (
        "../outside-repository.json"
    )
    tampered_policy = tmp_path / "path-escape-policy.json"
    tampered_policy.write_text(json.dumps(policy, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(DesignContractAuditError, match="escapes repository"):
        build_design_contract_audit(REPO_ROOT, policy_path=tampered_policy)


@pytest.mark.parametrize("field", ["file_count", "listing_sha256"])
def test_legacy_baseline_rejects_category_reconciliation_tampering(field: str) -> None:
    manifest = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(manifest)
    category = tampered["tracked_tree"]["categories"]["database_rebuild_inputs"]
    if field == "file_count":
        category[field] += 1
    else:
        category[field] = "0" * 64

    with pytest.raises(DesignContractAuditError, match="does not reconcile"):
        validate_legacy_baseline_manifest(tampered)


def test_design_contract_output_is_deterministic(tmp_path: Path) -> None:
    first_json = tmp_path / "first.json"
    first_markdown = tmp_path / "first.md"
    second_json = tmp_path / "second.json"
    second_markdown = tmp_path / "second.md"

    write_design_contract_audit(REPO_ROOT, first_json, first_markdown)
    write_design_contract_audit(REPO_ROOT, second_json, second_markdown)

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_markdown.read_bytes() == second_markdown.read_bytes()
