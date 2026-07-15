from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "progressive-design-contract-audit-v1"
POLICY_SCHEMA_VERSION = "progressive-design-evidence-policy-v1"
BASELINE_SCHEMA_VERSION = "legacy-baseline-snapshot-v1"
EXPECTED_ITEM_COUNT = 89
EXPECTED_LEGACY_SOURCE_COMMIT = "2db016af915ada1912afe91d315defe844dba09d"
EXPECTED_EXTERNAL_NO_GO_IDS = ("18.4-10", "22.4-10")
PASS_STATUSES = frozenset({"PASS_LOCAL", "PASS_CONTROLLED"})
VALID_STATUSES = frozenset(
    {"PASS_LOCAL", "PASS_CONTROLLED", "NO_GO_GATE", "NO_GO_EXTERNAL"}
)

CHECKLIST_PATTERN = re.compile(r"^- \[ \] (?P<requirement>.+?)\s*$")
SECTION_18_PATTERN = re.compile(r"^### (?P<section>18\.[3-6])\s")

SECTION_22_HEADINGS = {
    "### 名称与边界": ("22.1", "名称与边界"),
    "### 数据与算法": ("22.2", "数据与算法"),
    "### 业务与安全": ("22.3", "业务与安全"),
    "### 工程与验收": ("22.4", "工程与验收"),
}

EXPECTED_SECTION_COUNTS = {
    "18.3": 10,
    "18.4": 10,
    "18.5": 10,
    "18.6": 8,
    "22.1": 5,
    "22.2": 7,
    "22.3": 10,
    "22.4": 17,
    "23": 12,
}


class DesignContractAuditError(ValueError):
    """Raised when the design contract or its evidence is incomplete or inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DesignContractAuditError(
            f"cannot read JSON evidence {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DesignContractAuditError(f"JSON evidence must be an object: {path}")
    return payload


def _sha256_text(text: str) -> str:
    canonical = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evidence_sha256(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return _sha256_text(text)


def _strip_requirement_punctuation(value: str) -> str:
    return value.rstrip().rstrip("；。").strip()


def extract_design_contract_items(source_path: Path) -> list[dict[str, Any]]:
    """Extract the 77 checklist clauses and 12 final criteria from sections 18-23."""

    lines = source_path.read_text(encoding="utf-8-sig").splitlines()
    items: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    active_section: str | None = None
    active_heading: str | None = None
    in_section_22 = False
    in_section_23_requirements = False

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        section_18 = SECTION_18_PATTERN.match(line)
        if section_18:
            active_section = section_18.group("section")
            active_heading = line.removeprefix("### ")
            in_section_22 = False
            in_section_23_requirements = False
            continue

        if line.startswith("## 22. "):
            active_section = None
            active_heading = None
            in_section_22 = True
            in_section_23_requirements = False
            continue
        if in_section_22 and line in SECTION_22_HEADINGS:
            active_section, active_heading = SECTION_22_HEADINGS[line]
            continue

        if line.startswith("## 23. "):
            active_section = None
            active_heading = None
            in_section_22 = False
            in_section_23_requirements = False
            continue
        if line == "同时必须满足：":
            active_section = "23"
            active_heading = "系统最终完成标准"
            in_section_23_requirements = True
            continue
        if in_section_23_requirements and line.startswith("完成以上内容后"):
            active_section = None
            in_section_23_requirements = False
            continue

        if active_section == "23" and in_section_23_requirements:
            if not line.startswith("- "):
                continue
            requirement = _strip_requirement_punctuation(line.removeprefix("- "))
        elif active_section in EXPECTED_SECTION_COUNTS and active_section != "23":
            match = CHECKLIST_PATTERN.match(line)
            if match is None:
                continue
            requirement = _strip_requirement_punctuation(match.group("requirement"))
        else:
            continue

        counters[active_section] += 1
        items.append(
            {
                "item_id": f"{active_section}-{counters[active_section]:02d}",
                "source_section": active_section,
                "source_heading": active_heading,
                "source_line": line_number,
                "requirement": requirement,
            }
        )

    actual_counts = Counter(item["source_section"] for item in items)
    if dict(actual_counts) != EXPECTED_SECTION_COUNTS:
        raise DesignContractAuditError(
            "design checklist section counts changed: "
            f"expected={EXPECTED_SECTION_COUNTS}, actual={dict(actual_counts)}"
        )
    if len(items) != EXPECTED_ITEM_COUNT:
        raise DesignContractAuditError(
            f"design checklist must contain {EXPECTED_ITEM_COUNT} items, found {len(items)}"
        )
    item_ids = [item["item_id"] for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise DesignContractAuditError(
            "design checklist contains duplicate item identifiers"
        )
    return items


def _safe_evidence_path(repo_root: Path, relative: str) -> Path:
    candidate_path = Path(relative)
    if not relative or candidate_path.is_absolute():
        raise DesignContractAuditError(
            f"evidence path must be repository-relative: {relative!r}"
        )
    candidate = (repo_root / candidate_path).resolve()
    if candidate != repo_root and repo_root not in candidate.parents:
        raise DesignContractAuditError(f"evidence path escapes repository: {relative}")
    if not candidate.is_file():
        raise DesignContractAuditError(f"evidence file is missing: {relative}")
    return candidate


def _validate_evidence_groups(
    repo_root: Path,
    policy: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    definitions = policy.get("status_definitions")
    if not isinstance(definitions, dict) or not VALID_STATUSES.issubset(definitions):
        raise DesignContractAuditError(
            "policy must define every supported evidence status"
        )
    groups = policy.get("evidence_groups")
    if not isinstance(groups, dict) or not groups:
        raise DesignContractAuditError("policy must contain evidence groups")

    validated: dict[str, dict[str, Any]] = {}
    for group_name, group in groups.items():
        if not isinstance(group, dict):
            raise DesignContractAuditError(
                f"evidence group must be an object: {group_name}"
            )
        status = group.get("status")
        if status not in VALID_STATUSES:
            raise DesignContractAuditError(
                f"evidence group {group_name} has unsupported status {status!r}"
            )
        evidence = group.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise DesignContractAuditError(f"evidence group is empty: {group_name}")
        checked_evidence: list[dict[str, Any]] = []
        for entry in evidence:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise DesignContractAuditError(
                    f"evidence group {group_name} contains an invalid path entry"
                )
            relative = entry["path"]
            path = _safe_evidence_path(repo_root, relative)
            contains = entry.get("contains")
            if contains is not None:
                if not isinstance(contains, str) or not contains:
                    raise DesignContractAuditError(
                        f"evidence marker must be a non-empty string: {relative}"
                    )
                try:
                    evidence_text = path.read_text(encoding="utf-8-sig")
                except UnicodeDecodeError as exc:
                    raise DesignContractAuditError(
                        f"text marker cannot be checked in binary evidence: {relative}"
                    ) from exc
                if contains not in evidence_text:
                    raise DesignContractAuditError(
                        f"evidence marker {contains!r} is missing from {relative}"
                    )
            checked_evidence.append(
                {
                    "path": relative.replace("\\", "/"),
                    "contains": contains,
                    "sha256": _evidence_sha256(path),
                }
            )
        validated[group_name] = {
            "status": status,
            "scope": str(group.get("scope", "")),
            "limitation": str(group.get("limitation", "")),
            "evidence": checked_evidence,
        }
    return validated


def _validate_assignments(
    items: list[dict[str, Any]],
    policy: dict[str, Any],
    groups: dict[str, dict[str, Any]],
) -> dict[str, str]:
    assignments = policy.get("assignments")
    if not isinstance(assignments, dict):
        raise DesignContractAuditError("policy assignments must be an object")
    if set(assignments) != set(groups):
        missing_groups = sorted(set(groups) - set(assignments))
        unknown_groups = sorted(set(assignments) - set(groups))
        raise DesignContractAuditError(
            f"assignment groups differ from evidence groups: "
            f"missing={missing_groups}, unknown={unknown_groups}"
        )

    expected_ids = {item["item_id"] for item in items}
    item_to_group: dict[str, str] = {}
    duplicates: list[str] = []
    for group_name, item_ids in assignments.items():
        if not isinstance(item_ids, list) or not item_ids:
            raise DesignContractAuditError(f"assignment group is empty: {group_name}")
        for item_id in item_ids:
            if not isinstance(item_id, str):
                raise DesignContractAuditError(
                    f"assignment group {group_name} contains a non-string item"
                )
            if item_id in item_to_group:
                duplicates.append(item_id)
            item_to_group[item_id] = group_name

    actual_ids = set(item_to_group)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if duplicates or missing or extra:
        raise DesignContractAuditError(
            f"design evidence assignments are not one-to-one: "
            f"duplicates={sorted(set(duplicates))}, missing={missing}, extra={extra}"
        )
    external_ids = sorted(
        item_id
        for item_id, group_name in item_to_group.items()
        if groups[group_name]["status"] == "NO_GO_EXTERNAL"
    )
    if external_ids != list(EXPECTED_EXTERNAL_NO_GO_IDS):
        raise DesignContractAuditError(
            "external No-Go items changed without an explicit contract decision: "
            f"expected={list(EXPECTED_EXTERNAL_NO_GO_IDS)}, actual={external_ids}"
        )
    return item_to_group


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_git_sha1(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def validate_legacy_baseline_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the committed baseline manifest without requiring ignored binary snapshots."""

    if manifest.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise DesignContractAuditError("legacy baseline manifest schema is unsupported")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise DesignContractAuditError("legacy baseline metadata is missing")
    if metadata.get("source_commit") != EXPECTED_LEGACY_SOURCE_COMMIT:
        raise DesignContractAuditError(
            "legacy baseline source commit is not the frozen baseline"
        )
    if not _is_git_sha1(metadata.get("source_tree")):
        raise DesignContractAuditError("legacy baseline source tree hash is invalid")
    if not _is_sha256(metadata.get("git_archive_sha256")):
        raise DesignContractAuditError("legacy baseline Git archive hash is invalid")

    tracked_tree = manifest.get("tracked_tree")
    if (
        not isinstance(tracked_tree, dict)
        or int(tracked_tree.get("file_count", 0)) <= 0
    ):
        raise DesignContractAuditError("legacy tracked tree is empty")
    if int(tracked_tree.get("blob_bytes", 0)) <= 0:
        raise DesignContractAuditError("legacy tracked tree byte count is empty")
    categories = tracked_tree.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise DesignContractAuditError("legacy baseline categories are missing")
    category_summaries: dict[str, Any] = {}
    for category_name, category in categories.items():
        if not isinstance(category, dict):
            raise DesignContractAuditError(
                f"legacy category is invalid: {category_name}"
            )
        entries = category.get("entries")
        if not isinstance(entries, list) or not entries:
            raise DesignContractAuditError(f"legacy category is empty: {category_name}")
        paths: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise DesignContractAuditError(
                    f"legacy category contains an invalid entry: {category_name}"
                )
            path = entry.get("path")
            if not isinstance(path, str) or not path:
                raise DesignContractAuditError(
                    f"legacy category contains an invalid path: {category_name}"
                )
            if not _is_git_sha1(entry.get("object")):
                raise DesignContractAuditError(
                    f"legacy Git object hash is invalid: {path}"
                )
            size = entry.get("bytes")
            if not isinstance(size, int) or size < 0:
                raise DesignContractAuditError(
                    f"legacy Git object size is invalid: {path}"
                )
            paths.append(path)
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise DesignContractAuditError(
                f"legacy category paths must be unique and sorted: {category_name}"
            )
        expected_count = len(entries)
        expected_bytes = sum(entry["bytes"] for entry in entries)
        expected_hash = _canonical_json_sha256(entries)
        if category.get("file_count") != expected_count:
            raise DesignContractAuditError(
                f"legacy category file count does not reconcile: {category_name}"
            )
        if category.get("blob_bytes") != expected_bytes:
            raise DesignContractAuditError(
                f"legacy category byte count does not reconcile: {category_name}"
            )
        if category.get("listing_sha256") != expected_hash:
            raise DesignContractAuditError(
                f"legacy category listing hash does not reconcile: {category_name}"
            )
        category_summaries[category_name] = {
            "file_count": expected_count,
            "blob_bytes": expected_bytes,
            "listing_sha256": expected_hash,
        }

    database = manifest.get("database_snapshot")
    if not isinstance(database, dict) or not _is_sha256(database.get("sha256")):
        raise DesignContractAuditError("legacy database snapshot hash is invalid")
    tables = database.get("tables")
    indexes = database.get("indexes")
    if not isinstance(tables, list) or not tables:
        raise DesignContractAuditError("legacy database table manifest is empty")
    if not isinstance(indexes, list) or not indexes:
        raise DesignContractAuditError("legacy database index manifest is empty")
    if database.get("table_count") != len(tables):
        raise DesignContractAuditError("legacy database table count does not reconcile")
    if database.get("index_count") != len(indexes):
        raise DesignContractAuditError("legacy database index count does not reconcile")
    if not isinstance(database.get("bytes"), int) or database["bytes"] <= 0:
        raise DesignContractAuditError("legacy database byte count is invalid")
    table_names: list[str] = []
    for table in tables:
        if not isinstance(table, dict) or not isinstance(table.get("name"), str):
            raise DesignContractAuditError("legacy database table entry is invalid")
        if not _is_sha256(table.get("schema_sha256")) or not _is_sha256(
            table.get("rows_sha256")
        ):
            raise DesignContractAuditError(
                f"legacy database table hash is invalid: {table.get('name')}"
            )
        if not isinstance(table.get("row_count"), int) or table["row_count"] < 0:
            raise DesignContractAuditError(
                f"legacy database row count is invalid: {table.get('name')}"
            )
        if not isinstance(table.get("columns"), list) or not table["columns"]:
            raise DesignContractAuditError(
                f"legacy database columns are missing: {table.get('name')}"
            )
        table_names.append(table["name"])
    if len(table_names) != len(set(table_names)):
        raise DesignContractAuditError("legacy database table names are not unique")
    index_names: list[str] = []
    for index in indexes:
        if not isinstance(index, dict) or not isinstance(index.get("name"), str):
            raise DesignContractAuditError("legacy database index entry is invalid")
        if not _is_sha256(index.get("schema_sha256")):
            raise DesignContractAuditError(
                f"legacy database index hash is invalid: {index.get('name')}"
            )
        index_names.append(index["name"])
    if len(index_names) != len(set(index_names)):
        raise DesignContractAuditError("legacy database index names are not unique")

    rag = manifest.get("rag_index_snapshot")
    if not isinstance(rag, dict) or not _is_sha256(rag.get("sha256")):
        raise DesignContractAuditError("legacy RAG snapshot hash is invalid")
    if not isinstance(rag.get("document_count"), int) or rag["document_count"] <= 0:
        raise DesignContractAuditError("legacy RAG snapshot document count is empty")
    if not isinstance(rag.get("bytes"), int) or rag["bytes"] <= 0:
        raise DesignContractAuditError("legacy RAG snapshot byte count is invalid")

    reproduction = manifest.get("reproduction")
    if not isinstance(reproduction, dict):
        raise DesignContractAuditError(
            "legacy baseline reproduction contract is missing"
        )
    if reproduction.get("network_required") is not False:
        raise DesignContractAuditError("legacy baseline reproduction must be offline")
    if reproduction.get("writes_formal_response_state") is not False:
        raise DesignContractAuditError(
            "legacy baseline reproduction must not write formal response state"
        )
    return {
        "source_commit": metadata["source_commit"],
        "source_tree": metadata["source_tree"],
        "git_archive_sha256": metadata["git_archive_sha256"],
        "tracked_file_count": tracked_tree["file_count"],
        "tracked_blob_bytes": tracked_tree["blob_bytes"],
        "categories": category_summaries,
        "database": {
            "sha256": database["sha256"],
            "bytes": database["bytes"],
            "table_count": database["table_count"],
            "index_count": database["index_count"],
        },
        "rag_index": {
            "sha256": rag["sha256"],
            "bytes": rag["bytes"],
            "document_count": rag["document_count"],
        },
        "offline_reproducible": True,
    }


def build_design_contract_audit(
    repo_root: Path,
    *,
    source_path: Path | None = None,
    policy_path: Path | None = None,
    baseline_manifest_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    policy_path = policy_path or (
        repo_root / "docs/progressive_upgrade/design_contract_evidence_policy.json"
    )
    baseline_manifest_path = baseline_manifest_path or (
        repo_root / "output/acceptance/legacy_baseline_manifest.json"
    )
    policy = _load_json(policy_path)
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise DesignContractAuditError("design evidence policy schema is unsupported")
    if policy.get("expected_item_count") != EXPECTED_ITEM_COUNT:
        raise DesignContractAuditError(
            "design evidence policy item count is not frozen at 89"
        )
    source_document = policy.get("source_document")
    if not isinstance(source_document, str) or not source_document:
        raise DesignContractAuditError(
            "design evidence policy source document is missing"
        )
    source_path = source_path or (repo_root / source_document)
    if source_path.resolve() != (repo_root / source_document).resolve():
        raise DesignContractAuditError(
            "design source path differs from the policy contract"
        )

    items = extract_design_contract_items(source_path)
    groups = _validate_evidence_groups(repo_root, policy)
    item_to_group = _validate_assignments(items, policy, groups)
    baseline_manifest = _load_json(baseline_manifest_path)
    baseline_summary = validate_legacy_baseline_manifest(baseline_manifest)

    audited_items: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    for item in items:
        group_name = item_to_group[item["item_id"]]
        group = groups[group_name]
        status_counts[group["status"]] += 1
        group_counts[group_name] += 1
        audited_items.append(
            {
                **item,
                "status": group["status"],
                "evidence_group": group_name,
                "evidence": [entry["path"] for entry in group["evidence"]],
                "limitation": group["limitation"],
            }
        )

    controlled_count = sum(status_counts[status] for status in PASS_STATUSES)
    external_ids = [
        item["item_id"] for item in audited_items if item["status"] == "NO_GO_EXTERNAL"
    ]
    summary = {
        "item_count": len(audited_items),
        "status_counts": {
            status: status_counts[status]
            for status in (
                "PASS_LOCAL",
                "PASS_CONTROLLED",
                "NO_GO_GATE",
                "NO_GO_EXTERNAL",
            )
        },
        "controlled_or_local_pass_count": controlled_count,
        "external_no_go_count": len(external_ids),
        "external_no_go_ids": external_ids,
        "all_items_accounted": len(audited_items) == EXPECTED_ITEM_COUNT,
        "controlled_scope_complete": controlled_count
        == EXPECTED_ITEM_COUNT - len(external_ids),
        "full_production_complete": False,
        "overall": "CONTROLLED_CONTRACT_ACCOUNTED_EXTERNAL_NO_GO",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "source_document": source_document,
            "source_contract_sha256": _canonical_json_sha256(items),
            "policy": str(policy_path.resolve().relative_to(repo_root)).replace(
                "\\", "/"
            ),
            "policy_sha256": _evidence_sha256(policy_path),
            "baseline_manifest": str(
                baseline_manifest_path.resolve().relative_to(repo_root)
            ).replace("\\", "/"),
            "baseline_manifest_sha256": _evidence_sha256(baseline_manifest_path),
            "scope": "sections 18.3-18.6, 22 and 23 controlled acceptance contract",
        },
        "summary": summary,
        "section_counts": {
            section: EXPECTED_SECTION_COUNTS[section]
            for section in EXPECTED_SECTION_COUNTS
        },
        "evidence_groups": {
            group_name: {**group, "assigned_item_count": group_counts[group_name]}
            for group_name, group in groups.items()
        },
        "items": audited_items,
        "legacy_baseline": baseline_summary,
        "interpretation": (
            "The 87 local or controlled clauses are evidenced without claiming production "
            "completion. Items 18.4-10 and 22.4-10 remain external No-Go until real legacy "
            "traffic, in-flight events, archives, accounts, and write permissions are verified."
        ),
    }


def render_design_contract_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    status_counts = summary["status_counts"]
    lines = [
        "# 渐进式设计合同逐条审计",
        "",
        f"- 合同条目：{summary['item_count']}/{EXPECTED_ITEM_COUNT}",
        f"- 本地/受控通过：{summary['controlled_or_local_pass_count']}",
        f"- 外部 No-Go：{summary['external_no_go_count']}",
        f"- 外部条目：`{', '.join(summary['external_no_go_ids'])}`",
        f"- 总体边界：`{summary['overall']}`",
        "",
        "> 该报告证明 89 条显式合同均有唯一归属和证据。生产旧流量归零、无在途事件、数据归档恢复与账号撤权仍需真实生产证据。",
        "",
        "## 状态统计",
        "",
        "| 状态 | 条目数 |",
        "|---|---:|",
    ]
    for status in ("PASS_LOCAL", "PASS_CONTROLLED", "NO_GO_GATE", "NO_GO_EXTERNAL"):
        lines.append(f"| `{status}` | {status_counts[status]} |")
    lines.extend(
        [
            "",
            "## 旧系统基线",
            "",
            f"- Git：`{report['legacy_baseline']['source_commit']}`",
            f"- SQLite：`{report['legacy_baseline']['database']['sha256']}`",
            f"- RAG 索引：`{report['legacy_baseline']['rag_index']['sha256']}`",
            "- 重建要求：离线、不写正式响应状态。",
            "",
            "## 89 条合同账本",
            "",
            "| ID | 原文 | 状态 | 证据组 | 原文行 |",
            "|---|---|---|---|---:|",
        ]
    )
    for item in report["items"]:
        requirement = item["requirement"].replace("|", "\\|")
        lines.append(
            f"| `{item['item_id']}` | {requirement} | `{item['status']}` | "
            f"`{item['evidence_group']}` | {item['source_line']} |"
        )
    lines.extend(
        [
            "",
            "## 外部边界",
            "",
            "- `18.4-10`：只有真实旧接口调用归零后才允许移除。",
            "- `22.4-10`：只有真实旧写流量归零、无在途事件、归档可恢复且账号/写权限撤销后，旧链路退役条件才满足。",
            "",
        ]
    )
    return "\n".join(lines)


def write_design_contract_audit(
    repo_root: Path,
    json_target: Path,
    markdown_target: Path,
    *,
    source_path: Path | None = None,
    policy_path: Path | None = None,
    baseline_manifest_path: Path | None = None,
) -> tuple[Path, Path]:
    report = build_design_contract_audit(
        repo_root,
        source_path=source_path,
        policy_path=policy_path,
        baseline_manifest_path=baseline_manifest_path,
    )
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_target.write_text(
        render_design_contract_markdown(report), encoding="utf-8"
    )
    return json_target, markdown_target
