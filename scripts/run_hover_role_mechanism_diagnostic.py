from __future__ import annotations

import gzip
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCORED = ROOT / ".cache/benchmarks/hover/scored_verification_roles_blind.jsonl"
EVIDENCE = (
    ROOT
    / "output/rag_evaluation/hover_verification_roles/"
    "hover_verification_roles_cases.jsonl.gz"
)
OUTPUT = (
    ROOT
    / "output/rag_evaluation/hover_verification_roles/"
    "hover_role_mechanism_diagnostic.json"
)
EXPECTED_HASHES = {
    "scored": "ecf761b27d91165b46eda4992ac151d6e5f018a5ca559e9c1417cd4d1721217a",
    "evidence": "73f5c62c0d120fef2e5c6cc328d7e3e2f9e2103b8a91013aa15edb0484d8bb6e",
}
GENERIC_ROLES = (
    "direct_answer_support",
    "entity_and_scope",
    "corroborating_evidence",
    "contradiction_detection",
)
VERIFICATION_ROLES = (
    "claim_support",
    "claim_refutation",
    "entity_bridge",
    "cross_document_chain",
)
BUDGETS = (512, 1024, 1500)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordinal_rank(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float), kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(array)), 6),
        "median": round(float(np.median(array)), 6),
        "p05": round(float(np.quantile(array, 0.05)), 6),
        "p95": round(float(np.quantile(array, 0.95)), 6),
    }


def build_report() -> dict[str, Any]:
    actual = {"scored": _sha256(SCORED), "evidence": _sha256(EVIDENCE)}
    if actual != EXPECTED_HASHES:
        raise ValueError(f"v39 mechanism source hashes changed: {actual}")
    correlations = {"generic": [], "verification": [], "matched_cross_family": []}
    distinct = {"generic": [], "verification": []}
    above_threshold = {"generic": [], "verification": []}
    within_candidate_std = {"generic": [], "verification": []}
    scored_rows = 0
    with SCORED.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            scored_rows += 1
            scores = row["candidate_scores"]
            families = {
                "generic": GENERIC_ROLES,
                "verification": VERIFICATION_ROLES,
            }
            matrices: dict[str, np.ndarray] = {}
            rank_matrices: dict[str, np.ndarray] = {}
            for name, roles in families.items():
                matrix = np.asarray(
                    [
                        [float(candidate["role_scores"][role]) for candidate in scores]
                        for role in roles
                    ]
                )
                matrices[name] = matrix
                rank_matrix = np.asarray(
                    [_ordinal_rank(values.tolist()) for values in matrix]
                )
                rank_matrices[name] = rank_matrix
                for first, second in itertools.combinations(range(4), 2):
                    correlations[name].append(
                        float(np.corrcoef(rank_matrix[first], rank_matrix[second])[0, 1])
                    )
                distinct[name].append(len(set(np.argmax(matrix, axis=1).tolist())))
                above_threshold[name].append(float(np.mean(np.sum(matrix >= 0.55, axis=0))))
                within_candidate_std[name].append(float(np.mean(np.std(matrix, axis=0))))
            for index in range(4):
                correlations["matched_cross_family"].append(
                    float(
                        np.corrcoef(
                            rank_matrices["generic"][index],
                            rank_matrices["verification"][index],
                        )[0, 1]
                    )
                )
    selection = {
        str(budget): {
            "generic_verification_ordered_identity": [],
            "verification_cross_ordered_identity": [],
            "verification_cross_set_jaccard": [],
        }
        for budget in BUDGETS
    }
    evidence_rows = 0
    with gzip.open(EVIDENCE, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            evidence_rows += 1
            for budget in BUDGETS:
                methods = row["configurations"][str(budget)]["methods"]
                generic = methods["frc_generic_roles_v39"]["selected_ids"]
                verification = methods["frc_verification_roles_v39"]["selected_ids"]
                cross = methods["cross_encoder_topk"]["selected_ids"]
                target = selection[str(budget)]
                target["generic_verification_ordered_identity"].append(generic == verification)
                target["verification_cross_ordered_identity"].append(verification == cross)
                union = set(verification) | set(cross)
                target["verification_cross_set_jaccard"].append(
                    len(set(verification) & set(cross)) / len(union) if union else 1.0
                )
    if scored_rows != 4000 or evidence_rows != 4000:
        raise ValueError("v39 mechanism diagnostic requires all 4000 cases")
    return {
        "schema_version": "frc-hover-role-mechanism-diagnostic-v1",
        "metadata": {
            "experiment_id": "FRC-HOVER-VERIFICATION-ROLES-V39",
            "post_result_diagnostic": True,
            "cases": scored_rows,
            "source_sha256": actual,
            "gold_used_for_role_or_selection_diagnostic": False,
            "raw_text_exported": False,
        },
        "role_signal": {
            "rank_correlations": {
                name: _summary(values) for name, values in correlations.items()
            },
            "generic": {
                "mean_distinct_role_argmax": round(float(np.mean(distinct["generic"])), 6),
                "all_roles_share_argmax_rate": round(
                    float(np.mean(np.asarray(distinct["generic"]) == 1)), 6
                ),
                "mean_roles_above_0_55_per_candidate": round(
                    float(np.mean(above_threshold["generic"])), 6
                ),
                "mean_within_candidate_role_std": round(
                    float(np.mean(within_candidate_std["generic"])), 6
                ),
            },
            "verification": {
                "mean_distinct_role_argmax": round(
                    float(np.mean(distinct["verification"])), 6
                ),
                "all_roles_share_argmax_rate": round(
                    float(np.mean(np.asarray(distinct["verification"]) == 1)), 6
                ),
                "mean_roles_above_0_55_per_candidate": round(
                    float(np.mean(above_threshold["verification"])), 6
                ),
                "mean_within_candidate_role_std": round(
                    float(np.mean(within_candidate_std["verification"])), 6
                ),
            },
        },
        "selection_collapse": {
            budget: {
                metric: round(float(np.mean(values)), 6)
                for metric, values in metrics.items()
            }
            for budget, metrics in selection.items()
        },
        "interpretation": {
            "status": "STATIC_ROLE_SIGNAL_COLLAPSE_OBSERVED",
            "diagnosis": "The static role prompts produce nearly collinear candidate rankings and usually select the same evidence as direct cross-encoder top-k.",
            "causal_claim": False,
            "next_test": "Prospectively test claim-conditioned atomic queries and threshold-free rank coverage on IDs not used in this diagnostic.",
            "gate_2": "NO-GO/SHADOW",
        },
    }


def render(report: dict[str, Any]) -> str:
    role = report["role_signal"]
    lines = [
        "# HoVer v39 角色信号退化诊断",
        "",
        "- 状态：`STATIC_ROLE_SIGNAL_COLLAPSE_OBSERVED`",
        f"- 样例：{report['metadata']['cases']}（诊断不使用 gold）",
        "- 通用/核验角色内部平均 Spearman："
        f"{role['rank_correlations']['generic']['mean']:.6f} / "
        f"{role['rank_correlations']['verification']['mean']:.6f}",
        "- 通用/核验角色共享同一首选候选比例："
        f"{role['generic']['all_roles_share_argmax_rate']:.6f} / "
        f"{role['verification']['all_roles_share_argmax_rate']:.6f}",
        "",
        "## 选择退化",
        "",
        "| Token 预算 | 通用=核验（有序） | 核验=交叉编码器（有序） | 核验/交叉集合 Jaccard |",
        "|---:|---:|---:|---:|",
    ]
    for budget, metrics in report["selection_collapse"].items():
        lines.append(
            f"| {budget} | {metrics['generic_verification_ordered_identity']:.6f} | "
            f"{metrics['verification_cross_ordered_identity']:.6f} | "
            f"{metrics['verification_cross_set_jaccard']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "这是看过 v39 结果后的机制诊断，不是独立性能评测。它只说明静态角色"
            "提示在当前重排器上高度共线，不能改变 Gate 2，也不能用于声称 v41 有效。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown = OUTPUT.with_suffix(".md")
    markdown.write_text(render(report), encoding="utf-8", newline="\n")
    print(OUTPUT)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
