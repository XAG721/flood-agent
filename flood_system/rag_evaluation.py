from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .models import CorpusType, RAGDocument
from .rag import SimpleRAGStore, _normalize_text, _tokenize


RAG_METHODS = ("bm25", "dense_top_k", "hybrid", "mmr", "rerank", "setr", "frc_select")
ROLE_IDS = ("condition", "object", "responsibility", "procedure", "exception", "attribution")


@dataclass(frozen=True)
class RAGEvaluationCase:
    case_id: str
    query: str
    slots: list[str]
    relevant_doc_ids: list[str]
    required_roles: list[str]
    corpus: CorpusType = CorpusType.POLICY


@dataclass(frozen=True)
class RAGMethodResult:
    case_id: str
    method: str
    selected_doc_ids: list[str]
    evidence_recall: float
    role_coverage: float
    answer_f1: float
    task_element_completeness: float
    citation_precision: float
    unsupported_evidence_ratio: float
    latency_ms: float


class RAGBaselineEvaluator:
    """Deterministic comparison harness for retrieval/set-selection methods.

    The dense baseline uses a reproducible hashed n-gram vector so the local
    acceptance suite has no model download dependency. It is an engineering
    baseline, not a substitute for a publication-grade neural embedding run.
    """

    def __init__(self, documents: list[RAGDocument]) -> None:
        self.documents = list(documents)

    @classmethod
    def from_benchmark(cls, path: str | Path) -> tuple["RAGBaselineEvaluator", list[RAGEvaluationCase], dict]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        documents = [RAGDocument.model_validate(item) for item in payload["documents"]]
        cases = [
            RAGEvaluationCase(
                case_id=item["case_id"],
                query=item["query"],
                slots=item["slots"],
                relevant_doc_ids=item["relevant_doc_ids"],
                required_roles=item["required_roles"],
                corpus=CorpusType(item.get("corpus", "policy")),
            )
            for item in payload["cases"]
        ]
        return cls(documents), cases, dict(payload.get("metadata", {}))

    def retrieve(
        self,
        method: str,
        case: RAGEvaluationCase,
        *,
        top_k: int = 4,
        token_budget: int | None = 520,
    ) -> list[RAGDocument]:
        documents = [item for item in self.documents if item.corpus == case.corpus]
        if method == "bm25":
            return self._bm25(documents, case.query)[:top_k]
        if method == "dense_top_k":
            return self._dense(documents, case.query)[:top_k]
        if method == "hybrid":
            return self._hybrid(documents, case.query)[:top_k]
        if method == "mmr":
            return self._mmr(documents, case.query, top_k)
        if method == "rerank":
            return self._rerank(documents, case.query, case.slots)[:top_k]
        if method == "setr":
            return self._setr(documents, case.query, case.required_roles, top_k, token_budget)
        if method == "frc_select":
            return self._frc_select(documents, case, top_k=top_k, token_budget=token_budget)
        raise ValueError(f"unsupported RAG evaluation method: {method}")

    def evaluate(
        self,
        cases: list[RAGEvaluationCase],
        *,
        methods: Iterable[str] = RAG_METHODS,
        top_k: int = 4,
        token_budget: int | None = 520,
    ) -> dict:
        results: list[RAGMethodResult] = []
        for case in cases:
            for method in methods:
                started = time.perf_counter()
                selected = self.retrieve(method, case, top_k=top_k, token_budget=token_budget)
                latency_ms = (time.perf_counter() - started) * 1000
                results.append(self._score(case, method, selected, latency_ms))
        aggregates = {}
        for method in methods:
            rows = [item for item in results if item.method == method]
            aggregates[method] = {
                field: round(sum(getattr(item, field) for item in rows) / max(1, len(rows)), 4)
                for field in (
                    "evidence_recall",
                    "role_coverage",
                    "answer_f1",
                    "task_element_completeness",
                    "citation_precision",
                    "unsupported_evidence_ratio",
                    "latency_ms",
                )
            }
        return {
            "methods": list(methods),
            "case_count": len(cases),
            "top_k": top_k,
            "token_budget": token_budget,
            "aggregates": aggregates,
            "cases": [asdict(item) for item in results],
            "limitations": [
                "Dense Top-K 使用本地确定性哈希 n-gram 向量，仅作为无外部模型依赖的工程基线。",
                "区县标注集为小规模演示集，不能替代 ConditionalQA、MultiHop-RAG 或 HotpotQA 的正式复现实验。",
                "Answer F1 以相关证据集合 F1 作为可重复代理指标，未调用生成模型裁判。",
            ],
        }

    def evaluate_ablations(
        self,
        cases: list[RAGEvaluationCase],
        *,
        top_k: int = 4,
        token_budget: int = 520,
    ) -> dict:
        variants = {
            "full_frc_select": lambda case: self.retrieve("frc_select", case, top_k=top_k, token_budget=token_budget),
            "without_role": lambda case: self._frc_select(
                [item for item in self.documents if item.corpus == case.corpus],
                case,
                top_k=top_k,
                token_budget=token_budget,
                include_roles=False,
            ),
            "without_field": lambda case: self._frc_select(
                [item for item in self.documents if item.corpus == case.corpus],
                case,
                top_k=top_k,
                token_budget=token_budget,
                include_fields=False,
            ),
            "without_budget": lambda case: self.retrieve("frc_select", case, top_k=top_k, token_budget=None),
            "without_trust_conflict": lambda case: self.retrieve("setr", case, top_k=top_k, token_budget=token_budget),
            "without_set_objective": lambda case: self.retrieve("hybrid", case, top_k=top_k, token_budget=token_budget),
        }
        output = {}
        for name, selector in variants.items():
            rows = []
            for case in cases:
                started = time.perf_counter()
                selected = selector(case)
                rows.append(self._score(case, name, selected, (time.perf_counter() - started) * 1000))
            output[name] = {
                field: round(sum(getattr(item, field) for item in rows) / max(1, len(rows)), 4)
                for field in ("evidence_recall", "role_coverage", "citation_precision", "unsupported_evidence_ratio", "latency_ms")
            }
        return {
            "variants": output,
            "interpretation": {
                "without_role": "移除功能角色约束，检验角色互补性对证据选择的贡献。",
                "without_field": "移除任务字段查询约束，检验字段完整性目标的贡献。",
                "without_budget": "移除证据预算约束，观察覆盖收益与证据成本控制的权衡。",
                "without_trust_conflict": "使用仅覆盖优先的 SetR，移除 FRC 的可信度与冲突惩罚。",
                "without_set_objective": "退化为混合 Top-K，移除集合互补性目标。",
            },
        }

    @staticmethod
    def _frc_select(
        documents: list[RAGDocument],
        case: RAGEvaluationCase,
        *,
        top_k: int,
        token_budget: int | None,
        include_roles: bool = True,
        include_fields: bool = True,
    ) -> list[RAGDocument]:
        return SimpleRAGStore(documents).query_evidence_set(
            case.corpus,
            case.query,
            top_k=top_k,
            candidate_k=max(20, len(documents)),
            token_budget=token_budget,
            slots=case.slots if include_fields else [],
            required_roles=case.required_roles if include_roles else [],
        )

    @staticmethod
    def _score(case: RAGEvaluationCase, method: str, selected: list[RAGDocument], latency_ms: float) -> RAGMethodResult:
        selected_ids = {item.doc_id for item in selected}
        expected_ids = set(case.relevant_doc_ids)
        true_positive = len(selected_ids & expected_ids)
        recall = true_positive / max(1, len(expected_ids))
        precision = true_positive / max(1, len(selected_ids))
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        selected_roles = {role for item in selected for role in RAGBaselineEvaluator._roles(item)}
        required_roles = set(case.required_roles)
        role_coverage = len(selected_roles & required_roles) / max(1, len(required_roles))
        return RAGMethodResult(
            case_id=case.case_id,
            method=method,
            selected_doc_ids=[item.doc_id for item in selected],
            evidence_recall=round(recall, 4),
            role_coverage=round(role_coverage, 4),
            answer_f1=round(f1, 4),
            task_element_completeness=round(role_coverage, 4),
            citation_precision=round(precision, 4),
            unsupported_evidence_ratio=round(1 - precision, 4),
            latency_ms=round(latency_ms, 4),
        )

    @staticmethod
    def _roles(document: RAGDocument) -> set[str]:
        value = document.metadata.get("evidence_roles", [])
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",")]
        return {str(item) for item in value if str(item) in ROLE_IDS}

    @staticmethod
    def _token_cost(document: RAGDocument) -> int:
        configured = document.metadata.get("token_cost")
        if configured is not None:
            return max(1, int(configured))
        return max(1, math.ceil(len(document.content) / 4))

    @staticmethod
    def _bm25(documents: list[RAGDocument], query: str) -> list[RAGDocument]:
        query_tokens = _tokenize(query)
        doc_tokens = [_tokenize(f"{item.title} {item.content}") for item in documents]
        average_length = sum(len(tokens) for tokens in doc_tokens) / max(1, len(doc_tokens))
        scored = []
        for index, (document, tokens) in enumerate(zip(documents, doc_tokens, strict=True)):
            score = 0.0
            for token in query_tokens:
                document_frequency = sum(token in candidate for candidate in doc_tokens)
                idf = math.log(1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5))
                frequency = 1.0 if token in tokens else 0.0
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / max(1, average_length))
                score += idf * frequency * 2.5 / max(1e-9, denominator)
            scored.append((score, index, document))
        scored.sort(key=lambda item: (-item[0], item[2].doc_id))
        return [item[2] for item in scored]

    @staticmethod
    def _vector(text: str, dimensions: int = 384) -> list[float]:
        vector = [0.0] * dimensions
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    @classmethod
    def _dense(cls, documents: list[RAGDocument], query: str) -> list[RAGDocument]:
        query_vector = cls._vector(query)
        scored = []
        for document in documents:
            vector = cls._vector(f"{document.title} {document.content}")
            score = sum(left * right for left, right in zip(query_vector, vector, strict=True))
            scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], item[1].doc_id))
        return [item[1] for item in scored]

    @classmethod
    def _hybrid(cls, documents: list[RAGDocument], query: str) -> list[RAGDocument]:
        rankings = [cls._bm25(documents, query), cls._dense(documents, query)]
        scores: dict[str, float] = {}
        by_id = {item.doc_id: item for item in documents}
        for ranking in rankings:
            for rank, document in enumerate(ranking, start=1):
                scores[document.doc_id] = scores.get(document.doc_id, 0.0) + 1 / (60 + rank)
        return [by_id[doc_id] for doc_id in sorted(scores, key=lambda item: (-scores[item], item))]

    @classmethod
    def _mmr(cls, documents: list[RAGDocument], query: str, top_k: int) -> list[RAGDocument]:
        ranking = cls._hybrid(documents, query)
        relevance = {item.doc_id: 1 - rank / max(1, len(ranking)) for rank, item in enumerate(ranking)}
        vectors = {item.doc_id: cls._vector(f"{item.title} {item.content}") for item in ranking}
        selected: list[RAGDocument] = []
        remaining = list(ranking)
        while remaining and len(selected) < top_k:
            def score(item: RAGDocument) -> float:
                redundancy = max(
                    (sum(a * b for a, b in zip(vectors[item.doc_id], vectors[other.doc_id], strict=True)) for other in selected),
                    default=0.0,
                )
                return 0.7 * relevance[item.doc_id] - 0.3 * redundancy
            best = max(remaining, key=lambda item: (score(item), item.doc_id))
            selected.append(best)
            remaining.remove(best)
        return selected

    @classmethod
    def _rerank(cls, documents: list[RAGDocument], query: str, slots: list[str]) -> list[RAGDocument]:
        candidates = cls._hybrid(documents, query)[:20]
        query_tokens = _tokenize(query)
        normalized_query = _normalize_text(query)
        scored = []
        for rank, document in enumerate(candidates):
            text = f"{document.title} {document.content}"
            tokens = _tokenize(text)
            coverage = len(tokens & query_tokens) / max(1, len(query_tokens))
            slot_support = sum(bool(tokens & _tokenize(slot)) for slot in slots) / max(1, len(slots))
            phrase = int(normalized_query in _normalize_text(text))
            scored.append((3 * coverage + 2 * slot_support + phrase - rank / 100, document))
        scored.sort(key=lambda item: (-item[0], item[1].doc_id))
        return [item[1] for item in scored]

    @classmethod
    def _setr(
        cls,
        documents: list[RAGDocument],
        query: str,
        required_roles: list[str],
        top_k: int,
        token_budget: int | None,
    ) -> list[RAGDocument]:
        candidates = cls._hybrid(documents, query)
        selected: list[RAGDocument] = []
        covered: set[str] = set()
        used = 0
        while candidates and len(selected) < top_k:
            eligible = [item for item in candidates if token_budget is None or used + cls._token_cost(item) <= token_budget]
            if not eligible:
                break
            best = max(
                eligible,
                key=lambda item: (
                    len((cls._roles(item) & set(required_roles)) - covered),
                    -candidates.index(item),
                    item.doc_id,
                ),
            )
            selected.append(best)
            covered.update(cls._roles(best))
            used += cls._token_cost(best)
            candidates.remove(best)
        return selected


def render_rag_evaluation_markdown(
    report: dict,
    ablations: dict,
    metadata: dict,
    conflict_report: dict | None = None,
    gate_report: dict | None = None,
) -> str:
    lines = [
        "# V3 RAG 技术评测报告",
        "",
        f"- 数据集：{metadata.get('name', '未命名区县标注集')}",
        f"- 标注性质：{metadata.get('annotation_scope', '本地小规模人工标注演示集')}",
        f"- 用例数：{report['case_count']}",
        f"- Top-K：{report['top_k']}；证据预算：{report['token_budget']} tokens",
        "",
        "## 方法对比",
        "",
        "| 方法 | Evidence Recall | Role Coverage | Answer F1 | 任务要素完整率 | 引用正确率 | 无依据证据比例 | 延迟 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in RAG_METHODS:
        row = report["aggregates"][method]
        lines.append(
            f"| {method} | {row['evidence_recall']:.4f} | {row['role_coverage']:.4f} | "
            f"{row['answer_f1']:.4f} | {row['task_element_completeness']:.4f} | "
            f"{row['citation_precision']:.4f} | {row['unsupported_evidence_ratio']:.4f} | {row['latency_ms']:.4f} |"
        )
    lines.extend(["", "## FRC-Select 消融", "", "| 变体 | Evidence Recall | Role Coverage | 引用正确率 | 无依据证据比例 | 延迟 ms |", "|---|---:|---:|---:|---:|---:|"])
    for name, row in ablations["variants"].items():
        lines.append(
            f"| {name} | {row['evidence_recall']:.4f} | {row['role_coverage']:.4f} | "
            f"{row['citation_precision']:.4f} | {row['unsupported_evidence_ratio']:.4f} | {row['latency_ms']:.4f} |"
        )
    if conflict_report:
        lines.extend(
            [
                "",
                "## 冲突检测",
                "",
                f"- 受控模拟用例：{conflict_report['case_count']}",
                f"- Precision：{conflict_report['precision']:.4f}",
                f"- Recall：{conflict_report['recall']:.4f}",
                f"- F1：{conflict_report['f1']:.4f}",
            ]
        )
    if gate_report:
        lines.extend(
            [
                "",
                "## Gate 2 判定",
                "",
                f"- 状态：{gate_report['status']}",
                f"- 相对最强基线字段覆盖增益：{gate_report['coverage_gain_percentage_points']:.4f} 个百分点",
                f"- Full 相对 w/o Role / w/o Field 的引用正确率提升："
                f"{gate_report['full_vs_without_role_citation_gain']:.4f} / "
                f"{gate_report['full_vs_without_field_citation_gain']:.4f}",
            ]
        )
        for criterion, passed in gate_report["criteria"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'}：{criterion}")
    lines.extend(["", "## 解释与限制", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("- 本报告可复现本地工程选择逻辑；正式论文结论仍需真实神经向量模型、公开基准与独立人工标注。")
    return "\n".join(lines) + "\n"


def evaluate_frc_gate(report: dict, ablations: dict, conflict_report: dict) -> dict:
    """Evaluate the controlled Gate 2 thresholds without overriding a NO-GO result."""

    aggregates = report["aggregates"]
    full = aggregates["frc_select"]
    baseline_coverages = [
        row["task_element_completeness"]
        for method, row in aggregates.items()
        if method != "frc_select"
    ]
    strongest_baseline = max(baseline_coverages, default=0.0)
    coverage_gain = round((full["task_element_completeness"] - strongest_baseline) * 100, 4)
    variants = ablations["variants"]
    role_gain = round(full["citation_precision"] - variants["without_role"]["citation_precision"], 4)
    field_gain = round(full["citation_precision"] - variants["without_field"]["citation_precision"], 4)
    undisclosed_conflicts = sum(
        len(set(row.get("expected", [])) - set(row.get("predicted", [])))
        for row in conflict_report.get("rows", [])
    )
    criteria = {
        "核心任务字段覆盖率不低于0.90": full["task_element_completeness"] >= 0.90,
        "引用支持精度不低于0.95": full["citation_precision"] >= 0.95,
        "无依据关键字段率为0": full["unsupported_evidence_ratio"] == 0,
        "冲突检测F1不低于0.85": conflict_report["f1"] >= 0.85,
        "未披露关键冲突数为0": undisclosed_conflicts == 0,
        "相同预算下字段覆盖率相对最强基线提高至少5个百分点": coverage_gain >= 5.0,
        "Full引用正确率严格优于w/o Role和w/o Field": role_gain > 0 and field_gain > 0,
    }
    return {
        "status": "GO" if all(criteria.values()) else "NO-GO",
        "criteria": criteria,
        "strongest_baseline_task_element_completeness": strongest_baseline,
        "frc_task_element_completeness": full["task_element_completeness"],
        "coverage_gain_percentage_points": coverage_gain,
        "undisclosed_critical_conflicts": undisclosed_conflicts,
        "full_vs_without_role_citation_gain": role_gain,
        "full_vs_without_field_citation_gain": field_gain,
    }
