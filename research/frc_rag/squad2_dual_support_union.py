"""Fourth disjoint SQuAD 2.0 dual-support union experiment (v61)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import research.frc_rag.squad2_extractive_support_gate as v59
import research.frc_rag.squad2_generative_answerability_gate as v58
import research.frc_rag.squad2_structured_span_gate as v60
from research.frc_rag.rgb_cost_aware_frc import read_jsonl


SCHEMA_VERSION = "frc-squad2-dual-support-union-v61"
EXPERIMENT_ID = "FRC-SQUAD2-DUAL-SUPPORT-UNION-V61"
DATASET_ID = "squad2_fourth_disjoint_dual_support_evidence_selection_v61"
CAPABILITY = "closed_paragraph_dual_support_union_and_evidence_selection"
PROTOCOL_SHA256 = "8e07c2404c22a8c19fc879fe18f567ef3a5fec396f8213a657d2c663f9e1cfcd"

STAGES = v60.STAGES
SOURCE_FILES = v60.SOURCE_FILES
TARGET_CASES = v60.TARGET_CASES
TARGET_PER_GROUP = v59.TARGET_PER_GROUP
MAX_CASES_PER_PARAGRAPH = v59.MAX_CASES_PER_PARAGRAPH
MAX_CASES_PER_ARTICLE_PER_STATE = v59.MAX_CASES_PER_ARTICLE_PER_STATE
MINIMUM_PARAGRAPHS = v59.MINIMUM_PARAGRAPHS
MINIMUM_ARTICLES = v59.MINIMUM_ARTICLES
BUDGETS = v60.BUDGETS
METHODS = v60.METHODS
CANDIDATE = v60.CANDIDATE
SAMPLE_SALTS = {
    "development": "FRC-SQUAD2-V61-DISJOINT-DEVELOPMENT|",
    "confirmation": "FRC-SQUAD2-V61-CONFIRMATION|",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def read_stage_source(source_root: Path, stage: str) -> dict[str, Any]:
    return v60.read_stage_source(source_root, stage)


def load_prior_exclusion_union(paths: Sequence[Path]) -> set[str]:
    if len(paths) != 3:
        raise ValueError("SQuAD2 v61 requires v58, v59 and v60 exclusion maps")
    groups = [v59.load_v58_excluded_commitments(path) for path in paths]
    union = set().union(*groups)
    if len(union) != 1800:
        raise ValueError("SQuAD2 v61 prior exclusion union must contain 1,800 ids")
    return union


def _sample_key(row: dict[str, Any], stage: str) -> tuple[str, str]:
    return (
        _hash(
            SAMPLE_SALTS[stage],
            str(row["answer_state"]),
            str(row["article_id"]),
            str(row["paragraph_id"]),
            str(row["raw_id"]),
        ),
        str(row["raw_id"]),
    )


def select_disjoint_balanced_sample(
    source: dict[str, Any],
    *,
    stage: str,
    excluded_commitments: set[str],
    target_per_group: int = TARGET_PER_GROUP,
    maximum_cases_per_paragraph: int = MAX_CASES_PER_PARAGRAPH,
    maximum_cases_per_article_per_state: int = MAX_CASES_PER_ARTICLE_PER_STATE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported SQuAD2 v61 stage: {stage}")
    rows, schema = v58.extract_cases(source)
    available = [
        row for row in rows if _hash(str(row["raw_id"])) not in excluded_commitments
    ]
    groups = ("answer_bearing", "no_answer")
    by_group = {
        group: sorted(
            [row for row in available if row["answer_state"] == group],
            key=lambda row: _sample_key(row, stage),
        )
        for group in groups
    }
    if any(len(by_group[group]) < target_per_group for group in groups):
        raise ValueError("SQuAD2 v61 source lacks the disjoint answer-state quota")
    selected: list[dict[str, Any]] = []
    selected_by_group: Counter[str] = Counter()
    paragraph_counts: Counter[str] = Counter()
    article_state_counts: Counter[tuple[str, str]] = Counter()
    cursors = {group: 0 for group in groups}
    while any(selected_by_group[group] < target_per_group for group in groups):
        progressed = False
        for group in groups:
            if selected_by_group[group] >= target_per_group:
                continue
            values = by_group[group]
            while cursors[group] < len(values):
                row = values[cursors[group]]
                cursors[group] += 1
                paragraph_id = str(row["paragraph_id"])
                article_state = (str(row["article_id"]), group)
                if paragraph_counts[paragraph_id] >= maximum_cases_per_paragraph:
                    continue
                if (
                    article_state_counts[article_state]
                    >= maximum_cases_per_article_per_state
                ):
                    continue
                selected.append(row)
                selected_by_group[group] += 1
                paragraph_counts[paragraph_id] += 1
                article_state_counts[article_state] += 1
                progressed = True
                break
        if not progressed:
            raise ValueError("SQuAD2 v61 caps prevent the disjoint balanced sample")
    selected.sort(key=lambda row: (str(row["answer_state"]), _sample_key(row, stage)))
    overlap = sum(_hash(str(row["raw_id"])) in excluded_commitments for row in selected)
    return selected, {
        "stage": stage,
        "target_cases": target_per_group * 2,
        "selected_cases": len(selected),
        "eligible_answer_state_counts_after_exclusion": dict(
            Counter(row["answer_state"] for row in available)
        ),
        "selected_answer_state_counts": dict(selected_by_group),
        "selected_paragraphs": len(paragraph_counts),
        "selected_articles": len({str(row["article_id"]) for row in selected}),
        "maximum_cases_per_paragraph": max(paragraph_counts.values(), default=0),
        "maximum_cases_per_article_per_answer_state": max(
            article_state_counts.values(), default=0
        ),
        "prior_excluded_commitment_count": len(excluded_commitments),
        "selected_prior_commitment_overlap": overlap,
        "selected_v58_commitment_overlap": overlap,
        **schema,
    }


def prepare_blind_cases(
    selected: Sequence[dict[str, Any]], tokenizer: Any, *, stage: str
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    prepared, maps, verifier_inputs, structural = v60.prepare_blind_cases(
        selected, tokenizer, stage=stage
    )
    for row in prepared:
        row["schema_version"] = SCHEMA_VERSION
        row["dataset_id"] = DATASET_ID
        row["capability"] = CAPABILITY
    for row in maps:
        row["schema_version"] = "frc-squad2-v61-candidate-map-v1"
    for row in verifier_inputs:
        row["schema_version"] = "frc-squad2-v61-verifier-input-v1"
        row["dataset_id"] = DATASET_ID
    structural["schema_version"] = "frc-squad2-v61-structural-census-v1"
    return prepared, maps, verifier_inputs, structural


def build_deterministic_queries(
    prepared_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = v60.build_deterministic_queries(prepared_rows)
    for row in rows:
        row["schema_version"] = SCHEMA_VERSION
        row["dataset_id"] = DATASET_ID
    return rows


def validate_query_cache(
    prepared_rows: Sequence[dict[str, Any]], query_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if list(query_rows) != build_deterministic_queries(prepared_rows):
        raise ValueError("SQuAD2 v61 deterministic query cache changed")
    return {
        "rows": len(query_rows),
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "gold_fields_visible_to_generator": False,
        "deterministic_templates": True,
    }


class FrozenSquad2Scorer(v60.FrozenSquad2Scorer):
    """Frozen BGE scorer with v61 metadata labels."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
        return rows


def combine_support_decisions(
    binary_text: str, structured_text: str, context: str
) -> dict[str, Any]:
    binary_decision = v58.normalize_support_decision(binary_text)
    structured_decision = v60.parse_structured_support(structured_text, context)
    support_passed = (
        binary_decision == "SUPPORTED" or structured_decision == "SUPPORTED_SPAN"
    )
    return {
        "binary_decision": binary_decision,
        "binary_malformed_fail_closed_used": binary_decision is None,
        "structured_decision": structured_decision,
        "structured_invalid_fail_closed_used": structured_decision is None,
        "support_passed": support_passed,
    }


class LocalQwenDualSupportVerifier:
    """One frozen model executing the exact v58 and v60 support prompts."""

    def __init__(
        self,
        *,
        model_path: Path,
        batch_size: int = 16,
        maximum_input_tokens: int = 3072,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.maximum_input_tokens = maximum_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()

    def _chat_prompt(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate(self, prompts: list[str], *, max_new_tokens: int) -> list[str]:
        encoded = self.tokenizer(
            [self._chat_prompt(prompt) for prompt in prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.maximum_input_tokens,
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_length = encoded.input_ids.shape[-1]
        return [
            value.strip()
            for value in self.tokenizer.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )
        ]

    def verify(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        output_path: Path,
        cache_key: str,
    ) -> list[dict[str, Any]]:
        existing = list(read_jsonl(output_path)) if output_path.is_file() else []
        expected_ids = [str(row["id"]) for row in rows]
        if [str(row.get("id", "")) for row in existing] != expected_ids[
            : len(existing)
        ]:
            raise ValueError("SQuAD2 v61 verifier cache is not a valid prefix")
        if any(row.get("cache_key") != cache_key for row in existing):
            raise ValueError("SQuAD2 v61 verifier cache key changed")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if existing else "w"
        with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
            for start in range(len(existing), len(rows), self.batch_size):
                batch = list(rows[start : start + self.batch_size])
                binary_raw = self._generate(
                    [v58.support_prompt(row) for row in batch], max_new_tokens=4
                )
                structured_raw = self._generate(
                    [v60.structured_prompt(row) for row in batch], max_new_tokens=40
                )
                for source, binary, structured in zip(
                    batch, binary_raw, structured_raw, strict=True
                ):
                    combined = combine_support_decisions(
                        binary, structured, str(source["context"])
                    )
                    support_passed = bool(combined["support_passed"])
                    prediction = {
                        "schema_version": "frc-squad2-v61-dual-decision-v1",
                        "experiment_id": EXPERIMENT_ID,
                        "stage": str(source["stage"]),
                        "id": str(source["id"]),
                        "cache_key": cache_key,
                        "binary_raw_prediction": binary,
                        "binary_raw_sha256": hashlib.sha256(
                            binary.encode("utf-8")
                        ).hexdigest(),
                        "binary_decision": combined["binary_decision"],
                        "binary_malformed_fail_closed_used": combined[
                            "binary_malformed_fail_closed_used"
                        ],
                        "structured_raw_prediction": structured,
                        "structured_raw_sha256": hashlib.sha256(
                            structured.encode("utf-8")
                        ).hexdigest(),
                        "structured_decision": combined["structured_decision"],
                        "structured_invalid_fail_closed_used": combined[
                            "structured_invalid_fail_closed_used"
                        ],
                        "decision": "SUPPORTED_UNION"
                        if support_passed
                        else "UNSUPPORTED_UNION",
                        "support_passed": support_passed,
                        "invalid_fail_closed_used": combined[
                            "structured_invalid_fail_closed_used"
                        ],
                        "raw_prediction_sha256": hashlib.sha256(
                            (binary + "\x1f" + structured).encode("utf-8")
                        ).hexdigest(),
                        "raw_prediction_codepoints": len(binary) + len(structured),
                        "gold_fields_visible_to_verifier": False,
                    }
                    handle.write(
                        json.dumps(
                            prediction,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    existing.append(prediction)
                handle.flush()
                print(
                    f"verified {min(start + len(batch), len(rows))}/{len(rows)} "
                    "SQuAD2 v61 dual cases",
                    flush=True,
                )
        return existing


def validate_verifier_cache(
    verifier_inputs: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    cache_key: str,
) -> dict[str, Any]:
    expected = [str(row["id"]) for row in verifier_inputs]
    if [str(row["id"]) for row in decisions] != expected:
        raise ValueError("SQuAD2 v61 verifier cache is incomplete or out of order")
    if any(row.get("cache_key") != cache_key for row in decisions):
        raise ValueError("SQuAD2 v61 verifier cache key changed")
    binary_bad = sum(
        bool(row["binary_malformed_fail_closed_used"]) for row in decisions
    )
    structured_bad = sum(
        bool(row["structured_invalid_fail_closed_used"]) for row in decisions
    )
    return {
        "rows": len(decisions),
        "binary_malformed_count": binary_bad,
        "binary_malformed_rate": binary_bad / len(decisions) if decisions else 1.0,
        "structured_invalid_count": structured_bad,
        "structured_invalid_rate": (
            structured_bad / len(decisions) if decisions else 1.0
        ),
        "invalid_output_count": structured_bad,
        "invalid_output_rate": structured_bad / len(decisions) if decisions else 1.0,
        "gold_fields_visible_to_verifier": False,
    }


def build_gold_rows(
    source: dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    structural_census: dict[str, Any],
) -> list[dict[str, Any]]:
    return v60.build_gold_rows(
        source,
        candidate_maps,
        scored_rows,
        structural_census=structural_census,
    )


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    value = v60.build_candidate_coverage(gold_rows, sampling)
    value["schema_version"] = "frc-squad2-v61-candidate-coverage-v1"
    value["experiment_id"] = EXPERIMENT_ID
    return value


def evaluate_stage(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    support_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    verifier_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
    *,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report, evidence = v60.evaluate_stage(
        gold_rows,
        scored_rows,
        support_rows,
        query_summary,
        verifier_summary,
        source_artifacts,
        stage=stage,
    )
    report["schema_version"] = "frc-squad2-dual-support-report-v61"
    report["experiment_id"] = EXPERIMENT_ID
    report["metadata"]["dataset_id"] = DATASET_ID
    report["metadata"]["v60_case_level_artifact_reused"] = False
    analysis = report["analysis"]
    analysis["mechanism"] = (
        "OR union of exact frozen v58 binary and v60 structured-span support "
        "components with v59-frozen downstream selectors"
    )
    checks = analysis["support_checks"]
    checks.pop("answer_recall_drop_vs_ungated_exact_anchor_at_most_0_1", None)
    checks.pop("answer_bearing_span_pass_rate_at_least_0_75", None)
    checks.pop("no_answer_rejection_rate_at_least_0_65", None)
    checks.pop("candidate_no_answer_abstention_accuracy_at_least_0_65", None)
    checks.pop("candidate_abstention_rate_at_least_0_2", None)
    verifier = analysis["support_verifier"]
    candidate = analysis["aggregates"][CANDIDATE]
    checks["answer_bearing_union_pass_rate_at_least_0_8"] = (
        verifier["answer_bearing_span_pass_rate"] >= 0.8
    )
    checks["no_answer_union_rejection_rate_at_least_0_6"] = (
        verifier["no_answer_rejection_rate"] >= 0.6
    )
    checks["binary_malformed_rate_equals_0"] = (
        verifier_summary["binary_malformed_rate"] == 0.0
    )
    checks["structured_invalid_rate_at_most_0_02"] = (
        verifier_summary["structured_invalid_rate"] <= 0.02
    )
    checks["candidate_answer_recall_at_least_0_75"] = (
        candidate["answer_macro_recall"] >= 0.75
    )
    checks["candidate_no_answer_abstention_accuracy_at_least_0_6"] = (
        candidate["no_answer_abstention_accuracy"] >= 0.6
    )
    checks["candidate_abstention_rate_at_least_0_15"] = (
        candidate["abstention_rate"] >= 0.15
    )
    checks["selected_prior_commitment_overlap_equals_0"] = (
        source_artifacts["sampling"]["selected_prior_commitment_overlap"] == 0
    )
    supported = all(checks.values())
    if stage == "development":
        status = (
            "SQUAD2_V61_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_CONFIRMATION"
            if supported
            else "SQUAD2_V61_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
        )
    else:
        status = (
            "SQUAD2_V61_DUAL_SUPPORT_FRC_FEASIBILITY_ESTABLISHED"
            if supported
            else "SQUAD2_V61_DUAL_SUPPORT_FRC_FEASIBILITY_NOT_ESTABLISHED"
        )
    outcome = analysis["outcome"]
    outcome["status"] = status
    outcome["support_established"] = supported
    outcome["confirmation_open_authorized"] = stage == "development" and supported
    analysis["support_verifier"]["binary_malformed_rate"] = verifier_summary[
        "binary_malformed_rate"
    ]
    analysis["support_verifier"]["structured_invalid_rate"] = verifier_summary[
        "structured_invalid_rate"
    ]
    return report, evidence


def write_report(
    report: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    json_path: Path,
    markdown_path: Path,
    evidence_path: Path,
) -> None:
    v60.write_report(report, evidence, json_path, markdown_path, evidence_path)
    text = markdown_path.read_text(encoding="utf-8")
    text = text.replace(
        "# SQuAD 2.0 structured span gate", "# SQuAD 2.0 dual support union", 1
    ).replace(", v60)", ", v61)", 1)
    markdown_path.write_text(text, encoding="utf-8", newline="\n")


def validate_protocol(protocol_path: Path) -> dict[str, Any]:
    if _sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("SQuAD2 v61 protocol hash changed")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    if value["prior_boundary"]["dev_content_opened"] is not False:
        raise ValueError("SQuAD2 v61 dev boundary changed")
    prompts = value["model_and_prompts"]
    if prompts["binary_prompt"] != v58.SUPPORT_PROMPT:
        raise ValueError("SQuAD2 v61 binary prompt changed")
    if prompts["structured_prompt"] != v60.STRUCTURED_PROMPT:
        raise ValueError("SQuAD2 v61 structured prompt changed")
    if value["source_and_scope"]["expected_prior_exclusion_commitments"] != 1800:
        raise ValueError("SQuAD2 v61 exclusion count changed")
    return value


def validate_source_contract(
    source_registration_path: Path,
    *,
    protocol_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path)
    if (
        _sha256(source_registration_path)
        != protocol["source_and_scope"]["source_registration_sha256"]
    ):
        raise ValueError("SQuAD2 v61 source registration changed")
    return v60.validate_source_contract(
        source_registration_path,
        protocol_path=(
            protocol_path.resolve().parents[2]
            / "docs/progressive_upgrade/squad2_structured_span_gate_protocol_v60.json"
        ),
        source_root=source_root,
    )


def validate_implementation_registration(
    registration_path: Path,
    *,
    protocol_path: Path,
    source_registration_path: Path,
    module_path: Path,
    runner_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    expected = {
        "protocol": _sha256(protocol_path),
        "source_registration": _sha256(source_registration_path),
        "module": _sha256(module_path),
        "runner": _sha256(runner_path),
        "tests": _sha256(test_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("SQuAD2 v61 implementation registration changed")
    if value.get("v61_sample_constructed_before_registration") is not False:
        raise ValueError("SQuAD2 v61 sample was constructed before freeze")
    if value.get("dev_content_read") is not False:
        raise ValueError("SQuAD2 v61 dev was opened before freeze")
    return value


__all__ = [
    "BUDGETS",
    "CANDIDATE",
    "EXPERIMENT_ID",
    "FrozenSquad2Scorer",
    "LocalQwenDualSupportVerifier",
    "SOURCE_FILES",
    "STAGES",
    "TARGET_CASES",
    "build_candidate_coverage",
    "build_deterministic_queries",
    "build_gold_rows",
    "combine_support_decisions",
    "evaluate_stage",
    "load_prior_exclusion_union",
    "prepare_blind_cases",
    "read_stage_source",
    "select_disjoint_balanced_sample",
    "validate_implementation_registration",
    "validate_protocol",
    "validate_query_cache",
    "validate_source_contract",
    "validate_verifier_cache",
    "write_report",
]
