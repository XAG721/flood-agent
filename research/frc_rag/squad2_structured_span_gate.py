"""Third disjoint SQuAD 2.0 structured-span support-gate experiment (v60)."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import research.frc_rag.squad2_extractive_support_gate as v59
import research.frc_rag.squad2_generative_answerability_gate as v58
from research.frc_rag.evidence_inference_low_core_divergence_atomic_roles import (
    _round_for_display,
)
from research.frc_rag.rgb_cost_aware_frc import read_jsonl


SCHEMA_VERSION = "frc-squad2-structured-span-gate-v60"
EXPERIMENT_ID = "FRC-SQUAD2-STRUCTURED-SPAN-GATE-V60"
DATASET_ID = "squad2_third_disjoint_structured_span_evidence_selection_v60"
CAPABILITY = "closed_paragraph_structured_span_support_and_evidence_selection"
PROTOCOL_SHA256 = "8033a31ce20219cdd1fa654f13fbd53fd24b7bae4c126035d14595c7c12940d1"

STAGES = v59.STAGES
SOURCE_FILES = v59.SOURCE_FILES
TARGET_CASES = v59.TARGET_CASES
TARGET_PER_GROUP = v59.TARGET_PER_GROUP
MAX_CASES_PER_PARAGRAPH = v59.MAX_CASES_PER_PARAGRAPH
MAX_CASES_PER_ARTICLE_PER_STATE = v59.MAX_CASES_PER_ARTICLE_PER_STATE
MINIMUM_PARAGRAPHS = v59.MINIMUM_PARAGRAPHS
MINIMUM_ARTICLES = v59.MINIMUM_ARTICLES
BUDGETS = v59.BUDGETS
METHODS = v59.METHODS
CANDIDATE = v59.CANDIDATE
GATED_NON_FRC_METHODS = v59.GATED_NON_FRC_METHODS
GATED_FRC_CONTROLS = v59.GATED_FRC_CONTROLS
EXACT_ANCHOR_GATED = v59.EXACT_ANCHOR_GATED
EXACT_ANCHOR_UNGATED = v59.EXACT_ANCHOR_UNGATED
SAMPLE_SALTS = {
    "development": "FRC-SQUAD2-V60-DISJOINT-DEVELOPMENT|",
    "confirmation": "FRC-SQUAD2-V60-CONFIRMATION|",
}

STRUCTURED_PROMPT = (
    "Determine whether the paragraph contains a contiguous span that directly "
    "answers the requested fact. Return exactly one line in one of these forms:\n"
    "SPAN: <copy the shortest exact answer span from the paragraph>\n"
    "UNSUPPORTED\n"
    "Do not add explanation, markdown, quotation commentary, or outside "
    "knowledge. Use UNSUPPORTED for a related topic, a plausible guess, or an "
    "unstated assumption.\n\nQuestion: {question}\n\nParagraph:\n{context}"
    "\n\nOutput:"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def read_stage_source(source_root: Path, stage: str) -> dict[str, Any]:
    return v59.read_stage_source(source_root, stage)


def load_prior_exclusion_union(paths: Sequence[Path]) -> set[str]:
    if len(paths) != 2:
        raise ValueError("SQuAD2 v60 requires exactly the v58 and v59 exclusion maps")
    groups = [v59.load_v58_excluded_commitments(path) for path in paths]
    union = set().union(*groups)
    if len(union) != 1200:
        raise ValueError("SQuAD2 v60 prior exclusion union must contain 1,200 ids")
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
        raise ValueError(f"Unsupported SQuAD2 v60 stage: {stage}")
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
        raise ValueError("SQuAD2 v60 source lacks the disjoint answer-state quota")
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
            raise ValueError("SQuAD2 v60 caps prevent the disjoint balanced sample")
    selected.sort(key=lambda row: (str(row["answer_state"]), _sample_key(row, stage)))
    selected_overlap = sum(
        _hash(str(row["raw_id"])) in excluded_commitments for row in selected
    )
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
        "selected_prior_commitment_overlap": selected_overlap,
        "selected_v58_commitment_overlap": selected_overlap,
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
    prepared, maps, verifier_inputs, structural = v59.prepare_blind_cases(
        selected, tokenizer, stage=stage
    )
    for row in prepared:
        row["schema_version"] = SCHEMA_VERSION
        row["dataset_id"] = DATASET_ID
        row["capability"] = CAPABILITY
    for row in maps:
        row["schema_version"] = "frc-squad2-v60-candidate-map-v1"
    for row in verifier_inputs:
        row["schema_version"] = "frc-squad2-v60-verifier-input-v1"
        row["dataset_id"] = DATASET_ID
    structural["schema_version"] = "frc-squad2-v60-structural-census-v1"
    return prepared, maps, verifier_inputs, structural


def build_deterministic_queries(
    prepared_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = v59.build_deterministic_queries(prepared_rows)
    for row in rows:
        row["schema_version"] = SCHEMA_VERSION
        row["dataset_id"] = DATASET_ID
    return rows


def validate_query_cache(
    prepared_rows: Sequence[dict[str, Any]], query_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if list(query_rows) != build_deterministic_queries(prepared_rows):
        raise ValueError("SQuAD2 v60 deterministic query cache changed")
    return {
        "rows": len(query_rows),
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "gold_fields_visible_to_generator": False,
        "deterministic_templates": True,
    }


class FrozenSquad2Scorer(v59.FrozenSquad2Scorer):
    """Frozen BGE scorer with v60 metadata labels."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
        return rows


def structured_prompt(row: dict[str, Any]) -> str:
    if v58._contains_forbidden_blind_key(row):
        raise ValueError("SQuAD2 v60 prompt received a forbidden field")
    return STRUCTURED_PROMPT.format(
        question=str(row["question"]), context=str(row["context"])
    )


def _strip_paired_wrapper(value: str) -> str:
    pairs = (('"', '"'), ("'", "'"), ("`", "`"))
    result = value.strip()
    for left, right in pairs:
        if len(result) >= 2 and result.startswith(left) and result.endswith(right):
            return result[len(left) : -len(right)].strip()
    return result


def parse_structured_support(text: str, context: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    first = lines[0]
    if re.match(r"^UNSUPPORTED\b", first, flags=re.IGNORECASE):
        return "UNSUPPORTED"
    candidate = re.sub(
        r"^(?:SPAN|ANSWER)\s*:\s*", "", first, count=1, flags=re.IGNORECASE
    )
    candidate = _strip_paired_wrapper(candidate)
    if candidate and candidate.casefold() in context.casefold():
        return "SUPPORTED_SPAN"
    quoted = re.findall(r'"([^"\r\n]+)"|\'([^\'\r\n]+)\'|`([^`\r\n]+)`', text)
    for groups in quoted:
        value = next((item.strip() for item in groups if item.strip()), "")
        if value and value.casefold() in context.casefold():
            return "SUPPORTED_SPAN"
    return None


class LocalQwenStructuredVerifier:
    """Frozen local deterministic tagged-span verifier."""

    def __init__(
        self,
        *,
        model_path: Path,
        batch_size: int = 16,
        max_new_tokens: int = 40,
        maximum_input_tokens: int = 3072,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
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

    def verify(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        output_path: Path,
        cache_key: str,
    ) -> list[dict[str, Any]]:
        existing = list(read_jsonl(output_path)) if output_path.is_file() else []
        expected_ids = [str(row["id"]) for row in rows]
        existing_ids = [str(row.get("id", "")) for row in existing]
        if existing_ids != expected_ids[: len(existing_ids)]:
            raise ValueError("SQuAD2 v60 verifier cache is not a valid prefix")
        if any(row.get("cache_key") != cache_key for row in existing):
            raise ValueError("SQuAD2 v60 verifier cache key changed")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if existing else "w"
        with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
            for start in range(len(existing), len(rows), self.batch_size):
                batch = list(rows[start : start + self.batch_size])
                prompts = [self._chat_prompt(structured_prompt(row)) for row in batch]
                encoded = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.maximum_input_tokens,
                ).to(self.model.device)
                with self.torch.inference_mode():
                    generated = self.model.generate(
                        **encoded,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                prompt_length = encoded.input_ids.shape[-1]
                decoded = self.tokenizer.batch_decode(
                    generated[:, prompt_length:], skip_special_tokens=True
                )
                for source, raw in zip(batch, decoded, strict=True):
                    value = raw.strip()
                    decision = parse_structured_support(value, str(source["context"]))
                    prediction = {
                        "schema_version": "frc-squad2-v60-structured-decision-v1",
                        "experiment_id": EXPERIMENT_ID,
                        "stage": str(source["stage"]),
                        "id": str(source["id"]),
                        "cache_key": cache_key,
                        "raw_prediction": value,
                        "raw_prediction_sha256": hashlib.sha256(
                            value.encode("utf-8")
                        ).hexdigest(),
                        "raw_prediction_codepoints": len(value),
                        "decision": decision,
                        "support_passed": decision == "SUPPORTED_SPAN",
                        "invalid_fail_closed_used": decision is None,
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
                    "SQuAD2 v60 cases",
                    flush=True,
                )
        return existing


def validate_verifier_cache(
    verifier_inputs: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    cache_key: str,
) -> dict[str, Any]:
    return v59.validate_verifier_cache(verifier_inputs, decisions, cache_key=cache_key)


def build_gold_rows(
    source: dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    structural_census: dict[str, Any],
) -> list[dict[str, Any]]:
    return v59.build_gold_rows(
        source,
        candidate_maps,
        scored_rows,
        structural_census=structural_census,
    )


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    value = v59.build_candidate_coverage(gold_rows, sampling)
    value["schema_version"] = "frc-squad2-v60-candidate-coverage-v1"
    value["experiment_id"] = EXPERIMENT_ID
    value["selected_prior_commitment_overlap"] = sampling[
        "selected_prior_commitment_overlap"
    ]
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
    report, evidence = v59.evaluate_stage(
        gold_rows,
        scored_rows,
        support_rows,
        query_summary,
        verifier_summary,
        source_artifacts,
        stage=stage,
    )
    report["schema_version"] = "frc-squad2-structured-span-report-v60"
    report["experiment_id"] = EXPERIMENT_ID
    report["metadata"]["dataset_id"] = DATASET_ID
    report["metadata"]["v58_case_level_artifact_reused"] = False
    report["metadata"]["v59_case_level_artifact_reused"] = False
    report["analysis"]["mechanism"] = (
        "v59-frozen downstream selectors under one shared tagged-span Qwen "
        "decision with pre-registered generic wrapper normalization"
    )
    report["analysis"]["support_checks"][
        "selected_prior_commitment_overlap_equals_0"
    ] = source_artifacts["sampling"]["selected_prior_commitment_overlap"] == 0
    supported = all(report["analysis"]["support_checks"].values())
    if stage == "development":
        status = (
            "SQUAD2_V60_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_CONFIRMATION"
            if supported
            else "SQUAD2_V60_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
        )
    else:
        status = (
            "SQUAD2_V60_STRUCTURED_SPAN_FRC_FEASIBILITY_ESTABLISHED"
            if supported
            else "SQUAD2_V60_STRUCTURED_SPAN_FRC_FEASIBILITY_NOT_ESTABLISHED"
        )
    outcome = report["analysis"]["outcome"]
    outcome["status"] = status
    outcome["support_established"] = supported
    outcome["confirmation_open_authorized"] = stage == "development" and supported
    return report, evidence


def write_report(
    report: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    json_path: Path,
    markdown_path: Path,
    evidence_path: Path,
) -> None:
    rounded = _round_for_display(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(rounded, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    analysis = rounded["analysis"]
    verifier = analysis["support_verifier"]
    candidate = analysis["aggregates"][CANDIDATE]
    lines = [
        f"# SQuAD 2.0 structured span gate ({rounded['metadata']['stage']}, v60)",
        "",
        f"- Status: `{analysis['outcome']['status']}`",
        f"- Cases/articles/paragraphs: {rounded['metadata']['cases']}/{rounded['metadata']['articles']}/{rounded['metadata']['paragraphs']}",
        f"- Verifier balanced accuracy: {verifier['balanced_accuracy']:.6f}",
        f"- Answer-bearing span pass rate: {verifier['answer_bearing_span_pass_rate']:.6f}",
        f"- No-answer rejection rate: {verifier['no_answer_rejection_rate']:.6f}",
        f"- Invalid-output rate: {verifier['invalid_output_rate']:.6f}",
        f"- Candidate utility F1: {candidate['answer_or_abstention_macro_f1']:.6f}",
        "- Selector adoption: `false`",
        "- Gate 2: `NO-GO/SHADOW`",
        "",
        "## Family comparisons",
        "",
    ]
    for name, value in analysis["family_comparison"].items():
        lines.append(
            f"- `{name}`: {value['point']:+.6f} "
            f"(95% CI [{value['ci_low']:+.6f}, {value['ci_high']:+.6f}])"
        )
    lines.extend(["", "## Support checks", ""])
    lines.extend(
        f"- `{name}`: `{str(value).lower()}`"
        for name, value in analysis["support_checks"].items()
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a third disjoint balanced SQuAD 2.0 mechanism sample. It is not official answer-string scoring, hidden-test evaluation, SetR reproduction, selector adoption, or flood-domain validation.",
            "",
        ]
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in evidence:
                compressed.write(
                    (
                        json.dumps(
                            _round_for_display(row),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )


def validate_protocol(protocol_path: Path) -> dict[str, Any]:
    if _sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("SQuAD2 v60 protocol hash changed")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    if value["prior_boundary"]["dev_content_opened"] is not False:
        raise ValueError("SQuAD2 v60 dev boundary changed")
    if value["frozen_structured_prompt"] != STRUCTURED_PROMPT:
        raise ValueError("SQuAD2 v60 structured prompt changed")
    if value["scope"]["expected_prior_exclusion_commitments"] != 1200:
        raise ValueError("SQuAD2 v60 exclusion count changed")
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
        != protocol["source_and_split"]["source_registration_sha256"]
    ):
        raise ValueError("SQuAD2 v60 source registration changed")
    return v59.validate_source_contract(
        source_registration_path,
        protocol_path=(
            protocol_path.resolve().parents[2]
            / "docs/progressive_upgrade/squad2_extractive_support_gate_protocol_v59.json"
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
        raise ValueError("SQuAD2 v60 implementation registration changed")
    if value.get("v60_sample_constructed_before_registration") is not False:
        raise ValueError("SQuAD2 v60 sample was constructed before freeze")
    if value.get("dev_content_read") is not False:
        raise ValueError("SQuAD2 v60 dev was opened before freeze")
    return value


__all__ = [
    "BUDGETS",
    "CANDIDATE",
    "EXPERIMENT_ID",
    "FrozenSquad2Scorer",
    "LocalQwenStructuredVerifier",
    "SOURCE_FILES",
    "STAGES",
    "STRUCTURED_PROMPT",
    "TARGET_CASES",
    "build_candidate_coverage",
    "build_deterministic_queries",
    "build_gold_rows",
    "evaluate_stage",
    "load_prior_exclusion_union",
    "parse_structured_support",
    "prepare_blind_cases",
    "read_stage_source",
    "select_disjoint_balanced_sample",
    "structured_prompt",
    "validate_implementation_registration",
    "validate_protocol",
    "validate_query_cache",
    "validate_source_contract",
    "validate_verifier_cache",
    "write_report",
]
