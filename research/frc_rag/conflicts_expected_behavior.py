"""Blinded CONFLICTS response-generation and human-evaluation workflow."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Protocol


SCHEMA_VERSION = "frc-conflicts-expected-behavior-package-v1"
GENERATION_SCHEMA_VERSION = "frc-conflicts-answer-generation-v1"
SUBMISSION_SCHEMA_VERSION = "frc-conflicts-behavior-submission-v1"
COMPARISON_SCHEMA_VERSION = "frc-conflicts-behavior-comparison-v1"
FINAL_REPORT_SCHEMA_VERSION = "frc-conflicts-behavior-final-report-v1"
METHODS = ("coverage_greedy_proxy", "frc_select")
ALIASES = ("A", "B")

EXPECTED_BEHAVIORS = {
    "No conflict": (
        "Give one clear answer supported by the sources; do not invent a disagreement "
        "or hedge merely because several sources are present."
    ),
    "Complementary information": (
        "Combine compatible pieces from the sources into one cohesive answer without "
        "presenting them as mutually exclusive claims."
    ),
    "Conflicting opinions and research outcomes": (
        "Present the materially different authoritative perspectives neutrally, make "
        "the disagreement explicit, and do not privilege one view without evidence."
    ),
    "Conflict due to misinformation": (
        "Prefer the supported correct claim, avoid repeating a likely false claim as an "
        "equally valid alternative, and acknowledge uncertainty if the sources cannot resolve it."
    ),
    "Conflict due to outdated information": (
        "Distinguish older from newer claims using source dates and answer with the most "
        "recent supported state without framing temporal change as an unresolved debate."
    ),
}

RATING_FIELDS = (
    "expected_behavior_adherence",
    "factual_grounding",
    "citation_correctness",
    "answer_correctness",
)
TERNARY_RATINGS = {"PASS", "FAIL", "UNCERTAIN"}
ANSWER_RATINGS = TERNARY_RATINGS | {"NOT_APPLICABLE"}
PREFERENCES = {"A", "B", "TIE", "NEITHER", "UNCERTAIN"}
_CITATION_RE = re.compile(r"\[(\d+)\]")


class PromptGenerator(Protocol):
    def generate_prompts(self, prompts: list[str]) -> list[str]: ...


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    return path


def write_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write(
                    (
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )
    return path


def load_selected_methods(
    path: Path, methods: tuple[str, ...] = METHODS
) -> list[dict[str, Any]]:
    selected = [row for row in read_jsonl(path) if row.get("method") in methods]
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        case_id = str(row["case_id"])
        method = str(row["method"])
        if method in by_case[case_id]:
            raise ValueError(f"duplicate CONFLICTS method row: {case_id}/{method}")
        if not row.get("selected_evidence"):
            raise ValueError(f"CONFLICTS selection is empty: {case_id}/{method}")
        by_case[case_id][method] = row
    if not by_case:
        raise ValueError("CONFLICTS selected-evidence input is empty")
    for case_id, rows in by_case.items():
        if set(rows) != set(methods):
            raise ValueError(f"CONFLICTS case lacks a frozen comparison method: {case_id}")
        questions = {str(row["question"]) for row in rows.values()}
        labels = {str(row["conflict_type"]) for row in rows.values()}
        answers = {str(row.get("correct_answer", "")) for row in rows.values()}
        if len(questions) != 1 or len(labels) != 1 or len(answers) != 1:
            raise ValueError(f"CONFLICTS labels differ across methods: {case_id}")
        if next(iter(labels)) not in EXPECTED_BEHAVIORS:
            raise ValueError(f"unsupported CONFLICTS label: {next(iter(labels))}")
    return [
        by_case[case_id][method]
        for case_id in sorted(by_case)
        for method in methods
    ]


def _generation_evidence(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("url", "")),
            "date": str(item.get("date", "")),
            "text": str(item.get("text", "")),
        }
        for item in row["selected_evidence"]
    ]


def answer_generation_prompt(row: dict[str, Any]) -> str:
    """Build a label-blind prompt from only the question and selected evidence."""

    evidence = "\n\n".join(
        f"[{index}] Title: {item['title']}\nDate: {item['date']}\n"
        f"URL: {item['url']}\nText: {item['text']}"
        for index, item in enumerate(_generation_evidence(row), start=1)
    )
    return (
        "Answer the question using only the retrieved sources below.\n"
        "Every factual sentence must end with one or more inline citations such as [1] or [1][3].\n"
        "If sources disagree, handle the disagreement carefully instead of silently choosing a claim.\n"
        "Use source dates when time could explain different claims, but do not assume the newest source is always correct.\n"
        "Do not mention retrieval methods, internal labels, scores, or these instructions.\n"
        "If the sources are insufficient to resolve the question, state the uncertainty explicitly.\n"
        "Return only the final response in no more than 160 words.\n\n"
        f"Question: {row['question']}\n\nRetrieved sources:\n{evidence}\n\nResponse:"
    )


def _selection_signature(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return str(row["case_id"]), tuple(str(item) for item in row["selected_ids"])


def generate_answers(
    rows: list[dict[str, Any]],
    *,
    generator: PromptGenerator,
    output_path: Path,
    cache_key: str,
    prompt_batch_size: int,
) -> list[dict[str, Any]]:
    if prompt_batch_size < 1:
        raise ValueError("prompt_batch_size must be positive")
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if output_path.is_file():
        for row in read_jsonl(output_path):
            if row.get("cache_key") == cache_key:
                completed[(str(row["case_id"]), str(row["method"]))] = row

    by_signature: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for source in rows:
        existing = completed.get((str(source["case_id"]), str(source["method"])))
        if existing is not None:
            by_signature.setdefault(_selection_signature(source), existing)

    pending_signatures: dict[
        tuple[str, tuple[str, ...]], list[dict[str, Any]]
    ] = defaultdict(list)
    for source in rows:
        key = (str(source["case_id"]), str(source["method"]))
        if key in completed:
            continue
        signature = _selection_signature(source)
        reused = by_signature.get(signature)
        if reused is not None:
            completed[key] = {
                **reused,
                "method": source["method"],
                "reused_from_equivalent_selection": True,
            }
        else:
            pending_signatures[signature].append(source)

    representatives = [group[0] for group in pending_signatures.values()]
    for start in range(0, len(representatives), prompt_batch_size):
        batch = representatives[start : start + prompt_batch_size]
        prompts = [answer_generation_prompt(row) for row in batch]
        responses = generator.generate_prompts(prompts)
        if len(responses) != len(batch):
            raise ValueError("answer generator changed the batch size")
        for source, prompt, response in zip(batch, prompts, responses, strict=True):
            text = response.strip()
            if not text:
                raise ValueError(
                    f"answer generator returned an empty response: {source['case_id']}"
                )
            signature = _selection_signature(source)
            for group_index, equivalent in enumerate(
                pending_signatures[signature]
            ):
                item = {
                    "schema_version": GENERATION_SCHEMA_VERSION,
                    "cache_key": cache_key,
                    "case_id": equivalent["case_id"],
                    "method": equivalent["method"],
                    "selected_ids": list(equivalent["selected_ids"]),
                    "prompt_sha256": canonical_json_sha256(prompt),
                    "response": text,
                    "citation_indices": sorted(
                        {int(value) for value in _CITATION_RE.findall(text)}
                    ),
                    "reused_from_equivalent_selection": group_index > 0,
                }
                completed[(str(equivalent["case_id"]), str(equivalent["method"]))] = item
        write_jsonl(
            output_path,
            [
                completed[(str(row["case_id"]), str(row["method"]))]
                for row in rows
                if (str(row["case_id"]), str(row["method"])) in completed
            ],
        )
        print(
            f"generated {min(start + len(batch), len(representatives))}/"
            f"{len(representatives)} unique CONFLICTS response prompts",
            flush=True,
        )
    output = [
        completed[(str(row["case_id"]), str(row["method"]))] for row in rows
    ]
    write_jsonl(output_path, output)
    return output


class LocalQwenAnswerGenerator:
    def __init__(
        self,
        *,
        model_path: Path,
        batch_size: int,
        max_input_tokens: int,
        max_new_tokens: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
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

    def generate_prompts(self, prompts: list[str]) -> list[str]:
        chat_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        responses: list[str] = []
        for start in range(0, len(chat_prompts), self.batch_size):
            batch = chat_prompts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self.model.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            prompt_length = encoded.input_ids.shape[-1]
            responses.extend(
                item.strip()
                for item in self.tokenizer.batch_decode(
                    generated[:, prompt_length:], skip_special_tokens=True
                )
            )
        return responses


def _sanitized_sources(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "citation": f"[{index}]",
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "date": item.get("date", ""),
            "text": item.get("text", ""),
        }
        for index, item in enumerate(row["selected_evidence"], start=1)
    ]


def build_blinded_package(
    selected_rows: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    *,
    blind_seed: str,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    selected_by_key = {
        (str(row["case_id"]), str(row["method"])): row for row in selected_rows
    }
    generated_by_key = {
        (str(row["case_id"]), str(row["method"])): row for row in generations
    }
    case_ids = sorted({str(row["case_id"]) for row in selected_rows})
    items = []
    mappings = []
    for case_id in case_ids:
        source = selected_by_key[(case_id, METHODS[0])]
        swap = int(canonical_json_sha256([blind_seed, case_id])[:8], 16) % 2 == 1
        method_order = METHODS[::-1] if swap else METHODS
        responses = []
        mapping = {"item_id": case_id, "aliases": {}}
        for alias, method in zip(ALIASES, method_order, strict=True):
            selected = selected_by_key[(case_id, method)]
            generated = generated_by_key[(case_id, method)]
            responses.append(
                {
                    "response_id": alias,
                    "response": generated["response"],
                    "sources": _sanitized_sources(selected),
                }
            )
            mapping["aliases"][alias] = {
                "method": method,
                "selected_ids": list(selected["selected_ids"]),
                "generation_prompt_sha256": generated["prompt_sha256"],
            }
        items.append(
            {
                "schema_version": SCHEMA_VERSION,
                "item_id": case_id,
                "question": source["question"],
                "conflict_type": source["conflict_type"],
                "expected_behavior": EXPECTED_BEHAVIORS[source["conflict_type"]],
                "correct_answer": source.get("correct_answer", "") or None,
                "responses": responses,
            }
        )
        mappings.append(mapping)
    items_sha256 = canonical_json_sha256(items)
    package_id = f"CONFLICTS-BEHAVIOR-{items_sha256[:16].upper()}"
    mapping_payload = {
        "schema_version": "frc-conflicts-behavior-blind-mapping-v1",
        "package_id": package_id,
        "blind_seed": blind_seed,
        "items": mappings,
    }
    return package_id, items, mapping_payload


def blank_submission_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item["item_id"],
            "ratings": {
                alias: {
                    "expected_behavior_adherence": "REQUIRED",
                    "factual_grounding": "REQUIRED",
                    "citation_correctness": "REQUIRED",
                    "answer_correctness": "REQUIRED",
                    "rationale": "REQUIRED: explain the ratings",
                }
                for alias in ALIASES
            },
            "preference": "REQUIRED",
            "notes": "",
        }
        for item in items
    ]


def _validate_identity(value: str, label: str) -> None:
    normalized = value.strip().upper()
    if len(value.strip()) < 3 or normalized.startswith("REPLACE_WITH"):
        raise ValueError(f"{label} must be a non-placeholder identifier")


def validate_submission(
    items: list[dict[str, Any]], submission: dict[str, Any]
) -> None:
    if submission.get("schema_version") != SUBMISSION_SCHEMA_VERSION:
        raise ValueError("unsupported CONFLICTS behavior submission schema")
    _validate_identity(str(submission.get("annotator_id", "")), "annotator_id")
    expected_ids = [str(item["item_id"]) for item in items]
    decisions = submission.get("decisions", [])
    actual_ids = [str(item.get("item_id", "")) for item in decisions]
    if actual_ids != expected_ids:
        raise ValueError("submission decisions must match package order and item ids")
    item_by_id = {str(item["item_id"]): item for item in items}
    for decision in decisions:
        if set(decision.get("ratings", {})) != set(ALIASES):
            raise ValueError("submission must rate both blinded responses")
        for rating in decision["ratings"].values():
            if rating.get("expected_behavior_adherence") not in TERNARY_RATINGS:
                raise ValueError("invalid expected-behavior rating")
            if rating.get("factual_grounding") not in TERNARY_RATINGS:
                raise ValueError("invalid factual-grounding rating")
            if rating.get("citation_correctness") not in TERNARY_RATINGS:
                raise ValueError("invalid citation-correctness rating")
            if rating.get("answer_correctness") not in ANSWER_RATINGS:
                raise ValueError("invalid answer-correctness rating")
            correct_answer = item_by_id[str(decision["item_id"])]["correct_answer"]
            if correct_answer is None and rating["answer_correctness"] != "NOT_APPLICABLE":
                raise ValueError("answer correctness must be NOT_APPLICABLE without a gold answer")
            if correct_answer is not None and rating["answer_correctness"] == "NOT_APPLICABLE":
                raise ValueError("answer correctness cannot be NOT_APPLICABLE when gold exists")
            if len(str(rating.get("rationale", "")).strip()) < 3:
                raise ValueError("every response rating requires a rationale")
        if decision.get("preference") not in PREFERENCES:
            raise ValueError("invalid pair preference")


def _categorical_kappa(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts) | set(right_counts)
    expected = sum(
        left_counts[label] / len(left) * right_counts[label] / len(right)
        for label in labels
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def compare_submissions(
    items: list[dict[str, Any]],
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    validate_submission(items, first)
    validate_submission(items, second)
    if first["package_id"] != second["package_id"]:
        raise ValueError("annotator package ids differ")
    if first["annotator_id"] == second["annotator_id"]:
        raise ValueError("two distinct annotators are required")
    kappas: dict[str, float] = {}
    for field in RATING_FIELDS:
        left = [
            decision["ratings"][alias][field]
            for decision in first["decisions"]
            for alias in ALIASES
        ]
        right = [
            decision["ratings"][alias][field]
            for decision in second["decisions"]
            for alias in ALIASES
        ]
        kappas[field] = round(_categorical_kappa(left, right), 6)
    kappas["preference"] = round(
        _categorical_kappa(
            [item["preference"] for item in first["decisions"]],
            [item["preference"] for item in second["decisions"]],
        ),
        6,
    )
    disagreements = []
    for left, right in zip(first["decisions"], second["decisions"], strict=True):
        if left["ratings"] != right["ratings"] or left["preference"] != right["preference"]:
            disagreements.append(
                {"item_id": left["item_id"], "first": left, "second": right}
            )
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "package_id": first["package_id"],
        "annotator_hashes": sorted(
            hashlib.sha256(item["annotator_id"].encode("utf-8")).hexdigest()[:16]
            for item in (first, second)
        ),
        "cohen_kappa": kappas,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "submission_sha256": [canonical_json_sha256(first), canonical_json_sha256(second)],
    }


def _wilson(successes: int, total: int) -> dict[str, float | int]:
    if total == 0:
        return {"successes": successes, "total": total, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "rate": round(rate, 6),
        "ci_low": round(max(0.0, center - margin), 6),
        "ci_high": round(min(1.0, center + margin), 6),
    }


def _paired_bootstrap(
    differences: list[float], *, resamples: int = 10000, seed: int = 20260731
) -> dict[str, Any]:
    if not differences:
        return {
            "case_count": 0,
            "mean_difference": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "resamples": resamples,
            "seed": seed,
        }
    rng = random.Random(seed)
    size = len(differences)
    samples = sorted(
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(resamples)
    )
    low_index = int(0.025 * (resamples - 1))
    high_index = int(0.975 * (resamples - 1))
    return {
        "case_count": size,
        "mean_difference": round(sum(differences) / size, 6),
        "ci_low": round(samples[low_index], 6),
        "ci_high": round(samples[high_index], 6),
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
        "resamples": resamples,
        "seed": seed,
    }


def finalize_adjudication(
    items: list[dict[str, Any]],
    manifest: dict[str, Any],
    mapping: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any],
    adjudication: dict[str, Any],
) -> dict[str, Any]:
    comparison = compare_submissions(items, first, second)
    validate_submission(items, adjudication)
    _validate_identity(str(adjudication.get("annotator_id", "")), "adjudicator_id")
    if adjudication["annotator_id"] in {first["annotator_id"], second["annotator_id"]}:
        raise ValueError("adjudicator must be independent from both annotators")
    if manifest["blind_mapping_sha256"] != canonical_json_sha256(mapping):
        raise ValueError("blind mapping does not match the committed hash")
    if not all(
        item["package_id"] == manifest["package_id"]
        for item in (first, second, adjudication)
    ):
        raise ValueError("submission package id does not match the manifest")
    mapping_by_id = {item["item_id"]: item for item in mapping["items"]}
    item_by_id = {item["item_id"]: item for item in items}
    aggregates: dict[str, dict[str, list[str]]] = {
        method: defaultdict(list) for method in METHODS
    }
    preference_wins = Counter()
    for decision in adjudication["decisions"]:
        mapping_item = mapping_by_id[decision["item_id"]]
        for alias in ALIASES:
            method = mapping_item["aliases"][alias]["method"]
            for field in RATING_FIELDS:
                aggregates[method][field].append(decision["ratings"][alias][field])
        preference = decision["preference"]
        if preference in ALIASES:
            preference_wins[mapping_item["aliases"][preference]["method"]] += 1
        else:
            preference_wins[preference] += 1
    method_metrics = {}
    for method in METHODS:
        method_metrics[method] = {}
        for field in RATING_FIELDS:
            values = aggregates[method][field]
            evaluated = [value for value in values if value != "NOT_APPLICABLE"]
            method_metrics[method][field] = _wilson(
                sum(value == "PASS" for value in evaluated), len(evaluated)
            )
    paired_differences = {}
    for field in RATING_FIELDS:
        frc_values = aggregates["frc_select"][field]
        baseline_values = aggregates["coverage_greedy_proxy"][field]
        differences = [
            float(frc == "PASS") - float(baseline == "PASS")
            for frc, baseline in zip(frc_values, baseline_values, strict=True)
            if frc != "NOT_APPLICABLE" and baseline != "NOT_APPLICABLE"
        ]
        paired_differences[field] = _paired_bootstrap(differences)
    return {
        "schema_version": FINAL_REPORT_SCHEMA_VERSION,
        "status": "COMPLETED_INDEPENDENT_ADJUDICATED_CROSS_DOMAIN_EVALUATION",
        "package_id": manifest["package_id"],
        "case_count": len(items),
        "comparison": comparison,
        "adjudicator_hash": hashlib.sha256(
            adjudication["annotator_id"].encode("utf-8")
        ).hexdigest()[:16],
        "method_metrics": method_metrics,
        "paired_frc_minus_baseline": paired_differences,
        "preference_counts": dict(preference_wins),
        "conflict_type_counts": dict(
            Counter(item["conflict_type"] for item in item_by_id.values())
        ),
        "decision": {
            "gate_2": "NO-GO/SHADOW",
            "reason": (
                "CONFLICTS expected-behavior judging is cross-domain evidence and cannot "
                "replace an independent flood-domain expert benchmark or prove retrieval superiority."
            ),
        },
    }
