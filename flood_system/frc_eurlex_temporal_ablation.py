from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from flood_system.frc_conflicts_evaluation import (
    approximate_token_count,
    bm25_scores,
    local_model_snapshot,
    minmax,
    read_jsonl,
    stable_hash,
    write_jsonl,
)
from flood_system.frc_public_evidence import paired_bootstrap


EURLEX_SOURCE_SCHEMA = "eurlex-effective-expiry-source-v1"
EURLEX_ABLATION_SCHEMA = "frc-eurlex-effective-expiry-ablation-v1"
EURLEX_SOURCE_NAME = "EU Publications Office CELLAR and EUR-Lex"
EURLEX_SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"
EURLEX_CONTENT_URL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"
)
EURLEX_REUSE_NOTICE = (
    "https://eur-lex.europa.eu/content/help/data-reuse/"
    "reuse-contents-eurlex-details.html?locale=en"
)
EURLEX_ELI_DOCUMENTATION = (
    "https://eur-lex.europa.eu/eli-register/eu_publications_office.html"
)
EURLEX_SNAPSHOT_DATE = "2026-07-14"

DEFAULT_SEED = 20260714
DEFAULT_FAMILY_QUOTAS = {"decision": 12, "directive": 6, "regulation": 12}
DEFAULT_DISTRACTOR_PAIRS = 2
DEFAULT_TOP_K = 1
DEFAULT_TOKEN_BUDGET = 512
DEFAULT_EVIDENCE_CHARS = 1800

EURLEX_METHODS = (
    "bm25_top1",
    "cross_encoder_top1",
    "applicability_filtered_cross_encoder_top1",
    "frc_full",
    "w/o_applicability",
)

EURLEX_SPARQL_QUERY = """PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT DISTINCT
  ?old ?oldcelex ?oldtype ?olddateforce ?olddateend
  ?new ?newcelex ?newtype ?newdateforce ?newdateend
WHERE {
  ?new cdm:resource_legal_repeals_resource_legal ?old ;
       cdm:resource_legal_id_celex ?newcelex ;
       cdm:work_has_resource-type ?newtype ;
       cdm:resource_legal_date_entry-into-force ?newdateforce ;
       cdm:resource_legal_date_end-of-validity ?newdateend .
  ?old cdm:resource_legal_id_celex ?oldcelex ;
       cdm:work_has_resource-type ?oldtype ;
       cdm:resource_legal_date_entry-into-force ?olddateforce ;
       cdm:resource_legal_date_end-of-validity ?olddateend .
  FILTER(STRSTARTS(STR(?oldcelex), "3"))
  FILTER(STRSTARTS(STR(?newcelex), "3"))
  FILTER(?olddateforce >= "1950-01-01"^^xsd:date)
  FILTER(?newdateforce >= "2015-01-01"^^xsd:date)
  FILTER(?newdateforce < "2026-01-01"^^xsd:date)
  FILTER(?newdateforce = (?olddateend + "P1D"^^xsd:dayTimeDuration))
  FILTER(?newdateend >= ?newdateforce)
}
ORDER BY DESC(?newdateforce)
LIMIT 500
"""

_TYPE_FAMILIES = {
    "DEC": "decision",
    "DEC_DEL": "decision",
    "DEC_IMPL": "decision",
    "DIR": "directive",
    "DIR_DEL": "directive",
    "DIR_IMPL": "directive",
    "REG": "regulation",
    "REG_DEL": "regulation",
    "REG_IMPL": "regulation",
}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
_TOPIC_STOPWORDS = {
    "amending",
    "and",
    "commission",
    "council",
    "decision",
    "delegated",
    "directive",
    "european",
    "for",
    "implementing",
    "of",
    "on",
    "parliament",
    "recast",
    "regulation",
    "repealing",
    "the",
    "to",
    "union",
    "with",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class _EurLexHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._suppressed = 0
        self._title_depth = 0
        self._paragraph_depth = 0
        self._title_parts: list[str] = []
        self._metadata_title = ""
        self._fallback_title_parts: list[str] = []
        self._fallback_title_current = False
        self._fallback_title_closed = False
        self._paragraph_parts: list[str] = []
        self._current_paragraph: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag in {"script", "style"}:
            self._suppressed += 1
            return
        if self._suppressed:
            return
        attributes = dict(attrs)
        if tag == "meta" and str(attributes.get("name") or "").lower() == "dc.description":
            self._metadata_title = canonical_space(
                str(attributes.get("content") or "")
            )
        classes = set(str(attributes.get("class") or "").split())
        if "eli-main-title" in classes:
            self._title_depth += 1
        elif self._title_depth and tag == "div":
            self._title_depth += 1
        if tag == "p":
            self._fallback_title_current = bool(
                not self._fallback_title_closed
                and classes.intersection({"doc-ti", "oj-doc-ti"})
            )
            if (
                self._fallback_title_parts
                and not self._fallback_title_current
                and not self._title_depth
            ):
                self._fallback_title_closed = True
            self._paragraph_depth += 1
            if self._paragraph_depth == 1:
                self._current_paragraph = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._suppressed:
            self._suppressed -= 1
            return
        if self._suppressed:
            return
        if tag == "p" and self._paragraph_depth:
            self._paragraph_depth -= 1
            if self._paragraph_depth == 0:
                paragraph = canonical_space(" ".join(self._current_paragraph))
                if paragraph:
                    self._paragraph_parts.append(paragraph)
                    if self._fallback_title_current:
                        self._fallback_title_parts.append(paragraph)
                self._current_paragraph = []
                self._fallback_title_current = False
        if tag == "div" and self._title_depth:
            self._title_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._suppressed:
            return
        value = canonical_space(data)
        if not value:
            return
        if self._title_depth:
            self._title_parts.append(value)
        if self._paragraph_depth:
            self._current_paragraph.append(value)

    def result(self) -> dict[str, str]:
        title = canonical_space(
            " ".join(self._title_parts or self._fallback_title_parts)
            or self._metadata_title
        )
        paragraphs = []
        for paragraph in self._paragraph_parts:
            if not paragraphs or paragraphs[-1] != paragraph:
                paragraphs.append(paragraph)
        text = "\n".join(paragraphs)
        if not title or len(text) < 200:
            raise ValueError("EUR-Lex HTML did not contain a usable English legal text")
        return {"title": title, "text": text}


def extract_eurlex_html(value: str) -> dict[str, str]:
    parser = _EurLexHTMLParser()
    parser.feed(value)
    parser.close()
    return parser.result()


def _binding_value(row: dict[str, Any], name: str) -> str:
    value = row.get(name, {}).get("value")
    if not isinstance(value, str) or not value:
        raise ValueError(f"CELLAR row is missing {name}")
    return value


def _resource_type(value: str) -> str:
    return value.rsplit("/", 1)[-1]


def parse_eurlex_repeal_pairs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    bindings = payload.get("results", {}).get("bindings")
    if not isinstance(bindings, list):
        raise ValueError("CELLAR SPARQL response has no bindings")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in bindings:
        old_celex = _binding_value(row, "oldcelex")
        new_celex = _binding_value(row, "newcelex")
        old_type = _resource_type(_binding_value(row, "oldtype"))
        new_type = _resource_type(_binding_value(row, "newtype"))
        family = _TYPE_FAMILIES.get(new_type)
        if family is None or old_type not in _TYPE_FAMILIES:
            continue
        old_from = date.fromisoformat(_binding_value(row, "olddateforce"))
        old_to = date.fromisoformat(_binding_value(row, "olddateend"))
        new_from = date.fromisoformat(_binding_value(row, "newdateforce"))
        new_to = date.fromisoformat(_binding_value(row, "newdateend"))
        if old_from < date(1950, 1, 1):
            continue
        if not date(2015, 1, 1) <= new_from < date(2026, 1, 1):
            continue
        if old_to + timedelta(days=1) != new_from:
            continue
        if old_from > old_to or new_from > new_to:
            continue
        grouped[(old_celex, new_celex)].append(
            {
                "old_celex": old_celex,
                "new_celex": new_celex,
                "old_cellar_uri": _binding_value(row, "old"),
                "new_cellar_uri": _binding_value(row, "new"),
                "old_type": old_type,
                "new_type": new_type,
                "family": family,
                "old_from": old_from,
                "old_to": old_to,
                "new_from": new_from,
                "new_to": new_to,
            }
        )
    pairs = []
    for rows in grouped.values():
        boundaries = {(row["old_to"], row["new_from"]) for row in rows}
        if len(boundaries) != 1:
            continue
        canonical = min(rows, key=lambda row: (row["old_from"], row["new_to"]))
        old_from = min(row["old_from"] for row in rows)
        new_to = max(row["new_to"] for row in rows)
        if (canonical["old_to"] - old_from).days < 30:
            continue
        pairs.append(
            {
                "pair_id": f"{canonical['old_celex']}--{canonical['new_celex']}",
                "family": canonical["family"],
                "old": {
                    "celex": canonical["old_celex"],
                    "cellar_uri": canonical["old_cellar_uri"],
                    "resource_type": canonical["old_type"],
                    "valid_from": old_from.isoformat(),
                    "valid_to": canonical["old_to"].isoformat(),
                    "content_url": EURLEX_CONTENT_URL.format(
                        celex=canonical["old_celex"]
                    ),
                },
                "new": {
                    "celex": canonical["new_celex"],
                    "cellar_uri": canonical["new_cellar_uri"],
                    "resource_type": canonical["new_type"],
                    "valid_from": canonical["new_from"].isoformat(),
                    "valid_to": new_to.isoformat(),
                    "content_url": EURLEX_CONTENT_URL.format(
                        celex=canonical["new_celex"]
                    ),
                },
            }
        )
    return sorted(pairs, key=lambda row: row["pair_id"])


def select_eurlex_source_pairs(
    pairs: Iterable[dict[str, Any]],
    *,
    family_quotas: dict[str, int] | None = None,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    quotas = dict(family_quotas or DEFAULT_FAMILY_QUOTAS)
    selected = []
    for family in sorted(quotas):
        family_rows = [row for row in pairs if row["family"] == family]
        ranked = sorted(
            family_rows,
            key=lambda row: (
                sha256_text(f"{seed}:{row['pair_id']}"),
                row["pair_id"],
            ),
        )
        requested = int(quotas[family])
        if len(ranked) < requested:
            raise ValueError(
                f"EUR-Lex family {family} has {len(ranked)} pairs, needs {requested}"
            )
        selected.extend(ranked[:requested])
    return sorted(selected, key=lambda row: row["pair_id"])


def build_eurlex_source_manifest(
    selected_pairs: list[dict[str, Any]],
    *,
    documents_dir: Path,
    query_result_path: Path,
    seed: int = DEFAULT_SEED,
    family_quotas: dict[str, int] | None = None,
) -> dict[str, Any]:
    pairs = []
    for pair in selected_pairs:
        output_pair = {
            "pair_id": pair["pair_id"],
            "family": pair["family"],
        }
        for side in ("old", "new"):
            source = dict(pair[side])
            document_path = documents_dir / f"{source['celex']}.json"
            if not document_path.is_file():
                raise FileNotFoundError(document_path)
            document = json.loads(document_path.read_text(encoding="utf-8"))
            if document.get("celex") != source["celex"]:
                raise ValueError(f"EUR-Lex document ID mismatch: {document_path}")
            text_hash = sha256_text(str(document.get("text") or ""))
            if text_hash != document.get("canonical_text_sha256"):
                raise ValueError(f"EUR-Lex canonical text hash mismatch: {document_path}")
            output_pair[side] = {
                **source,
                "title": document["title"],
                "raw_html_sha256": document["raw_html_sha256"],
                "canonical_text_sha256": text_hash,
            }
        pairs.append(output_pair)
    quotas = dict(family_quotas or DEFAULT_FAMILY_QUOTAS)
    return {
        "schema_version": EURLEX_SOURCE_SCHEMA,
        "metadata": {
            "source": EURLEX_SOURCE_NAME,
            "snapshot_date": EURLEX_SNAPSHOT_DATE,
            "language": "ENG",
            "sparql_endpoint": EURLEX_SPARQL_ENDPOINT,
            "sparql_query_sha256": sha256_text(EURLEX_SPARQL_QUERY),
            "sparql_result_sha256": sha256(query_result_path),
            "reuse_notice": EURLEX_REUSE_NOTICE,
            "eli_documentation": EURLEX_ELI_DOCUMENTATION,
            "selection_seed": seed,
            "family_quotas": quotas,
            "selection_policy": (
                "stable hash within decision/directive/regulation families; only one-day "
                "adjacent repeal boundaries with a unique effective-expiry boundary and "
                "at least 30 days of prior validity"
            ),
        },
        "pairs": pairs,
    }


def load_eurlex_source(
    manifest_path: Path, documents_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EURLEX_SOURCE_SCHEMA:
        raise ValueError("unsupported EUR-Lex source manifest")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("EUR-Lex source manifest contains no pairs")
    loaded = []
    for pair in pairs:
        item = {"pair_id": pair["pair_id"], "family": pair["family"]}
        for side in ("old", "new"):
            expected = pair[side]
            document_path = documents_dir / f"{expected['celex']}.json"
            document = json.loads(document_path.read_text(encoding="utf-8"))
            actual_hash = sha256_text(str(document.get("text") or ""))
            if actual_hash != expected["canonical_text_sha256"]:
                raise ValueError(f"EUR-Lex source drift: {expected['celex']}")
            if document.get("raw_html_sha256") != expected["raw_html_sha256"]:
                raise ValueError(f"EUR-Lex raw source drift: {expected['celex']}")
            item[side] = {
                **expected,
                "text": document["text"],
            }
        loaded.append(item)
    return manifest, loaded


def _topic_from_titles(old_title: str, new_title: str) -> str:
    old_words = [word.lower() for word in _WORD_RE.findall(old_title)]
    new_words = [word.lower() for word in _WORD_RE.findall(new_title)]
    shared = set(old_words) & set(new_words)
    topic = []
    for word in new_words + old_words:
        if word in _TOPIC_STOPWORDS or len(word) < 4:
            continue
        if word in shared and word not in topic:
            topic.append(word)
    if len(topic) < 3:
        for word in new_words + old_words:
            if word in _TOPIC_STOPWORDS or len(word) < 4 or word in topic:
                continue
            topic.append(word)
            if len(topic) >= 10:
                break
    return " ".join(topic[:12])


def _candidate(side: dict[str, Any], *, origin: str) -> dict[str, Any]:
    text = canonical_space(f"{side['title']} {side['text']}")[:DEFAULT_EVIDENCE_CHARS]
    return {
        "id": side["celex"],
        "celex": side["celex"],
        "title": side["title"],
        "text": text,
        "token_count": approximate_token_count(text),
        "valid_from": side["valid_from"],
        "valid_to": side["valid_to"],
        "source_url": side["content_url"],
        "origin": origin,
    }


def build_eurlex_temporal_cases(
    pairs: list[dict[str, Any]],
    *,
    distractor_pairs: int = DEFAULT_DISTRACTOR_PAIRS,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    if len(pairs) <= distractor_pairs:
        raise ValueError("EUR-Lex source has too few pairs for distractors")
    cases = []
    for pair in pairs:
        other_pairs = sorted(
            [row for row in pairs if row["pair_id"] != pair["pair_id"]],
            key=lambda row: (
                sha256_text(f"{seed}:{pair['pair_id']}:{row['pair_id']}"),
                row["pair_id"],
            ),
        )[:distractor_pairs]
        topic = _topic_from_titles(pair["old"]["title"], pair["new"]["title"])
        snapshots = (
            ("last_valid_day", pair["old"]["valid_to"], "old", "new"),
            ("first_effective_day", pair["new"]["valid_from"], "new", "old"),
        )
        for boundary, as_of_date, gold_side, superseded_side in snapshots:
            candidates = [
                _candidate(pair["old"], origin="target_old"),
                _candidate(pair["new"], origin="target_new"),
            ]
            for distractor in other_pairs:
                candidates.extend(
                    [
                        _candidate(distractor["old"], origin="distractor_old"),
                        _candidate(distractor["new"], origin="distractor_new"),
                    ]
                )
            candidates = sorted(candidates, key=lambda item: item["id"])
            gold_id = pair[gold_side]["celex"]
            cases.append(
                {
                    "id": f"{pair['pair_id']}::{boundary}",
                    "pair_id": pair["pair_id"],
                    "family": pair["family"],
                    "boundary": boundary,
                    "as_of_date": as_of_date,
                    "question": (
                        "Select the governing EU legal evidence for the following subject "
                        f"as applicable on {as_of_date}: {topic}."
                    ),
                    "gold_evidence_id": gold_id,
                    "superseded_pair_evidence_id": pair[superseded_side]["celex"],
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                }
            )
    expected = len(pairs) * 2
    if len(cases) != expected:
        raise ValueError(f"built {len(cases)} of {expected} EUR-Lex cases")
    return sorted(cases, key=lambda row: row["id"])


class EurLexRealScorer:
    def __init__(
        self,
        *,
        reranker_model: str,
        device: str,
        hf_home: Path,
        rerank_batch_size: int = 16,
    ) -> None:
        import os

        os.environ["HF_HOME"] = str(hf_home)
        os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "hub")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import CrossEncoder

        self.rerank_batch_size = rerank_batch_size
        self.reranker = CrossEncoder(
            str(local_model_snapshot(hf_home, reranker_model)), device=device
        )

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pairs = [
            (str(case["question"]), str(candidate["text"]))
            for case in cases
            for candidate in case["candidates"]
        ]
        raw = self.reranker.predict(
            pairs,
            batch_size=self.rerank_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        output = []
        offset = 0
        for case in cases:
            candidates = case["candidates"]
            lexical = bm25_scores(
                str(case["question"]),
                [str(candidate["text"]) for candidate in candidates],
            )
            cross = minmax(raw[offset : offset + len(candidates)])
            offset += len(candidates)
            output.append(
                {
                    **case,
                    "candidates": [
                        {
                            **candidate,
                            "scores": {
                                "bm25": float(lexical[index]),
                                "cross_encoder": float(cross[index]),
                            },
                        }
                        for index, candidate in enumerate(candidates)
                    ],
                }
            )
        print(f"reranked {len(pairs)} EUR-Lex candidate pairs", flush=True)
        return output


def score_eurlex_cases(
    cases: list[dict[str, Any]],
    *,
    output_path: Path,
    metadata_path: Path,
    source_manifest_sha256: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    case_signature = stable_hash(
        [
            {
                "id": case["id"],
                "question": case["question"],
                "gold_evidence_id": case["gold_evidence_id"],
                "candidates": [
                    {
                        "id": candidate["id"],
                        "text": candidate["text"],
                        "token_count": candidate["token_count"],
                        "valid_from": candidate["valid_from"],
                        "valid_to": candidate["valid_to"],
                    }
                    for candidate in case["candidates"]
                ],
            }
            for case in cases
        ]
    )
    expected = {
        "source_manifest_sha256": source_manifest_sha256,
        "case_signature": case_signature,
        "config": config,
        "cases": len(cases),
    }
    if output_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata == expected:
            cached = list(read_jsonl(output_path))
            if len(cached) == len(cases):
                return cached
    scorer = EurLexRealScorer(
        reranker_model=str(config["reranker_model"]),
        device=str(config["device"]),
        hf_home=Path(str(config["hf_home"])),
        rerank_batch_size=int(config["rerank_batch_size"]),
    )
    scored = scorer.score_cases(cases)
    write_jsonl(output_path, scored)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return scored


def _is_applicable(case: dict[str, Any], candidate: dict[str, Any]) -> bool:
    as_of = date.fromisoformat(case["as_of_date"])
    valid_from = date.fromisoformat(candidate["valid_from"])
    valid_to = date.fromisoformat(candidate["valid_to"])
    return valid_from <= as_of <= valid_to


def select_eurlex_evidence(
    case: dict[str, Any],
    method: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    if method not in EURLEX_METHODS:
        raise ValueError(f"unsupported EUR-Lex method: {method}")
    candidates = list(case["candidates"])
    if method in {"applicability_filtered_cross_encoder_top1", "frc_full"}:
        candidates = [item for item in candidates if _is_applicable(case, item)]
    score_name = "bm25" if method == "bm25_top1" else "cross_encoder"
    ranked = sorted(
        candidates,
        key=lambda item: (-float(item["scores"][score_name]), str(item["id"])),
    )
    selected = []
    token_cost = 0
    for candidate in ranked:
        candidate_cost = int(candidate.get("token_count", 1))
        if token_cost + candidate_cost > token_budget:
            continue
        selected.append(candidate)
        token_cost += candidate_cost
        if len(selected) == top_k:
            break
    return selected


def select_eurlex_methods(
    cases: Iterable[dict[str, Any]],
    *,
    methods: Iterable[str] = EURLEX_METHODS,
    top_k: int = DEFAULT_TOP_K,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        for method in methods:
            selected = select_eurlex_evidence(
                case, method, top_k=top_k, token_budget=token_budget
            )
            rows.append(
                {
                    "case_id": case["id"],
                    "pair_id": case["pair_id"],
                    "family": case["family"],
                    "boundary": case["boundary"],
                    "as_of_date": case["as_of_date"],
                    "gold_evidence_id": case["gold_evidence_id"],
                    "superseded_pair_evidence_id": case[
                        "superseded_pair_evidence_id"
                    ],
                    "method": method,
                    "candidate_count": len(case["candidates"]),
                    "selected_ids": [item["id"] for item in selected],
                    "selected_evidence": selected,
                }
            )
    return rows


def _score_row(row: dict[str, Any]) -> dict[str, Any]:
    selected = row["selected_evidence"]
    selected_first = selected[0] if selected else {}
    applicable = bool(selected_first) and (
        date.fromisoformat(selected_first["valid_from"])
        <= date.fromisoformat(row["as_of_date"])
        <= date.fromisoformat(selected_first["valid_to"])
    )
    return {
        "case_id": row["case_id"],
        "pair_id": row["pair_id"],
        "family": row["family"],
        "boundary": row["boundary"],
        "as_of_date": row["as_of_date"],
        "method": row["method"],
        "gold_evidence_id": row["gold_evidence_id"],
        "selected_ids": row["selected_ids"],
        "metrics": {
            "exact_evidence_accuracy": float(
                selected_first.get("id") == row["gold_evidence_id"]
            ),
            "validity_accuracy": float(applicable),
            "invalid_applicability_rate": float(not applicable),
            "wrong_boundary_version_rate": float(
                selected_first.get("id") == row["superseded_pair_evidence_id"]
            ),
            "token_cost": sum(
                int(item.get("token_count", 1)) for item in selected
            ),
        },
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    metric_names = tuple(rows[0]["metrics"]) if rows else ()
    return {
        "cases": len(rows),
        **{
            name: round(
                sum(float(row["metrics"][name]) for row in rows) / len(rows), 6
            )
            for name in metric_names
        },
    }


def _paired_comparison(
    by_method: dict[str, list[dict[str, Any]]], left: str, right: str
) -> dict[str, Any]:
    left_rows = {row["case_id"]: row for row in by_method[left]}
    right_rows = {row["case_id"]: row for row in by_method[right]}
    if set(left_rows) != set(right_rows):
        raise ValueError(f"paired EUR-Lex cases differ for {left} and {right}")
    metrics = (
        "exact_evidence_accuracy",
        "validity_accuracy",
        "invalid_applicability_rate",
        "wrong_boundary_version_rate",
    )
    return {
        "left": left,
        "right": right,
        "direction": "left_minus_right",
        "metrics": {
            metric: paired_bootstrap(
                [
                    float(left_rows[case_id]["metrics"][metric])
                    - float(right_rows[case_id]["metrics"][metric])
                    for case_id in sorted(left_rows)
                ],
                seed=DEFAULT_SEED,
            )
            for metric in metrics
        },
    }


def build_eurlex_ablation_report(
    *,
    cases: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    config: dict[str, Any],
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
    scored_path: Path,
) -> dict[str, Any]:
    results = [_score_row(row) for row in selected_rows]
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_method[row["method"]].append(row)
    if set(by_method) != set(EURLEX_METHODS):
        raise ValueError("EUR-Lex method coverage is incomplete")
    aggregates = {method: _aggregate(rows) for method, rows in by_method.items()}
    baselines = (
        "bm25_top1",
        "cross_encoder_top1",
        "applicability_filtered_cross_encoder_top1",
    )
    strongest_baseline = max(
        baselines,
        key=lambda method: (
            float(aggregates[method]["exact_evidence_accuracy"]), method
        ),
    )
    paired = {
        "full_minus_w_o_applicability": _paired_comparison(
            by_method, "frc_full", "w/o_applicability"
        ),
        "full_minus_strongest_baseline": _paired_comparison(
            by_method, "frc_full", strongest_baseline
        ),
    }
    full_by_case = {row["case_id"]: row for row in by_method["frc_full"]}
    without_by_case = {
        row["case_id"]: row for row in by_method["w/o_applicability"]
    }
    selection_changed = sum(
        full_by_case[case_id]["selected_ids"]
        != without_by_case[case_id]["selected_ids"]
        for case_id in full_by_case
    )
    by_boundary = []
    for boundary in sorted({case["boundary"] for case in cases}):
        row_groups = {
            method: [row for row in rows if row["boundary"] == boundary]
            for method, rows in by_method.items()
        }
        by_boundary.append(
            {
                "boundary": boundary,
                "cases": len(row_groups["frc_full"]),
                "frc_full_exact_evidence_accuracy": _aggregate(
                    row_groups["frc_full"]
                )["exact_evidence_accuracy"],
                "w_o_applicability_exact_evidence_accuracy": _aggregate(
                    row_groups["w/o_applicability"]
                )["exact_evidence_accuracy"],
            }
        )
    by_family = []
    for family in sorted({case["family"] for case in cases}):
        row_groups = {
            method: [row for row in rows if row["family"] == family]
            for method, rows in by_method.items()
        }
        by_family.append(
            {
                "family": family,
                "cases": len(row_groups["frc_full"]),
                "frc_full_exact_evidence_accuracy": _aggregate(
                    row_groups["frc_full"]
                )["exact_evidence_accuracy"],
                "w_o_applicability_exact_evidence_accuracy": _aggregate(
                    row_groups["w/o_applicability"]
                )["exact_evidence_accuracy"],
            }
        )
    baseline_delta = paired["full_minus_strongest_baseline"]["metrics"][
        "exact_evidence_accuracy"
    ]
    return {
        "schema_version": EURLEX_ABLATION_SCHEMA,
        "dataset": EURLEX_SOURCE_NAME,
        "status": "RUN_PUBLIC_OFFICIAL_EFFECTIVE_EXPIRY_REAL_MODEL",
        "metadata": {
            "public_dataset": True,
            "official_temporal_metadata": True,
            "real_model_scores": True,
            "source_snapshot_date": source_manifest["metadata"]["snapshot_date"],
            "source_manifest_sha256": sha256(source_manifest_path),
            "source_query_sha256": source_manifest["metadata"][
                "sparql_query_sha256"
            ],
            "source_reuse_notice": EURLEX_REUSE_NOTICE,
            "scored_cases_sha256": sha256(scored_path),
            "pair_count": len(source_manifest["pairs"]),
            "case_count": len(cases),
            "candidate_occurrences": sum(
                len(case["candidates"]) for case in cases
            ),
            "family_counts": {
                family: sum(
                    pair["family"] == family for pair in source_manifest["pairs"]
                )
                for family in sorted({pair["family"] for pair in source_manifest["pairs"]})
            },
            "methods": list(EURLEX_METHODS),
            "models": {"reranker": config["reranker_model"]},
            "selection_parameters": {
                "top_k": config["top_k"],
                "token_budget": config["token_budget"],
                "distractor_pairs": config["distractor_pairs"],
                "seed": config["seed"],
            },
            "frozen_protocol": (
                "thirty CELEX repeal pairs with a unique one-day adjacent expiry/effective "
                "boundary; evaluate the last valid day of the old act and first effective "
                "day of the new act; add two deterministic unrelated repeal-pair distractors"
            ),
            "gold_usage": (
                "CELEX repeal relation and applicability dates determine labels after retrieval; "
                "the scorer sees only the natural-language question and candidate legal text, "
                "while frc_full and the fair filtered baseline may use validity metadata"
            ),
        },
        "coverage": {
            "effective_dates": "IDENTIFIABLE_OFFICIAL_CELLAR_METADATA",
            "expiry_dates": "IDENTIFIABLE_OFFICIAL_CELLAR_METADATA",
            "adjacent_repeal_boundary": "IDENTIFIABLE",
            "old_and_new_boundary_snapshots": "RUN",
        },
        "aggregates": aggregates,
        "by_boundary": by_boundary,
        "by_family": by_family,
        "strongest_baseline_by_exact_evidence_accuracy": strongest_baseline,
        "paired_comparisons": paired,
        "selection_changed_cases": selection_changed,
        "decision": {
            "full_strictly_better_than_w_o_applicability": (
                paired["full_minus_w_o_applicability"]["metrics"][
                    "exact_evidence_accuracy"
                ]["mean_difference"]
                > 0.0
            ),
            "full_exact_gain_over_strongest_baseline_at_least_0_05": (
                float(baseline_delta["mean_difference"]) >= 0.05
            ),
            "gate_2": "NO-GO",
            "reason": (
                "Official effective and expiry dates make temporal applicability identifiable, "
                "but the fair applicability-filtered Cross-Encoder baseline receives the same "
                "metadata and the corpus is EU law rather than flood-response evidence."
            ),
        },
        "limitations": [
            "The corpus covers EU legal acts and does not establish performance on district flood-response documents.",
            "CELLAR may expose multiple partial-application dates; the frozen slice keeps only repeal pairs with one unique adjacent expiry/effective boundary.",
            "EUR-Lex legal texts and consolidated representations are reused for research and are not legal advice or an official legal edition claim.",
            "The experiment evaluates evidence selection at the document boundary, not answer generation or article-level expected-behavior adherence.",
            "FRC and the fair filtered Cross-Encoder baseline intentionally share validity metadata, preventing an unfair metadata advantage.",
        ],
        "case_results": results,
    }


def render_eurlex_ablation_markdown(report: dict[str, Any]) -> str:
    full = report["aggregates"]["frc_full"]
    without = report["aggregates"]["w/o_applicability"]
    paired = report["paired_comparisons"]["full_minus_w_o_applicability"][
        "metrics"
    ]["exact_evidence_accuracy"]
    lines = [
        "# EUR-Lex effective/expiry applicability ablation",
        "",
        f"- Status: `{report['status']}`",
        f"- Gate 2: `{report['decision']['gate_2']}`",
        f"- Source snapshot: `{report['metadata']['source_snapshot_date']}`",
        f"- Pairs / cases: {report['metadata']['pair_count']} / {report['metadata']['case_count']}",
        "",
        "## Aggregate results",
        "",
        "| Method | Exact evidence | Validity accuracy | Invalid applicability | Wrong boundary version |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, values in report["aggregates"].items():
        lines.append(
            f"| `{method}` | {values['exact_evidence_accuracy']:.6f} | "
            f"{values['validity_accuracy']:.6f} | "
            f"{values['invalid_applicability_rate']:.6f} | "
            f"{values['wrong_boundary_version_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired applicability effect",
            "",
            f"Full exact evidence accuracy is {full['exact_evidence_accuracy']:.6f}; "
            f"`w/o Applicability` is {without['exact_evidence_accuracy']:.6f}. "
            f"The paired difference is {paired['mean_difference']:+.6f} with 95% CI "
            f"[{paired['ci_low']:+.6f}, {paired['ci_high']:+.6f}].",
            "",
            f"Strongest fair baseline: `{report['strongest_baseline_by_exact_evidence_accuracy']}`.",
            "",
            "## Decision",
            "",
            report["decision"]["reason"],
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"
