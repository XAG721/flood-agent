from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.frc_rag.hover_verification_roles import (  # noqa: E402
    FrozenHoVerScorer,
    evaluate_scored_cases,
    extract_official_candidates,
    inspect_database,
    load_articles,
    load_frozen_tokenizer,
    load_registration,
    load_report,
    load_source_rows,
    prepare_dataset,
    privacy_audit,
    read_jsonl,
    score_cases_resumable,
    sha256,
    validate_preparation_summary,
    validate_source_files,
    write_jsonl,
    write_report,
)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen v39 HoVer task-specific role experiment."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/hover_verification_roles_protocol.json"
        ),
    )
    parser.add_argument(
        "--execution",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/hover_verification_roles_execution.json"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(".cache/benchmarks/hover/hover_dev_release_v1.1.json"),
    )
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=Path(
            ".cache/benchmarks/hover/dev_tfidf_doc_retrieval_results.json"
        ),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(".cache/benchmarks/hover/wiki_wo_links.db"),
    )
    parser.add_argument(
        "--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache")
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path(
            ".cache/benchmarks/hover/prepared_verification_roles_blind.jsonl"
        ),
    )
    parser.add_argument(
        "--scored",
        type=Path,
        default=Path(
            ".cache/benchmarks/hover/scored_verification_roles_blind.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/hover_verification_roles"),
    )
    parser.add_argument(
        "--census-output",
        type=Path,
        default=Path(".cache/benchmarks/hover/v39_census.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--case-batch-size", type=int, default=8)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--census-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    mode_count = sum((args.census_only, args.prepare_only, args.evaluate_only))
    if mode_count > 1:
        parser.error(
            "--census-only, --prepare-only and --evaluate-only are mutually exclusive"
        )
    if args.resamples <= 0:
        parser.error("--resamples must be positive")

    protocol_path = _resolve(args.protocol)
    execution_path = _resolve(args.execution)
    dataset_path = _resolve(args.dataset)
    retrieval_path = _resolve(args.retrieval)
    database_path = _resolve(args.database)
    hf_home = _resolve(args.hf_home)
    prepared_path = _resolve(args.prepared)
    scored_path = _resolve(args.scored)
    output_dir = _resolve(args.output_dir)
    census_path = _resolve(args.census_output)

    if sha256(protocol_path) != (
        "82ec16abcde6b28a86a9dd6f2fe3da26496cf14d718dcb1e388fb9341e1c8e82"
    ):
        raise ValueError("HoVer protocol changed after preregistration")
    validate_source_files(dataset_path, retrieval_path, database_path)
    dataset_rows, retrieval_rows = load_source_rows(dataset_path, retrieval_path)
    retrieval, requested_titles = extract_official_candidates(retrieval_rows)
    del retrieval
    database_schema = inspect_database(database_path)
    if database_schema["quick_check"] != "ok":
        raise ValueError("HoVer Wikipedia database quick_check failed")
    articles = load_articles(
        database_path, requested_titles, schema=database_schema
    )
    tokenizer = load_frozen_tokenizer(hf_home)
    prepared, gold, summary = prepare_dataset(
        dataset_rows, retrieval_rows, articles, tokenizer
    )
    write_jsonl(prepared_path, prepared)

    source_hashes = {
        "dataset": sha256(dataset_path),
        "official_tfidf_candidates": sha256(retrieval_path),
        "official_wikipedia_database": sha256(database_path),
    }
    census = {
        "structural_census": summary,
        "database": {
            "schema": database_schema,
            "requested_top20_unique_titles": len(requested_titles),
            "resolved_requested_titles": len(articles),
        },
        "source_hashes": source_hashes,
        "prepared_blind_sha256": sha256(prepared_path),
    }
    census_path.parent.mkdir(parents=True, exist_ok=True)
    census_path.write_text(
        json.dumps(census, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if args.census_only:
        print(
            json.dumps(
                {
                    "status": "HOVER_POST_DOWNLOAD_CENSUS_COMPLETE",
                    "census_path": str(census_path),
                    "census_sha256": sha256(census_path),
                    **census,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    _, execution = load_registration(protocol_path, execution_path)
    validate_preparation_summary(summary, execution)
    if execution["source_hashes"] != source_hashes:
        raise ValueError("HoVer registered source hashes changed")
    if execution["database_registration"] != census["database"]:
        raise ValueError("HoVer database registration changed")
    if execution["prepared_blind_sha256"] != census["prepared_blind_sha256"]:
        raise ValueError("HoVer blind preparation fingerprint changed")
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "HOVER_BLIND_PREPARATION_COMPLETE",
                    "summary": summary,
                    "prepared_sha256": sha256(prepared_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.evaluate_only:
        scored = list(read_jsonl(scored_path))
        if [row.get("id") for row in scored] != [
            row.get("id") for row in prepared
        ]:
            raise ValueError("HoVer score cache is incomplete or reordered")
    else:
        scorer = FrozenHoVerScorer(
            hf_home=hf_home,
            device=args.device,
            embedding_batch_size=args.embedding_batch_size,
            reranker_batch_size=args.reranker_batch_size,
            use_fp16=args.device.startswith("cuda"),
        )
        scored = score_cases_resumable(
            prepared,
            scorer,
            scored_path,
            case_batch_size=args.case_batch_size,
        )
    report, evidence = evaluate_scored_cases(
        gold, scored, resamples=args.resamples
    )
    audit = privacy_audit(evidence)
    if not audit["passed"]:
        raise AssertionError(f"HoVer privacy audit failed: {audit['leaks']}")
    source_paths = {
        "dataset": dataset_path,
        "execution": execution_path,
        "implementation": REPO_ROOT
        / "research/frc_rag/hover_verification_roles.py",
        "official_tfidf_candidates": retrieval_path,
        "official_wikipedia_database": database_path,
        "prepared_blind": prepared_path,
        "protocol": protocol_path,
        "scored_blind": scored_path,
    }
    paths = write_report(
        report, evidence, output_dir, source_paths=source_paths
    )
    first = {name: sha256(path) for name, path in paths.items()}
    paths = write_report(
        report, evidence, output_dir, source_paths=source_paths
    )
    second = {name: sha256(path) for name, path in paths.items()}
    if first != second:
        raise AssertionError("HoVer report outputs are not byte deterministic")
    loaded = load_report(paths["json"], paths["evidence"])
    comparison = loaded["analysis"]["family_comparison"]
    print(
        json.dumps(
            {
                "status": loaded["analysis"]["outcome"]["status"],
                "cases": loaded["metadata"]["cases"],
                "candidate_chunks": loaded["metadata"]["candidate_chunks"],
                "strongest_non_frc": comparison[
                    "observed_strongest_non_frc"
                ],
                "verification_minus_generic": comparison[
                    "verification_minus_generic"
                ],
                "verification_minus_strongest": comparison[
                    "verification_minus_observed_strongest"
                ],
                "simultaneous_ci": comparison[
                    "verification_minus_bootstrap_strongest_simultaneous"
                ],
                "privacy_audit": audit,
                "output_hashes": second,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    for path in paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
