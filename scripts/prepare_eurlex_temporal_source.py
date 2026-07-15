from __future__ import annotations
# ruff: noqa: E402 -- research stays outside the runtime wheel.

import argparse
import sys
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.eurlex_temporal_ablation import (
    DEFAULT_FAMILY_QUOTAS,
    DEFAULT_SEED,
    EURLEX_CONTENT_URL,
    EURLEX_SPARQL_ENDPOINT,
    EURLEX_SPARQL_QUERY,
    build_eurlex_source_manifest,
    extract_eurlex_html,
    parse_eurlex_repeal_pairs,
    select_eurlex_source_pairs,
    sha256_text,
)


USER_AGENT = "flood-agent-research/0.3 (+https://github.com/XAG721/flood-agent)"


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _request(url: str, *, accept: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Accept": accept, "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def _download_query(path: Path) -> None:
    query = urllib.parse.urlencode({"query": EURLEX_SPARQL_QUERY})
    payload = _request(
        f"{EURLEX_SPARQL_ENDPOINT}?{query}",
        accept="application/sparql-results+json",
    )
    parsed = json.loads(payload.decode("utf-8"))
    _write_text(path, json.dumps(parsed, ensure_ascii=False, indent=2))


def _download_document(celex: str, documents_dir: Path, *, refresh: bool) -> Path:
    output_path = documents_dir / f"{celex}.json"
    if output_path.is_file() and not refresh:
        return output_path
    url = EURLEX_CONTENT_URL.format(celex=celex)
    payload = _request(url, accept="text/html,application/xhtml+xml")
    decoded = payload.decode("utf-8")
    try:
        extracted = extract_eurlex_html(decoded)
    except ValueError as exc:
        failure_path = documents_dir / f"{celex}.failed.html"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_bytes(payload)
        raise ValueError(
            f"EUR-Lex document {celex} was not usable; saved {failure_path}"
        ) from exc
    document = {
        "celex": celex,
        "content_url": url,
        "title": extracted["title"],
        "text": extracted["text"],
        "raw_html_sha256": hashlib.sha256(payload).hexdigest(),
        "canonical_text_sha256": sha256_text(extracted["text"]),
    }
    _write_text(output_path, json.dumps(document, ensure_ascii=False, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a reproducible EUR-Lex repeal-boundary source slice with official "
            "CELLAR effective/expiry metadata and English legal text hashes."
        )
    )
    parser.add_argument(
        "--query-result",
        type=Path,
        default=Path(".cache/benchmarks/eurlex_temporal/repeal_pairs.sparql.json"),
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=Path(".cache/benchmarks/eurlex_temporal/documents"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("benchmarks/eurlex_temporal_selection.json"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--decisions", type=int, default=DEFAULT_FAMILY_QUOTAS["decision"])
    parser.add_argument("--directives", type=int, default=DEFAULT_FAMILY_QUOTAS["directive"])
    parser.add_argument(
        "--regulations", type=int, default=DEFAULT_FAMILY_QUOTAS["regulation"]
    )
    parser.add_argument("--refresh-query", action="store_true")
    parser.add_argument("--refresh-documents", action="store_true")
    args = parser.parse_args()

    if args.refresh_query or not args.query_result.is_file():
        _download_query(args.query_result)
    payload = json.loads(args.query_result.read_text(encoding="utf-8"))
    pairs = parse_eurlex_repeal_pairs(payload)
    quotas = {
        "decision": args.decisions,
        "directive": args.directives,
        "regulation": args.regulations,
    }
    selected = select_eurlex_source_pairs(
        pairs, family_quotas=quotas, seed=args.seed
    )
    celex_ids = sorted(
        {pair[side]["celex"] for pair in selected for side in ("old", "new")}
    )
    for index, celex in enumerate(celex_ids, start=1):
        print(f"fetching {index}/{len(celex_ids)} {celex}", flush=True)
        _download_document(
            celex, args.documents_dir, refresh=args.refresh_documents
        )
        print(f"downloaded {index}/{len(celex_ids)} EUR-Lex documents", flush=True)
    manifest = build_eurlex_source_manifest(
        selected,
        documents_dir=args.documents_dir,
        query_result_path=args.query_result,
        seed=args.seed,
        family_quotas=quotas,
    )
    _write_text(args.output_manifest, json.dumps(manifest, ensure_ascii=False, indent=2))
    print(args.output_manifest)


if __name__ == "__main__":
    main()
