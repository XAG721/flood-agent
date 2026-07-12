from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


OFFICIAL_REPOSITORIES = {
    "MultiHop-RAG": "https://github.com/yixuantt/MultiHop-RAG.git",
    "ConditionalQA": "https://github.com/haitian-sun/ConditionalQA.git",
    "HotpotQA": "https://github.com/hotpotqa/hotpot.git",
}
HOTPOT_HF_REPO = "hotpotqa/hotpot_qa"
HOTPOT_VALIDATION_FILE = "distractor/validation-00000-of-00001.parquet"


def clone_if_missing(url: str, target: Path) -> None:
    if (target / ".git").is_dir():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch official public RAG benchmark sources.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/benchmarks"))
    args = parser.parse_args()

    clone_if_missing(OFFICIAL_REPOSITORIES["MultiHop-RAG"], args.cache_dir / "MultiHop-RAG")
    clone_if_missing(OFFICIAL_REPOSITORIES["ConditionalQA"], args.cache_dir / "ConditionalQA")
    clone_if_missing(OFFICIAL_REPOSITORIES["HotpotQA"], args.cache_dir / "hotpot")

    from huggingface_hub import hf_hub_download

    hotpot_dir = args.cache_dir / "hotpot_hf"
    hotpot_path = hf_hub_download(
        repo_id=HOTPOT_HF_REPO,
        repo_type="dataset",
        filename=HOTPOT_VALIDATION_FILE,
        local_dir=hotpot_dir,
    )
    print(f"MultiHop-RAG: {args.cache_dir / 'MultiHop-RAG'}")
    print(f"ConditionalQA: {args.cache_dir / 'ConditionalQA'}")
    print(f"HotpotQA: {hotpot_path}")


if __name__ == "__main__":
    main()
