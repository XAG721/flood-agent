from __future__ import annotations

from scripts.run_frc_chunk_length_sensitivity import build_chunked_cases, chunk_text


class FakeTokenizer:
    def __init__(self) -> None:
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        output = []
        for word in text.split():
            if word not in self.word_to_id:
                token_id = len(self.word_to_id) + 1
                self.word_to_id[word] = token_id
                self.id_to_word[token_id] = word
            output.append(self.word_to_id[word])
        return output

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(self.id_to_word[token_id] for token_id in token_ids)


def test_chunk_text_uses_fixed_overlap_without_duplicate_tail():
    tokenizer = FakeTokenizer()

    chunks = chunk_text(
        "zero one two three four five six seven eight nine",
        tokenizer,
        chunk_length=4,
        overlap_ratio=0.25,
    )

    assert chunks == [
        ("zero one two three", 4),
        ("three four five six", 4),
        ("six seven eight nine", 4),
    ]


def test_chunked_cases_keep_parent_ids_for_evidence_scoring():
    source = {
        "dataset": "conditionalqa",
        "id": "case-1",
        "question": "What?",
        "required_roles": ["answer"],
        "gold_evidence_ids": ["parent-a"],
        "candidates": [
            {"id": "parent-a", "text": "one two three four five"},
        ],
    }

    [case] = build_chunked_cases(
        [source],
        FakeTokenizer(),
        chunk_length=3,
        overlap_ratio=0.0,
    )

    assert [item["parent_id"] for item in case["candidates"]] == ["parent-a", "parent-a"]
    assert case["gold_evidence_ids"] == ["parent-a"]
