from __future__ import annotations

from scripts.check_tokenizer import analyze_text, percentile, script_category


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


def test_script_categories() -> None:
    assert script_category("বাংলা") == "bengali"
    assert script_category("bangla") == "latin"
    assert script_category("বাংলা bangla") == "mixed"
    assert script_category("😭") == "other"


def test_grapheme_measurement_handles_combining_text() -> None:
    result = analyze_text(FakeTokenizer(), "ক্\u200dষ", 10)
    assert result["codepoints"] == 4
    assert result["graphemes"] == 1
    assert result["truncated"] is False


def test_percentile() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 3.0
    assert percentile([], 0.5) == 0.0
