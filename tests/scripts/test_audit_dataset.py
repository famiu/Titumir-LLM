from __future__ import annotations

from scripts.audit_dataset import audit_examples, stratified_sample


def make_example(index: int, topic: str, content: str = "বাংলা text") -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"{content} {index}"},
            {"role": "assistant", "content": f"reply {index}"},
        ],
        "metadata": {"topic": topic},
    }


def test_audit_reports_topics_scripts_and_duplicates() -> None:
    examples = [make_example(1, "a"), make_example(2, "b"), make_example(1, "a")]
    report = audit_examples(examples)
    assert report["examples"] == 3
    assert report["topics"] == {"a": 2, "b": 1}
    assert report["scripts"]["mixed"] == 3
    assert report["normalized_duplicates"] == 1


def test_stratified_sample_is_deterministic() -> None:
    examples = [make_example(index, str(index % 2)) for index in range(10)]
    first = stratified_sample(examples, 4)
    second = stratified_sample(examples, 4)
    assert first == second
    assert len(first) == 4
