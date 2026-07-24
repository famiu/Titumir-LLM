"""Audit conversation datasets without modifying source records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts._data import atomic_text_writer, conversation_key, validate_conversation
from scripts.check_tokenizer import script_category
from training.config import load_config


def load_examples(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load valid examples and collect source diagnostics."""
    examples = []
    errors = []
    with open(path, encoding="utf-8") as file:
        for line_num, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                examples.append(validate_conversation(value, f"{path}:{line_num}"))
            except (json.JSONDecodeError, ValueError) as error:
                errors.append(str(error))
    return examples, errors


def audit_examples(examples: list[dict[str, Any]], comparison_keys: set[str] | None = None) -> dict[str, Any]:
    """Compute deterministic quality and diversity statistics."""
    topics = Counter()
    scripts = Counter()
    normalized = Counter()
    phrases = Counter()
    lengths = []
    overlap = 0
    for example in examples:
        metadata = example.get("metadata")
        topic = metadata.get("topic", "unknown") if isinstance(metadata, dict) else "unknown"
        topics[topic] += 1
        text = " ".join(message["content"] for message in example["messages"])
        scripts[script_category(text)] += 1
        key = conversation_key(example)
        normalized[key] += 1
        overlap += int(comparison_keys is not None and key in comparison_keys)
        words = text.lower().split()
        lengths.append(len(words))
        phrases.update(" ".join(words[index : index + 3]) for index in range(max(0, len(words) - 2)))

    repeated_phrases = [
        {"phrase": phrase, "count": count}
        for phrase, count in phrases.most_common(20)
        if count > 1 and len(phrase) >= 6
    ]
    return {
        "examples": len(examples),
        "topics": dict(sorted(topics.items())),
        "scripts": dict(sorted(scripts.items())),
        "normalized_duplicates": sum(count - 1 for count in normalized.values() if count > 1),
        "comparison_overlap": overlap,
        "word_length": {
            "min": min(lengths, default=0),
            "max": max(lengths, default=0),
            "mean": sum(lengths) / len(lengths) if lengths else 0,
        },
        "repeated_trigrams": repeated_phrases,
    }


def stratified_sample(examples: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Select a deterministic topic/script-stratified human-audit sample."""
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        metadata = example.get("metadata")
        topic = metadata.get("topic", "unknown") if isinstance(metadata, dict) else "unknown"
        text = " ".join(message["content"] for message in example["messages"])
        strata[(topic, script_category(text))].append(example)
    for values in strata.values():
        values.sort(key=lambda example: hashlib.sha256(conversation_key(example).encode()).hexdigest())

    sample = []
    while len(sample) < size and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(sample) < size:
                sample.append(strata[key].pop(0))
    return sample


def audit_dataset(
    config_path: str | None = None,
    dataset_path: str | None = None,
    output: str | None = None,
    sample_output: str | None = None,
    sample_size: int = 100,
    compare: str | None = None,
) -> None:
    """Audit a merged or explicitly supplied conversation dataset."""
    config = load_config(config_path)
    path = Path(dataset_path or config.profile.local_dataset)
    examples, errors = load_examples(path)
    comparison_keys = None
    if compare is not None:
        compared, comparison_errors = load_examples(Path(compare))
        errors.extend(comparison_errors)
        comparison_keys = {conversation_key(example) for example in compared}
    report = audit_examples(examples, comparison_keys)
    report["path"] = str(path)
    report["errors"] = errors

    output_path = Path(output or f"{path}.audit.json")
    with atomic_text_writer(output_path) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    if sample_output is not None:
        with atomic_text_writer(sample_output) as file:
            for example in stratified_sample(examples, sample_size):
                file.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"Audit report written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit a conversation JSONL dataset")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset path; defaults to the profile dataset")
    parser.add_argument("-o", "--output", type=str, default=None, help="Audit report path")
    parser.add_argument("--sample-output", type=str, default=None, help="Optional human-audit sample JSONL")
    parser.add_argument("--sample-size", type=int, default=100, help="Human-audit sample size")
    parser.add_argument("--compare", type=str, default=None, help="Optional dataset to check for normalized overlap")
    args = parser.parse_args()
    audit_dataset(
        config_path=args.config,
        dataset_path=args.dataset,
        output=args.output,
        sample_output=args.sample_output,
        sample_size=args.sample_size,
        compare=args.compare,
    )
