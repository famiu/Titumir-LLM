import argparse
import json
import os
from pathlib import Path

from scripts._data import atomic_text_writer, conversation_key, validate_conversation
from training.config import load_config


def merge_datasets(config_path: str | None = None) -> None:
    """Merge all refined JSONL files into a single deduplicated dataset."""
    config = load_config(config_path)
    input_dir = config.profile.refined_data_dir
    output_file = config.profile.local_dataset

    input_path = Path(input_dir)
    output_path = Path(output_file)
    files = sorted(f for f in input_path.glob("*.jsonl") if f.resolve() != output_path.resolve())

    if not files:
        print(f"No JSONL files found in {input_dir}")
        return

    print(f"Found {len(files)} files in {input_dir}:")
    for f in files:
        print(f"  {f.name}")

    seen_exact = set()
    seen_normalized = set()
    total_examples = 0
    exact_duplicates = 0
    normalized_duplicates = 0

    os.makedirs(output_path.parent, exist_ok=True)
    with atomic_text_writer(output_path) as output:
        for path in files:
            added = 0
            with open(path, encoding="utf-8") as file:
                for line_num, line in enumerate(file, 1):
                    if not line.strip():
                        continue
                    try:
                        example = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Malformed JSON in {path} at line {line_num}: {error}") from error
                    example = validate_conversation(example, f"{path}:{line_num}")
                    exact_key = json.dumps(
                        example["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    normalized_key = conversation_key(example)
                    if exact_key in seen_exact:
                        exact_duplicates += 1
                        continue
                    if normalized_key in seen_normalized:
                        normalized_duplicates += 1
                        continue
                    seen_exact.add(exact_key)
                    seen_normalized.add(normalized_key)
                    output.write(json.dumps(example, ensure_ascii=False) + "\n")
                    added += 1
                    total_examples += 1
            print(f"  {path.name} — {added} unique examples added")

    print(
        f"Done — {total_examples} examples written to {output_file} "
        f"({exact_duplicates} exact, {normalized_duplicates} normalization-equivalent duplicates removed)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge refined datasets")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    merge_datasets(config_path=args.config)
