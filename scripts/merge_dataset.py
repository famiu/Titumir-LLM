import argparse
import json
import os
from pathlib import Path

from training.config import load_config


def merge_datasets(config_path: str | None = None) -> None:
    """Merge all refined JSONL files into a single deduplicated dataset."""
    config = load_config(config_path)
    input_dir = config.paths.refined_data_dir
    output_file = config.paths.default_dataset

    input_path = Path(input_dir)
    output_path = Path(output_file)
    files = sorted(f for f in input_path.glob("*.jsonl") if f.resolve() != output_path.resolve())

    if not files:
        print(f"No JSONL files found in {input_dir}")
        return

    print(f"Found {len(files)} files in {input_dir}:")
    for f in files:
        print(f"  {f.name}")

    seen = set()
    examples = []

    for path in files:
        before = len(examples)
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                content = ex["messages"][-1]["content"]
                if content not in seen:
                    seen.add(content)
                    examples.append(ex)
        added = len(examples) - before
        print(f"  {path.name} — {added} unique examples added")

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Done — {len(examples)} examples written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge refined datasets")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    merge_datasets(config_path=args.config)
