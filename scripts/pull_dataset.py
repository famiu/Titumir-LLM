import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

from scripts._data import atomic_text_writer, validate_conversation
from training.config import load_config


def pull_dataset(config_path: str | None = None, overwrite: bool = False) -> None:
    """Pull dataset from HuggingFace Hub."""
    load_dotenv()
    config = load_config(config_path)

    if config.profile.hf_dataset is None:
        print(f"No HF dataset configured for profile '{config.profile.name}' — skipping pull")
        return

    output_path = Path(config.profile.local_dataset)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Dataset already exists: {output_path}. Use --overwrite to replace it.")

    token = os.environ.get("HF_TOKEN")
    dataset = load_dataset(config.profile.hf_dataset, split="train", token=token)

    os.makedirs(config.profile.refined_data_dir, exist_ok=True)
    with atomic_text_writer(output_path) as file:
        for index, example in enumerate(dataset):
            example = validate_conversation(example, f"Hub dataset row {index}")
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Pulled {len(dataset)} examples to {config.profile.local_dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull dataset from HuggingFace Hub")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing local dataset")
    args = parser.parse_args()
    pull_dataset(config_path=args.config, overwrite=args.overwrite)
