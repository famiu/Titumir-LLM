import argparse
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

from training.config import load_config


def push_dataset(config_path: str | None = None) -> None:
    """Push dataset to HuggingFace Hub."""
    load_dotenv()
    config = load_config(config_path)

    if config.profile.hf_dataset is None:
        print(f"No HF dataset configured for profile '{config.profile.name}' — skipping push")
        return

    dataset_path = config.profile.local_dataset
    if not Path(dataset_path).exists():
        print(f"Dataset file not found: {dataset_path}")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required to push a dataset")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset.push_to_hub(config.profile.hf_dataset, token=token)
    print(f"Pushed {len(dataset)} examples to {config.profile.hf_dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push dataset to HuggingFace Hub")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    push_dataset(config_path=args.config)
