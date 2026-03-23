import argparse
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

from training.config import load_config


def push_dataset(config_path: str | None = None) -> None:
    """Push dataset to HuggingFace Hub."""
    load_dotenv()
    login()
    config = load_config(config_path)
    dataset_path = config.paths.local_dataset

    if not Path(dataset_path).exists():
        print(f"Dataset file not found: {dataset_path}")
        return

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset.push_to_hub(config.paths.hf_dataset)
    print(f"Pushed {len(dataset)} examples to {config.paths.hf_dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push dataset to HuggingFace Hub")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    push_dataset(config_path=args.config)
