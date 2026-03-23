import argparse
import json
import os

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

from training.config import load_config


def pull_dataset(config_path: str | None = None) -> None:
    """Pull dataset from HuggingFace Hub."""
    load_dotenv()
    login()
    config = load_config(config_path)
    dataset = load_dataset(config.paths.hf_dataset, split="train")

    os.makedirs(config.paths.refined_data_dir, exist_ok=True)
    with open(config.paths.local_dataset, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Pulled {len(dataset)} examples to {config.paths.local_dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull dataset from HuggingFace Hub")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    pull_dataset(config_path=args.config)
