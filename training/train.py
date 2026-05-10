from dotenv import load_dotenv

load_dotenv()

import argparse
import gc

from unsloth import FastLanguageModel  # isort: skip
import torch

from training.config import load_config
from training.cpt import run_cpt
from training.export_to_gguf import export_gguf
from training.sft import run_sft


def run_pipeline(config_path: str | None = None) -> None:
    """Run full training pipeline: CPT → SFT → export, single model load."""
    config = load_config(config_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
    )

    model, tokenizer = run_cpt(config_path, model=model, tokenizer=tokenizer)
    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = run_sft(config_path, model=model, tokenizer=tokenizer)
    gc.collect()
    torch.cuda.empty_cache()

    export_gguf(config_path, model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full training pipeline: CPT → SFT → export")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    run_pipeline(config_path=args.config)
