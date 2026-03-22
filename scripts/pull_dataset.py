import json
import os

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

from training.config import DEFAULT_DATASET, HF_DATASET, REFINED_DATA_DIR

load_dotenv()
login()

dataset = load_dataset(HF_DATASET, split="train")

os.makedirs(REFINED_DATA_DIR, exist_ok=True)
with open(DEFAULT_DATASET, "w", encoding="utf-8") as f:
    for example in dataset:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"✓ Pulled {len(dataset)} examples to {DEFAULT_DATASET}")
