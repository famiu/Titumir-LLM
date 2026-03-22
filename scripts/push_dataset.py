from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

from training.config import DEFAULT_DATASET, HF_DATASET

load_dotenv()
login()

dataset = load_dataset("json", data_files=DEFAULT_DATASET, split="train")
dataset.push_to_hub(HF_DATASET)
print(f"✓ Pushed {len(dataset)} examples to {HF_DATASET}")
