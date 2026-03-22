# Default — show available recipes
default:
    @just --list

# ── Training Pipeline ──────────────────────────────────────────────────────────

# Run full training pipeline: CPT → SFT → export
train: cpt sft export

# Stage 1: Continued pretraining on raw Bengali text
cpt:
    uv run training/cpt.py

# Stage 2: Supervised finetuning on conversational dataset
sft dataset="":
    uv run training/sft.py {{dataset}}

# Stage 3: Merge and export to GGUF
export:
    uv run training/export_to_gguf.py

# ── Data Pipeline ──────────────────────────────────────────────────────────────

# Generate synthetic training data
generate-dataset filename="":
    uv run scripts/generate_dataset.py {{filename}}

# Refine all unprocessed datasets, or a specific file if provided
refine-dataset filename="":
    uv run scripts/refine_dataset.py {{filename}}

# Merge all refined datasets
merge-dataset:
    uv run scripts/merge_dataset.py

# Push merged dataset to HuggingFace Hub
push-dataset:
    uv run scripts/push_dataset.py

# Pull dataset from HuggingFace Hub for local training
pull-dataset:
    uv run scripts/pull_dataset.py

# ── Utilities ──────────────────────────────────────────────────────────────────

# Check tokenizer efficiency on Bengali text
check-tokenizer:
    uv run scripts/check_tokenizer.py

# Lint and format all Python files
lint:
    uv run ruff check .
    uv run ruff format .
