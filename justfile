# Default — show available recipes
default:
    @just --list

# ── Training Pipeline ──────────────────────────────────────────────────────────

# Run full training pipeline: CPT → SFT → export
train *args:
    just cpt {{args}}
    just sft {{args}}
    just export {{args}}

# Run dry-run pipeline. Pass --regenerate/-r to regenerate the test dataset first.
[arg("regenerate", long="regenerate", short="r", value="true")]
dry-run regenerate="false":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{regenerate}}" = "true" ]; then
        just generate-dataset --config configs/dry_run.yaml dry_run.jsonl
        just refine-dataset --config configs/dry_run.yaml dry_run.jsonl
        just merge-dataset --config configs/dry_run.yaml
    fi
    just train --config configs/dry_run.yaml

# Stage 1: Continued pretraining on raw Bengali text
cpt *args:
    uv run training/cpt.py {{args}}

# Stage 2: Supervised finetuning on conversational dataset
sft *args:
    uv run training/sft.py {{args}}

# Stage 3: Merge and export to GGUF
export *args:
    uv run training/export_to_gguf.py {{args}}

# ── Data Pipeline ──────────────────────────────────────────────────────────────

# Generate synthetic training data
generate-dataset *args:
    uv run scripts/generate_dataset.py {{args}}

# Refine all unprocessed datasets, or a specific file if provided
refine-dataset *args:
    uv run scripts/refine_dataset.py {{args}}

# Merge all refined datasets
merge-dataset *args:
    uv run scripts/merge_dataset.py {{args}}

# Push merged dataset to HuggingFace Hub
push-dataset *args:
    uv run scripts/push_dataset.py {{args}}

# Pull dataset from HuggingFace Hub for local training
pull-dataset *args:
    uv run scripts/pull_dataset.py {{args}}

# ── Utilities ──────────────────────────────────────────────────────────────────

# Check tokenizer efficiency on Bengali text
check-tokenizer *args:
    uv run scripts/check_tokenizer.py {{args}}

# Lint and format all Python files
lint:
    uv run ruff check .
    uv run ruff format .
