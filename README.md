# Titumir LLM

A Bengali language finetuning pipeline for Qwen3.5 9B, optimized for colloquial
Bangladeshi speech including Bengali script, romanized Bengali, and Banglish
code-switching.

> [!CAUTION]
> Training a model using outputs from another model (knowledge distillation) may violate the base model's terms of service and is legally dubious. This project is designed for fine-tuning on independently sourced data or distilling from models that explicitly allow it, not distillation from proprietary models. Anyone using this project for distillation does so entirely at their own risk.

## Overview

Titumir is a two-stage finetuning pipeline:

1. **Continued Pretraining (CPT)** — adapts the base model to colloquial Bengali
   register using social media and web text corpora
2. **Supervised Finetuning (SFT)** — teaches the model specific conversational
   behavior using a synthetic Bengali social media dataset

The resulting model is designed to produce natural, human-sounding Bengali/Banglish
output suitable for persona-driven conversational agents.

## Requirements

- Python 3.11–3.13
- CUDA 12.4 compatible GPU (24GB+ VRAM recommended for training)
- [uv](https://github.com/astral-sh/uv)
- [just](https://github.com/casey/just)

## Setup

### Dependencies

```bash
git clone https://github.com/famiuhaque/titumir-llm
cd titumir-llm
uv sync
```

### Environment Variables

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
```

`.env`:

```bash
OPENROUTER_API_KEY="sk-or-..."
HF_TOKEN="hf_..."
# UNSLOTH_LLAMA_CPP_PATH="/path/to/llama.cpp"
```

| Variable                 | Purpose                                                                           |
| ------------------------ | --------------------------------------------------------------------------------- |
| `OPENROUTER_API_KEY`     | Data generation & refinement via OpenRouter                                       |
| `HF_TOKEN`               | Push/pull datasets to/from HuggingFace Hub                                        |
| `UNSLOTH_LLAMA_CPP_PATH` | Pre-built llama.cpp path for GGUF export, built by Unsloth automatically if unset |

## Usage

### Data Pipeline

```bash
# Generate synthetic training data
just generate-dataset

# Generate with a specific output filename
just generate-dataset my_dataset.jsonl

# Refine all unprocessed datasets
just refine-dataset

# Refine a specific file by name
just refine-dataset my_dataset.jsonl

# Merge all refined datasets into the final training file
just merge-dataset

# Push merged dataset to HuggingFace Hub
just push-dataset

# Pull dataset from HuggingFace Hub to local
just pull-dataset
```

Use `--config` on any command to use a custom configuration file.

### Training Pipeline

```bash
# Run full training pipeline: CPT → SFT → export
just train

# Or run stages individually
just cpt
just sft
just export
```

### Dry Run

```bash
# CPT → SFT → export using existing test data
just dry-run

# Regenerate dataset first, then run full pipeline
just dry-run -r
```

### Utilities

```bash
# Check tokenizer efficiency on Bengali text
just check-tokenizer

# Lint and format
just lint
```

## Configuration

All configuration is managed via `configs/config.yaml`. The config uses a nested YAML structure organized into sections:

- **profile** — Data profile information
- **model** — Base model configuration
- **cpt_training** — Continued pretraining settings (datasets, max examples, output/checkpoint dirs, LoRA params)
- **sft_training** — Supervised finetuning settings (output/checkpoint dirs)
- **generation** — Dataset generation LLM settings (endpoint, model, temperature, max tokens, batch size, timeout) and prompt template
- **refinement** — Dataset refinement LLM settings (endpoint, model, temperature, max tokens, batch size, timeout) and prompt template
- **export** — Model export path and quantization method
- **topics** — List of topics for dataset generation

### Using Custom Configs

Pass a custom config file with `--config`:

```bash
just generate-dataset --config configs/custom.yaml
just refine-dataset --config configs/custom.yaml
just merge-dataset --config configs/custom.yaml
just cpt --config configs/custom.yaml
just sft --config configs/custom.yaml
just export --config configs/custom.yaml
```

See `configs/config.yaml` for the default configuration.

## License

Apache 2.0 — see [LICENSE](LICENSE)
