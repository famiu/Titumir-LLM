# Titumir LLM

A Bengali language finetuning pipeline for Qwen3.5 9B, optimized for colloquial
Bangladeshi speech including Bengali script, romanized Bengali, and Banglish
code-switching.

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

```bash
git clone https://github.com/famiuhaque/titumir-llm
cd titumir-llm
uv sync
```

Copy the example env file and fill in your API keys:
```bash
cp .env.example .env
```

`.env`:
```bash
OPENROUTER_API_KEY="sk-or-..."
HF_TOKEN="hf_..."
```

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

### Training Pipeline
```bash
# Run full training pipeline: CPT → SFT → export
just train

# Or run stages individually
just cpt
just sft                                # loads dataset from HuggingFace Hub
just sft data/refined/my_dataset.jsonl  # loads dataset from local file
just export
```

### Utilities
```bash
# Check tokenizer efficiency on Bengali text
just check-tokenizer

# Lint and format
just lint
```

## Configuration

All training parameters are in `training/config.py`:

## CPT Data Sources

| Dataset | Type | Weight |
|---|---|---|
| BanglishRev | E-commerce reviews (Bengali/Banglish) | 40% |
| Ben-Sarc | Facebook comments | 25% |
| CC100 Bengali | Web text | 20% |
| CC100 Bengali Romanized | Romanized web text | 10% |
| Wikipedia Bengali | Formal prose | 5% |

## License

Apache 2.0 — see [LICENSE](LICENSE)
