# Titumir LLM

A Bengali language finetuning pipeline for researching whether Qwen3.5 9B can be
adapted to colloquial Bangladeshi speech, including Bengali script, romanized
Bengali, and Banglish code-switching.

> [!CAUTION]
> Training a model using outputs from another model (knowledge distillation) may violate the base model's terms of service and is legally dubious. This project is designed for fine-tuning on independently sourced data or distilling from models that explicitly allow it, not distillation from proprietary models. Anyone using this project for distillation does so entirely at their own risk.

## Overview

Titumir is a two-stage finetuning pipeline:

1. **Continued Pretraining (CPT)** — adapts the base model to colloquial Bengali
   register using social media and web text corpora
2. **Supervised Finetuning (SFT)** — teaches the model specific conversational
   behavior using a synthetic Bengali social media dataset

Naturalness and downstream usefulness are research goals, not established outcomes.
Use held-out losses, generated comparison sets, dataset audits, and human review before
making quality claims.

## Requirements

- Python 3.11–3.13
- NVIDIA GPU with BF16 or FP16 support (24GB+ VRAM recommended for training)
- An NVIDIA driver compatible with the CUDA 12.8 runtime used by the locked PyTorch wheel
- A C/C++ compiler, CUDA development toolkit, and Ninja for the intentional xFormers source build
- [uv](https://github.com/astral-sh/uv)
- [just](https://github.com/casey/just)

## Setup

### Dependencies

```bash
git clone https://github.com/famiuhaque/titumir-llm
cd titumir-llm
uv sync
```

`uv sync` builds xFormers from source so it matches the locked PyTorch/CUDA stack. This can take several minutes and
requires a working compiler and CUDA headers. The local CUDA toolkit version and the CUDA runtime bundled with the
PyTorch wheel are separate; the NVIDIA driver must be new enough for the bundled CUDA 12.8 runtime.

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

# Continue an interrupted named generation run
just generate-dataset my_dataset.jsonl --resume

# Refine all unprocessed datasets
just refine-dataset

# Refine a specific file by name
just refine-dataset my_dataset.jsonl

# Continue interrupted refinement checkpoints
just refine-dataset my_dataset.jsonl --resume

# Merge all refined datasets into the final training file
just merge-dataset

# Push merged dataset to HuggingFace Hub
just push-dataset

# Pull dataset from HuggingFace Hub to local
just pull-dataset

# Explicitly overwrite the existing local dataset copy; the remote Hub dataset is unchanged
just pull-dataset --overwrite
```

Use `--config` on any command to use a custom configuration file.
Named generation outputs are protected from accidental replacement; pass `--overwrite`
only when replacement is intentional.

### Training Pipeline

```bash
# Run full training pipeline: CPT → SFT → export
just train

# Or run stages individually
just cpt
just sft
just export
```

Set `resume_from_checkpoint: true` in the relevant training section to resume the latest
Trainer checkpoint in its output directory, or set it to a specific checkpoint path.
Each stage writes a `run_manifest.json` containing the effective config, package versions,
dataset fingerprints, hardware metadata, git revision, and metrics. Environment values and
API keys are never included.

### Dry Run

```bash
# CPT → SFT → export using existing test data
just dry-run

# Regenerate dataset first, then run full pipeline
just dry-run -r
```

The dry run uses a much smaller Qwen2.5 model and validates pipeline wiring only. It does not validate Qwen3.5 model
compatibility, production VRAM requirements, final quality, or production GGUF behavior.

### Utilities

```bash
# Check tokenizer efficiency on Bengali text
just check-tokenizer

# Measure corpus-level tokenizer statistics and truncation
just check-tokenizer --dataset data/default/default_merged.jsonl

# Audit schema, topic/script balance, duplication, phrasing, and overlap
just audit-dataset

# Write a deterministic sample for human review
just audit-dataset --sample-output evaluation/human_sample.jsonl

# Compare base, CPT, and SFT generations without assigning a quality score
just evaluate-models

# Optionally smoke-test a GGUF through a configured llama.cpp executable
just evaluate-models --gguf export/titumir_9b_gguf/model.gguf

# Lint and format
just lint

# Run tests
just test
```

## Configuration

All configuration is managed via `configs/config.yaml`. The config uses a nested YAML structure organized into sections:

- **profile** — Data profile information
- **seed** — Shared seed for dataset operations and training
- **model** — Base model configuration
- **cpt_training** — CPT datasets, LoRA settings, packed training, 1% source-aware evaluation, and resume behavior
- **sft_training** — SFT settings, grouped evaluation, assistant-only loss, and resume behavior
- **generation** — Dataset generation LLM settings (endpoint, model, temperature, max tokens, batch size, timeout) and prompt template
- **refinement** — Dataset refinement LLM settings (endpoint, model, temperature, max tokens, batch size, timeout) and prompt template
- **export** — Model export path and quantization method
- **evaluation** — Versioned comparison prompts and generation length
- **topics** — List of topics for dataset generation

### Using Custom Configs

Pass a custom config file with `--config`:

```bash
just generate-dataset --config configs/custom.yaml
just refine-dataset --config configs/custom.yaml
just merge-dataset --config configs/custom.yaml
just audit-dataset --config configs/custom.yaml
just cpt --config configs/custom.yaml
just sft --config configs/custom.yaml
just export --config configs/custom.yaml
just evaluate-models --config configs/custom.yaml
```

See `configs/config.yaml` for the default configuration.

## Research Interpretation

- CPT evaluation loss and perplexity measure held-out next-token prediction, not conversational quality.
- SFT evaluation loss is grouped by normalized conversation and topic where metadata is available, but remains a synthetic-data metric.
- `just evaluate-models` emits deterministic comparison outputs for human inspection; it is not a benchmark score.
- GGUF smoke testing checks that inference executes, not that quantization preserved every behavior.
- Generator and refiner models can share biases and stylistic fingerprints. Audit a stratified sample with human reviewers.
- Review [DATA_SOURCES.md](DATA_SOURCES.md) before training or redistributing artifacts.
- Use [RESEARCH_REPORT_TEMPLATE.md](RESEARCH_REPORT_TEMPLATE.md) when recording experiments and model releases.

## License

Apache 2.0 — see [LICENSE](LICENSE)

> [!NOTE]
> Much of this project is written using AI coding tools (e.g. OpenCode). I use this project as a testing ground for evaluating the performance of AI tools and keeping up with their workflows. I supervise the overall structure of the project, but I don't care to control the minute details.
