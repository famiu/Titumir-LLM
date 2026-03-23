# AGENTS.md

## Stack
- Python 3.11–3.13, Pydantic v2, PyYAML
- Unsloth, TRL (SFTTrainer), HuggingFace datasets/transformers
- CUDA 12.4, 24GB+ VRAM recommended
- uv for dependency management, just for task running
- ruff for linting/formatting

## Commands
- `uv sync` — install dependencies
- `just lint` — ruff check + ruff format
- `just train --config configs/config.yaml` — full pipeline: CPT → SFT → export
- `just cpt` — continued pretraining
- `just sft` — supervised finetuning
- `just export` — merge adapter + export GGUF
- `just generate-dataset` — generate synthetic SFT data
- `just refine-dataset` — LLM-based quality filtering
- `just merge-dataset` — deduplicate + merge refined JSONLs
- All training/data commands accept `--config <path>`

## Project Structure
```
training/        config.py, cpt.py, sft.py, export_to_gguf.py
scripts/         generate, refine, merge, push, pull, check_tokenizer
configs/         YAML config files
data/            unprocessed/, refined/, removed/
```

## Conventions
- Config loading: `from training.config import load_config; config = load_config(config_path)`
- Config fields: add Pydantic field with default → add validator if needed → add YAML key to
  every file in `configs/` → reference as `config.section.field`
- Imports: sorted by ruff isort, `from __future__ import annotations` in config modules
- Type hints on all function signatures
- Scripts use `argparse` with `-c`/`--config` argument
- Comments only when necessary — don't over-comment, but use them where intent isn't obvious
- Use semantic commit messages (https://www.conventionalcommits.org/en/v1.0.0/)

## Boundaries
### Always do
- Run `just lint` before finishing any change
- Update all YAML files in `configs/` when adding or changing config fields
- Update README.md when user-facing workflow changes (new commands, changed flags, new config fields)
- Update AGENTS.md when conventions or commands change

### Ask first
- Before changing training hyperparameters or LoRA defaults
- Before adding new dependencies
- Before modifying the data pipeline scripts
- Before making any commits

### Never do
- Never read or modify `.env` — it contains secrets
- Never commit `.env` or credentials
- Never modify `unsloth_compiled_cache/`
- Never push to or pull from remote
