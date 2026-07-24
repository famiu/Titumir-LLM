"""Generate comparable outputs from base and finetuned checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from unsloth import FastLanguageModel  # isort: skip
import torch

from scripts._data import atomic_text_writer
from training.config import load_config


def generate_responses(model_path: str, prompts: list[str], max_new_tokens: int, max_seq_length: int) -> list[str]:
    """Generate deterministic responses for one model checkpoint."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    responses = []
    for prompt in prompts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[1] :]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True))
    return responses


def run_gguf_smoke(gguf_path: Path, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    """Run one optional llama.cpp smoke prompt."""
    configured = os.environ.get("UNSLOTH_LLAMA_CPP_PATH")
    if not configured:
        return {"status": "skipped", "reason": "UNSLOTH_LLAMA_CPP_PATH is not set"}
    executable = Path(configured)
    if executable.is_dir():
        executable = executable / "llama-cli"
    if not executable.is_file():
        return {"status": "skipped", "reason": f"llama-cli not found at {executable}"}
    result = subprocess.run(
        [str(executable), "-m", str(gguf_path), "-p", prompt, "-n", str(max_new_tokens)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-2000:],
    }


def evaluate_models(config_path: str | None = None, output: str | None = None, gguf: str | None = None) -> None:
    """Compare generations without assigning an unsupported quality score."""
    config = load_config(config_path)
    if not config.evaluation.prompts:
        raise ValueError("evaluation.prompts must contain at least one prompt")
    model_paths = {
        "base": config.model.name,
        "cpt": config.cpt_training.checkpoint,
        "sft": config.sft_training.checkpoint,
    }
    results: dict[str, Any] = {
        "prompts": config.evaluation.prompts,
        "models": {},
        "note": "Generation comparison only; these outputs are not an automatic quality benchmark.",
    }
    for label, model_path in model_paths.items():
        if label != "base" and not Path(model_path).is_dir():
            results["models"][label] = {"path": model_path, "status": "missing"}
            continue
        responses = generate_responses(
            model_path,
            config.evaluation.prompts,
            config.evaluation.max_new_tokens,
            config.model.max_seq_length,
        )
        results["models"][label] = {"path": model_path, "status": "evaluated", "responses": responses}

    if gguf is not None:
        results["gguf_smoke"] = run_gguf_smoke(
            Path(gguf), config.evaluation.prompts[0], config.evaluation.max_new_tokens
        )

    output_path = Path(output or "evaluation/results.json")
    with atomic_text_writer(output_path) as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f"Evaluation comparison written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare base, CPT, SFT, and optional GGUF outputs")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--gguf", type=str, default=None, help="Optional GGUF file for a llama.cpp smoke test")
    args = parser.parse_args()
    evaluate_models(config_path=args.config, output=args.output, gguf=args.gguf)
