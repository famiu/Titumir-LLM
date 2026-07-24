"""Runtime helpers shared by training stages."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from scripts._data import atomic_text_writer
from training.config import Config


def configure_seed(seed: int) -> None:
    """Configure the random seed used by training libraries."""
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precision_args() -> dict[str, bool]:
    """Select the best supported mixed precision mode."""
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    bf16 = torch.cuda.is_bf16_supported()
    return {"bf16": bf16, "fp16": not bf16}


def resolve_resume_checkpoint(value: bool | str, output_dir: str) -> str | None:
    """Resolve an explicit or latest trainer checkpoint."""
    if value is False:
        return None
    if isinstance(value, str):
        path = Path(value)
        if not path.is_dir():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return str(path)
    checkpoint = get_last_checkpoint(output_dir) if Path(output_dir).is_dir() else None
    if checkpoint is None:
        raise FileNotFoundError(f"No trainer checkpoint found in {output_dir}")
    return checkpoint


def ensure_trainable(model: Any) -> None:
    """Fail early if a reloaded adapter has no trainable parameters."""
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Reloaded model has no trainable parameters; verify the PEFT checkpoint")


def _git_metadata() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain"], capture_output=True, check=True, text=True).stdout
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def write_run_manifest(
    stage: str,
    config: Config,
    output_dir: str,
    metrics: dict[str, Any],
    dataset: dict[str, Any],
) -> None:
    """Write reproducibility metadata without reading or serializing secrets."""
    packages = {}
    for package in ("torch", "transformers", "trl", "unsloth", "datasets", "peft"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    hardware = {"cuda_available": torch.cuda.is_available(), "gpu": None}
    if torch.cuda.is_available():
        hardware["gpu"] = torch.cuda.get_device_name(0)

    manifest = {
        "stage": stage,
        "config": config.model_dump(mode="json"),
        "git": _git_metadata(),
        "packages": packages,
        "hardware": hardware,
        "metrics": metrics,
        "dataset": dataset,
    }
    with atomic_text_writer(Path(output_dir) / "run_manifest.json") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, default=str)
