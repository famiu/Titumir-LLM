"""Configuration management for the Titumir LLM project."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Configuration for data paths."""

    unprocessed_data_dir: str = "data/unprocessed"
    refined_data_dir: str = "data/refined"
    removed_data_dir: str = "data/removed"
    default_dataset: str = "data/refined/bangla_sft_merged.jsonl"
    hf_dataset: str = "famiu/titumir-sft-dataset"


class ModelConfig(BaseModel):
    """Configuration for the base model."""

    name: str = "Qwen/Qwen3.5-9B"
    max_seq_length: int = 2048


class CPTConfig(BaseModel):
    """Configuration for continued pretraining."""

    max_examples: int = 500_000
    output_dir: str = "./checkpoints/cpt"
    checkpoint: str = "./checkpoints/cpt_final"


class SFTConfig(BaseModel):
    """Configuration for supervised finetuning."""

    output_dir: str = "./checkpoints/sft"
    checkpoint: str = "./checkpoints/sft_final"


class ExportConfig(BaseModel):
    """Configuration for model export."""

    path: str = "./export/titumir_9b"


class LLMEndpointConfig(BaseModel):
    """Configuration for LLM API endpoint."""

    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key_env: str = "OPENROUTER_API_KEY"
    model: str = "google/gemini-3.1-flash-lite-preview"
    temperature: float = 0.9
    max_tokens: int = 4000

    def get_api_key(self) -> str | None:
        """Get the API key from the environment variable."""
        return os.environ.get(self.api_key_env)


class LLMConfig(BaseModel):
    """Configuration for LLM settings."""

    generation: LLMEndpointConfig = Field(default_factory=LLMEndpointConfig)
    refinement: LLMEndpointConfig | None = None

    def get_refinement_config(self) -> LLMEndpointConfig:
        """Get refinement config, falling back to generation config if not set."""
        return self.refinement or self.generation


class PromptsConfig(BaseModel):
    """Configuration for prompts."""

    generation: str = ""
    refinement: str = ""


class TopicEntry(BaseModel):
    """A topic entry for dataset generation."""

    topic: str
    count: int


class Config(BaseModel):
    """Root configuration class."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    cpt: CPTConfig = Field(default_factory=CPTConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    topics: list[TopicEntry] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        if "topics" in data and isinstance(data["topics"], list):
            data["topics"] = [
                {"topic": t[0], "count": t[1]} if isinstance(t, list | tuple) else t for t in data["topics"]
            ]

        return cls.model_validate(data)


@lru_cache(maxsize=1)
def _load_default_config() -> Config:
    """Load the default configuration with caching."""
    default_path = Path("configs/config.yaml")
    if default_path.exists():
        return Config.from_yaml(default_path)
    return Config()


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a YAML file, or use default if path is None."""
    if path is None:
        return _load_default_config()

    config_path = Path(path)
    return Config.from_yaml(config_path)


def get_config() -> Config:
    """Get the default configuration."""
    return _load_default_config()


def reset_config_cache() -> None:
    """Clear the default configuration cache."""
    _load_default_config.cache_clear()
