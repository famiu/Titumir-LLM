"""Configuration management for the Titumir LLM project."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class PathsConfig(BaseModel):
    """Configuration for data paths."""

    unprocessed_data_dir: str = "data/unprocessed"
    refined_data_dir: str = "data/refined"
    removed_data_dir: str = "data/removed"
    local_dataset: str = "data/refined/bangla_sft_merged.jsonl"
    hf_dataset: str = "famiu/titumir-sft-dataset"


class ModelConfig(BaseModel):
    """Configuration for the base model."""

    name: str = "Qwen/Qwen3.5-9B"
    max_seq_length: int = 2048

    @field_validator("max_seq_length")
    @classmethod
    def max_seq_length_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_seq_length must be positive")
        return v


class CPTTrainingConfig(BaseModel):
    """Configuration for continued pretraining."""

    max_examples: int = 500_000
    output_dir: str = "./checkpoints/cpt"
    checkpoint: str = "./checkpoints/cpt_final"
    lora_r: int = 16
    lora_alpha: int = 32
    batch_size: int = 4
    gradient_accumulation_steps: int = 4

    @field_validator("max_examples", "lora_r", "batch_size", "gradient_accumulation_steps")
    @classmethod
    def must_be_positive(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v

    @model_validator(mode="after")
    def lora_alpha_gte_r(self) -> CPTTrainingConfig:
        if self.lora_alpha < self.lora_r:
            raise ValueError("lora_alpha must be >= lora_r")
        return self


class SFTTrainingConfig(BaseModel):
    """Configuration for supervised finetuning."""

    output_dir: str = "./checkpoints/sft"
    checkpoint: str = "./checkpoints/sft_final"
    lora_r: int = 32
    lora_alpha: int = 64
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    eval_split: float | None = None

    @field_validator("lora_r", "batch_size", "gradient_accumulation_steps")
    @classmethod
    def must_be_positive(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v

    @field_validator("eval_split")
    @classmethod
    def eval_split_range(cls, v: float | None) -> float | None:
        if v is not None and (v <= 0 or v >= 1):
            raise ValueError("eval_split must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def lora_alpha_gte_r(self) -> SFTTrainingConfig:
        if self.lora_alpha < self.lora_r:
            raise ValueError("lora_alpha must be >= lora_r")
        return self


class GenerationConfig(BaseModel):
    """Configuration for dataset generation."""

    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key_env: str = "OPENROUTER_API_KEY"
    model: str = "CHANGE_ME"
    temperature: float = 0.9
    max_tokens: int = 4000
    batch_size: int = 20
    batch_timeout: int = 120
    prompt: str = ""

    @field_validator("temperature")
    @classmethod
    def temperature_range(cls, v: float) -> float:
        if v < 0 or v > 2:
            raise ValueError("temperature must be between 0 and 2")
        return v

    @field_validator("max_tokens", "batch_size", "batch_timeout")
    @classmethod
    def must_be_positive(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v

    def get_api_key(self) -> str | None:
        """Get the API key from the environment variable."""
        return os.environ.get(self.api_key_env)


class RefinementConfig(BaseModel):
    """Configuration for dataset refinement."""

    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key_env: str = "OPENROUTER_API_KEY"
    model: str = "CHANGE_ME"
    temperature: float = 0.1
    max_tokens: int = 1000
    batch_size: int = 40
    batch_timeout: int = 120
    prompt: str = ""

    @field_validator("temperature")
    @classmethod
    def temperature_range(cls, v: float) -> float:
        if v < 0 or v > 2:
            raise ValueError("temperature must be between 0 and 2")
        return v

    @field_validator("max_tokens", "batch_size", "batch_timeout")
    @classmethod
    def must_be_positive(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v

    def get_api_key(self) -> str | None:
        """Get the API key from the environment variable."""
        return os.environ.get(self.api_key_env)


class ExportConfig(BaseModel):
    """Configuration for model export."""

    path: str = "./export/titumir_9b"
    quantization_method: str = "q4_k_m"


class TopicEntry(BaseModel):
    """A topic entry for dataset generation."""

    topic: str
    count: int

    @field_validator("count")
    @classmethod
    def count_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("topic count must be positive")
        return v


class Config(BaseModel):
    """Root configuration class."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    cpt_training: CPTTrainingConfig = Field(default_factory=CPTTrainingConfig)
    sft_training: SFTTrainingConfig = Field(default_factory=SFTTrainingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    topics: list[TopicEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def topics_required_if_generation(self) -> Config:
        if self.topics is not None and len(self.topics) == 0 and self.generation.prompt:
            raise ValueError("topics list must not be empty if generation prompt is configured")
        return self

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


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a YAML file, or use default if path is None."""
    if path is None:
        default_path = Path("configs/config.yaml")
        if default_path.exists():
            return Config.from_yaml(default_path)
        return Config()

    return Config.from_yaml(Path(path))
