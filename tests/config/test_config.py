from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from training.config import (
    ApiConfigBase,
    Config,
    CPTDatasetEntry,
    CPTTrainingConfig,
    GenerationConfig,
    ModelConfig,
    ProfileConfig,
    RefinementConfig,
    SFTTrainingConfig,
)


class TestProfileConfig:
    def test_unprocessed_data_dir(self) -> None:
        profile = ProfileConfig(name="myprofile", data_root="data")
        assert profile.unprocessed_data_dir == "data/myprofile/unprocessed"

    def test_refined_data_dir(self) -> None:
        profile = ProfileConfig(name="myprofile", data_root="data")
        assert profile.refined_data_dir == "data/myprofile/refined"

    def test_removed_data_dir(self) -> None:
        profile = ProfileConfig(name="myprofile", data_root="data")
        assert profile.removed_data_dir == "data/myprofile/removed"

    def test_local_dataset(self) -> None:
        profile = ProfileConfig(name="myprofile", data_root="data")
        assert profile.local_dataset == "data/myprofile/myprofile_merged.jsonl"


class TestModelConfig:
    def test_max_seq_length_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(max_seq_length=0)

    def test_max_seq_length_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(max_seq_length=-1)


class TestCPTDatasetEntry:
    def test_probability_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            CPTDatasetEntry(path="some/path", probability=0)

    def test_probability_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            CPTDatasetEntry(path="some/path", probability=-0.5)


class TestCPTTrainingConfig:
    def test_max_examples_zero_raises(self) -> None:
        cfg = CPTTrainingConfig.model_construct(datasets=[])
        with pytest.raises(ValidationError) as exc_info:
            CPTTrainingConfig(max_examples=0, datasets=cfg.datasets, lora_r=16, lora_alpha=32)
        assert "max_examples" in str(exc_info.value)

    def test_lora_r_zero_raises(self) -> None:
        entry = CPTDatasetEntry(path="x", probability=1.0)
        with pytest.raises(ValidationError) as exc_info:
            CPTTrainingConfig(datasets=[entry], lora_r=0, lora_alpha=32)
        assert "lora_r" in str(exc_info.value)

    def test_lora_alpha_less_than_r_raises(self) -> None:
        entry = CPTDatasetEntry(path="x", probability=1.0)
        with pytest.raises(ValidationError) as exc_info:
            CPTTrainingConfig(datasets=[entry], lora_r=8, lora_alpha=4)
        assert "lora_alpha" in str(exc_info.value)

    def test_batch_size_zero_raises(self) -> None:
        entry = CPTDatasetEntry(path="x", probability=1.0)
        with pytest.raises(ValidationError) as exc_info:
            CPTTrainingConfig(datasets=[entry], batch_size=0)
        assert "batch_size" in str(exc_info.value)

    def test_grad_accum_zero_raises(self) -> None:
        entry = CPTDatasetEntry(path="x", probability=1.0)
        with pytest.raises(ValidationError) as exc_info:
            CPTTrainingConfig(datasets=[entry], grad_accum=0)
        assert "grad_accum" in str(exc_info.value)

    def test_epochs_zero_raises(self) -> None:
        entry = CPTDatasetEntry(path="x", probability=1.0)
        with pytest.raises(ValidationError) as exc_info:
            CPTTrainingConfig(datasets=[entry], epochs=0)
        assert "epochs" in str(exc_info.value)

    def test_learning_rate_zero_raises(self) -> None:
        entry = CPTDatasetEntry(path="x", probability=1.0)
        with pytest.raises(ValidationError) as exc_info:
            CPTTrainingConfig(datasets=[entry], learning_rate=0)
        assert "learning_rate" in str(exc_info.value)

    def test_empty_datasets_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CPTTrainingConfig(datasets=[])
        assert "datasets" in str(exc_info.value)

    def test_probabilities_sum_not_one_raises(self) -> None:
        entry_a = CPTDatasetEntry(path="x", probability=0.3)
        entry_b = CPTDatasetEntry(path="y", probability=0.4)
        with pytest.raises(ValidationError) as exc_info:
            CPTTrainingConfig(datasets=[entry_a, entry_b])
        assert "probability" in str(exc_info.value).lower() or "sum" in str(exc_info.value).lower()

    def test_valid_config_passes(self) -> None:
        entry = CPTDatasetEntry(path="x", probability=1.0)
        cfg = CPTTrainingConfig(datasets=[entry])
        assert cfg.datasets == [entry]


class TestSFTTrainingConfig:
    def test_batch_size_zero_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SFTTrainingConfig(batch_size=0)
        assert "batch_size" in str(exc_info.value)

    def test_grad_accum_zero_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SFTTrainingConfig(grad_accum=0)
        assert "grad_accum" in str(exc_info.value)

    def test_epochs_zero_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SFTTrainingConfig(epochs=0)
        assert "epochs" in str(exc_info.value)

    def test_learning_rate_zero_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SFTTrainingConfig(learning_rate=0)
        assert "learning_rate" in str(exc_info.value)

    def test_eval_split_zero_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SFTTrainingConfig(eval_split=0)
        assert "eval_split" in str(exc_info.value)

    def test_eval_split_one_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SFTTrainingConfig(eval_split=1)
        assert "eval_split" in str(exc_info.value)

    def test_eval_split_negative_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SFTTrainingConfig(eval_split=-0.1)
        assert "eval_split" in str(exc_info.value)

    def test_eval_split_none_valid(self) -> None:
        cfg = SFTTrainingConfig(eval_split=None)
        assert cfg.eval_split is None


class TestApiConfigBase:
    def test_temperature_negative_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ApiConfigBase(
                endpoint="http://x",
                api_key_env="X",
                model="y",
                temperature=-0.1,
                max_tokens=100,
                batch_size=10,
            )
        assert "temperature" in str(exc_info.value)

    def test_temperature_above_two_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ApiConfigBase(
                endpoint="http://x",
                api_key_env="X",
                model="y",
                temperature=2.1,
                max_tokens=100,
                batch_size=10,
            )
        assert "temperature" in str(exc_info.value)

    def test_max_tokens_zero_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ApiConfigBase(
                endpoint="http://x",
                api_key_env="X",
                model="y",
                temperature=0.5,
                max_tokens=0,
                batch_size=10,
            )
        assert "max_tokens" in str(exc_info.value)

    def test_batch_size_zero_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ApiConfigBase(
                endpoint="http://x",
                api_key_env="X",
                model="y",
                temperature=0.5,
                max_tokens=100,
                batch_size=0,
            )
        assert "batch_size" in str(exc_info.value)

    def test_batch_timeout_zero_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ApiConfigBase(
                endpoint="http://x",
                api_key_env="X",
                model="y",
                temperature=0.5,
                max_tokens=100,
                batch_size=10,
                batch_timeout=0,
            )
        assert "batch_timeout" in str(exc_info.value)

    def test_max_retries_zero_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ApiConfigBase(
                endpoint="http://x",
                api_key_env="X",
                model="y",
                temperature=0.5,
                max_tokens=100,
                batch_size=10,
                max_retries=0,
            )
        assert "max_retries" in str(exc_info.value)

    def test_max_workers_zero_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ApiConfigBase(
                endpoint="http://x",
                api_key_env="X",
                model="y",
                temperature=0.5,
                max_tokens=100,
                batch_size=10,
                max_workers=0,
            )
        assert "max_workers" in str(exc_info.value)

    def test_max_workers_none_valid(self) -> None:
        cfg = ApiConfigBase(
            endpoint="http://x",
            api_key_env="X",
            model="y",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
            max_workers=None,
        )
        assert cfg.max_workers is None

    def test_get_max_workers_returns_configured(self) -> None:
        cfg = ApiConfigBase(
            endpoint="http://x",
            api_key_env="X",
            model="y",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
            max_workers=4,
        )
        assert cfg.get_max_workers() == 4

    def test_get_max_workers_auto_calculates(self) -> None:
        cfg = ApiConfigBase(
            endpoint="http://x",
            api_key_env="X",
            model="y",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
            max_workers=None,
        )
        expected = min(32, (os.cpu_count() or 1) * 4)
        assert cfg.get_max_workers() == expected

    def test_get_api_key_returns_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
        cfg = ApiConfigBase(
            endpoint="http://x",
            api_key_env="OPENROUTER_API_KEY",
            model="y",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
        )
        assert cfg.get_api_key() == "test-key-123"

    def test_get_api_key_returns_none_for_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_KEY", raising=False)
        cfg = ApiConfigBase(
            endpoint="http://x",
            api_key_env="MISSING_KEY",
            model="y",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
        )
        assert cfg.get_api_key() is None


class TestGenerationConfig:
    def test_default_endpoint(self) -> None:
        cfg = GenerationConfig(
            model="test",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
        )
        assert cfg.endpoint == "https://openrouter.ai/api/v1/chat/completions"

    def test_default_api_key_env(self) -> None:
        cfg = GenerationConfig(
            model="test",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
        )
        assert cfg.api_key_env == "OPENROUTER_API_KEY"


class TestRefinementConfig:
    def test_default_endpoint(self) -> None:
        cfg = RefinementConfig(
            model="test",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
        )
        assert cfg.endpoint == "https://openrouter.ai/api/v1/chat/completions"

    def test_default_api_key_env(self) -> None:
        cfg = RefinementConfig(
            model="test",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
        )
        assert cfg.api_key_env == "OPENROUTER_API_KEY"


class TestConfig:
    @pytest.mark.parametrize("seed", [0, 4_294_967_295])
    def test_seed_boundaries_valid(self, dummy_config: Config, seed: int) -> None:
        data = dummy_config.model_dump()
        data["seed"] = seed
        assert Config.model_validate(data).seed == seed

    @pytest.mark.parametrize("seed", [-1, 4_294_967_296])
    def test_seed_out_of_range_raises(self, dummy_config: Config, seed: int) -> None:
        data = dummy_config.model_dump()
        data["seed"] = seed
        with pytest.raises(ValidationError, match="seed"):
            Config.model_validate(data)

    def test_empty_topics_with_generation_prompt_raises(self) -> None:
        cfg_dict = {
            "profile": {"name": "test", "data_root": "data"},
            "model": {"name": "test", "max_seq_length": 512},
            "cpt_training": {
                "datasets": [{"path": "x", "probability": 1.0}],
                "output_dir": "./o",
                "checkpoint": "./c",
            },
            "sft_training": {"output_dir": "./o", "checkpoint": "./c"},
            "generation": {
                "model": "test",
                "temperature": 0.5,
                "max_tokens": 100,
                "batch_size": 10,
                "prompt": "some prompt",
            },
            "refinement": {
                "model": "test",
                "temperature": 0.5,
                "max_tokens": 100,
                "batch_size": 10,
            },
            "topics": [],
        }
        with pytest.raises(ValidationError) as exc_info:
            Config.model_validate(cfg_dict)
        assert "topics" in str(exc_info.value).lower()

    def test_empty_topics_with_empty_prompt_valid(self) -> None:
        cfg_dict = {
            "profile": {"name": "test", "data_root": "data"},
            "model": {"name": "test", "max_seq_length": 512},
            "cpt_training": {
                "datasets": [{"path": "x", "probability": 1.0}],
                "output_dir": "./o",
                "checkpoint": "./c",
            },
            "sft_training": {"output_dir": "./o", "checkpoint": "./c"},
            "generation": {
                "model": "test",
                "temperature": 0.5,
                "max_tokens": 100,
                "batch_size": 10,
                "prompt": "",
            },
            "refinement": {
                "model": "test",
                "temperature": 0.5,
                "max_tokens": 100,
                "batch_size": 10,
            },
            "topics": [],
        }
        cfg = Config.model_validate(cfg_dict)
        assert cfg.topics == []

    def test_load_config_defaults_to_config_yaml(self) -> None:
        from training.config import load_config

        cfg = load_config()
        assert cfg.profile.name == "default"

    def test_load_config_nonexistent_file_raises(self) -> None:
        from training.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_cpt_training_is_required(self) -> None:
        with pytest.raises(ValueError, match="cpt_training"):
            Config.model_validate({"profile": {"name": "test"}})

    def test_from_yaml_loads_all_sections(self, dummy_config: Config) -> None:
        assert dummy_config.profile.name == "dry_run"
        assert dummy_config.model.name == "unsloth/Qwen2.5-0.5B-Instruct"
        assert len(dummy_config.cpt_training.datasets) == 1
        assert len(dummy_config.sft_training.output_dir) > 0
        assert dummy_config.export.quantization_method == "q4_k_m"

    def test_topics_list_format_converted(self, tmp_path: pytest.TempPathFactory) -> None:
        yaml_content = """\
profile:
  name: test
  data_root: data
cpt_training:
  datasets:
    - path: x
      probability: 1.0
  output_dir: ./o
  checkpoint: ./c
sft_training:
  output_dir: ./o
  checkpoint: ./c
generation:
  model: test
  temperature: 0.5
  max_tokens: 100
  batch_size: 10
refinement:
  model: test
  temperature: 0.5
  max_tokens: 100
  batch_size: 10
topics:
  - ["topic one", 10]
  - ["topic two", 20]
"""
        config_file = tmp_path / "test_topics.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")
        cfg = Config.from_yaml(config_file)
        assert len(cfg.topics) == 2
        assert cfg.topics[0].topic == "topic one"
        assert cfg.topics[0].count == 10
        assert cfg.topics[1].topic == "topic two"
        assert cfg.topics[1].count == 20
