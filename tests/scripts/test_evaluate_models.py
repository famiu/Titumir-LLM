from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

from scripts.evaluate_models import evaluate_models, run_gguf_smoke


def test_evaluate_models_writes_stable_result(tmp_path) -> None:
    output = tmp_path / "results.json"
    with (
        patch("scripts.evaluate_models.load_config") as load_config,
        patch("scripts.evaluate_models.generate_responses", return_value=["response"]) as generate,
        patch("scripts.evaluate_models.torch.cuda.is_available", return_value=True),
        patch("scripts.evaluate_models.torch.cuda.empty_cache") as empty_cache,
    ):
        config = load_config.return_value
        config.model.name = "base"
        config.model.max_seq_length = 128
        config.cpt_training.checkpoint = str(tmp_path / "missing-cpt")
        config.sft_training.checkpoint = str(tmp_path / "missing-sft")
        config.evaluation.prompts = ["prompt"]
        config.evaluation.max_new_tokens = 16
        evaluate_models(output=str(output))

    result = json.loads(output.read_text())
    assert result["models"]["base"]["responses"] == ["response"]
    assert result["models"]["cpt"]["status"] == "missing"
    assert generate.call_count == 1
    empty_cache.assert_called_once()


def test_model_failure_is_recorded_and_later_checkpoints_continue(tmp_path) -> None:
    output = tmp_path / "results.json"
    cpt_path = tmp_path / "cpt"
    sft_path = tmp_path / "sft"
    cpt_path.mkdir()
    sft_path.mkdir()
    with (
        patch("scripts.evaluate_models.load_config") as load_config,
        patch(
            "scripts.evaluate_models.generate_responses",
            side_effect=[RuntimeError("base load failed"), ["cpt response"], ["sft response"]],
        ) as generate,
        patch("scripts.evaluate_models.gc.collect") as collect,
        patch("scripts.evaluate_models.torch.cuda.is_available", return_value=True),
        patch("scripts.evaluate_models.torch.cuda.empty_cache") as empty_cache,
    ):
        config = load_config.return_value
        config.model.name = "base"
        config.model.max_seq_length = 128
        config.cpt_training.checkpoint = str(cpt_path)
        config.sft_training.checkpoint = str(sft_path)
        config.evaluation.prompts = ["prompt"]
        config.evaluation.max_new_tokens = 16
        evaluate_models(output=str(output))

    result = json.loads(output.read_text())
    assert result["models"]["base"] == {"path": "base", "status": "failed", "error": "base load failed"}
    assert result["models"]["cpt"]["responses"] == ["cpt response"]
    assert result["models"]["sft"]["responses"] == ["sft response"]
    assert generate.call_count == 3
    assert collect.call_count == 3
    assert empty_cache.call_count == 3


def test_gguf_smoke_skips_without_llama_cpp(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising=False)
    result = run_gguf_smoke(tmp_path / "model.gguf", "prompt", 10)
    assert result["status"] == "skipped"


def test_gguf_smoke_failure_preserves_model_results(tmp_path) -> None:
    output = tmp_path / "results.json"
    with (
        patch("scripts.evaluate_models.load_config") as load_config,
        patch("scripts.evaluate_models.generate_responses", return_value=["response"]),
        patch(
            "scripts.evaluate_models.run_gguf_smoke",
            side_effect=subprocess.TimeoutExpired("llama-cli", 300),
        ),
    ):
        config = load_config.return_value
        config.model.name = "base"
        config.model.max_seq_length = 128
        config.cpt_training.checkpoint = str(tmp_path / "missing-cpt")
        config.sft_training.checkpoint = str(tmp_path / "missing-sft")
        config.evaluation.prompts = ["prompt"]
        config.evaluation.max_new_tokens = 16
        evaluate_models(output=str(output), gguf=str(tmp_path / "model.gguf"))

    result = json.loads(output.read_text())
    assert result["models"]["base"]["responses"] == ["response"]
    assert result["gguf_smoke"]["status"] == "failed"
    assert "timed out" in result["gguf_smoke"]["error"]
