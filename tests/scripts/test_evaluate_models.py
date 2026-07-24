from __future__ import annotations

import json
from unittest.mock import patch

from scripts.evaluate_models import evaluate_models, run_gguf_smoke


def test_evaluate_models_writes_stable_result(tmp_path) -> None:
    output = tmp_path / "results.json"
    with (
        patch("scripts.evaluate_models.load_config") as load_config,
        patch("scripts.evaluate_models.generate_responses", return_value=["response"]) as generate,
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


def test_gguf_smoke_skips_without_llama_cpp(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising=False)
    result = run_gguf_smoke(tmp_path / "model.gguf", "prompt", 10)
    assert result["status"] == "skipped"
