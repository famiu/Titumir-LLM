from __future__ import annotations

from unittest.mock import MagicMock, patch

from datasets import Dataset

from training.config import CPTDatasetEntry
from training.cpt import run_cpt


def test_cpt_caps_examples_filters_text_and_evaluates() -> None:
    source = Dataset.from_dict({"text": ["এক", "দুই", "তিন", "চার", "", "পাঁচ"]})
    model = MagicMock()
    tokenizer = MagicMock()
    peft_model = MagicMock()
    trainer = MagicMock()
    trainer.evaluate.return_value = {"eval_loss": 1.0}

    with (
        patch("training.cpt.load_config") as load_config,
        patch("training.cpt.load_dataset", return_value=source) as load_dataset,
        patch("training.cpt.FastLanguageModel.get_peft_model", return_value=peft_model),
        patch("training.cpt.SFTTrainer", return_value=trainer) as trainer_class,
        patch("training.cpt.write_run_manifest") as write_manifest,
    ):
        config = load_config.return_value
        config.seed = 42
        config.model.max_seq_length = 128
        config.cpt_training.datasets = [
            CPTDatasetEntry(
                path="test/source",
                split="train",
                column="text",
                probability=1.0,
                revision="abc123",
                retrieved_at="2026-07-24",
            )
        ]
        config.cpt_training.max_examples = 100
        config.cpt_training.eval_split = 0.2
        config.cpt_training.packing = True
        config.cpt_training.resume_from_checkpoint = False
        config.cpt_training.lora_r = 8
        config.cpt_training.lora_alpha = 16
        config.cpt_training.learning_rate = 1e-5
        config.cpt_training.epochs = 1
        config.cpt_training.batch_size = 1
        config.cpt_training.grad_accum = 1
        config.cpt_training.output_dir = "output"
        config.cpt_training.checkpoint = "final"

        run_cpt(model=model, tokenizer=tokenizer)

    kwargs = trainer_class.call_args.kwargs
    assert len(kwargs["train_dataset"]) == 4
    assert len(kwargs["eval_dataset"]) == 1
    assert kwargs["args"].packing is True
    assert kwargs["args"].eval_strategy == "epoch"
    trainer.train.assert_called_once_with(resume_from_checkpoint=None)
    trainer.evaluate.assert_called_once()
    load_dataset.assert_called_once_with("test/source", revision="abc123", split="train")
    manifest_dataset = write_manifest.call_args.args[4]
    assert manifest_dataset["sources"][0]["revision"] == "abc123"
    assert manifest_dataset["sources"][0]["retrieved_at"] == "2026-07-24"
