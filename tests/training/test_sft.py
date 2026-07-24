from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from training.sft import run_sft, split_conversations


def make_example(index: int, topic: str = "topic") -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"post {index}"},
            {"role": "assistant", "content": f"reply {index}"},
        ],
        "metadata": {"topic": topic},
    }


def test_grouped_split_is_deterministic_and_disjoint() -> None:
    examples = [make_example(index, "a" if index < 10 else "b") for index in range(20)]
    duplicate = json.loads(json.dumps(examples[0]))
    examples.append(duplicate)

    train, evaluation = split_conversations(examples, 0.2)
    train_keys = {json.dumps(example["messages"], sort_keys=True) for example in train}
    eval_keys = {json.dumps(example["messages"], sort_keys=True) for example in evaluation}
    assert train_keys.isdisjoint(eval_keys)
    assert split_conversations(examples, 0.2) == (train, evaluation)


def test_sft_uses_prompt_completion_and_evaluation(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    examples = [make_example(index) for index in range(10)]
    dataset_path.write_text("".join(json.dumps(example) + "\n" for example in examples))
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.chat_template = "template"
    trainer = MagicMock()

    with (
        patch("training.sft.load_config") as load_config,
        patch("training.sft.SFTTrainer", return_value=trainer) as trainer_class,
        patch("training.sft.write_run_manifest"),
    ):
        config = load_config.return_value
        config.seed = 42
        config.profile.local_dataset = str(dataset_path)
        config.profile.hf_dataset = None
        config.model.max_seq_length = 128
        config.sft_training.eval_split = 0.2
        config.sft_training.assistant_only_loss = True
        config.sft_training.resume_from_checkpoint = False
        config.sft_training.learning_rate = 1e-4
        config.sft_training.epochs = 1
        config.sft_training.batch_size = 1
        config.sft_training.grad_accum = 1
        config.sft_training.output_dir = "output"
        config.sft_training.checkpoint = "final"
        run_sft(model=model, tokenizer=tokenizer)

    kwargs = trainer_class.call_args.kwargs
    assert "prompt" in kwargs["train_dataset"].column_names
    assert "completion" in kwargs["train_dataset"].column_names
    assert kwargs["args"].completion_only_loss is True
    assert kwargs["args"].eval_strategy == "epoch"
    trainer.train.assert_called_once_with(resume_from_checkpoint=None)


def test_sft_reports_malformed_line(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps(make_example(1)) + "\nnot json\n")
    tokenizer = MagicMock()
    tokenizer.chat_template = "template"

    with patch("training.sft.load_config") as load_config:
        config = load_config.return_value
        config.seed = 42
        config.profile.local_dataset = str(dataset_path)
        config.profile.hf_dataset = None
        with __import__("pytest").raises(ValueError, match="line 2"):
            run_sft(model=MagicMock(), tokenizer=tokenizer)
