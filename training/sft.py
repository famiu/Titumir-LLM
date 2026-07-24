import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from unsloth import FastLanguageModel  # isort: skip
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer

from scripts._data import conversation_key, validate_conversation
from training.config import load_config
from training.runtime import (
    configure_seed,
    ensure_trainable,
    precision_args,
    resolve_resume_checkpoint,
    write_run_manifest,
)


def fingerprint_examples(examples: list[dict]) -> str:
    """Hash materialized examples using stable canonical JSON."""
    digest = hashlib.sha256()
    for example in examples:
        encoded = json.dumps(example, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def split_conversations(examples: list[dict], eval_split: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split normalized conversation groups while preserving topic proportions."""
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for example in examples:
        metadata = example.get("metadata")
        topic = metadata.get("topic", "unknown") if isinstance(metadata, dict) else "unknown"
        grouped[topic][conversation_key(example)].append(example)

    train = []
    evaluation = []
    deferred_singletons = []
    for topic, groups_by_key in sorted(grouped.items()):
        groups = list(groups_by_key.values())
        groups.sort(
            key=lambda group: hashlib.sha256(f"{seed}:{topic}:{conversation_key(group[0])}".encode()).hexdigest()
        )
        if len(groups) == 1:
            deferred_singletons.append(groups[0])
            continue
        eval_count = max(1, min(len(groups) - 1, round(len(groups) * eval_split)))
        for group in groups[:eval_count]:
            evaluation.extend(group)
        for group in groups[eval_count:]:
            train.extend(group)

    for group in deferred_singletons:
        train.extend(group)
    if not evaluation and len(deferred_singletons) > 1:
        evaluation.extend(deferred_singletons.pop())
        train = [example for group in deferred_singletons for example in group]
    if not train or not evaluation:
        raise ValueError("SFT dataset is too small for the configured grouped evaluation split")
    return train, evaluation


def run_sft(config_path: str | None = None, model=None, tokenizer=None) -> tuple:
    """Run supervised finetuning on conversational dataset."""
    config = load_config(config_path)
    model_cfg = config.model
    cpt_cfg = config.cpt_training
    sft_cfg = config.sft_training
    configure_seed(config.seed)

    if model is None or tokenizer is None:
        if not os.path.isdir(cpt_cfg.checkpoint):
            raise FileNotFoundError(
                f"CPT checkpoint not found at '{cpt_cfg.checkpoint}'. "
                "Run 'just cpt' first or set 'cpt_training.checkpoint' in your config."
            )

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cpt_cfg.checkpoint,
            max_seq_length=model_cfg.max_seq_length,
            load_in_4bit=model_cfg.load_in_4bit,
        )
        ensure_trainable(model)

    local_path = Path(config.profile.local_dataset)
    if local_path.exists():
        data = []
        with open(local_path, encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Malformed JSON in {local_path} at line {line_num}: {error}") from error
                data.append(validate_conversation(example, f"{local_path}:{line_num}"))
        print(f"Loaded {len(data)} examples from {local_path}")
    elif config.profile.hf_dataset is not None:
        print(f"Local dataset not found at {local_path} — falling back to HuggingFace Hub")
        try:
            hub_dataset = load_dataset(
                config.profile.hf_dataset,
                split="train",
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset '{config.profile.hf_dataset}' from HuggingFace Hub. "
                "Check your internet connection and HF_TOKEN."
            ) from e
        data = [
            validate_conversation(example, f"{config.profile.hf_dataset} row {index}")
            for index, example in enumerate(hub_dataset)
        ]
        print(f"Loaded {len(data)} examples from {config.profile.hf_dataset}")
    else:
        raise FileNotFoundError(
            f"No local dataset found at {local_path} and no HF dataset "
            f"configured for profile '{config.profile.name}'. "
            "Run the data pipeline first or set 'profile.hf_dataset' in your config."
        )

    if not data:
        raise ValueError("SFT dataset is empty")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The selected tokenizer does not define a chat template required for SFT")

    eval_dataset = None
    if sft_cfg.eval_split is not None and sft_cfg.eval_split > 0:
        train_examples, eval_examples = split_conversations(data, sft_cfg.eval_split, config.seed)
        print(f"Split dataset: {len(train_examples)} train, {len(eval_examples)} eval")
    else:
        train_examples = data
        eval_examples = []

    def prepare_example(example: dict) -> dict:
        metadata = example.get("metadata")
        if sft_cfg.assistant_only_loss:
            prepared = {
                "prompt": [example["messages"][0]],
                "completion": [example["messages"][1]],
            }
        else:
            prepared = {"messages": example["messages"]}
        if metadata is not None:
            prepared["metadata"] = metadata
        return prepared

    prepared_train = [prepare_example(example) for example in train_examples]
    prepared_eval = [prepare_example(example) for example in eval_examples]
    train_fingerprint = fingerprint_examples(prepared_train)
    eval_fingerprint = fingerprint_examples(prepared_eval) if prepared_eval else None
    dataset = Dataset.from_list(prepared_train).shuffle(seed=config.seed)
    if prepared_eval:
        eval_dataset = Dataset.from_list(prepared_eval).shuffle(seed=config.seed)

    precision = precision_args()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            max_length=model_cfg.max_seq_length,
            completion_only_loss=sft_cfg.assistant_only_loss,
            packing=False,
            learning_rate=sft_cfg.learning_rate,
            num_train_epochs=sft_cfg.epochs,
            per_device_train_batch_size=sft_cfg.batch_size,
            gradient_accumulation_steps=sft_cfg.grad_accum,
            **precision,
            seed=config.seed,
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,
            output_dir=sft_cfg.output_dir,
            warmup_steps=0.05,
            lr_scheduler_type="cosine",
            report_to="none",
            eval_strategy="epoch" if eval_dataset is not None else "no",
        ),
    )

    print("Starting SFT...")
    resume_checkpoint = resolve_resume_checkpoint(sft_cfg.resume_from_checkpoint, sft_cfg.output_dir)
    train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)

    model.save_pretrained(sft_cfg.checkpoint)
    tokenizer.save_pretrained(sft_cfg.checkpoint)
    metrics = getattr(train_output, "metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    write_run_manifest(
        "sft",
        config,
        sft_cfg.output_dir,
        metrics,
        {
            "train_fingerprint": train_fingerprint,
            "eval_fingerprint": eval_fingerprint,
            "train_examples": len(dataset),
            "eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
            "resume_checkpoint": resume_checkpoint,
        },
    )
    print(f"SFT complete — saved to {sft_cfg.checkpoint}")

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run supervised finetuning")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    run_sft(config_path=args.config)
