import argparse
import math
from collections import Counter

from unsloth import FastLanguageModel  # isort: skip
from datasets import concatenate_datasets, interleave_datasets, load_dataset
from trl import SFTConfig, SFTTrainer

from training.config import load_config
from training.runtime import configure_seed, precision_args, resolve_resume_checkpoint, write_run_manifest


def run_cpt(config_path: str | None = None, model=None, tokenizer=None) -> tuple:
    """Run continued pretraining on raw Bengali text, prioritizing colloquial sources."""
    config = load_config(config_path)
    model_cfg = config.model
    cpt_cfg = config.cpt_training
    configure_seed(config.seed)

    if model is None or tokenizer is None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_cfg.name,
            max_seq_length=model_cfg.max_seq_length,
            load_in_4bit=model_cfg.load_in_4bit,
        )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cpt_cfg.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=cpt_cfg.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    # ── Load and interleave datasets from config ─────────────────────────
    loaded_datasets = []
    eval_datasets = []
    probabilities = []

    for entry in cpt_cfg.datasets:
        load_kwargs = {}
        if entry.config:
            load_kwargs["name"] = entry.config
        if entry.revision:
            load_kwargs["revision"] = entry.revision
        try:
            ds = load_dataset(entry.path, **load_kwargs, split=entry.split)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CPT dataset '{entry.path}' (split={entry.split}, config={entry.config}). "
                "Check your internet connection and that the dataset exists."
            ) from e
        print(f"Loaded {entry.path} [{entry.split}]: {len(ds)} examples, columns: {ds.column_names}")
        if entry.column not in ds.column_names:
            raise ValueError(
                f"CPT dataset '{entry.path}' does not contain configured column '{entry.column}'. "
                f"Available columns: {ds.column_names}"
            )
        if entry.column != "text":
            ds = ds.rename_column(entry.column, "text")
        ds = ds.select_columns(["text"])
        before_filter = len(ds)
        ds = ds.filter(lambda example: isinstance(example["text"], str) and bool(example["text"].strip()))
        print(f"  Kept {len(ds)}/{before_filter} non-empty examples")
        if len(ds) < 2:
            raise ValueError(f"CPT dataset '{entry.path}' needs at least two non-empty examples for evaluation")
        split = ds.train_test_split(test_size=cpt_cfg.eval_split, seed=config.seed)
        source_name = entry.path if entry.config is None else f"{entry.path}/{entry.config}"
        train_ds = split["train"].add_column("source", [source_name] * len(split["train"]))
        eval_ds = split["test"].add_column("source", [source_name] * len(split["test"]))
        loaded_datasets.append(train_ds)
        eval_datasets.append(eval_ds)
        probabilities.append(entry.probability)

    interleaved = interleave_datasets(
        loaded_datasets,
        probabilities=probabilities,
        seed=config.seed,
        stopping_strategy="all_exhausted_without_replacement",
    )
    available = len(interleaved)
    selected = min(cpt_cfg.max_examples, available)
    if selected < cpt_cfg.max_examples:
        print(f"Requested {cpt_cfg.max_examples} CPT examples, but only {available} are available")
    raw_dataset = interleaved.shuffle(seed=config.seed).select(range(selected))
    eval_dataset = concatenate_datasets(eval_datasets).shuffle(seed=config.seed)
    eval_limit = min(len(eval_dataset), max(1, round(cpt_cfg.max_examples * cpt_cfg.eval_split)))
    eval_dataset = eval_dataset.select(range(eval_limit))

    print(f"Total CPT examples: {len(raw_dataset)}")
    realized = Counter(raw_dataset["source"])
    for source, count in sorted(realized.items()):
        print(f"  Realized source mix: {source}: {count} ({count / len(raw_dataset):.1%})")

    precision = precision_args()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=raw_dataset.remove_columns("source"),
        eval_dataset=eval_dataset.remove_columns("source"),
        args=SFTConfig(
            dataset_text_field="text",
            max_length=model_cfg.max_seq_length,
            packing=cpt_cfg.packing,
            learning_rate=cpt_cfg.learning_rate,
            num_train_epochs=cpt_cfg.epochs,
            per_device_train_batch_size=cpt_cfg.batch_size,
            gradient_accumulation_steps=cpt_cfg.grad_accum,
            **precision,
            seed=config.seed,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            output_dir=cpt_cfg.output_dir,
            warmup_steps=0.05,
            lr_scheduler_type="cosine",
            report_to="none",
            eval_strategy="epoch",
        ),
    )

    print("Starting CPT...")
    resume_checkpoint = resolve_resume_checkpoint(cpt_cfg.resume_from_checkpoint, cpt_cfg.output_dir)
    train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss")
    if eval_loss is not None:
        perplexity = math.exp(eval_loss) if eval_loss < 100 else math.inf
        print(f"CPT eval loss: {eval_loss:.4f}, perplexity: {perplexity:.4f}")

    model.save_pretrained(cpt_cfg.checkpoint)
    tokenizer.save_pretrained(cpt_cfg.checkpoint)
    train_metrics = getattr(train_output, "metrics", {})
    if not isinstance(train_metrics, dict):
        train_metrics = {}
    write_run_manifest(
        "cpt",
        config,
        cpt_cfg.output_dir,
        {**train_metrics, **eval_metrics},
        {
            "train_fingerprint": raw_dataset._fingerprint,
            "eval_fingerprint": eval_dataset._fingerprint,
            "train_examples": len(raw_dataset),
            "eval_examples": len(eval_dataset),
            "source_counts": dict(realized),
            "sources": [entry.model_dump(mode="json") for entry in cpt_cfg.datasets],
            "resume_checkpoint": resume_checkpoint,
        },
    )
    print(f"CPT complete — saved to {cpt_cfg.checkpoint}")

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run continued pretraining")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    run_cpt(config_path=args.config)
