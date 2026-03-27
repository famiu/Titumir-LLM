import argparse

from unsloth import FastLanguageModel  # isort: skip
from datasets import interleave_datasets, load_dataset
from trl import SFTConfig, SFTTrainer

from training.config import load_config


def run_cpt(config_path: str | None = None) -> None:
    """Run continued pretraining on raw Bengali text, prioritizing colloquial sources."""
    config = load_config(config_path)
    model_cfg = config.model
    cpt_cfg = config.cpt_training

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
        random_state=42,
    )

    # ── Load and interleave datasets from config ─────────────────────────
    loaded_datasets = []
    probabilities = []

    for entry in cpt_cfg.datasets:
        load_kwargs = {}
        if entry.config:
            load_kwargs["name"] = entry.config
        ds = load_dataset(entry.path, **load_kwargs, split=entry.split)
        print(f"Loaded {entry.path} [{entry.split}]: {len(ds)} examples, columns: {ds.column_names}")
        if entry.column != "text":
            ds = ds.rename_column(entry.column, "text")
        ds = ds.select_columns(["text"])
        loaded_datasets.append(ds)
        probabilities.append(entry.probability)

    raw_dataset = (
        interleave_datasets(
            loaded_datasets,
            probabilities=probabilities,
            seed=42,
            stopping_strategy="all_exhausted_without_replacement",
        )
        .shuffle(seed=42)
        .select(range(cpt_cfg.max_examples))
    )

    print(f"Total CPT examples: {len(raw_dataset)}")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=raw_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=model_cfg.max_seq_length,
            learning_rate=cpt_cfg.learning_rate,
            num_train_epochs=cpt_cfg.epochs,
            per_device_train_batch_size=cpt_cfg.batch_size,
            gradient_accumulation_steps=cpt_cfg.grad_accum,
            bf16=True,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            output_dir=cpt_cfg.output_dir,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            report_to="none",
        ),
    )

    print("Starting CPT...")
    trainer.train()

    model.save_pretrained(cpt_cfg.checkpoint)
    tokenizer.save_pretrained(cpt_cfg.checkpoint)
    print(f"CPT complete — saved to {cpt_cfg.checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run continued pretraining")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    run_cpt(config_path=args.config)
