import argparse
import json

from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from training.config import load_config


def run_sft(config_path: str | None = None, dataset_path: str | None = None) -> None:
    """Run supervised finetuning on conversational dataset."""
    config = load_config(config_path)
    model_cfg = config.model
    cpt_cfg = config.cpt_training
    sft_cfg = config.sft_training

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cpt_cfg.checkpoint,
        max_seq_length=model_cfg.max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=sft_cfg.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=sft_cfg.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if dataset_path is not None:
        with open(dataset_path, encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(data)} examples from {dataset_path}")
        dataset = Dataset.from_list(data)
    else:
        dataset = load_dataset(config.paths.hf_dataset, split="train")
        print(f"Loaded {len(dataset)} examples from {config.paths.hf_dataset}")

    def format_example(example: dict) -> dict:
        """Format a single example using the model's chat template."""
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = dataset.map(format_example).shuffle(seed=42)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=model_cfg.max_seq_length,
            learning_rate=2e-4,
            num_train_epochs=3,
            per_device_train_batch_size=sft_cfg.batch_size,
            gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
            bf16=True,
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,
            output_dir=sft_cfg.output_dir,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            report_to="none",
        ),
    )

    print("Starting SFT...")
    trainer.train()

    model.save_pretrained(sft_cfg.checkpoint)
    tokenizer.save_pretrained(sft_cfg.checkpoint)
    print(f"SFT complete — saved to {sft_cfg.checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run supervised finetuning")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    parser.add_argument("dataset", nargs="?", type=str, default=None, help="Optional path to local dataset")
    args = parser.parse_args()
    run_sft(config_path=args.config, dataset_path=args.dataset)
