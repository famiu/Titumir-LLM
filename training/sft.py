import json
import sys

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from training.config import (
    CPT_CHECKPOINT,
    HF_DATASET,
    MAX_SEQ_LENGTH,
    SFT_CHECKPOINT,
    SFT_OUTPUT_DIR,
)


def run_sft(dataset_path: str | None = None) -> None:
    """Run supervised finetuning on conversational dataset."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CPT_CHECKPOINT,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if dataset_path is not None:
        # Load from local file
        with open(dataset_path, encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(data)} examples from {dataset_path}")
        dataset = Dataset.from_list(data)
    else:
        # Load from HuggingFace Hub
        from datasets import load_dataset

        dataset = load_dataset(HF_DATASET, split="train")
        print(f"Loaded {len(dataset)} examples from {HF_DATASET}")

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
            max_seq_length=MAX_SEQ_LENGTH,
            learning_rate=2e-4,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            bf16=True,
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,
            output_dir=SFT_OUTPUT_DIR,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            report_to="none",
        ),
    )

    print("Starting SFT...")
    trainer.train()

    model.save_pretrained(SFT_CHECKPOINT)
    tokenizer.save_pretrained(SFT_CHECKPOINT)
    print(f"SFT complete — saved to {SFT_CHECKPOINT}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
    run_sft(path)
