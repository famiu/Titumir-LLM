from datasets import interleave_datasets, load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from training.config import (
    CPT_CHECKPOINT,
    CPT_MAX_EXAMPLES,
    CPT_OUTPUT_DIR,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
)


def run_cpt() -> None:
    """Run continued pretraining on raw Bengali text, prioritizing colloquial sources."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── Colloquial / social media sources (priority) ───────────────────────

    # BanglishRev — 1.74M e-commerce reviews in Bengali/English/Banglish
    banglish_rev = load_dataset(
        "BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset",
        split="train",
    )
    assert banglish_rev.column_names is not None
    print(f"BanglishRev columns: {banglish_rev.column_names}")
    banglish_rev = banglish_rev.map(
        lambda x: {"text": x["review"]},
        remove_columns=list(banglish_rev.column_names),
    )

    # Ben-Sarc — 25,636 Facebook comments
    ben_sarc = load_dataset("sanzanalora/Ben-Sarc", split="train")
    assert ben_sarc.column_names is not None
    print(f"Ben-Sarc columns: {ben_sarc.column_names}")
    ben_sarc = ben_sarc.map(
        lambda x: {"text": x["Comments"]},
        remove_columns=list(ben_sarc.column_names),
    )

    # CC100 Bengali — broad web text
    cc100_bn = load_dataset("statmt/cc100", lang="bn", split="train[:60000]")
    cc100_bn = cc100_bn.select_columns(["text"])

    # CC100 Bengali Romanized — Banglish patterns
    cc100_bn_rom = load_dataset("statmt/cc100", lang="bn_rom", split="train[:20000]")
    cc100_bn_rom = cc100_bn_rom.select_columns(["text"])

    # ── Formal sources (lower priority, foundation only) ───────────────────

    # Wikipedia — formal Bengali, kept small to avoid register bias
    wiki = load_dataset("wikimedia/wikipedia", "20231101.bn", split="train[:10000]")
    assert wiki.column_names is not None
    wiki = wiki.map(
        lambda x: {"text": x["text"]},
        remove_columns=list(wiki.column_names),
    )

    # ── Interleave with explicit sampling probabilities ────────────────────
    # BanglishRev 40% — largest and most colloquial
    # Ben-Sarc    25% — direct social media register
    # CC100 bn    20% — broad colloquial web text
    # CC100 rom   10% — Banglish romanization patterns
    # Wikipedia    5% — formal foundation only
    raw_dataset = (
        interleave_datasets(
            [banglish_rev, ben_sarc, cc100_bn, cc100_bn_rom, wiki],
            probabilities=[0.40, 0.25, 0.20, 0.10, 0.05],
            seed=42,
            stopping_strategy="all_exhausted",
        )
        .shuffle(seed=42)
        .select(range(CPT_MAX_EXAMPLES))
    )

    print(f"Total CPT examples: {len(raw_dataset)}")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=raw_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            learning_rate=5e-6,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            bf16=True,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            output_dir=CPT_OUTPUT_DIR,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            report_to="none",
        ),
    )

    print("Starting CPT...")
    trainer.train()

    model.save_pretrained(CPT_CHECKPOINT)
    tokenizer.save_pretrained(CPT_CHECKPOINT)
    print(f"CPT complete — saved to {CPT_CHECKPOINT}")


if __name__ == "__main__":
    run_cpt()
