from unsloth import FastLanguageModel

from training.config import EXPORT_PATH, MAX_SEQ_LENGTH, SFT_CHECKPOINT


def export_gguf() -> None:
    """Merge SFT adapter and export to GGUF for local inference."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_CHECKPOINT,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model.save_pretrained_gguf(
        EXPORT_PATH,
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"Export complete — {EXPORT_PATH}.Q4_K_M.gguf")


if __name__ == "__main__":
    export_gguf()
