import argparse

from unsloth import FastLanguageModel

from training.config import load_config


def export_gguf(config_path: str | None = None) -> None:
    """Merge SFT adapter and export to GGUF for local inference."""
    config = load_config(config_path)
    model_cfg = config.model
    sft_cfg = config.sft
    export_cfg = config.export

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_cfg.checkpoint,
        max_seq_length=model_cfg.max_seq_length,
        load_in_4bit=True,
    )

    model.save_pretrained_gguf(
        export_cfg.path,
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"Export complete — {export_cfg.path}.Q4_K_M.gguf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to GGUF format")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    export_gguf(config_path=args.config)
