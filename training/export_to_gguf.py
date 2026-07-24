import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from unsloth import FastLanguageModel  # isort: skip

from scripts._data import atomic_text_writer, file_sha256
from training.config import load_config


def export_gguf(config_path: str | None = None, model=None, tokenizer=None) -> None:
    """Merge SFT adapter and export to GGUF for local inference."""
    config = load_config(config_path)
    model_cfg = config.model
    sft_cfg = config.sft_training
    export_cfg = config.export

    if model is None or tokenizer is None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_cfg.checkpoint,
            max_seq_length=model_cfg.max_seq_length,
            load_in_4bit=model_cfg.load_in_4bit,
        )

    model.save_pretrained_gguf(
        export_cfg.path,
        tokenizer,
        quantization_method=export_cfg.quantization_method,
    )
    output_dir = Path(f"{export_cfg.path}_gguf")
    if output_dir.is_dir():
        files = [
            {"name": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in sorted(output_dir.glob("*.gguf"))
        ]
        with atomic_text_writer(output_dir / "export_manifest.json") as file:
            json.dump(
                {
                    "source_checkpoint": sft_cfg.checkpoint,
                    "quantization_method": export_cfg.quantization_method,
                    "files": files,
                },
                file,
                indent=2,
            )
    print(f"Export complete — {export_cfg.path}_gguf/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to GGUF format")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    export_gguf(config_path=args.config)
