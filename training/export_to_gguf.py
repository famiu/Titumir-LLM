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
    export_prefix = Path(export_cfg.path)
    output_candidates = [export_prefix, Path(f"{export_cfg.path}_gguf")]
    gguf_files = []
    for candidate in output_candidates:
        if candidate.is_dir():
            gguf_files.extend(candidate.glob("*.gguf"))
        elif candidate.is_file() and candidate.suffix == ".gguf":
            gguf_files.append(candidate)
    gguf_files.extend(export_prefix.parent.glob(f"{export_prefix.name}*.gguf"))
    gguf_files = sorted(set(path.resolve() for path in gguf_files))
    if not gguf_files:
        raise FileNotFoundError(f"GGUF export completed but no output file was found for prefix {export_cfg.path}")

    manifest_dir = gguf_files[0].parent
    files = [{"name": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size} for path in gguf_files]
    with atomic_text_writer(manifest_dir / "export_manifest.json") as file:
        json.dump(
            {
                "source_checkpoint": sft_cfg.checkpoint,
                "quantization_method": export_cfg.quantization_method,
                "files": files,
            },
            file,
            indent=2,
        )
    print(f"Export complete — {manifest_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to GGUF format")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    export_gguf(config_path=args.config)
