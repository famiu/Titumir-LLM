from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from training.export_to_gguf import export_gguf


class TestExportGguf:
    def test_calls_save_pretrained_gguf_with_correct_args(self, tmp_path: pytest.TempPathFactory) -> None:
        with (
            patch("training.export_to_gguf.FastLanguageModel.from_pretrained") as mock_load,
            patch("training.export_to_gguf.load_config") as mock_load_cfg,
        ):
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            export_dir = tmp_path / "export_gguf"
            mock_model.save_pretrained_gguf.side_effect = lambda *args, **kwargs: (
                export_dir.mkdir(),
                (export_dir / "model.gguf").write_bytes(b"gguf"),
            )
            mock_load.return_value = (mock_model, mock_tokenizer)

            mock_cfg = mock_load_cfg.return_value
            mock_cfg.model.name = "test/model"
            mock_cfg.model.max_seq_length = 512
            mock_cfg.model.load_in_4bit = True
            mock_cfg.sft_training.checkpoint = "./checkpoints/sft_final"
            mock_cfg.export.path = str(tmp_path / "export")
            mock_cfg.export.quantization_method = "q4_k_m"

            export_gguf()

            mock_model.save_pretrained_gguf.assert_called_once()
            call_args = mock_model.save_pretrained_gguf.call_args
            assert call_args[0][0] == str(tmp_path / "export")
            assert call_args[1]["quantization_method"] == "q4_k_m"

    def test_uses_provided_model_and_tokenizer(self, tmp_path: pytest.TempPathFactory) -> None:
        with patch("training.export_to_gguf.load_config") as mock_load_cfg:
            mock_cfg = mock_load_cfg.return_value
            mock_cfg.model.name = "test/model"
            mock_cfg.model.max_seq_length = 512
            mock_cfg.model.load_in_4bit = True
            mock_cfg.sft_training.checkpoint = "./checkpoints/sft_final"
            mock_cfg.export.path = str(tmp_path / "export")
            mock_cfg.export.quantization_method = "q4_k_m"

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            export_dir = tmp_path / "export_gguf"
            mock_model.save_pretrained_gguf.side_effect = lambda *args, **kwargs: (
                export_dir.mkdir(),
                (export_dir / "model.gguf").write_bytes(b"gguf"),
            )

            with patch("training.export_to_gguf.FastLanguageModel.from_pretrained") as mock_load:
                export_gguf(model=mock_model, tokenizer=mock_tokenizer)

            mock_load.assert_not_called()

            mock_model.save_pretrained_gguf.assert_called_once()
            call_args = mock_model.save_pretrained_gguf.call_args
            assert call_args[1]["quantization_method"] == "q4_k_m"
            assert (export_dir / "export_manifest.json").exists()

    def test_manifest_excludes_unchanged_stale_files(self, tmp_path: pytest.TempPathFactory) -> None:
        export_dir = tmp_path / "export_gguf"
        export_dir.mkdir()
        (export_dir / "stale.gguf").write_bytes(b"stale")
        model = MagicMock()
        tokenizer = MagicMock()
        model.save_pretrained_gguf.side_effect = lambda *args, **kwargs: (export_dir / "current.gguf").write_bytes(
            b"current"
        )

        with patch("training.export_to_gguf.load_config") as load_config:
            config = load_config.return_value
            config.sft_training.checkpoint = "checkpoint"
            config.export.path = str(tmp_path / "export")
            config.export.quantization_method = "q4_k_m"
            export_gguf(model=model, tokenizer=tokenizer)

        import json

        manifest = json.loads((export_dir / "export_manifest.json").read_text())
        assert [entry["name"] for entry in manifest["files"]] == ["current.gguf"]
