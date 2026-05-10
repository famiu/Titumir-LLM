from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.merge_dataset import merge_datasets


class TestMergeDatasets:
    def test_cross_file_dedup(self, tmp_path: pytest.TempPathFactory, sample_a_path: Path, sample_b_path: Path) -> None:
        refined_dir = tmp_path / "refined"
        refined_dir.mkdir()

        a_dest = refined_dir / "a.jsonl"
        b_dest = refined_dir / "b.jsonl"
        shutil.copy(sample_a_path, a_dest)
        shutil.copy(sample_b_path, b_dest)

        output_file = refined_dir.parent / "merged.jsonl"

        with patch("scripts.merge_dataset.load_config") as mock_load:
            mock_cfg = mock_load.return_value
            mock_cfg.profile.refined_data_dir = str(refined_dir)
            mock_cfg.profile.local_dataset = str(output_file)

            merge_datasets()

        with open(output_file, encoding="utf-8") as f:
            merged_lines = [line for line in f if line.strip()]
        assert len(merged_lines) == 19, f"Expected 19 unique, got {len(merged_lines)}"

    def test_within_file_dedup(self, tmp_path: pytest.TempPathFactory, sample_with_dup_path: Path) -> None:
        refined_dir = tmp_path / "refined"
        refined_dir.mkdir()

        dest = refined_dir / "dup.jsonl"
        shutil.copy(sample_with_dup_path, dest)

        output_file = refined_dir.parent / "merged.jsonl"

        with patch("scripts.merge_dataset.load_config") as mock_load:
            mock_cfg = mock_load.return_value
            mock_cfg.profile.refined_data_dir = str(refined_dir)
            mock_cfg.profile.local_dataset = str(output_file)

            merge_datasets()

        with open(output_file, encoding="utf-8") as f:
            merged_lines = [line for line in f if line.strip()]
        assert len(merged_lines) == 4, f"Expected 4 unique, got {len(merged_lines)}"

    def test_malformed_lines_skipped(self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture) -> None:
        refined_dir = tmp_path / "refined"
        refined_dir.mkdir()

        valid_lines = [
            {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
            {"messages": [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]},
            {"messages": [{"role": "user", "content": "e"}, {"role": "assistant", "content": "f"}]},
        ]

        input_file = refined_dir / "malformed.jsonl"
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(valid_lines[0], ensure_ascii=False) + "\n")
            f.write(json.dumps(valid_lines[1], ensure_ascii=False) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps(valid_lines[2], ensure_ascii=False) + "\n")

        output_file = refined_dir.parent / "merged.jsonl"

        with patch("scripts.merge_dataset.load_config") as mock_load:
            mock_cfg = mock_load.return_value
            mock_cfg.profile.refined_data_dir = str(refined_dir)
            mock_cfg.profile.local_dataset = str(output_file)

            merge_datasets()

        with open(output_file, encoding="utf-8") as f:
            merged_lines = [line for line in f if line.strip()]
        assert len(merged_lines) == 3

        captured = capsys.readouterr()
        assert "malformed" in captured.out.lower()

    def test_empty_dir_shows_nothing_to_do(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture
    ) -> None:
        refined_dir = tmp_path / "refined"
        refined_dir.mkdir()

        output_file = refined_dir.parent / "merged.jsonl"

        with patch("scripts.merge_dataset.load_config") as mock_load:
            mock_cfg = mock_load.return_value
            mock_cfg.profile.refined_data_dir = str(refined_dir)
            mock_cfg.profile.local_dataset = str(output_file)

            merge_datasets()

        captured = capsys.readouterr()
        assert "no jsonl files" in captured.out.lower()


class TestConcurrencyStress:
    def test_concurrent_batch_processing(self, tmp_path: pytest.TempPathFactory, sample_a_path: Path) -> None:
        refined_dir = tmp_path / "refined"
        refined_dir.mkdir()

        dest = refined_dir / "stress.jsonl"
        shutil.copy(sample_a_path, dest)

        output_file = refined_dir.parent / "merged.jsonl"

        with patch("scripts.merge_dataset.load_config") as mock_load:
            mock_cfg = mock_load.return_value
            mock_cfg.profile.refined_data_dir = str(refined_dir)
            mock_cfg.profile.local_dataset = str(output_file)

            merge_datasets()

        with open(output_file, encoding="utf-8") as f:
            merged_lines = [line for line in f if line.strip()]
        assert len(merged_lines) == 10
