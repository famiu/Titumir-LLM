from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.refine_dataset import check_batch_with_retry, refine_dataset, refine_file
from training.config import RefinementConfig

PRIME_INDICES = [2, 3, 5, 7]


def prime_reasons_response(remove_indices: list[int]) -> dict:
    return {
        "keep": [i for i in range(10) if i not in remove_indices],
        "remove": remove_indices,
        "reasons": {str(i): f"reason for index {i}" for i in remove_indices},
    }


class TestCheckBatchWithRetry:
    @pytest.mark.parametrize(
        "mock_response,expected_kept,expected_removed_count",
        [
            (prime_reasons_response(PRIME_INDICES), 6, 4),
            ({"keep": list(range(10)), "remove": [], "reasons": {}}, 10, 0),
        ],
    )
    def test_check_batch_with_retry_cases(
        self,
        mock_response: dict | None,
        expected_kept: int,
        expected_removed_count: int,
        sample_a_path: Path,
    ) -> None:
        examples = [json.loads(line) for line in open(sample_a_path)]
        cfg = RefinementConfig(
            endpoint="http://x",
            api_key_env="X",
            model="test",
            temperature=0.1,
            max_tokens=100,
            batch_size=10,
        )
        prompt = "Check these examples"

        with patch("scripts.refine_dataset.call_llm", return_value=mock_response):
            batch_idx, kept, removed = check_batch_with_retry(0, examples, 0, cfg, prompt)

        assert len(kept) == expected_kept
        assert len(removed) == expected_removed_count

    @pytest.mark.parametrize("mock_response", [None, "not a dict", {"keep": [0], "remove": []}])
    def test_invalid_decision_raises(self, mock_response: object, sample_a_path: Path) -> None:
        examples = [json.loads(line) for line in open(sample_a_path)]
        cfg = RefinementConfig(
            endpoint="http://x",
            api_key_env="X",
            model="test",
            temperature=0.1,
            max_tokens=100,
            batch_size=10,
        )
        with (
            patch("scripts.refine_dataset.call_llm", return_value=mock_response),
            pytest.raises((RuntimeError, ValueError)),
        ):
            check_batch_with_retry(0, examples, 0, cfg, "")

    def test_missing_reason_fallback(self, sample_a_path: Path) -> None:
        examples = [json.loads(line) for line in open(sample_a_path)]
        cfg = RefinementConfig(
            endpoint="http://x",
            api_key_env="X",
            model="test",
            temperature=0.1,
            max_tokens=100,
            batch_size=10,
        )
        mock_response = {"keep": [i for i in range(10) if i != 2], "remove": [2], "reasons": {}}

        with patch("scripts.refine_dataset.call_llm", return_value=mock_response):
            batch_idx, kept, removed = check_batch_with_retry(0, examples, 0, cfg, "")

        assert len(kept) == 9
        assert len(removed) == 1
        assert removed[0]["reason"] == "no reason given"


class TestRefineFile:
    def test_keeps_and_removed_files_written(self, tmp_path: pytest.TempPathFactory, sample_a_path: Path) -> None:
        refined_dir = tmp_path / "refined"
        removed_dir = tmp_path / "removed"
        refined_dir.mkdir()
        removed_dir.mkdir()

        input_file = tmp_path / "input.jsonl"
        input_file.write_bytes(sample_a_path.read_bytes())

        cfg = RefinementConfig(
            endpoint="http://x",
            api_key_env="X",
            model="test",
            temperature=0.1,
            max_tokens=100,
            batch_size=10,
        )

        mock_response = prime_reasons_response(PRIME_INDICES)

        with patch("scripts.refine_dataset.call_llm", return_value=mock_response):
            refine_file(input_file, str(refined_dir), str(removed_dir), cfg, "", batch_size=10)

        kept_file = refined_dir / "input.jsonl"
        removed_file = removed_dir / "input.jsonl"

        kept_lines = [l for l in open(kept_file) if l.strip()]
        removed_lines = [l for l in open(removed_file) if l.strip()]

        assert len(kept_lines) == 6
        assert len(removed_lines) == 4

    def test_malformed_lines_abort_without_output(self, tmp_path: pytest.TempPathFactory) -> None:
        refined_dir = tmp_path / "refined"
        removed_dir = tmp_path / "removed"
        refined_dir.mkdir()
        removed_dir.mkdir()

        valid_lines = [json.loads(l) for l in open("tests/fixtures/sample_a.jsonl")][:4]
        input_file = tmp_path / "malformed.jsonl"
        with open(input_file, "w", encoding="utf-8") as f:
            for ex in valid_lines:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            f.write("this is not json\n")

        cfg = RefinementConfig(
            endpoint="http://x",
            api_key_env="X",
            model="test",
            temperature=0.1,
            max_tokens=100,
            batch_size=10,
        )

        with pytest.raises(ValueError, match="line 5"):
            refine_file(input_file, str(refined_dir), str(removed_dir), cfg, "", batch_size=10)
        assert not (refined_dir / "malformed.jsonl").exists()

    def test_failed_batch_can_resume(self, tmp_path: pytest.TempPathFactory, sample_a_path: Path) -> None:
        refined_dir = tmp_path / "refined"
        removed_dir = tmp_path / "removed"
        refined_dir.mkdir()
        removed_dir.mkdir()
        input_file = tmp_path / "input.jsonl"
        examples = [json.loads(line) for line in open(sample_a_path)][:4]
        input_file.write_text("".join(json.dumps(example) + "\n" for example in examples))
        cfg = RefinementConfig(
            endpoint="http://x",
            api_key_env="X",
            model="test",
            temperature=0.1,
            max_tokens=100,
            batch_size=2,
            max_workers=1,
        )
        decision = {"keep": [0, 1], "remove": [], "reasons": {}}

        with (
            patch("scripts.refine_dataset.call_llm", side_effect=[decision, None]),
            pytest.raises(RuntimeError, match="completed work is saved"),
        ):
            refine_file(input_file, str(refined_dir), str(removed_dir), cfg, "prompt", batch_size=2)

        state_file = refined_dir / "input.jsonl.state.json"
        assert state_file.exists()
        assert not (refined_dir / "input.jsonl").exists()

        with patch("scripts.refine_dataset.call_llm", return_value=decision) as mock_call:
            refine_file(input_file, str(refined_dir), str(removed_dir), cfg, "prompt", batch_size=2, resume=True)

        assert mock_call.call_count == 1
        assert len((refined_dir / "input.jsonl").read_text().splitlines()) == 4
        assert not state_file.exists()


class TestRefineDataset:
    def test_already_refined_file_skipped(
        self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture
    ) -> None:
        unprocessed = tmp_path / "unprocessed"
        refined = tmp_path / "refined"
        removed = tmp_path / "removed"
        unprocessed.mkdir()
        refined.mkdir()
        removed.mkdir()

        input_file = unprocessed / "file1.jsonl"
        input_file.write_text('{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"bye"}]}\n')

        refined_file = refined / "file1.jsonl"
        refined_file.write_text('{"messages":[{"role":"user","content":"refined"}]}\n')

        cfg = RefinementConfig(
            endpoint="http://x",
            api_key_env="X",
            model="test",
            temperature=0.1,
            max_tokens=100,
            batch_size=10,
        )

        with patch("scripts.refine_dataset.load_config") as mock_load:
            mock_cfg = mock_load.return_value
            mock_cfg.profile.unprocessed_data_dir = str(unprocessed)
            mock_cfg.profile.refined_data_dir = str(refined)
            mock_cfg.profile.removed_data_dir = str(removed)
            mock_cfg.refinement = cfg
            mock_cfg.refinement.prompt = "check"

            refine_dataset()

        captured = capsys.readouterr()
        assert "Skipping" in captured.out or "already refined" in captured.out

    def test_empty_unprocessed_dir(self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture) -> None:
        unprocessed = tmp_path / "unprocessed"
        refined = tmp_path / "refined"
        removed = tmp_path / "removed"
        unprocessed.mkdir()
        refined.mkdir()
        removed.mkdir()

        with patch("scripts.refine_dataset.load_config") as mock_load:
            mock_cfg = mock_load.return_value
            mock_cfg.profile.unprocessed_data_dir = str(unprocessed)
            mock_cfg.profile.refined_data_dir = str(refined)
            mock_cfg.profile.removed_data_dir = str(removed)
            mock_cfg.refinement.model = "test"
            mock_cfg.refinement.prompt = "check"
            mock_cfg.refinement.get_api_key.return_value = "key"

            refine_dataset()

        captured = capsys.readouterr()
        assert "nothing to do" in captured.out.lower() or "No JSONL files" in captured.out

    def test_specific_file_not_found(self, tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture) -> None:
        unprocessed = tmp_path / "unprocessed"
        unprocessed.mkdir()

        with patch("scripts.refine_dataset.load_config") as mock_load:
            mock_cfg = mock_load.return_value
            mock_cfg.profile.unprocessed_data_dir = str(unprocessed)
            mock_cfg.profile.refined_data_dir = str(tmp_path / "refined")
            mock_cfg.profile.removed_data_dir = str(tmp_path / "removed")
            mock_cfg.refinement.model = "test"
            mock_cfg.refinement.prompt = "check"
            mock_cfg.refinement.get_api_key.return_value = "key"

            refine_dataset(input_file="nonexistent.jsonl")

        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "nonexistent" in captured.out.lower()
