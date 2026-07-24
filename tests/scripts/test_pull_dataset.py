from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from scripts.pull_dataset import pull_dataset


def test_pull_uses_token_and_writes_valid_dataset(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "profile" / "merged.jsonl"
    refined = tmp_path / "profile" / "refined"
    monkeypatch.setenv("HF_TOKEN", "test-token")
    example = {"messages": [{"role": "user", "content": "post"}, {"role": "assistant", "content": "reply"}]}

    with (
        patch("scripts.pull_dataset.load_config") as load_config,
        patch("scripts.pull_dataset.load_dataset", return_value=[example]) as load_dataset,
    ):
        config = load_config.return_value
        config.profile.name = "test"
        config.profile.hf_dataset = "owner/dataset"
        config.profile.local_dataset = str(output)
        config.profile.refined_data_dir = str(refined)
        pull_dataset()

    load_dataset.assert_called_once_with("owner/dataset", split="train", token="test-token")
    assert json.loads(output.read_text())["messages"] == example["messages"]


def test_pull_refuses_to_overwrite(tmp_path: pytest.TempPathFactory) -> None:
    output = tmp_path / "merged.jsonl"
    output.write_text("existing\n")
    with patch("scripts.pull_dataset.load_config") as load_config:
        config = load_config.return_value
        config.profile.name = "test"
        config.profile.hf_dataset = "owner/dataset"
        config.profile.local_dataset = str(output)
        with pytest.raises(FileExistsError):
            pull_dataset()
    assert output.read_text() == "existing\n"
