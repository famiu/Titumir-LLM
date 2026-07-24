from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scripts.push_dataset import push_dataset


def test_push_uses_explicit_token(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("{}\n")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    dataset = MagicMock()

    with (
        patch("scripts.push_dataset.load_config") as load_config,
        patch("scripts.push_dataset.load_dataset", return_value=dataset),
    ):
        config = load_config.return_value
        config.profile.name = "test"
        config.profile.hf_dataset = "owner/dataset"
        config.profile.local_dataset = str(dataset_path)
        push_dataset()

    dataset.push_to_hub.assert_called_once_with("owner/dataset", token="test-token")


def test_push_requires_token(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("{}\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with patch("scripts.push_dataset.load_config") as load_config, patch("scripts.push_dataset.load_dotenv"):
        config = load_config.return_value
        config.profile.name = "test"
        config.profile.hf_dataset = "owner/dataset"
        config.profile.local_dataset = str(dataset_path)
        with pytest.raises(ValueError, match="HF_TOKEN"):
            push_dataset()
