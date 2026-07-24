from __future__ import annotations

from unittest.mock import patch

import pytest

from training.runtime import precision_args, resolve_resume_checkpoint


def test_precision_prefers_bf16() -> None:
    with (
        patch("training.runtime.torch.cuda.is_available", return_value=True),
        patch("training.runtime.torch.cuda.is_bf16_supported", return_value=True),
    ):
        assert precision_args() == {"bf16": True, "fp16": False}


def test_precision_falls_back_to_fp16() -> None:
    with (
        patch("training.runtime.torch.cuda.is_available", return_value=True),
        patch("training.runtime.torch.cuda.is_bf16_supported", return_value=False),
    ):
        assert precision_args() == {"bf16": False, "fp16": True}


def test_resolve_explicit_resume_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    assert resolve_resume_checkpoint(str(checkpoint), str(tmp_path)) == str(checkpoint)


def test_auto_resume_requires_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No trainer checkpoint"):
        resolve_resume_checkpoint(True, str(tmp_path))
