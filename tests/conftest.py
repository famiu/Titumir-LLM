from __future__ import annotations

from pathlib import Path

import pytest

from training.config import Config


@pytest.fixture
def dummy_config() -> Config:
    return Config.from_yaml("configs/dry_run.yaml")


@pytest.fixture
def sample_a_path() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_a.jsonl"


@pytest.fixture
def sample_b_path() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_b.jsonl"


@pytest.fixture
def sample_with_dup_path() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_with_dup.jsonl"


@pytest.fixture
def fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts._llm.time.sleep", lambda s: None)
