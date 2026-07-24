from __future__ import annotations

import pytest
import requests
import responses

from scripts._llm import call_llm, retry_delay


class TestRetryDelay:
    def test_attempt_0(self) -> None:
        assert retry_delay(0) == 2.0

    def test_attempt_1(self) -> None:
        assert retry_delay(1) == 4.0

    def test_attempt_2(self) -> None:
        assert retry_delay(2) == 8.0

    def test_attempt_3(self) -> None:
        assert retry_delay(3) == 16.0

    def test_attempt_4(self) -> None:
        assert retry_delay(4) == 32.0

    def test_attempt_5(self) -> None:
        assert retry_delay(5) == 64.0

    def test_attempt_6_capped(self) -> None:
        assert retry_delay(6) == 120.0

    def test_attempt_7_still_capped(self) -> None:
        assert retry_delay(7) == 120.0


@pytest.mark.usefixtures("fast_retry")
class TestCallLlm:
    @responses.activate
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"choices": [{"message": {"content": '{"key": "val"}'}}]},
        )
        from training.config import ApiConfigBase

        cfg = ApiConfigBase(
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key_env="OPENROUTER_API_KEY",
            model="test",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
        )
        result = call_llm(cfg, [{"role": "user", "content": "hello"}])
        assert result == {"key": "val"}

    @responses.activate
    def test_429_retry_then_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "fake-key")
        responses.add(
            responses.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            status=429,
        )
        responses.add(
            responses.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )
        from training.config import ApiConfigBase

        cfg = ApiConfigBase(
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key_env="X",
            model="test",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
        )
        result = call_llm(cfg, [{"role": "user", "content": "hello"}])
        assert result == {"ok": True}
        assert len(responses.calls) == 2

    @responses.activate
    def test_500_exhausts_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "fake-key")
        for _ in range(5):
            responses.add(
                responses.POST,
                "https://openrouter.ai/api/v1/chat/completions",
                status=500,
            )
        from training.config import ApiConfigBase

        cfg = ApiConfigBase(
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key_env="X",
            model="test",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
            max_retries=5,
        )
        result = call_llm(cfg, [{"role": "user", "content": "hello"}])
        assert result is None
        assert len(responses.calls) == 5

    @responses.activate
    def test_timeout_exhausts_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "fake-key")
        responses.add(
            responses.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            body=requests.exceptions.Timeout(),
        )
        from training.config import ApiConfigBase

        cfg = ApiConfigBase(
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key_env="X",
            model="test",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
        )
        result = call_llm(cfg, [{"role": "user", "content": "hello"}])
        assert result is None
        assert len(responses.calls) == 5

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_KEY", raising=False)
        from training.config import ApiConfigBase

        cfg = ApiConfigBase(
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key_env="MISSING_KEY",
            model="test",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
        )
        with pytest.raises(ValueError) as exc_info:
            call_llm(cfg, [{"role": "user", "content": "hello"}])
        assert "MISSING_KEY" in str(exc_info.value)

    @responses.activate
    def test_malformed_json_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "fake-key")
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"choices": [{"message": {"content": "not json"}}]},
        )
        from training.config import ApiConfigBase

        cfg = ApiConfigBase(
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key_env="X",
            model="test",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
        )
        result = call_llm(cfg, [{"role": "user", "content": "hello"}])
        assert result is None

    @responses.activate
    def test_markdown_fences_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "fake-key")
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"choices": [{"message": {"content": '```json\n{"k":"v"}\n```'}}]},
        )
        from training.config import ApiConfigBase

        cfg = ApiConfigBase(
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key_env="X",
            model="test",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
        )
        result = call_llm(cfg, [{"role": "user", "content": "hello"}])
        assert result == {"k": "v"}

    @responses.activate
    def test_expected_response_type_is_enforced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "fake-key")
        for _ in range(5):
            responses.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={"choices": [{"message": {"content": '{"not":"a list"}'}}]},
            )
        from training.config import ApiConfigBase

        cfg = ApiConfigBase(
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key_env="X",
            model="test",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
        )
        assert call_llm(cfg, [], expected_type=list) is None

    @responses.activate
    def test_reasoning_omitted_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "fake-key")
        responses.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"choices": [{"message": {"content": "{}"}}]},
        )
        from training.config import ApiConfigBase

        cfg = ApiConfigBase(
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key_env="X",
            model="test",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
        )
        call_llm(cfg, [])
        assert "reasoning" not in responses.calls[0].request.body.decode()
