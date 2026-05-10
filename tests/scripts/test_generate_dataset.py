from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts.generate_dataset import generate_dataset, generate_topic, is_valid_example


class TestIsValidExample:
    @pytest.mark.parametrize(
        "example,expected",
        [
            ({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hi"}]}, True),
            ({"messages": []}, False),
            ({"messages": [{"role": "user", "content": "hi"}]}, False),
            ({"messages": [{"role": "user", "content": "hi"}, "not a dict"]}, False),
            ({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}]}, False),
            ({"messages": [{"role": "user", "content": "   "}, {"role": "assistant", "content": "hi"}]}, False),
            ({"messages": [{"role": None, "content": "hi"}, {"role": "assistant", "content": "hi"}]}, False),
            ({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bye"}]}, True),
        ],
    )
    def test_is_valid_example(self, example: dict, expected: bool) -> None:
        assert is_valid_example(example) == expected


class TestGenerateTopic:
    def test_collects_examples_and_stops_on_none(self) -> None:
        from itertools import count

        from training.config import ApiConfigBase

        cfg = ApiConfigBase(
            endpoint="http://x",
            api_key_env="X",
            model="test",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
        )

        valid_batch = [
            {
                "messages": [
                    {"role": "user", "content": "post1"},
                    {"role": "assistant", "content": "reply1"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "post2"},
                    {"role": "assistant", "content": "reply2"},
                ]
            },
        ]

        call_count = 0

        def fake_call_llm(llm_cfg, messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return valid_batch
            return None

        with patch("scripts.generate_dataset.call_llm", side_effect=fake_call_llm):
            result = generate_topic(
                topic_idx=1,
                topic="test topic",
                examples_for_topic=2,
                batch_size=3,
                total_topics=1,
                llm_cfg=cfg,
                generation_prompt_template="Generate {n} examples about {topic}",
                global_batch_counter=count(1),
            )

        assert len(result) == 2
        for ex in result:
            assert is_valid_example(ex)


class TestGenerateDataset:
    def test_change_me_model_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        with patch("scripts.generate_dataset.load_config") as mock_load:
            mock_cfg = mock_load.return_value
            mock_cfg.profile.unprocessed_data_dir = str(tmp_path / "unprocessed")
            mock_cfg.generation.model = "CHANGE_ME"
            mock_cfg.generation.prompt = "some prompt"
            mock_cfg.generation.batch_size = 10
            mock_cfg.generation.get_max_workers.return_value = 1
            mock_cfg.topics = []
            mock_cfg.generation.get_api_key.return_value = "fake-key"

            with pytest.raises(ValueError) as exc_info:
                generate_dataset()
            assert "generation" in str(exc_info.value).lower()
            assert "model" in str(exc_info.value).lower()

    def test_empty_prompt_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        with patch("scripts.generate_dataset.load_config") as mock_load:
            mock_cfg = mock_load.return_value
            mock_cfg.profile.unprocessed_data_dir = str(tmp_path / "unprocessed")
            mock_cfg.generation.model = "test/model"
            mock_cfg.generation.prompt = ""
            mock_cfg.generation.batch_size = 10
            mock_cfg.generation.get_max_workers.return_value = 1
            mock_cfg.topics = []
            mock_cfg.generation.get_api_key.return_value = "fake-key"

            with pytest.raises(ValueError) as exc_info:
                generate_dataset()
            assert "generation" in str(exc_info.value).lower()
            assert "prompt" in str(exc_info.value).lower()

    def test_output_dir_created(self, tmp_path: pytest.TempPathFactory) -> None:
        out_dir = tmp_path / "unprocessed"
        out_dir.mkdir()

        valid_batch = [{"messages": [{"role": "user", "content": "p"}, {"role": "assistant", "content": "r"}]}]
        call_count = 0

        def fake_call_llm(llm_cfg, messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return valid_batch
            return None

        with patch("scripts.generate_dataset.load_config") as mock_load:
            mock_cfg = mock_load.return_value
            mock_cfg.profile.unprocessed_data_dir = str(out_dir)
            mock_cfg.generation.model = "test/model"
            mock_cfg.generation.prompt = "Generate {n} about {topic}"
            mock_cfg.generation.batch_size = 10
            mock_cfg.generation.get_max_workers.return_value = 1
            mock_cfg.generation.get_api_key.return_value = "fake-key"
            mock_cfg.topics = []

            with patch("scripts.generate_dataset.call_llm", side_effect=fake_call_llm):
                generate_dataset(filename="test_output.jsonl")

        output_file = out_dir / "test_output.jsonl"
        assert output_file.exists()
        assert output_file.name.endswith(".jsonl")
