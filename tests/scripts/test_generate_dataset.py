from __future__ import annotations

import json
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
            ({"messages": [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "bye"}]}, False),
            (
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "bye"},
                        {"role": "assistant", "content": "extra"},
                    ]
                },
                False,
            ),
            ({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bye"}]}, True),
        ],
    )
    def test_is_valid_example(self, example: dict, expected: bool) -> None:
        assert is_valid_example(example) == expected


class TestGenerateTopic:
    def test_collects_examples_and_stops_on_none(self) -> None:
        from itertools import count

        from training.config import GenerationConfig

        cfg = GenerationConfig(
            endpoint="http://x",
            api_key_env="X",
            model="test",
            temperature=0.5,
            max_tokens=100,
            batch_size=10,
            max_stalled_batches=2,
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

        def fake_call_llm(llm_cfg, messages, expected_type=None):
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
            assert ex["metadata"]["topic"] == "test topic"

    def test_stops_after_repeated_empty_batches(self) -> None:
        from itertools import count

        from training.config import GenerationConfig

        cfg = GenerationConfig(model="test", max_stalled_batches=2)
        with (
            patch("scripts.generate_dataset.call_llm", return_value=[]),
            pytest.raises(RuntimeError, match="stalled batches"),
        ):
            generate_topic(
                topic_idx=1,
                topic="test topic",
                examples_for_topic=2,
                batch_size=2,
                total_topics=1,
                llm_cfg=cfg,
                generation_prompt_template="Generate {n} examples about {topic}",
                global_batch_counter=count(1),
            )


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

        valid_batch = [{"messages": [{"role": "user", "content": "p"}, {"role": "assistant", "content": "r"}]}]
        call_count = 0

        def fake_call_llm(llm_cfg, messages, expected_type=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return valid_batch
            return None

        with patch("scripts.generate_dataset.load_config") as mock_load:
            mock_cfg = mock_load.return_value
            mock_cfg.profile.name = "test"
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

    def test_refuses_to_overwrite_existing_output(self, tmp_path: pytest.TempPathFactory) -> None:
        output_dir = tmp_path / "unprocessed"
        output_dir.mkdir()
        (output_dir / "existing.jsonl").write_text("existing\n")

        with patch("scripts.generate_dataset.load_config") as mock_load:
            mock_cfg = mock_load.return_value
            mock_cfg.profile.name = "test"
            mock_cfg.profile.unprocessed_data_dir = str(output_dir)
            mock_cfg.generation.model = "test/model"
            mock_cfg.generation.prompt = "Generate {n} about {topic}"
            mock_cfg.topics = []

            with pytest.raises(FileExistsError):
                generate_dataset(filename="existing.jsonl")

    def test_resume_uses_checkpointed_topics(self, tmp_path: pytest.TempPathFactory) -> None:
        output_dir = tmp_path / "unprocessed"
        output_dir.mkdir()
        output_file = output_dir / "resume.jsonl"
        state_file = output_dir / "resume.jsonl.state.json"
        completed_example = {
            "messages": [{"role": "user", "content": "p1"}, {"role": "assistant", "content": "r1"}],
            "metadata": {"topic": "topic one"},
        }

        with patch("scripts.generate_dataset.load_config") as mock_load:
            mock_cfg = mock_load.return_value
            mock_cfg.profile.name = "test"
            mock_cfg.profile.unprocessed_data_dir = str(output_dir)
            mock_cfg.generation.model = "test/model"
            mock_cfg.generation.prompt = "Generate {n} about {topic}"
            mock_cfg.generation.batch_size = 1
            mock_cfg.generation.get_max_workers.return_value = 1
            mock_cfg.topics = [
                type("Topic", (), {"topic": "topic one", "count": 1})(),
                type("Topic", (), {"topic": "topic two", "count": 1})(),
            ]
            identity = {
                "profile": "test",
                "model": "test/model",
                "prompt_sha256": __import__("hashlib").sha256(mock_cfg.generation.prompt.encode()).hexdigest(),
                "topics": [{"topic": "topic one", "count": 1}, {"topic": "topic two", "count": 1}],
            }
            state_file.write_text(
                json.dumps({"identity": identity, "completed": {"1": [completed_example]}}),
                encoding="utf-8",
            )
            generated_example = {
                "messages": [{"role": "user", "content": "p2"}, {"role": "assistant", "content": "r2"}]
            }
            with patch("scripts.generate_dataset.call_llm", return_value=[generated_example]) as mock_call:
                generate_dataset(filename="resume.jsonl", resume=True)

        assert mock_call.call_count == 1
        lines = [json.loads(line) for line in output_file.read_text().splitlines()]
        assert [line["messages"][0]["content"] for line in lines] == ["p1", "p2"]
        assert not state_file.exists()
