import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

import requests
from dotenv import load_dotenv

from training.config import GenerationConfig, load_config


class _SafeDict(dict):
    """Dict that returns {key} for missing keys, for safe str.format_map."""

    def __missing__(self, key):
        return "{" + key + "}"


load_dotenv()


def generate_batch_with_retry(
    topic_idx: int,
    batch_num: int,
    llm_cfg: GenerationConfig,
    generation_prompt: str,
) -> list[dict]:
    """Generate a single batch with retries. Returns examples or raises on total failure."""
    api_key = llm_cfg.get_api_key()
    if not api_key:
        raise ValueError(f"API key not found: set {llm_cfg.api_key_env} environment variable")

    for attempt in range(llm_cfg.max_retries):
        try:
            response = requests.post(
                llm_cfg.endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": llm_cfg.model,
                    "messages": [{"role": "user", "content": generation_prompt}],
                    "temperature": llm_cfg.temperature,
                    "max_tokens": llm_cfg.max_tokens,
                    "reasoning": {"effort": "none"},
                },
                timeout=llm_cfg.batch_timeout,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            print(f"  [topic {topic_idx} batch {batch_num}] JSON parse failed: {e} — retrying...")
            time.sleep(llm_cfg.retry_backoff[min(attempt, len(llm_cfg.retry_backoff) - 1)])

        except requests.exceptions.Timeout:
            wait = llm_cfg.retry_backoff[min(attempt, len(llm_cfg.retry_backoff) - 1)]
            print(
                f"  [topic {topic_idx} batch {batch_num}] Timed out after {llm_cfg.batch_timeout}s "
                f"(attempt {attempt + 1}/{llm_cfg.max_retries}) — retrying in {wait}s"
            )
            time.sleep(wait)

        except requests.exceptions.ConnectionError:
            wait = llm_cfg.retry_backoff[min(attempt, len(llm_cfg.retry_backoff) - 1)]
            print(
                f"  [topic {topic_idx} batch {batch_num}] Network error "
                f"(attempt {attempt + 1}/{llm_cfg.max_retries}) — retrying in {wait}s"
            )
            time.sleep(wait)

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                wait = llm_cfg.retry_backoff[min(attempt, len(llm_cfg.retry_backoff) - 1)] * 2
                print(f"  [topic {topic_idx} batch {batch_num}] Rate limited — retrying in {wait}s")
                time.sleep(wait)
            elif e.response.status_code >= 500:
                wait = llm_cfg.retry_backoff[min(attempt, len(llm_cfg.retry_backoff) - 1)]
                print(
                    f"  [topic {topic_idx} batch {batch_num}] Server error {e.response.status_code} "
                    f"— retrying in {wait}s"
                )
                time.sleep(wait)
            else:
                print(f"  [topic {topic_idx} batch {batch_num}] Client error {e.response.status_code} — skipping batch")
                return []

        except Exception as e:
            print(f"  [topic {topic_idx} batch {batch_num}] Unexpected error: {e} — retrying...")
            time.sleep(llm_cfg.retry_backoff[min(attempt, len(llm_cfg.retry_backoff) - 1)])

    print(f"  [topic {topic_idx} batch {batch_num}] All retries exhausted — skipping batch")
    return []


def is_valid_example(example: dict) -> bool:
    """Check that an example has the expected structure."""
    messages = example.get("messages", [])
    return (
        len(messages) >= 2
        and isinstance(messages[0], dict)
        and isinstance(messages[1], dict)
        and isinstance(messages[0].get("role"), str)
        and isinstance(messages[1].get("role"), str)
        and bool(messages[0].get("content", "").strip())
        and bool(messages[1].get("content", "").strip())
    )


def generate_topic(
    topic_idx: int,
    topic: str,
    examples_for_topic: int,
    batch_size: int,
    total_topics: int,
    llm_cfg: GenerationConfig,
    generation_prompt_template: str,
) -> list[dict]:
    """Generate all examples for a single topic sequentially."""
    print(f"\n[{topic_idx}/{total_topics}] Topic: {topic} ({examples_for_topic} examples)")
    topic_examples = []
    batch_num = 0

    while len(topic_examples) < examples_for_topic:
        batch_num += 1
        n = min(examples_for_topic - len(topic_examples), batch_size)
        print(f"  [topic {topic_idx}] Batch {batch_num} — requesting {n} examples...")

        generation_prompt = generation_prompt_template.format_map(_SafeDict(n=n, topic=topic))
        batch = generate_batch_with_retry(topic_idx, batch_num, llm_cfg, generation_prompt)
        valid = [
            {"messages": [{"role": m["role"], "content": m["content"]} for m in ex["messages"]]}
            for ex in batch
            if is_valid_example(ex)
        ]
        invalid = len(batch) - len(valid)

        if invalid:
            print(f"  [topic {topic_idx}] Dropped {invalid} malformed examples from batch")

        topic_examples.extend(valid)
        print(f"  [topic {topic_idx}] {len(topic_examples)}/{examples_for_topic} collected")

        if len(topic_examples) < examples_for_topic:
            time.sleep(3)

    return topic_examples[:examples_for_topic]


def generate_dataset(
    config_path: str | None = None,
    filename: str | None = None,
) -> None:
    """Generate full dataset across all topics using parallel workers."""
    config = load_config(config_path)
    gen_cfg = config.generation

    if gen_cfg.model == "CHANGE_ME":
        raise ValueError("Generation model not configured. Set 'model' in the 'generation' section of your config.")

    if not gen_cfg.prompt or not gen_cfg.prompt.strip():
        raise ValueError("Generation prompt not configured. Set 'prompt' in the 'generation' section of your config.")

    output_dir = config.paths.unprocessed_data_dir
    os.makedirs(output_dir, exist_ok=True)

    if filename is not None:
        if not filename.endswith(".jsonl"):
            filename = f"{filename}.jsonl"
        output_file = os.path.join(output_dir, filename)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"bangla_sft_{timestamp}.jsonl")

    total_topics = len(config.topics)
    total_written = 0
    write_lock = Lock()

    max_workers = gen_cfg.get_max_workers()
    print(f"Generating dataset with {max_workers} parallel topic workers")
    print(f"Output: {output_file}")
    print(f"Using LLM: {gen_cfg.model}")

    with open(output_file, "w", encoding="utf-8") as f:
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        generate_topic,
                        topic_idx,
                        topic_entry.topic,
                        topic_entry.count,
                        gen_cfg.batch_size,
                        total_topics,
                        gen_cfg,
                        gen_cfg.prompt,
                    ): topic_idx
                    for topic_idx, topic_entry in enumerate(config.topics, 1)
                }

                for future in as_completed(futures):
                    topic_idx = futures[future]
                    try:
                        examples = future.result()
                        with write_lock:
                            for example in examples:
                                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                                f.flush()
                                total_written += 1
                        print(f"  Topic {topic_idx} written — {len(examples)} examples ({total_written} total so far)")
                    except Exception as e:
                        print(f"  Topic {topic_idx} failed: {e}")

        except KeyboardInterrupt:
            print(f"\nInterrupted — {total_written} examples saved to {output_file}")
            return

    print(f"\nDone — {total_written} examples written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training dataset")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    parser.add_argument("filename", nargs="?", type=str, default=None, help="Output filename")
    args = parser.parse_args()
    generate_dataset(config_path=args.config, filename=args.filename)
