import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

import requests
from dotenv import load_dotenv

from training.config import GENERATION_PROMPT, TOPICS, UNPROCESSED_DATA_DIR

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

MAX_WORKERS = min(32, (os.cpu_count() or 1) * 4)
BATCH_TIMEOUT = 120
RETRY_BACKOFF = [2, 5, 10, 30, 60]
MAX_RETRIES = 5


def generate_batch_with_retry(
    topic: str,
    n: int,
    topic_idx: int,
    batch_num: int,
) -> list[dict]:
    """Generate a single batch with retries. Returns examples or raises on total failure."""
    prompt = GENERATION_PROMPT.format(n=n, topic=topic)

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "google/gemini-3.1-flash-lite-preview",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.9,
                    "max_tokens": 4000,
                    "reasoning": {"effort": "none"},
                },
                timeout=BATCH_TIMEOUT,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            print(f"  [topic {topic_idx} batch {batch_num}] ✗ JSON parse failed: {e} — retrying...")
            time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])

        except requests.exceptions.Timeout:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(
                f"  [topic {topic_idx} batch {batch_num}] ✗ Timed out after {BATCH_TIMEOUT}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES}) — retrying in {wait}s"
            )
            time.sleep(wait)

        except requests.exceptions.ConnectionError:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(
                f"  [topic {topic_idx} batch {batch_num}] ✗ Network error "
                f"(attempt {attempt + 1}/{MAX_RETRIES}) — retrying in {wait}s"
            )
            time.sleep(wait)

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)] * 2
                print(f"  [topic {topic_idx} batch {batch_num}] ✗ Rate limited — retrying in {wait}s")
                time.sleep(wait)
            elif e.response.status_code >= 500:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                print(
                    f"  [topic {topic_idx} batch {batch_num}] ✗ Server error {e.response.status_code} "
                    f"— retrying in {wait}s"
                )
                time.sleep(wait)
            else:
                print(
                    f"  [topic {topic_idx} batch {batch_num}] ✗ Client error {e.response.status_code} — skipping batch"
                )
                return []

        except Exception as e:
            print(f"  [topic {topic_idx} batch {batch_num}] ✗ Unexpected error: {e} — retrying...")
            time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])

    print(f"  [topic {topic_idx} batch {batch_num}] ✗ All retries exhausted — skipping batch")
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
) -> list[dict]:
    """Generate all examples for a single topic sequentially."""
    print(f"\n[{topic_idx}/{total_topics}] Topic: {topic} ({examples_for_topic} examples)")
    topic_examples = []
    batch_num = 0

    while len(topic_examples) < examples_for_topic:
        batch_num += 1
        n = min(examples_for_topic - len(topic_examples), batch_size)
        print(f"  [topic {topic_idx}] Batch {batch_num} — requesting {n} examples...")

        batch = generate_batch_with_retry(topic, n, topic_idx, batch_num)
        valid = [
            {"messages": [{"role": m["role"], "content": m["content"]} for m in ex["messages"]]}
            for ex in batch
            if is_valid_example(ex)
        ]
        invalid = len(batch) - len(valid)

        if invalid:
            print(f"  [topic {topic_idx}] ⚠ Dropped {invalid} malformed examples from batch")

        topic_examples.extend(valid)
        print(f"  [topic {topic_idx}] ✓ {len(topic_examples)}/{examples_for_topic} collected")

        if len(topic_examples) < examples_for_topic:
            time.sleep(3)

    return topic_examples[:examples_for_topic]


def generate_dataset(
    output_dir: str = UNPROCESSED_DATA_DIR,
    filename: str | None = None,
) -> None:
    """Generate full dataset across all topics using parallel workers."""
    os.makedirs(output_dir, exist_ok=True)

    if filename is not None:
        # ensure it ends with .jsonl
        if not filename.endswith(".jsonl"):
            filename = f"{filename}.jsonl"
        output_file = os.path.join(output_dir, filename)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"bangla_sft_{timestamp}.jsonl")

    batch_size = 20
    total_topics = len(TOPICS)
    total_written = 0
    write_lock = Lock()

    print(f"Generating dataset with {MAX_WORKERS} parallel topic workers")
    print(f"Output → {output_file}")

    # results keyed by topic_idx to preserve order on write
    results: dict[int, list[dict]] = {}

    with open(output_file, "w", encoding="utf-8") as f:
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        generate_topic,
                        topic_idx,
                        topic,
                        examples_for_topic,
                        batch_size,
                        total_topics,
                    ): topic_idx
                    for topic_idx, (topic, examples_for_topic) in enumerate(TOPICS, 1)
                }

                for future in as_completed(futures):
                    topic_idx = futures[future]
                    try:
                        examples = future.result()
                        with write_lock:
                            results[topic_idx] = examples
                            for example in examples:
                                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                                f.flush()
                                total_written += 1
                        print(
                            f"  ✓ Topic {topic_idx} written — {len(examples)} examples ({total_written} total so far)"
                        )
                    except Exception as e:
                        print(f"  ✗ Topic {topic_idx} failed: {e}")

        except KeyboardInterrupt:
            print(f"\n⚠ Interrupted — {total_written} examples saved to {output_file}")
            return

    print(f"\n✓ Done — {total_written} examples written to {output_file}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else None
    generate_dataset(filename=name)
