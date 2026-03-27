import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import count
from threading import Lock

from _llm import SafeDict, call_llm

from training.config import load_config


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
    llm_cfg,
    generation_prompt_template: str,
    global_batch_counter: count,
) -> list[dict]:
    """Generate all examples for a single topic sequentially."""
    print(f"\n[{topic_idx}/{total_topics}] Topic: {topic} ({examples_for_topic} examples)")
    topic_examples = []

    while len(topic_examples) < examples_for_topic:
        batch_num = next(global_batch_counter)
        n = min(examples_for_topic - len(topic_examples), batch_size)
        print(f"  Batch #{batch_num} [topic {topic_idx}] — requesting {n} examples...")

        generation_prompt = generation_prompt_template.format_map(SafeDict(n=n, topic=topic))
        batch = call_llm(llm_cfg, [{"role": "user", "content": generation_prompt}])

        if batch is None:
            print(f"  Batch #{batch_num} [topic {topic_idx}] failed — skipping")
            break

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
                global_batch_counter = count(1)
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
                        global_batch_counter,
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
                                total_written += 1
                            f.flush()
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
