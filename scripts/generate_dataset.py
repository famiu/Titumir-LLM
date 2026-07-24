import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import count
from pathlib import Path

from scripts._data import atomic_text_writer, validate_conversation, validate_prompt_fields
from scripts._llm import call_llm
from training.config import load_config


def is_valid_example(example: dict) -> bool:
    """Check that an example has the expected structure."""
    try:
        validate_conversation(example)
    except ValueError:
        return False
    return True


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
    stalled_batches = 0
    total_returned = 0
    total_invalid = 0

    while len(topic_examples) < examples_for_topic:
        batch_num = next(global_batch_counter)
        n = min(examples_for_topic - len(topic_examples), batch_size)
        print(f"  Batch #{batch_num} [topic {topic_idx}] — requesting {n} examples...")

        generation_prompt = generation_prompt_template.format(n=n, topic=topic)
        batch = call_llm(llm_cfg, [{"role": "user", "content": generation_prompt}], expected_type=list)

        if batch is None:
            stalled_batches += 1
            print(
                f"  Batch #{batch_num} [topic {topic_idx}] failed "
                f"({stalled_batches}/{llm_cfg.max_stalled_batches} stalled batches)"
            )
            if stalled_batches >= llm_cfg.max_stalled_batches:
                raise RuntimeError(
                    f"Topic {topic_idx} produced {len(topic_examples)}/{examples_for_topic} examples "
                    f"after {stalled_batches} stalled batches"
                )
            continue

        valid = []
        total_returned += len(batch)
        for example in batch:
            try:
                validated = validate_conversation(example)
                valid.append(
                    {
                        "messages": validated["messages"],
                        "metadata": {"topic": topic},
                    }
                )
            except ValueError:
                continue
        invalid = len(batch) - len(valid)
        total_invalid += invalid

        if invalid:
            print(f"  [topic {topic_idx}] Dropped {invalid} malformed examples from batch")

        if not valid:
            stalled_batches += 1
            if stalled_batches >= llm_cfg.max_stalled_batches:
                raise RuntimeError(
                    f"Topic {topic_idx} produced {len(topic_examples)}/{examples_for_topic} examples "
                    f"after {stalled_batches} stalled batches"
                )
            continue

        stalled_batches = 0
        topic_examples.extend(valid)
        print(f"  [topic {topic_idx}] {len(topic_examples)}/{examples_for_topic} collected")

    print(
        f"  [topic {topic_idx}] requested={examples_for_topic}, returned={total_returned}, "
        f"invalid={total_invalid}, accepted={examples_for_topic}"
    )
    return topic_examples[:examples_for_topic]


def generate_dataset(
    config_path: str | None = None,
    filename: str | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> None:
    """Generate full dataset across all topics using parallel workers."""
    config = load_config(config_path)
    gen_cfg = config.generation

    if gen_cfg.model == "CHANGE_ME":
        raise ValueError("Generation model not configured. Set 'model' in the 'generation' section of your config.")

    if not gen_cfg.prompt or not gen_cfg.prompt.strip():
        raise ValueError("Generation prompt not configured. Set 'prompt' in the 'generation' section of your config.")
    validate_prompt_fields(gen_cfg.prompt, {"n", "topic"})

    output_dir = config.profile.unprocessed_data_dir
    os.makedirs(output_dir, exist_ok=True)

    if filename is not None:
        if not filename.endswith(".jsonl"):
            filename = f"{filename}.jsonl"
        output_file = os.path.join(output_dir, filename)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"bangla_sft_{timestamp}.jsonl")

    output_path = Path(output_file)
    state_path = output_path.with_suffix(f"{output_path.suffix}.state.json")
    identity = {
        "profile": config.profile.name,
        "model": gen_cfg.model,
        "prompt_sha256": hashlib.sha256(gen_cfg.prompt.encode()).hexdigest(),
        "topics": [{"topic": entry.topic, "count": entry.count} for entry in config.topics],
    }

    if resume and overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    if overwrite:
        output_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
    elif output_path.exists() and not resume:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite or choose another filename.")

    state = {"identity": identity, "completed": {}}
    if resume:
        if not state_path.exists():
            if output_path.exists():
                print(f"Output is already complete: {output_path}")
                return
            raise FileNotFoundError(f"Resume state not found: {state_path}")
        with open(state_path, encoding="utf-8") as file:
            state = json.load(file)
        if state.get("identity") != identity:
            raise ValueError(f"Resume state is incompatible with the current config: {state_path}")

    completed: dict[str, list[dict]] = state["completed"]
    total_topics = len(config.topics)
    total_written = sum(len(examples) for examples in completed.values())
    failed_topics = []

    def save_state() -> None:
        with atomic_text_writer(state_path) as file:
            json.dump({"identity": identity, "completed": completed}, file, ensure_ascii=False)

    max_workers = gen_cfg.get_max_workers()
    print(f"Generating dataset with {max_workers} parallel topic workers")
    print(f"Output: {output_file}")
    print(f"Using LLM: {gen_cfg.model}")

    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {}
    try:
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
            if str(topic_idx) not in completed
        }

        for future in as_completed(futures):
            topic_idx = futures[future]
            try:
                examples = future.result()
                completed[str(topic_idx)] = examples
                total_written += len(examples)
                save_state()
                print(f"  Topic {topic_idx} checkpointed — {len(examples)} examples ({total_written} total so far)")
            except Exception as e:
                print(f"  Topic {topic_idx} failed: {e}")
                failed_topics.append(topic_idx)
    except KeyboardInterrupt as interrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        try:
            save_state()
            print(f"\nInterrupted — resume state saved to {state_path}")
        except OSError as error:
            print(f"\nInterrupted — failed to save resume state: {error}")
        raise interrupt
    else:
        executor.shutdown(wait=True)

    if failed_topics:
        save_state()
        failed = ", ".join(str(topic_idx) for topic_idx in sorted(failed_topics))
        raise RuntimeError(f"Generation incomplete; failed topic(s): {failed}. Resume with --resume.")

    with atomic_text_writer(output_path) as file:
        for topic_idx in range(1, total_topics + 1):
            for example in completed[str(topic_idx)]:
                file.write(json.dumps(example, ensure_ascii=False) + "\n")
    state_path.unlink(missing_ok=True)
    with atomic_text_writer(output_path.with_suffix(f"{output_path.suffix}.manifest.json")) as file:
        json.dump(
            {
                **identity,
                "topic_counts": {key: len(value) for key, value in completed.items()},
                "examples": total_written,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nDone — {total_written} examples written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training dataset")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    parser.add_argument("filename", nargs="?", type=str, default=None, help="Output filename")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Resume an interrupted named output")
    mode.add_argument("--overwrite", action="store_true", help="Replace an existing named output")
    args = parser.parse_args()
    generate_dataset(config_path=args.config, filename=args.filename, resume=args.resume, overwrite=args.overwrite)
