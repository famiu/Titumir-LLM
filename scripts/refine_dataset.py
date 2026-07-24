import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts._data import atomic_text_writer, file_sha256, validate_conversation
from scripts._llm import call_llm
from training.config import RefinementConfig, load_config


def check_batch_with_retry(
    batch_idx: int,
    batch: list[dict],
    start: int,
    llm_cfg: RefinementConfig,
    refinement_prompt: str,
) -> tuple[int, list[dict], list[dict]]:
    """Check a single batch with retries. Returns (batch_idx, kept, removed_with_reasons)."""
    formatted = []
    validated_batch = []
    for i, ex in enumerate(batch):
        ex = validate_conversation(ex, f"batch {batch_idx} example {i}")
        validated_batch.append(ex)
        post = ex["messages"][0]["content"]
        comment = ex["messages"][1]["content"]
        formatted.append(f"[{i}] Post: {post}\n    Comment: {comment}")

    prompt = "Check these training examples:\n\n" + "\n\n".join(formatted)

    result = call_llm(
        llm_cfg,
        [
            {"role": "system", "content": refinement_prompt},
            {"role": "user", "content": prompt},
        ],
        expected_type=dict,
    )

    if result is None:
        raise RuntimeError(f"Batch {batch_idx} refinement failed after all API retries")

    if not isinstance(result, dict):
        raise ValueError(f"Batch {batch_idx} refinement response must be an object")
    try:
        remove_indices = {int(index) for index in result.get("remove", [])}
        keep_indices = {int(index) for index in result.get("keep", [])}
    except (TypeError, ValueError) as error:
        raise ValueError(f"Batch {batch_idx} contains non-integer keep/remove indices") from error
    reasons = result.get("reasons", {})
    if not isinstance(reasons, dict):
        raise ValueError(f"Batch {batch_idx} reasons must be an object")

    valid_indices = set(range(len(validated_batch)))
    if remove_indices & keep_indices:
        raise ValueError(f"Batch {batch_idx} keep/remove indices overlap")
    if remove_indices | keep_indices != valid_indices:
        raise ValueError(f"Batch {batch_idx} keep/remove indices must partition the complete batch")

    kept = []
    removed = []
    for i, example in enumerate(validated_batch):
        if i in remove_indices:
            removed.append(
                {
                    "example": example,
                    "reason": reasons.get(str(i), "no reason given"),
                    "global_idx": start + i,
                }
            )
        else:
            kept.append(example)

    return batch_idx, kept, removed


def refine_file(
    input_file: Path,
    refined_dir: str,
    removed_dir: str,
    llm_cfg: RefinementConfig,
    refinement_prompt: str,
    batch_size: int,
    resume: bool = False,
) -> None:
    """Refine a single JSONL file."""
    kept_file = os.path.join(refined_dir, input_file.name)
    removed_file = os.path.join(removed_dir, input_file.name)
    state_file = Path(f"{kept_file}.state.json")

    all_examples = []
    with open(input_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON in {input_file} at line {line_num}: {e}") from e
            all_examples.append(validate_conversation(example, f"{input_file}:{line_num}"))

    total = len(all_examples)
    if total == 0:
        print(f"  {input_file.name}: Empty file, nothing to refine")
        return

    batches = []
    for i in range(0, total, batch_size):
        batches.append((i // batch_size, all_examples[i : i + batch_size], i))

    total_batches = len(batches)
    print(f"\nRefining {input_file.name} — {total} examples, {total_batches} batches")

    identity = {
        "input_sha256": file_sha256(input_file),
        "prompt_sha256": hashlib.sha256(refinement_prompt.encode()).hexdigest(),
        "model": llm_cfg.model,
        "batch_size": batch_size,
    }
    state = {"identity": identity, "completed": {}}
    if state_file.exists():
        if not resume:
            raise FileExistsError(f"Refinement state already exists: {state_file}. Resume with --resume.")
        with open(state_file, encoding="utf-8") as file:
            state = json.load(file)
        if state.get("identity") != identity:
            raise ValueError(f"Refinement state is incompatible with the current input/config: {state_file}")
    elif resume:
        raise FileNotFoundError(f"Refinement state not found: {state_file}")

    serialized_results: dict[str, dict[str, list[dict]]] = state["completed"]

    def save_state() -> None:
        with atomic_text_writer(state_file) as file:
            json.dump({"identity": identity, "completed": serialized_results}, file, ensure_ascii=False)

    completed = len(serialized_results)
    executor = ThreadPoolExecutor(max_workers=llm_cfg.get_max_workers())
    futures = {}
    try:
        futures = {
            executor.submit(
                check_batch_with_retry,
                idx,
                batch,
                start,
                llm_cfg,
                refinement_prompt,
            ): idx
            for idx, batch, start in batches
            if str(idx) not in serialized_results
        }

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                batch_idx, kept, removed = future.result()
            except Exception as error:
                save_state()
                for pending in futures:
                    pending.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(
                    f"Refinement stopped at batch {batch_idx}; completed work is saved in {state_file}"
                ) from error
            serialized_results[str(batch_idx)] = {"kept": kept, "removed": removed}
            completed += 1
            save_state()
            print(
                f"  [{completed}/{total_batches}] batch {batch_idx} checkpointed — "
                f"{len(kept)} kept, {len(removed)} removed"
            )
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        save_state()
        print(f"\nInterrupted during {input_file.name}; resume state saved to {state_file}")
        raise
    else:
        executor.shutdown(wait=True)

    total_kept = 0
    total_removed = 0

    with atomic_text_writer(kept_file) as kf, atomic_text_writer(removed_file) as rf:
        for batch_idx in range(total_batches):
            result = serialized_results[str(batch_idx)]
            kept = result["kept"]
            removed = result["removed"]
            for example in kept:
                kf.write(json.dumps(example, ensure_ascii=False) + "\n")
                total_kept += 1
            for entry in removed:
                rf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print(f"  [{entry['global_idx']:05d}] REMOVED — {entry['reason']}")
                print(f"         Post:    {entry['example']['messages'][0]['content'][:80]}")
                print(f"         Comment: {entry['example']['messages'][1]['content'][:80]}")
                total_removed += 1
    state_file.unlink(missing_ok=True)
    with atomic_text_writer(f"{kept_file}.manifest.json") as file:
        json.dump(
            {
                **identity,
                "input_examples": total,
                "kept_examples": total_kept,
                "removed_examples": total_removed,
                "retention_rate": total_kept / total,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    retention = f"{100 * total_kept // total}%" if total > 0 else "0%"
    print(f"  {input_file.name} done — {total_kept} kept, {total_removed} removed ({retention} retained)")


def refine_dataset(
    config_path: str | None = None,
    input_file: str | None = None,
    resume: bool = False,
) -> None:
    """Refine all unprocessed JSONL files that don't already have a refined counterpart."""
    config = load_config(config_path)
    ref_cfg = config.refinement
    input_dir = config.profile.unprocessed_data_dir
    refined_dir = config.profile.refined_data_dir
    removed_dir = config.profile.removed_data_dir

    unprocessed_path = Path(input_dir)
    refined_path = Path(refined_dir)

    if ref_cfg.model == "CHANGE_ME":
        raise ValueError("Refinement model not configured. Set 'model' in the 'refinement' section of your config.")

    if not ref_cfg.prompt or not ref_cfg.prompt.strip():
        raise ValueError("Refinement prompt not configured. Set 'prompt' in the 'refinement' section of your config.")

    print(f"Using LLM for refinement: {ref_cfg.model}")

    if input_file is not None:
        input_path = Path(input_dir) / input_file
        if not input_path.exists():
            print(f"File not found: {input_path}")
            return
        os.makedirs(refined_dir, exist_ok=True)
        os.makedirs(removed_dir, exist_ok=True)
        refine_file(
            input_path,
            refined_dir,
            removed_dir,
            ref_cfg,
            ref_cfg.prompt,
            ref_cfg.batch_size,
            resume=resume,
        )
        return

    if not unprocessed_path.exists():
        print(f"Input directory not found: {input_dir}")
        return

    os.makedirs(refined_dir, exist_ok=True)
    os.makedirs(removed_dir, exist_ok=True)

    all_files = sorted(unprocessed_path.glob("*.jsonl"))
    pending = [f for f in all_files if not (refined_path / f.name).exists()]
    skipped = len(all_files) - len(pending)

    if skipped:
        print(f"Skipping {skipped} already-refined files")
    if not pending:
        print("All files already refined — nothing to do")
        return

    print(f"Found {len(pending)} file(s) to refine:")
    for f in pending:
        print(f"  {f.name}")

    try:
        for file in pending:
            refine_file(
                file,
                refined_dir,
                removed_dir,
                ref_cfg,
                ref_cfg.prompt,
                ref_cfg.batch_size,
                resume=resume,
            )
    except KeyboardInterrupt:
        print("\nInterrupted")
        return

    print("\nAll done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine generated dataset")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    parser.add_argument("filename", nargs="?", type=str, default=None, help="Specific file to refine")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted refinement checkpoints")
    args = parser.parse_args()
    refine_dataset(config_path=args.config, input_file=args.filename, resume=args.resume)
