import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests
from dotenv import load_dotenv

from training.config import RefinementConfig, load_config

load_dotenv()

RETRY_BASE_DELAY = 2
RETRY_MAX_DELAY = 120


def _retry_delay(attempt: int) -> float:
    """Exponential backoff: base * 2^attempt, capped at max."""
    return min(RETRY_BASE_DELAY * (2**attempt), RETRY_MAX_DELAY)


def check_batch_with_retry(
    batch_idx: int,
    batch: list[dict],
    start: int,
    llm_cfg: RefinementConfig,
    refinement_prompt: str,
) -> tuple[int, list[dict], list[dict]]:
    """Check a single batch with retries. Returns (batch_idx, kept, removed_with_reasons)."""
    api_key = llm_cfg.get_api_key()
    if not api_key:
        raise ValueError(f"API key not found: set {llm_cfg.api_key_env} environment variable")

    formatted = []
    for i, ex in enumerate(batch):
        post = ex["messages"][0]["content"]
        comment = ex["messages"][1]["content"]
        formatted.append(f"[{i}] Post: {post[:200]}\n    Comment: {comment[:200]}")

    prompt = "Check these training examples:\n\n" + "\n\n".join(formatted)

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
                    "messages": [
                        {"role": "system", "content": refinement_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": llm_cfg.temperature,
                    "max_tokens": llm_cfg.max_tokens,
                    "reasoning": {"effort": "none"},
                },
                timeout=llm_cfg.batch_timeout,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned)

            remove_indices = set(result.get("remove", []))
            reasons = result.get("reasons", {})

            kept = []
            removed = []
            for i, example in enumerate(batch):
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

        except (json.JSONDecodeError, KeyError):
            wait = _retry_delay(attempt)
            print(
                f"  [batch {batch_idx}] Parse failed (attempt {attempt + 1}/{llm_cfg.max_retries}) — "
                f"retrying in {wait}s"
            )
            time.sleep(wait)

        except requests.exceptions.Timeout:
            wait = _retry_delay(attempt)
            print(
                f"  [batch {batch_idx}] Timed out after {llm_cfg.batch_timeout}s "
                f"(attempt {attempt + 1}/{llm_cfg.max_retries}) — retrying in {wait}s"
            )
            time.sleep(wait)

        except requests.exceptions.ConnectionError:
            wait = _retry_delay(attempt)
            print(
                f"  [batch {batch_idx}] Network error (attempt {attempt + 1}/{llm_cfg.max_retries}) — retrying in {wait}s"
            )
            time.sleep(wait)

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                wait = _retry_delay(attempt + 1)
                print(f"  [batch {batch_idx}] Rate limited — retrying in {wait}s")
                time.sleep(wait)
            elif e.response.status_code >= 500:
                wait = _retry_delay(attempt)
                print(f"  [batch {batch_idx}] Server error {e.response.status_code} — retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"  [batch {batch_idx}] Client error {e.response.status_code} — keeping entire batch")
                return batch_idx, batch, []

        except Exception as e:
            print(f"  [batch {batch_idx}] Unexpected error: {e} — keeping entire batch")
            return batch_idx, batch, []

    print(f"  [batch {batch_idx}] All {llm_cfg.max_retries} retries exhausted — keeping entire batch")
    return batch_idx, batch, []


def refine_file(
    input_file: Path,
    refined_dir: str,
    removed_dir: str,
    llm_cfg: RefinementConfig,
    refinement_prompt: str,
    batch_size: int,
) -> None:
    """Refine a single JSONL file."""
    kept_file = os.path.join(refined_dir, input_file.name)
    removed_file = os.path.join(removed_dir, input_file.name)

    all_examples = []
    with open(input_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                all_examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping malformed JSON at line {line_num}: {e}")

    total = len(all_examples)
    if total == 0:
        print(f"  {input_file.name}: Empty file, nothing to refine")
        return

    batches = []
    for i in range(0, total, batch_size):
        batches.append((i // batch_size, all_examples[i : i + batch_size], i))

    total_batches = len(batches)
    print(f"\nRefining {input_file.name} — {total} examples, {total_batches} batches")

    results: dict[int, tuple[list[dict], list[dict]]] = {}
    results_lock = Lock()
    completed = 0

    with ThreadPoolExecutor(max_workers=llm_cfg.get_max_workers()) as executor:
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
        }

        try:
            for future in as_completed(futures):
                batch_idx, kept, removed = future.result()
                with results_lock:
                    results[batch_idx] = (kept, removed)
                    completed += 1
                    print(
                        f"  [{completed}/{total_batches}] batch {batch_idx} done — "
                        f"{len(kept)} kept, {len(removed)} removed"
                    )

        except KeyboardInterrupt:
            print(f"\nInterrupted during {input_file.name}")
            raise

    total_kept = 0
    total_removed = 0

    with (
        open(kept_file, "w", encoding="utf-8") as kf,
        open(removed_file, "w", encoding="utf-8") as rf,
    ):
        for batch_idx in sorted(results.keys()):
            kept, removed = results[batch_idx]
            for example in kept:
                kf.write(json.dumps(example, ensure_ascii=False) + "\n")
                total_kept += 1
            for entry in removed:
                rf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print(f"  [{entry['global_idx']:05d}] REMOVED — {entry['reason']}")
                print(f"         Post:    {entry['example']['messages'][0]['content'][:80]}")
                print(f"         Comment: {entry['example']['messages'][1]['content'][:80]}")
                total_removed += 1

    retention = f"{100 * total_kept // total}%" if total > 0 else "0%"
    print(f"  {input_file.name} done — {total_kept} kept, {total_removed} removed ({retention} retained)")


def refine_dataset(
    config_path: str | None = None,
    input_file: str | None = None,
) -> None:
    """Refine all unprocessed JSONL files that don't already have a refined counterpart."""
    config = load_config(config_path)
    ref_cfg = config.refinement
    input_dir = config.paths.unprocessed_data_dir
    refined_dir = config.paths.refined_data_dir
    removed_dir = config.paths.removed_data_dir

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
            )
    except KeyboardInterrupt:
        print("\nInterrupted")
        return

    print("\nAll done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine generated dataset")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    parser.add_argument("filename", nargs="?", type=str, default=None, help="Specific file to refine")
    args = parser.parse_args()
    refine_dataset(config_path=args.config, input_file=args.filename)
