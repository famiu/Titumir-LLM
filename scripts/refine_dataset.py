import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests
from dotenv import load_dotenv

from training.config import REFINED_DATA_DIR, REFINEMENT_SYSTEM_PROMPT, REMOVED_DATA_DIR, UNPROCESSED_DATA_DIR

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

MAX_WORKERS = min(32, (os.cpu_count() or 1) * 4)
CHECKER_BATCH_SIZE = 40
MAX_RETRIES = 5
BATCH_TIMEOUT = 120
RETRY_BACKOFF = [2, 5, 10, 30, 60]


def check_batch_with_retry(
    batch_idx: int,
    batch: list[dict],
    start: int,
) -> tuple[int, list[dict], list[dict]]:
    """Check a single batch with retries. Returns (batch_idx, kept, removed_with_reasons)."""
    formatted = []
    for i, ex in enumerate(batch):
        post = ex["messages"][0]["content"]
        comment = ex["messages"][1]["content"]
        formatted.append(f"[{i}] Post: {post[:200]}\n    Comment: {comment[:200]}")

    prompt = "Check these training examples:\n\n" + "\n\n".join(formatted)

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
                    "messages": [
                        {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "reasoning": {"effort": "none"},
                },
                timeout=BATCH_TIMEOUT,
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
            print(f"  [batch {batch_idx}] ✗ Parse failed — keeping entire batch")
            return batch_idx, batch, []

        except requests.exceptions.Timeout:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(
                f"  [batch {batch_idx}] ✗ Timed out after {BATCH_TIMEOUT}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES}) — retrying in {wait}s"
            )
            time.sleep(wait)

        except requests.exceptions.ConnectionError:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"  [batch {batch_idx}] ✗ Network error (attempt {attempt + 1}/{MAX_RETRIES}) — retrying in {wait}s")
            time.sleep(wait)

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)] * 2
                print(f"  [batch {batch_idx}] ✗ Rate limited — retrying in {wait}s")
                time.sleep(wait)
            elif e.response.status_code >= 500:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                print(f"  [batch {batch_idx}] ✗ Server error {e.response.status_code} — retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"  [batch {batch_idx}] ✗ Client error {e.response.status_code} — keeping entire batch")
                return batch_idx, batch, []

        except Exception as e:
            print(f"  [batch {batch_idx}] ✗ Unexpected error: {e} — keeping entire batch")
            return batch_idx, batch, []

    print(f"  [batch {batch_idx}] ✗ All {MAX_RETRIES} retries exhausted — keeping entire batch")
    return batch_idx, batch, []


def refine_file(
    input_file: Path,
    refined_dir: str,
    removed_dir: str,
) -> None:
    """Refine a single JSONL file."""
    kept_file = os.path.join(refined_dir, input_file.name)
    removed_file = os.path.join(removed_dir, input_file.name)

    with open(input_file, encoding="utf-8") as f:
        all_examples = [json.loads(line) for line in f if line.strip()]

    total = len(all_examples)
    batches = []
    for i in range(0, total, CHECKER_BATCH_SIZE):
        batches.append((i // CHECKER_BATCH_SIZE, all_examples[i : i + CHECKER_BATCH_SIZE], i))

    total_batches = len(batches)
    print(f"\nRefining {input_file.name} — {total} examples, {total_batches} batches")

    results: dict[int, tuple[list[dict], list[dict]]] = {}
    results_lock = Lock()
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(check_batch_with_retry, idx, batch, start): idx for idx, batch, start in batches}

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
            print(f"\n⚠ Interrupted during {input_file.name}")
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
                print(f"  ✗ [{entry['global_idx']:05d}] REMOVED — {entry['reason']}")
                print(f"         Post:    {entry['example']['messages'][0]['content'][:80]}")
                print(f"         Comment: {entry['example']['messages'][1]['content'][:80]}")
                total_removed += 1

    print(
        f"  ✓ {input_file.name} done — "
        f"{total_kept} kept, {total_removed} removed "
        f"({100 * total_kept // total}% retained)"
    )


def refine_dataset(
    input_dir: str = UNPROCESSED_DATA_DIR,
    refined_dir: str = REFINED_DATA_DIR,
    removed_dir: str = REMOVED_DATA_DIR,
) -> None:
    """Refine all unprocessed JSONL files that don't already have a refined counterpart."""
    unprocessed_path = Path(input_dir)
    refined_path = Path(refined_dir)

    if not unprocessed_path.exists():
        print(f"✗ Input directory not found: {input_dir}")
        return

    os.makedirs(refined_dir, exist_ok=True)
    os.makedirs(removed_dir, exist_ok=True)

    all_files = sorted(unprocessed_path.glob("*.jsonl"))
    pending = [f for f in all_files if not (refined_path / f.name).exists()]
    skipped = len(all_files) - len(pending)

    if skipped:
        print(f"Skipping {skipped} already-refined files")
    if not pending:
        print("✓ All files already refined — nothing to do")
        return

    print(f"Found {len(pending)} file(s) to refine:")
    for f in pending:
        print(f"  {f.name}")

    try:
        for file in pending:
            refine_file(file, refined_dir, removed_dir)
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
        return

    print("\n✓ All done")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = Path(UNPROCESSED_DATA_DIR) / sys.argv[1]
        if not input_path.exists():
            print(f"✗ File not found: {input_path}")
            sys.exit(1)
        os.makedirs(REFINED_DATA_DIR, exist_ok=True)
        os.makedirs(REMOVED_DATA_DIR, exist_ok=True)
        refine_file(input_path, REFINED_DATA_DIR, REMOVED_DATA_DIR)
    else:
        # No argument — scan unprocessed directory
        refine_dataset()
