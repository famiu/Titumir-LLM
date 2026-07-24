import argparse
import json
from collections import defaultdict
from pathlib import Path

import regex
from transformers import AutoTokenizer

from scripts._data import validate_conversation
from training.config import load_config

TEST_STRINGS = [
    ("Pure Bengali script", "আজকে ঢাকায় অনেক জ্যাম ছিল, সত্যিই অনেক কষ্ট হয়েছে"),
    ("Colloquial Banglish", "bhai seriously কষ্ট হইছে, amader ki hobe 😭"),
    ("Romanized Bengali", "vai eta ki hoise, amio same rokom vabsi ekdom"),
    ("Heavy code-switching", "bhai এইটা দেইখা হাসতে হাসতে শেষ 💀 seriously amader ki hobe"),
    ("Pure English", "bhai seriously what is happening in this country today"),
    ("Political Bengali", "সরকারের দুর্নীতি আর সহ্য করা যাচ্ছে না, জনগণ এখন রাস্তায়"),
    ("Conjuncts", "রাষ্ট্রের শ্রদ্ধেয় শিক্ষক লক্ষ্মীপুরে সংস্কৃতি চর্চা করেন"),
    ("Digits and punctuation", "আজ ১২৩৪৫৬৭৮৯০ টাকা খরচ হয়েছে। কী অবস্থা!"),
    ("Marks", "চাঁদ, বাংলা, দুঃখ, রং এবং কাঁটা"),
    ("Joiners", "ক্\u200dষ এবং ক্\u200cষ দেখতে আলাদা হতে পারে"),
]


def script_category(text: str) -> str:
    """Classify text by Bengali and Latin script presence."""
    has_bengali = bool(regex.search(r"\p{Script=Bengali}", text))
    has_latin = bool(regex.search(r"\p{Script=Latin}", text))
    if has_bengali and has_latin:
        return "mixed"
    if has_bengali:
        return "bengali"
    if has_latin:
        return "latin"
    return "other"


def analyze_text(tokenizer, text: str, max_length: int) -> dict[str, float | int | str | bool]:
    """Measure tokenizer efficiency for one text."""
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    grapheme_count = len(regex.findall(r"\X", text))
    return {
        "category": script_category(text),
        "codepoints": len(text),
        "graphemes": grapheme_count,
        "tokens": token_count,
        "tokens_per_codepoint": token_count / len(text) if text else 0.0,
        "tokens_per_grapheme": token_count / grapheme_count if grapheme_count else 0.0,
        "truncated": token_count > max_length,
    }


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def check_tokenizer(config_path: str | None = None, dataset_path: str | None = None, limit: int = 10000) -> None:
    """Check tokenizer efficiency on Bengali text."""
    config = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    print(f"Model: {config.model.name}\n")
    print(f"{'Test':<25} {'Codepts':>7} {'Graphs':>7} {'Tokens':>7} {'Tok/Graph':>10} {'Script':>8}")
    print("-" * 78)

    for label, text in TEST_STRINGS:
        result = analyze_text(tokenizer, text, config.model.max_seq_length)
        print(
            f"{label:<25} {result['codepoints']:>7} {result['graphemes']:>7} {result['tokens']:>7} "
            f"{result['tokens_per_grapheme']:>10.2f} {result['category']:>8}"
        )

    if dataset_path is None:
        return

    by_category: dict[str, list[dict]] = defaultdict(list)
    with open(Path(dataset_path), encoding="utf-8") as file:
        for line_num, line in enumerate(file, 1):
            if sum(len(items) for items in by_category.values()) >= limit:
                break
            if not line.strip():
                continue
            try:
                example = validate_conversation(json.loads(line), f"{dataset_path}:{line_num}")
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON in {dataset_path} at line {line_num}: {error}") from error
            except ValueError as error:
                raise ValueError(f"Invalid conversation in {dataset_path} at line {line_num}: {error}") from error
            text = "\n".join(message["content"] for message in example["messages"])
            result = analyze_text(tokenizer, text, config.model.max_seq_length)
            by_category[str(result["category"])].append(result)

    print("\nCorpus summary")
    for category, results in sorted(by_category.items()):
        ratios = [float(result["tokens_per_grapheme"]) for result in results]
        truncated = sum(bool(result["truncated"]) for result in results)
        print(
            f"{category}: n={len(results)}, tok/grapheme p50={percentile(ratios, 0.5):.2f}, "
            f"p95={percentile(ratios, 0.95):.2f}, truncated={truncated / len(results):.1%}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check tokenizer efficiency on Bengali text")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--dataset", type=str, default=None, help="Optional conversation JSONL for corpus metrics")
    parser.add_argument("--limit", type=int, default=10000, help="Maximum corpus examples to inspect")
    args = parser.parse_args()
    check_tokenizer(config_path=args.config, dataset_path=args.dataset, limit=args.limit)
