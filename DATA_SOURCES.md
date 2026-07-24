# Data Sources And Provenance

This file records the sources referenced by the default research configuration. Dataset availability,
licenses, terms, and revisions can change independently of this repository. Verify the current upstream
dataset card and underlying corpus terms before every training run, publication, or model release.

The Apache-2.0 license in this repository covers the code only. It does not relicense datasets, generated
examples, model weights, or outputs from external model providers.

## Continued Pretraining Sources

| Dataset | Config and split | Immutable Hub revision | Retrieved | Intended signal | License/provenance action |
| --- | --- | --- | --- | --- | --- |
| [BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset](https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset) | `train`, column `review` | `38c97cd4255799359b612bafb721e3c442bc0851` | `2026-07-24` | Informal Bengali-English reviews | Verify the dataset card, original collection consent, commercial-use terms, and redistribution rights. Treat as unresolved until recorded in the run report. |
| [sanzanalora/Ben-Sarc](https://huggingface.co/datasets/sanzanalora/Ben-Sarc) | `train`, column `Comments` | `3bc2ba28968d2a18e17aebae85f7dab23d18302d` | `2026-07-24` | Bengali social comments and sarcasm | Verify the dataset card, upstream source, privacy handling, and license. Treat as unresolved until recorded in the run report. |
| [allenai/c4](https://huggingface.co/datasets/allenai/c4) | `bn`, `train[:60000]` | `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` | `2026-07-24` | Bengali web text | Review the dataset card and Common Crawl-derived terms. Record attribution and filtering obligations for the selected revision. |
| [statmt/cc100](https://huggingface.co/datasets/statmt/cc100) | `bn_rom`, `train[:20000]` | `8c658c983d32eab9170d77d416252cfaa0c23e96` | `2026-07-24` | Romanized Bengali web text | Review the dataset card, CC100 paper, source crawl terms, and redistribution conditions for the selected revision. |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | `20231101.bn`, `train[:10000]` | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | `2026-07-24` | Formal Bengali reference text | Wikimedia content generally carries attribution and share-alike obligations. Verify the exact dump terms and preserve required attribution. |

## Supervised Finetuning Data

The default SFT workflow generates post/comment pairs through a configured external LLM endpoint and then
uses an LLM-based refinement pass. Before using generated data, record:

- Generator and refiner provider, model identifiers, dates, and applicable terms.
- Prompt hashes and generated/refined dataset manifest hashes.
- Whether model-output training and redistribution are permitted.
- Topic counts, script distribution, retention rates, duplicate statistics, and failed batches.
- Human-review sampling method, reviewer instructions, agreement, and identified harms.

Generation and refinement manifests are written next to their JSONL outputs. `just audit-dataset` produces
additional quality and overlap statistics without modifying the source dataset.

## Known Data Risks

- Common Crawl-derived corpora can contain boilerplate, duplicated text, language-identification errors,
  personal information, abuse, and copyrighted material.
- Synthetic examples can reproduce provider biases, memorized text, prompt phrasing, and a narrow caricature
  of Banglish.
- The same or related web content may occur in C4, CC100, Wikipedia mirrors, synthetic outputs, and evaluation
  material.
- Deduplication builds a canonical JSON key from the ordered `user` and `assistant` roles and their content.
  Content is normalized to Unicode NFC and whitespace runs are collapsed for this comparison key only; emitted
  Bengali text remains unchanged. NFC composes canonically equivalent combining sequences where Unicode defines
  a composed form, while combining marks without such a form and explicit ZWJ/ZWNJ code points remain significant
  and are not stripped. Fuzzy similarities are surfaced through audit overlap and repeated-phrase statistics rather
  than automatically removed, avoiding silent loss of legitimate spelling, script, or dialect variation.
- A low training or evaluation loss does not establish factuality, safety, cultural representativeness, or
  human-perceived naturalness.

## Per-Run Checklist

1. Pin or record every dataset revision and generated file hash.
2. Resolve and record each source license and attribution requirement.
3. Run `just audit-dataset` and archive its report.
4. Export a topic/script-stratified human-review sample.
5. Check overlap against available evaluation and comparison corpora.
6. Record exclusions, filtering decisions, and unresolved provenance concerns.
7. Do not publish weights or datasets while material licensing or consent questions remain unresolved.
