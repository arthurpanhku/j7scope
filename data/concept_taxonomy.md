# Concept taxonomy

Classification of the probe corpus, primarily serving **M3** (graded-sharing hypothesis H3: abstract concepts share cross-lingual J-space directions more than concrete entity concepts do).

## Alignment protocol

- Each prompt exists as a zh/en pair sharing the same `id` across `probe_prompts_zh.jsonl` and `probe_prompts_en.jsonl`.
- Pairs are **strictly syntax-aligned**: same clause structure, same information order, same final-position cloze — only the language changes. This keeps "language difference" from being confounded with "phrasing difference."
- Every prompt ends right before the concept would be verbalized, so the J-lens readout at the final token position should surface the concept before the model says it.
- `expected` lists acceptable surface tokens per language, used by `j7scope.metrics.topk_overlap` scoring. Cross-lingual overlap is computed at the *concept* level (via the shared `concept` label / expected-token translation table), never by raw string match.

## Categories

| category | concept family | abstractness | count | paper-validated family |
|---|---|---|---|---|
| `deception` | deception / lying | abstract | 4 | deception & manipulation directions |
| `manipulation` | manipulation / coercion | abstract | 4 | deception & manipulation directions |
| `concession` | concession / compromise | abstract | 2 | deception & manipulation directions (negotiation) |
| `eval_awareness` | "this is a test/evaluation" | abstract | 4 | evaluation-awareness directions |
| `multihop` | 2-hop factual composition | concrete (target) | 4 | multi-hop reasoning directions |
| `emotion` | joy, anger, fear, envy, guilt, gratitude | abstract | 6 | M3 abstract pole |
| `entity` | cities, persons, rivers, companies, planets | concrete | 6 | M3 concrete pole |

## Abstractness field

The `abstractness` label follows the **readout target**, not the prompt topic:

- **abstract** — the position where the readout happens is about to verbalize an emotion, intent, or social-cognitive concept (deception, manipulation, concession, evaluation-awareness, emotions). H3 predicts these live in shared cross-lingual directions.
- **concrete** — the readout target is a specific named entity (Tokyo, Einstein, Thames…). H3 predicts these are more language-bound (tokenization/orthography-tied), hence lower cross-lingual overlap.

`multihop` targets are concrete entities but additionally require compositional reasoning; when analyzing M3, report it separately rather than pooling it into either pole.

## Known limitations / TODO

- 30 pairs is a starter set for M1 plumbing; scale each category to ≥20 pairs before drawing quantitative conclusions.
- zh cloze positions sometimes end with a classifier ("一种/一次/一场") that constrains the next-token distribution more than the English article does; when expanding the corpus, add classifier-free variants to control for this.
- Add a per-concept zh↔en expected-token translation table once the concept inventory stabilizes (currently implicit in `expected`).
