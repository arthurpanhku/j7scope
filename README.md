# TvinnHugr

> **Is there one workspace, or twin workspaces?**
> Testing cross-lingual generalization of the J-space / global workspace in language models.

Old Norse *tvinnr* ("twin, paired") + *hugr* ("mind, thought"). The name is the research question: when a language model runs in Chinese and in English, is its "global workspace" one shared *hugr*, or two separate *tvinnir hugir*?

**Status: early scaffold.** M1 (correlational analysis) is being set up on Qwen2.5-7B-Instruct. Nothing here is a result yet.

## Background

Anthropic's July 2026 paper [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) introduces the **J-lens**: map an intermediate-layer residual stream through the expected input–output Jacobian into final-layer coordinates, then decode with the model's own unembedding — reading out concepts the model is "inclined to say but hasn't said yet." The paper finds that the readable directions form a sparse subspace (the **J-space**, ~6–10% of activation variance) with far-above-chance broadcast read/write connectivity, functionally resembling the *global workspace* of cognitive science.

The paper's corpus is entirely English web text; the J-lens is fitted and validated within a single language. That leaves a question the paper does not answer:

> If the workspace is the model's genuine "concept level," it should be language-independent — the same concept, whether triggered in Chinese or in English, should land in nearby J-space directions. This assumption has never been tested.

The official companion code, [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens), is explicitly a reference implementation (not maintained, not accepting contributions), and the public open-model demos are single-language visualizers. The cross-lingual direction is open.

## Research question & hypotheses

Does the same concept ("deception", "manipulation", "concession", …) triggered in a Chinese context vs. an English context land in shared or separate J-space directions?

| | Hypothesis | What it would mean |
|---|---|---|
| **H1** | **Language-independent** — zh/en trigger highly overlapping directions | Supports the paper's core claim that the workspace is the model's genuine "thought level" |
| **H2** | **Language-specific** — zh/en directions barely overlap | The current J-space is closer to "surface language habit" than a language-independent concept space; a substantive counterexample to the paper's core claim |
| **H3** | **Graded sharing** — abstract concepts (emotions, intents) share strongly, concrete entity concepts (names, proper nouns) share weakly | The most plausible — and most informative — outcome |

## Method

### Model selection

Requirements: strong in both Chinese and English, dense transformer (no MoE, to keep Jacobians simple), well supported by nnsight / TransformerLens.

- **Primary:** Qwen2.5-7B-Instruct (balanced zh/en ability, mature community support)
- **Replication:** Yi-1.5-9B, InternLM2.5-7B — a conclusion that holds on only one model is weak evidence

### Parallel probe corpus

zh/en prompt pairs with strictly aligned syntax — only the language changes, so "language difference" is not confounded with "phrasing difference." The pairs cover concept families validated in the paper (deception/manipulation, evaluation awareness, multi-hop reasoning), plus abstract-vs-concrete pairs for M3. See [`data/`](data/) and [`data/concept_taxonomy.md`](data/concept_taxonomy.md).

### Analysis stages (cheap first, expensive later)

- **M1 — correlational (cheap, run first).** Fit a single language-agnostic J_l (a property of the model, fitted once), then read out top-k tokens for Chinese and English prompts at aligned positions. Compare the two sets of readout directions with CKA / SVCCA, with top-k readout overlap as an intuitive companion metric.
- **M2 — causal (expensive, only once M1 shows signal).** Activation patching: transplant the residual stream carrying "manipulation" from a Chinese context into an English forward pass. If the J-lens readout follows the *concept* rather than the *source language*, that is causal evidence, stronger than correlation.
- **M3 — graded (optional).** Compare cross-lingual overlap for abstract concepts vs. concrete entities to test H3.

### Metrics

- CKA / SVCCA representation similarity (cross-language vs. same-language different-prompt baselines)
- Top-k readout token overlap
- Fraction of patched runs whose readout follows the concept (causal metric)

## Repo layout

```
tvinnhugr/
├── data/
│   ├── probe_prompts_zh.jsonl     # parallel probe corpus, strictly syntax-aligned
│   ├── probe_prompts_en.jsonl
│   └── concept_taxonomy.md        # abstract/concrete taxonomy for M3
├── tvinnhugr/
│   ├── data.py                    # corpus loading & pairing
│   ├── fitting.py                 # J-lens fitting, adapted for Qwen2.5-style models
│   ├── patching.py                # M2 cross-lingual activation patching
│   ├── metrics.py                 # CKA / SVCCA / top-k overlap
│   └── viz.py                     # bilingual side-by-side readout views
├── notebooks/
│   └── walkthrough_zh_en.ipynb    # end-to-end M1 walkthrough
└── results/                       # generated figures & tables
```

## Quickstart

```bash
pip install -e .
# GPU with ~20 GB VRAM recommended for Qwen2.5-7B-Instruct in bf16
jupyter lab notebooks/walkthrough_zh_en.ipynb
```

```python
from tvinnhugr.data import load_parallel_pairs
from tvinnhugr.fitting import load_model, JLens

model, tok = load_model("Qwen/Qwen2.5-7B-Instruct")
jlens = JLens(model, tok, layer=18)
jlens.estimate_jacobian(corpus_prompts)          # fit once, language-agnostic

pairs = load_parallel_pairs("data")
h_zh = jlens.collect_residual(pairs["deception-01"]["zh"]["text"])
h_en = jlens.collect_residual(pairs["deception-01"]["en"]["text"])
print(jlens.readout(h_zh), jlens.readout(h_en))  # do the readouts agree?
```

## Roadmap

- [ ] M1 on Qwen2.5-7B-Instruct — deception/manipulation family first
- [ ] M1 across all concept families, with same-language baselines
- [ ] M2 activation patching (if M1 shows signal)
- [ ] M3 abstract vs. concrete gradient
- [ ] Cross-model replication (Yi-1.5-9B / InternLM2.5-7B)

## License & attribution

Apache-2.0 (see [LICENSE](LICENSE)). Portions adapted from [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) (Apache-2.0); upstream attribution is preserved in [NOTICE](NOTICE).

**TvinnHugr is an independent third-party research extension. It is not an official Anthropic project, and is not affiliated with or endorsed by Anthropic.**

## 中文简介

独立复现并扩展 Anthropic 的全局工作空间（global workspace）研究，检验语言模型内部"思维空间"（J-space）在中英双语下是否共享同一套概念表示：同一个概念（如"欺骗""操纵""让步"），用中文触发和用英文触发，落在的方向是同一个，还是两个平行世界。方法上先做便宜的相关性分析（CKA / SVCCA / top-k 重叠），有信号后再做因果的跨语言 activation patching。
