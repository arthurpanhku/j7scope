"""The rigor layer: per-token cross-lingual sharedness against a null baseline.

A single J-lens read-out at one generated token contains both zh and en tokens
(Qwen's vocab is bilingual). The question this module answers, per token, is:
**do the zh column and the en column point at the same concept, more than
chance?** That is the twin-workspace question (README §2) at token resolution.

Pipeline for one token:
1. Map each read-out token to a concept via a lexicon (raw "欺骗" and
   "deception" never string-match, so we compare in concept space).
2. `cross_lang_overlap` = overlap coefficient between the zh-column concepts and
   the en-column concepts at this token.
3. `null` = the same overlap but pairing this token's zh concepts against *other*
   tokens' en concepts (shuffled pairing) — the chance level (README §3.4).
4. `same_lang_baseline` = the achievable ceiling, estimated within-trace.
5. `sharedness` = (obs - null_mean) / (same_lang - null_mean), with a bootstrap CI.

This is the *single* implementation of the rigor numbers; the recorder bakes them
into traces and the viewer only displays them — nothing recomputes sharedness in
the frontend.

Provisional-for-P1 note: `same_lang_baseline` and the CI are estimated from a
single trace here. With a real multi-prompt corpus (P3) they should be replaced
by proper cross-prompt resampling; the function signatures are chosen so that
swap is local to this file.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# Fillers the mock (and real read-outs) emit that carry no concept; mapping them
# to None keeps them from creating spurious overlap.
_EPS = 1e-9


# --- concept lexicon -------------------------------------------------------

def build_lexicon(
    pairs: Optional[dict] = None,
    extra: Optional[Iterable[Sequence]] = None,
) -> Dict[str, str]:
    """Build a {surface_token: concept} map.

    ``pairs`` is the output of ``j7scope.data.load_parallel_pairs`` (optional):
    every ``expected`` token maps to its ``concept``. ``extra`` is an iterable of
    ``(concept, [tokens...])`` for tokens not in the corpus (e.g. the mock's
    synonym lists). Later sources win on conflict.
    """
    lex: Dict[str, str] = {}
    if pairs:
        for record_pair in pairs.values():
            for lang in ("zh", "en"):
                rec = record_pair[lang]
                for tok in rec.get("expected", []):
                    lex[tok] = rec["concept"]
    if extra:
        for concept, tokens in extra:
            for tok in tokens:
                lex[tok] = concept
    return lex


def concepts_of(tokens: Iterable[str], lexicon: Dict[str, str]) -> List[str]:
    """Ordered, de-duplicated concepts for a list of read-out tokens.

    Tokens absent from the lexicon (fillers, punctuation) are dropped.
    """
    out: List[str] = []
    seen = set()
    for tok in tokens:
        c = lexicon.get(tok)
        if c is not None and c not in seen:
            seen.add(c)
            out.append(c)
    return out


# --- overlap + null --------------------------------------------------------

def overlap_coef(a: Sequence[str], b: Sequence[str]) -> float:
    """|A ∩ B| / min(|A|, |B|) over concept sets. 0 if either is empty."""
    sa, sb = set(a), set(b)
    m = min(len(sa), len(sb))
    if m == 0:
        return 0.0
    return len(sa & sb) / m


def _percentile_summary(values: np.ndarray, *, metric: str, n: int) -> dict:
    if values.size == 0:
        return {"metric": metric, "mean": 0.0, "p05": 0.0, "p95": 0.0, "n": 0}
    return {
        "metric": metric,
        "mean": round(float(values.mean()), 4),
        "p05": round(float(np.percentile(values, 5)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "n": int(n),
    }


def null_distribution(
    zh_target: Sequence[str],
    en_pool: Sequence[Sequence[str]],
    *,
    metric: str = "concept_overlap",
) -> dict:
    """Shuffled-pairing null: this token's zh concepts vs every *other* token's
    en concepts. Returns {metric, mean, p05, p95, n}."""
    vals = np.array([overlap_coef(zh_target, en_other) for en_other in en_pool],
                    dtype=np.float64)
    return _percentile_summary(vals, metric=metric, n=vals.size)


def _bootstrap_ci(
    zh: Sequence[str], en: Sequence[str], *, null_mean: float, same_lang: float,
    n_boot: int, seed: int,
) -> List[float]:
    """95% CI on sharedness by resampling the two concept lists with replacement.

    Small-sample and provisional (top-k membership is the resampling unit); with
    a real corpus this becomes cross-prompt resampling.
    """
    if not zh or not en:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    za, ea = np.array(zh, dtype=object), np.array(en, dtype=object)
    denom = max(_EPS, same_lang - null_mean)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        zb = za[rng.integers(0, len(za), len(za))]
        eb = ea[rng.integers(0, len(ea), len(ea))]
        obs = overlap_coef(zb.tolist(), eb.tolist())
        samples[i] = (obs - null_mean) / denom
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


SHAREDNESS_DEFINITION = "(cross_lang_overlap - null.mean) / (same_lang_baseline - null.mean)"


def token_rigor(
    zh_concepts: Sequence[str],
    en_concepts: Sequence[str],
    en_pool: Sequence[Sequence[str]],
    *,
    same_lang_baseline: float,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Compute the full ``rigor`` block for one token (schema §3.2)."""
    obs = overlap_coef(zh_concepts, en_concepts)
    null = null_distribution(zh_concepts, en_pool)
    denom = max(_EPS, same_lang_baseline - null["mean"])
    value = (obs - null["mean"]) / denom
    ci = _bootstrap_ci(zh_concepts, en_concepts, null_mean=null["mean"],
                       same_lang=same_lang_baseline, n_boot=n_boot, seed=seed)
    return {
        "cross_lang_overlap": round(float(obs), 4),
        "same_lang_baseline": round(float(same_lang_baseline), 4),
        "null": null,
        "sharedness": {
            "value": round(float(value), 4),
            "ci95": ci,
            "definition": SHAREDNESS_DEFINITION,
        },
    }


def estimate_same_lang_baseline(
    zh_concept_lists: Sequence[Sequence[str]],
) -> float:
    """Within-trace ceiling: mean overlap between zh read-outs of *different*
    tokens that share a dominant concept.

    This approximates "how well does one language agree with itself across
    prompts" using tokens as pseudo-prompts. Falls back to 1.0 when there are
    too few comparable tokens. Provisional (see module docstring).
    """
    # Group token indices by their first (dominant) concept.
    by_concept: Dict[str, List[int]] = {}
    for i, cs in enumerate(zh_concept_lists):
        if cs:
            by_concept.setdefault(cs[0], []).append(i)
    overlaps: List[float] = []
    for idxs in by_concept.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                overlaps.append(overlap_coef(zh_concept_lists[idxs[a]],
                                             zh_concept_lists[idxs[b]]))
    if not overlaps:
        return 1.0
    return max(0.05, float(np.mean(overlaps)))


def compute_trace_rigor(
    tokens: Sequence[dict],
    lexicon: Dict[str, str],
    *,
    seed: int = 0,
) -> List[dict]:
    """Attach a ``rigor`` block to every token record in a trace.

    ``tokens`` are token records each carrying ``readout`` with ``zh`` / ``en``
    lists of ``{"token", ...}``. Returns the same records with ``rigor`` added.
    The null pool for a token is every *other* token's en concepts; the
    same-language baseline is estimated once per trace.
    """
    zh_concepts = [concepts_of([t["token"] for t in tok["readout"].get("zh", [])], lexicon)
                   for tok in tokens]
    en_concepts = [concepts_of([t["token"] for t in tok["readout"].get("en", [])], lexicon)
                   for tok in tokens]
    same_lang = estimate_same_lang_baseline(zh_concepts)

    out: List[dict] = []
    for i, tok in enumerate(tokens):
        en_pool = [en_concepts[j] for j in range(len(tokens)) if j != i]
        rigor = token_rigor(zh_concepts[i], en_concepts[i], en_pool,
                            same_lang_baseline=same_lang, seed=seed + i)
        rec = dict(tok)
        rec["rigor"] = rigor
        out.append(rec)
    return out
