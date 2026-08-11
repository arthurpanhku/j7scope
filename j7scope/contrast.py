"""Research-grade paired evidence for the J-Space Contrast Lab.

The browser is intentionally a renderer.  This module turns two Trace v1
sessions into a precomputed contrast record containing a paired permutation
null, a same-language ceiling, a bootstrap interval, concept-rank trajectories,
an answer-identity confound check, and exact provenance.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .rigor import (
    SHAREDNESS_DEFINITION,
    compute_pair_rigor,
    concepts_of,
    estimate_same_lang_baseline,
    overlap_coef,
)

CONTRAST_SCHEMA_VERSION = 1
PAIR_SHAREDNESS_DEFINITION = (
    "(paired_observed_mean - permutation_null.mean) / "
    "(same_lang_ceiling - permutation_null.mean)"
)
_EPS = 1e-9


def _row_concept(row: Mapping, lexicon: Mapping[str, str]) -> Optional[str]:
    concept = row.get("concept")
    if isinstance(concept, str) and concept:
        return concept
    token = row.get("token")
    if isinstance(token, str):
        return lexicon.get(token)
    return None


def token_concepts(token: Mapping, lexicon: Mapping[str, str]) -> List[str]:
    """Ordered concepts in a token's complete bilingual readout.

    Captured traces can either bake ``row.concept`` into each readout or rely on
    the corpus lexicon.  Supporting both makes imported JSONL traces portable.
    """
    out: List[str] = []
    seen = set()
    readout = token.get("readout", {})
    for bucket in ("zh", "en", "other"):
        for row in readout.get(bucket, []):
            concept = _row_concept(row, lexicon)
            if concept and concept not in seen:
                seen.add(concept)
                out.append(concept)
    return out


def _rank_for_concept(
    token: Mapping, concept: str, lexicon: Mapping[str, str]
) -> Optional[int]:
    ranks = []
    for bucket in ("zh", "en", "other"):
        for fallback_rank, row in enumerate(token.get("readout", {}).get(bucket, [])):
            if _row_concept(row, lexicon) == concept:
                ranks.append(int(row.get("rank", fallback_rank)) + 1)
    return min(ranks) if ranks else None


def concept_rank_trajectories(
    tokens_a: Sequence[Mapping],
    tokens_b: Sequence[Mapping],
    position_map: Sequence[Sequence[int]],
    lexicon: Mapping[str, str],
) -> List[dict]:
    concepts = sorted({
        concept
        for token in (*tokens_a, *tokens_b)
        for concept in token_concepts(token, lexicon)
    })
    rows = []
    for concept in concepts:
        rank_a = [_rank_for_concept(tokens_a[ia], concept, lexicon)
                  for ia, _ in position_map]
        rank_b = [_rank_for_concept(tokens_b[ib], concept, lexicon)
                  for _, ib in position_map]
        coverage = sum(rank is not None for rank in rank_a + rank_b)
        if coverage:
            rows.append({
                "concept": concept,
                "rank_a": rank_a,
                "rank_b": rank_b,
                "coverage": round(coverage / max(1, 2 * len(position_map)), 4),
            })
    return sorted(rows, key=lambda row: (-row["coverage"], row["concept"]))


def _permutation_null(
    concepts_a: Sequence[Sequence[str]],
    concepts_b: Sequence[Sequence[str]],
    *,
    n_permutations: int,
    seed: int,
) -> tuple[float, dict]:
    observed_values = [overlap_coef(a, b) for a, b in zip(concepts_a, concepts_b)]
    observed = float(np.mean(observed_values)) if observed_values else 0.0
    if len(concepts_a) < 2 or n_permutations < 1:
        return observed, {
            "metric": "paired_mean_concept_overlap",
            "mean": 0.0,
            "p05": 0.0,
            "p95": 0.0,
            "n": 0,
            "seed": seed,
        }

    rng = np.random.default_rng(seed)
    values = np.empty(n_permutations, dtype=np.float64)
    base = np.arange(len(concepts_b))
    for i in range(n_permutations):
        permuted = rng.permutation(base)
        values[i] = np.mean([
            overlap_coef(concepts_a[j], concepts_b[int(permuted[j])])
            for j in range(len(concepts_a))
        ])
    return observed, {
        "metric": "paired_mean_concept_overlap",
        "mean": round(float(values.mean()), 4),
        "p05": round(float(np.percentile(values, 5)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "n": int(n_permutations),
        "seed": int(seed),
    }


def _paired_bootstrap_ci(
    overlaps: Sequence[float],
    *,
    null_mean: float,
    ceiling: float,
    n_boot: int,
    seed: int,
) -> List[float]:
    if not overlaps or n_boot < 1:
        return [0.0, 0.0]
    values = np.asarray(overlaps, dtype=np.float64)
    rng = np.random.default_rng(seed)
    denominator = max(_EPS, ceiling - null_mean)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        picked = values[rng.integers(0, len(values), len(values))]
        samples[i] = (float(picked.mean()) - null_mean) / denominator
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def detect_answer_identity_confound(
    manifest_a: Mapping, manifest_b: Mapping
) -> dict:
    """Classify whether a correctness contrast controls answer identity.

    This is deliberately conservative: absent structured ``correct`` and
    ``answer_identity`` metadata, the tool reports ``not_assessable`` instead
    of inferring correctness from free text.
    """
    correct_a, correct_b = manifest_a.get("correct"), manifest_b.get("correct")
    answer_a, answer_b = manifest_a.get("answer_identity"), manifest_b.get("answer_identity")
    if not isinstance(correct_a, bool) or not isinstance(correct_b, bool):
        return {
            "status": "not_assessable",
            "controlled": None,
            "reason": "manifests need boolean correct and string answer_identity fields",
        }
    if correct_a == correct_b:
        return {
            "status": "not_applicable",
            "controlled": None,
            "reason": "the pair does not contrast correctness",
        }
    if not isinstance(answer_a, str) or not isinstance(answer_b, str):
        return {
            "status": "not_assessable",
            "controlled": None,
            "reason": "answer_identity is missing from one or both manifests",
        }
    controlled = answer_a == answer_b
    return {
        "status": "controlled" if controlled else "confounded",
        "controlled": controlled,
        "answer_identity_a": answer_a,
        "answer_identity_b": answer_b,
        "reason": (
            "correctness differs while answer identity is held constant"
            if controlled else
            "correctness and answer identity both change in this pair"
        ),
    }


def _trace_provenance(trace: Mapping) -> dict:
    manifest = trace["manifest"]
    jacobian = manifest.get("jacobian", {})
    capture = manifest.get("capture", {})
    return {
        "trace_id": manifest.get("trace_id"),
        "model": manifest.get("model"),
        "model_revision": manifest.get("revision"),
        "layer": manifest.get("layer"),
        "language": manifest.get("language"),
        "lens_sha1": jacobian.get("sha1"),
        "corpus_id": jacobian.get("corpus_id"),
        "estimator": jacobian.get("estimator"),
        "capture_tool": capture.get("tool"),
        "created_at": capture.get("created_at"),
        "is_demo": bool(manifest.get("is_demo", False)),
    }


def build_contrast(
    trace_a: Mapping,
    trace_b: Mapping,
    lexicon: Mapping[str, str],
    *,
    position_map: Optional[Sequence[Sequence[int]]] = None,
    mode: str = "cross_language",
    n_permutations: int = 1000,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Build a complete, serializable Contrast Lab evidence record."""
    tokens_a, tokens_b = trace_a["tokens"], trace_b["tokens"]
    if position_map is None:
        position_map = [[i, i] for i in range(min(len(tokens_a), len(tokens_b)))]
    position_map = [[int(ia), int(ib)] for ia, ib in position_map]
    if not position_map:
        raise ValueError("position_map must contain at least one aligned position")
    if any(ia < 0 or ia >= len(tokens_a) or ib < 0 or ib >= len(tokens_b)
           for ia, ib in position_map):
        raise IndexError("position_map references a token outside one of the traces")

    aligned_a = [token_concepts(tokens_a[ia], lexicon) for ia, _ in position_map]
    aligned_b = [token_concepts(tokens_b[ib], lexicon) for _, ib in position_map]
    # Empty/empty filler positions contain no hypothesis-bearing observation.
    # Including them would dilute both observed overlap and the permutation null
    # according to narrative length rather than representation quality.
    informative = [
        i for i, (concepts_a, concepts_b) in enumerate(zip(aligned_a, aligned_b))
        if concepts_a or concepts_b
    ]
    evidence_a = [aligned_a[i] for i in informative]
    evidence_b = [aligned_b[i] for i in informative]
    if not evidence_a:
        raise ValueError("the aligned pair contains no concepts in either trace")
    observed, null = _permutation_null(
        evidence_a, evidence_b, n_permutations=n_permutations, seed=seed
    )
    ceiling = float(np.mean([
        estimate_same_lang_baseline(evidence_a),
        estimate_same_lang_baseline(evidence_b),
    ]))
    denominator = max(_EPS, ceiling - null["mean"])
    sharedness = (observed - null["mean"]) / denominator
    overlaps = [overlap_coef(a, b) for a, b in zip(evidence_a, evidence_b)]
    ci = _paired_bootstrap_ci(
        overlaps,
        null_mean=null["mean"],
        ceiling=ceiling,
        n_boot=n_boot,
        seed=seed + 17,
    )
    pair_rigor = compute_pair_rigor(
        tokens_a, tokens_b, position_map, dict(lexicon), seed=seed
    )

    return {
        "schema_version": CONTRAST_SCHEMA_VERSION,
        "mode": mode,
        "trace_a": trace_a["manifest"].get("trace_id"),
        "trace_b": trace_b["manifest"].get("trace_id"),
        "position_map": position_map,
        "pair_rigor": pair_rigor,
        "calibration": {
            "n_aligned_positions": len(position_map),
            "n_informative_positions": len(informative),
            "informative_positions": informative,
            "observed": round(observed, 4),
            "null": null,
            "same_lang_ceiling": round(ceiling, 4),
            "sharedness": {
                "value": round(float(sharedness), 4),
                "ci95": ci,
                "definition": PAIR_SHAREDNESS_DEFINITION,
            },
            "token_sharedness_definition": SHAREDNESS_DEFINITION,
        },
        "concept_trajectories": concept_rank_trajectories(
            tokens_a, tokens_b, position_map, lexicon
        ),
        "answer_identity_check": detect_answer_identity_confound(
            trace_a["manifest"], trace_b["manifest"]
        ),
        "provenance": {
            "trace_a": _trace_provenance(trace_a),
            "trace_b": _trace_provenance(trace_b),
            "analysis": {
                "tool": "j7scope.contrast.build_contrast",
                "schema_version": CONTRAST_SCHEMA_VERSION,
                "n_permutations": int(n_permutations),
                "n_boot": int(n_boot),
                "seed": int(seed),
            },
        },
    }
