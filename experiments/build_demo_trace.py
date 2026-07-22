"""Build deterministic demo traces for the Replay viewer (platform P1).

This does not run a model. It synthesises read-outs from the probe corpus and
runs them through the *real* rigor pipeline (``j7scope.rigor``), so the demo's
sharedness / null numbers are honestly computed on clearly-marked synthetic data
(``is_demo: true``) — the same philosophy as ``build_demo_run.py``.

The narrative is designed to make the layered-rigor display meaningful:

* abstract-concept tokens  -> zh and en columns agree  -> sharedness above null
* filler tokens            -> no concept               -> sharedness ~0, in null band
* concrete-entity tokens   -> partial agreement (H3)    -> intermediate

It emits one English single trace, plus a zh/en parallel pair with align.json.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from j7scope import rigor
from j7scope.artifacts import write_json
from j7scope.data import load_parallel_pairs
from j7scope.trace import write_trace

# concept -> (zh words, en words). Concrete entity kept partly language-bound.
CONCEPTS = {
    "deception":    (["欺骗", "撒谎", "不诚实"], ["deception", "lying", "dishonesty"]),
    "manipulation": (["操纵", "摆布", "利用"],   ["manipulation", "coercion", "exploiting"]),
    "concession":   (["让步", "妥协", "退让"],   ["concession", "compromise", "yielding"]),
    "emotion":      (["情绪", "感受", "焦虑"],   ["emotion", "feeling", "anxiety"]),
    "entity":       (["巴黎", "法国"],           ["Paris", "France"]),
}
FILLER_ZH = ["的", "一种", "行为", "这", "属于"]
FILLER_EN = ["a", "kind", "of", "this", "is"]

# A session narrative: (surface_token, concept_or_None). Abstract concepts are
# interspersed with fillers; the concrete entity shows weaker cross-lingual tie.
NARRATIVE = [
    ("This", None), ("is", None), ("clearly", None),
    ("deception", "deception"), (",", None), ("a", None),
    ("manipulation", "manipulation"), ("of", None), ("trust", None), (".", None),
    ("She", None), ("felt", None), ("intense", None),
    ("emotion", "emotion"), (",", None), ("then", None), ("made", None), ("a", None),
    ("concession", "concession"), (".", None),
    ("They", None), ("met", None), ("in", None),
    ("Paris", "entity"), (".", None),
]

# Position-aligned Chinese narrative: same concept at the same index, different
# surface language — so Compare shows two genuinely different-language streams
# that (should) land on the same concepts.
ZH_NARRATIVE = [
    ("这", None), ("显然", None), ("是", None),
    ("欺骗", "deception"), ("，", None), ("一种", None),
    ("操纵", "manipulation"), ("了", None), ("信任", None), ("。", None),
    ("她", None), ("感到", None), ("强烈的", None),
    ("情绪", "emotion"), ("，", None), ("随后", None), ("做出", None), ("了", None),
    ("让步", "concession"), ("。", None),
    ("他们", None), ("相遇", None), ("于", None),
    ("巴黎", "entity"), ("。", None),
]


def _readout(concept, lang_leaning: str, entity_noise: bool) -> dict:
    """Build a bucketed readout {zh:[...], en:[...]} for one token.

    Concept tokens put the concept's words at the top of both columns. For the
    concrete entity we add extra distinct filler so the concept overlap is only
    partial (demonstrating the graded H3 result).
    """
    if concept is None:
        zh = list(FILLER_ZH)
        en = list(FILLER_EN)
    elif entity_noise:
        # Concrete entity: shared on the entity itself, but each language also
        # leans on a *different* concept, so the concept overlap is only partial
        # (the graded H3 result: concrete entities are less language-shared).
        zw, ew = CONCEPTS[concept]
        zh = list(zw) + [CONCEPTS["emotion"][0][0]]        # entity + 情绪
        en = list(ew) + [CONCEPTS["deception"][1][0]]      # entity + deception
    else:
        zw, ew = CONCEPTS[concept]
        zh = list(zw)
        en = list(ew)

    def rows(words):
        return [{"token": w, "score": round(11.5 - i * 0.7, 3), "rank": i}
                for i, w in enumerate(words)]

    return {"zh": rows(zh), "en": rows(en), "other": []}


def _tag_readout_concepts(readout: dict, lexicon) -> dict:
    """Annotate each read-out row with its concept so the client can highlight
    shared concepts without needing the lexicon."""
    for bucket in ("zh", "en", "other"):
        for row in readout.get(bucket, []):
            row["concept"] = lexicon.get(row["token"])
    return readout


def build_trace(narrative, language: str, lexicon, *, trace_id: str,
                parallel_group=None, seed: int = 0):
    raw_tokens = []
    for i, (surface, concept) in enumerate(narrative):
        readout = _tag_readout_concepts(
            _readout(concept, language, entity_noise=(concept == "entity")), lexicon)
        raw_tokens.append({
            "seq": i,
            "ts_rel": round(i * 0.12, 3),
            "token": surface,
            "token_script": "zh" if language == "zh" else "en",
            "readout": readout,
            "concept": concept,
        })
    tokens = rigor.compute_trace_rigor(raw_tokens, lexicon, seed=seed)

    sharedness = [t["rigor"]["sharedness"]["value"] for t in tokens]
    above = sum(1 for t in tokens
                if t["rigor"]["cross_lang_overlap"] > t["rigor"]["null"]["p95"])
    manifest = {
        "trace_id": trace_id,
        "kind": "parallel_member" if parallel_group else "single",
        "model": "demo-synthetic",
        "layer": 18,
        "language": language,
        "concept": None,
        "prompt": "(synthetic demo narrative)",
        "jacobian": {"corpus_id": "demo-synthetic", "sha1": None},
        "capture": {"tool": "build_demo_trace.py",
                    "created_at": datetime.now(timezone.utc).isoformat()},
        "is_demo": True,
        "doi": None,
        "parallel_group": parallel_group,
        "label": f"Demo · {trace_id}",
    }
    metrics = {
        "trace_id": trace_id,
        "n_tokens": len(tokens),
        "language": language,
        "mean_sharedness": round(sum(sharedness) / len(sharedness), 4),
        "tokens_above_null": above,
        "frac_above_null": round(above / len(tokens), 4),
    }
    return manifest, tokens, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "traces")
    args = parser.parse_args()

    pairs = load_parallel_pairs(args.data_dir)
    extra = [(concept, zw + ew) for concept, (zw, ew) in CONCEPTS.items()]
    lexicon = rigor.build_lexicon(pairs=pairs, extra=extra)

    index = []

    # 1) single English narrative (default replay)
    m, t, mt = build_trace(NARRATIVE, "en", lexicon, trace_id="demo-narrative-en")
    write_trace(args.out / m["trace_id"], manifest=m, tokens=t, metrics=mt)
    index.append({"trace_id": m["trace_id"], "label": m["label"],
                  "language": "en", "is_demo": True, "n_tokens": mt["n_tokens"],
                  "parallel_group": None})

    # 2) zh/en parallel pair (position-aligned, same narrative structure)
    group = "demo-parallel"
    m_en, t_en, mt_en = build_trace(NARRATIVE, "en", lexicon,
                                    trace_id=f"{group}-en", parallel_group=group)
    m_zh, t_zh, mt_zh = build_trace(ZH_NARRATIVE, "zh", lexicon,
                                    trace_id=f"{group}-zh", parallel_group=group)
    position_map = [[i, i] for i in range(len(NARRATIVE))]  # identical structure
    # Cross-trace rigor: A = en session, B = zh session, at aligned positions.
    pair_rigor = rigor.compute_pair_rigor(t_en, t_zh, position_map, lexicon)
    align = {
        "parallel_group": group,
        "members": {"en": m_en["trace_id"], "zh": m_zh["trace_id"]},
        "position_map": position_map,
        "pair_rigor": pair_rigor,
    }
    write_trace(args.out / m_en["trace_id"], manifest=m_en, tokens=t_en,
                metrics=mt_en, align=align)
    write_trace(args.out / m_zh["trace_id"], manifest=m_zh, tokens=t_zh,
                metrics=mt_zh, align=align)
    for m, mt, lang in ((m_en, mt_en, "en"), (m_zh, mt_zh, "zh")):
        index.append({"trace_id": m["trace_id"], "label": m["label"],
                      "language": lang, "is_demo": True,
                      "n_tokens": mt["n_tokens"], "parallel_group": group})

    write_json(args.out / "index.json", {"schema_version": 1, "traces": index})
    print(f"Wrote {len(index)} demo traces to {args.out}")
    for row in index:
        print(f"  - {row['trace_id']}  ({row['language']}, {row['n_tokens']} tokens)")


if __name__ == "__main__":
    main()
