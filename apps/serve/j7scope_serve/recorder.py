"""Record a live session into a Trace v1 artifact (the ``--record`` path).

The server buffers each generated token's bucketed read-out during a request;
at request end this module runs the *real* rigor pipeline over the whole session
(``j7scope.rigor``) and writes ``traces/<trace_id>/`` via ``j7scope.trace``.

Rigor is a post-pass — the null pool for a token is every *other* token in the
same session — so the live SSE stream stays lightweight while the recorded trace
carries the full rigor layer baked in.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from j7scope import rigor
from j7scope.trace import write_trace


def _dominant_language(tokens: Sequence[dict]) -> str:
    zh = sum(1 for t in tokens if t.get("token_script") == "zh")
    en = sum(1 for t in tokens if t.get("token_script") == "en")
    return "zh" if zh > en else "en"


def build_lexicon(repo_root: Optional[Path], backend) -> Dict[str, str]:
    """Corpus concepts + (for the mock) its synonym table."""
    pairs = None
    if repo_root is not None:
        try:
            from j7scope.data import load_parallel_pairs
            pairs = load_parallel_pairs(repo_root / "data")
        except Exception:
            pairs = None
    extra = None
    # MockBackend exposes a module-level lexicon; import lazily to avoid a hard dep.
    try:
        from j7scope_serve.backends import mock_concept_lexicon, MockBackend
        if isinstance(backend, MockBackend):
            lex = mock_concept_lexicon()
            extra = [(c, [t]) for t, c in lex.items()]
    except Exception:
        pass
    return rigor.build_lexicon(pairs=pairs, extra=extra)


def record_trace(
    record_dir: Path,
    *,
    backend,
    prompt: str,
    buffered: List[dict],
    lexicon: Dict[str, str],
    trace_id: Optional[str] = None,
    concept: Optional[str] = None,
    language: Optional[str] = None,
    capture_tool: str = "j7scope_serve --record",
    seed: int = 0,
) -> Optional[Path]:
    """Write one Trace v1 from buffered token records. Returns the trace dir.

    ``buffered`` items: {seq, ts_rel, token, token_script, readout}. Returns None
    if the session produced no tokens.
    """
    if not buffered:
        return None

    trace_id = trace_id or ("rec-" + uuid.uuid4().hex[:8])
    language = language or _dominant_language(buffered)

    tokens = rigor.compute_trace_rigor(buffered, lexicon, seed=seed)

    is_demo = bool(getattr(backend, "is_demo", False))
    manifest = {
        "trace_id": trace_id,
        "kind": "single",
        "model": backend.model_name,
        "revision": getattr(backend, "model_revision_resolved", None),
        "layer": backend.layer,
        "language": language,
        "concept": concept,
        "prompt": prompt,
        "jacobian": {
            "corpus_id": (
                "mock-synthetic"
                if is_demo
                else getattr(backend, "jacobian_corpus_id", "generic-v1")
            ),
            "estimator": getattr(
                backend, "jacobian_estimator",
                "synthetic" if is_demo else "unknown",
            ),
            "position": getattr(backend, "jacobian_position", None),
            "n_prompts": getattr(
                backend,
                "jacobian_n_prompts",
                len(getattr(backend, "jacobian_corpus", [])),
            ),
            "n_probes": getattr(backend, "n_probes", None),
            "seed": getattr(backend, "jacobian_seed", None),
            "sha1": getattr(backend, "jacobian_sha1", None),
        },
        "capture": {
            "tool": capture_tool,
            "backend": type(backend).__name__,
            "device": getattr(backend, "device", None),
            "dtype": getattr(backend, "dtype", None),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "is_demo": is_demo,
        "doi": None,
    }

    sharedness = [t["rigor"]["sharedness"]["value"] for t in tokens]
    above_null = sum(
        1 for t in tokens
        if t["rigor"]["cross_lang_overlap"] > t["rigor"]["null"]["p95"]
    )
    metrics = {
        "trace_id": trace_id,
        "n_tokens": len(tokens),
        "language": language,
        "mean_sharedness": round(sum(sharedness) / len(sharedness), 4),
        "tokens_above_null": above_null,
        "frac_above_null": round(above_null / len(tokens), 4),
    }

    return write_trace(record_dir / trace_id, manifest=manifest,
                       tokens=tokens, metrics=metrics)["manifest"].parent
