"""Trace Schema v1: the citable artifact of a recorded J-Space session.

A trace is one generation session written to ``traces/<trace_id>/``:

    manifest.json   session metadata + provenance + schema version
    tokens.jsonl    one row per generated token: readout + rigor layer
    metrics.json    session-level summary
    align.json      optional: position map to a parallel (other-language) trace

Traces are the object papers cite: permanent, reproducible, deep-linkable. The
rigor layer (shuffled-pair null, same-language baseline, sharedness + CI) is
baked in at capture time by ``j7scope.rigor`` — the frontend only displays it.

See docs/platform-plan.md §3 for the field-level specification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Union

from .artifacts import read_jsonl, write_json, write_jsonl

TRACE_SCHEMA_VERSION = 1

TRACE_FILES = {
    "manifest": "manifest.json",
    "tokens": "tokens.jsonl",
    "metrics": "metrics.json",
    "align": "align.json",   # optional
}

PathLike = Union[str, Path]

# Required keys for schema validation.
_MANIFEST_REQUIRED = (
    "schema_version", "trace_id", "kind", "model", "layer", "language",
    "prompt", "jacobian", "capture", "is_demo",
)
_TOKEN_REQUIRED = ("seq", "token", "readout")
_RIGOR_REQUIRED = ("cross_lang_overlap", "same_lang_baseline", "null", "sharedness")


def trace_paths(trace_dir: PathLike) -> dict:
    trace_dir = Path(trace_dir)
    return {name: trace_dir / fn for name, fn in TRACE_FILES.items()}


def write_trace(
    trace_dir: PathLike,
    *,
    manifest: Mapping,
    tokens: Iterable[Mapping],
    metrics: Mapping,
    align: Optional[Mapping] = None,
) -> dict:
    """Write a complete Trace v1. Stamps schema_version if absent."""
    manifest = dict(manifest)
    manifest.setdefault("schema_version", TRACE_SCHEMA_VERSION)
    paths = trace_paths(trace_dir)
    write_json(paths["manifest"], manifest)
    write_jsonl(paths["tokens"], tokens)
    write_json(paths["metrics"], metrics)
    if align is not None:
        write_json(paths["align"], align)
    return paths


def read_trace(trace_dir: PathLike) -> dict:
    """Read a Trace v1 into memory. ``align`` is None when absent."""
    import json
    paths = trace_paths(trace_dir)
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    tokens = read_jsonl(paths["tokens"])
    metrics = json.loads(Path(paths["metrics"]).read_text(encoding="utf-8"))
    align = None
    if Path(paths["align"]).exists():
        align = json.loads(Path(paths["align"]).read_text(encoding="utf-8"))
    return {"manifest": manifest, "tokens": tokens, "metrics": metrics, "align": align}


def validate_manifest(manifest: Mapping) -> List[str]:
    problems: List[str] = []
    for key in _MANIFEST_REQUIRED:
        if key not in manifest:
            problems.append(f"manifest missing required key: {key}")
    version = manifest.get("schema_version")
    if version is not None and version > TRACE_SCHEMA_VERSION:
        problems.append(
            f"manifest schema_version {version} is newer than supported "
            f"{TRACE_SCHEMA_VERSION}")
    if "is_demo" in manifest and not isinstance(manifest["is_demo"], bool):
        problems.append("manifest.is_demo must be a bool")
    return problems


def validate_tokens(tokens: Sequence[Mapping], *, require_rigor: bool = True) -> List[str]:
    problems: List[str] = []
    for i, tok in enumerate(tokens):
        for key in _TOKEN_REQUIRED:
            if key not in tok:
                problems.append(f"token[{i}] missing required key: {key}")
        readout = tok.get("readout")
        if not isinstance(readout, Mapping) or "zh" not in readout or "en" not in readout:
            problems.append(f"token[{i}].readout must have 'zh' and 'en' lists")
        if require_rigor:
            rigor = tok.get("rigor")
            if not isinstance(rigor, Mapping):
                problems.append(f"token[{i}] missing rigor block")
            else:
                for key in _RIGOR_REQUIRED:
                    if key not in rigor:
                        problems.append(f"token[{i}].rigor missing key: {key}")
    return problems


def validate_trace(trace: Mapping, *, require_rigor: bool = True) -> List[str]:
    """Return a list of problems ([] means valid). Does not raise."""
    problems = validate_manifest(trace.get("manifest", {}))
    problems += validate_tokens(trace.get("tokens", []), require_rigor=require_rigor)
    return problems


def validate_trace_dir(trace_dir: PathLike, *, require_rigor: bool = True) -> List[str]:
    return validate_trace(read_trace(trace_dir), require_rigor=require_rigor)
