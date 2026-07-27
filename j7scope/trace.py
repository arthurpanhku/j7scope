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

import json
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


def trace_index_row(trace_dir: PathLike) -> dict:
    """Build one gallery-index row from a trace directory."""
    trace = read_trace(trace_dir)
    manifest, tokens = trace["manifest"], trace["tokens"]
    return {
        "trace_id": manifest["trace_id"],
        "label": manifest.get("label", manifest["trace_id"]),
        "model": manifest.get("model"),
        "layer": manifest.get("layer"),
        "language": manifest.get("language", "?"),
        "concept": manifest.get("concept"),
        "is_demo": bool(manifest.get("is_demo", False)),
        "preview": bool(manifest.get("preview", False)),
        "doi": manifest.get("doi"),
        "n_tokens": len(tokens),
        "parallel_group": manifest.get("parallel_group"),
    }


def rebuild_trace_index(trace_root: PathLike) -> Path:
    """Rebuild ``index.json`` deterministically from all child traces."""
    trace_root = Path(trace_root)
    rows = [
        trace_index_row(trace_dir)
        for trace_dir in sorted(trace_root.iterdir())
        if trace_dir.is_dir() and (trace_dir / TRACE_FILES["manifest"]).exists()
    ]
    return write_json(
        trace_root / "index.json",
        {"schema_version": TRACE_SCHEMA_VERSION, "traces": rows},
    )


def validate_trace_gallery(trace_root: PathLike) -> List[str]:
    """Validate gallery indexing, trace schema, rigor, and provenance."""
    from .rigor import SHAREDNESS_DEFINITION

    trace_root = Path(trace_root)
    index_path = trace_root / "index.json"
    if not index_path.exists():
        return [f"gallery missing index: {index_path}"]
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"gallery index is unreadable: {exc}"]

    problems: List[str] = []
    if index.get("schema_version") != TRACE_SCHEMA_VERSION:
        problems.append(
            f"gallery index schema_version must be {TRACE_SCHEMA_VERSION}"
        )
    rows = index.get("traces")
    if not isinstance(rows, list):
        return problems + ["gallery index.traces must be a list"]

    indexed_ids = [
        row.get("trace_id") for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("trace_id"), str)
    ]
    if len(indexed_ids) != len(rows):
        problems.append("every gallery index row must be an object with trace_id")
    duplicates = sorted(
        trace_id for trace_id in set(indexed_ids)
        if indexed_ids.count(trace_id) > 1
    )
    if duplicates:
        problems.append(f"gallery index has duplicate trace_ids: {duplicates}")

    discovered_ids = sorted(
        path.name for path in trace_root.iterdir()
        if path.is_dir() and (path / TRACE_FILES["manifest"]).exists()
    )
    missing = sorted(set(discovered_ids) - set(indexed_ids))
    extra = sorted(set(indexed_ids) - set(discovered_ids))
    if missing:
        problems.append(f"traces missing from gallery index: {missing}")
    if extra:
        problems.append(f"gallery index references missing traces: {extra}")

    rows_by_id = {
        row["trace_id"]: row for row in rows
        if isinstance(row, Mapping) and row.get("trace_id") in discovered_ids
    }
    for trace_id in discovered_ids:
        trace_dir = trace_root / trace_id
        try:
            trace = read_trace(trace_dir)
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{trace_id}: unreadable trace: {exc}")
            continue
        problems.extend(
            f"{trace_id}: {problem}" for problem in validate_trace(trace)
        )
        manifest = trace["manifest"]
        tokens = trace["tokens"]
        metrics = trace["metrics"]
        if manifest.get("trace_id") != trace_id:
            problems.append(
                f"{trace_id}: manifest.trace_id does not match directory name"
            )
        if metrics.get("trace_id") != trace_id:
            problems.append(f"{trace_id}: metrics.trace_id does not match")
        if metrics.get("n_tokens") != len(tokens):
            problems.append(f"{trace_id}: metrics.n_tokens does not match tokens.jsonl")
        seqs = [token.get("seq") for token in tokens]
        if seqs != list(range(len(tokens))):
            problems.append(f"{trace_id}: token seq values must be contiguous from 0")

        row = rows_by_id.get(trace_id, {})
        expected_row = trace_index_row(trace_dir)
        for key in ("language", "is_demo", "n_tokens", "parallel_group"):
            if row.get(key) != expected_row[key]:
                problems.append(f"{trace_id}: index.{key} does not match trace")

        for token_index, token in enumerate(tokens):
            sharedness = token.get("rigor", {}).get("sharedness", {})
            if sharedness.get("definition") != SHAREDNESS_DEFINITION:
                problems.append(
                    f"{trace_id}: token[{token_index}] sharedness definition drifted"
                )

        if not manifest.get("is_demo", False):
            jacobian = manifest.get("jacobian", {})
            capture = manifest.get("capture", {})
            for key in ("revision",):
                if not manifest.get(key):
                    problems.append(f"{trace_id}: real trace missing manifest.{key}")
            for key in ("estimator", "sha1"):
                if not jacobian.get(key):
                    problems.append(f"{trace_id}: real trace missing jacobian.{key}")
            for key in ("device", "dtype"):
                if not capture.get(key):
                    problems.append(f"{trace_id}: real trace missing capture.{key}")

        align = trace.get("align")
        if align is not None:
            members = set(align.get("members", {}).values())
            unknown_members = sorted(members - set(discovered_ids))
            if unknown_members:
                problems.append(
                    f"{trace_id}: align references missing members {unknown_members}"
                )
            for pair_index, pair in enumerate(align.get("position_map", [])):
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or not all(isinstance(value, int) and value >= 0 for value in pair)
                ):
                    problems.append(
                        f"{trace_id}: align.position_map[{pair_index}] is invalid"
                    )
    return problems
