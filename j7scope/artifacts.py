"""Stable artifact schema for J7Scope experiment runs.

The frontend should never depend on notebook state, Python objects, or raw
tensor files. Experiment scripts write a small set of JSON/JSONL files through
this module; the J-Space Explorer reads those artifacts as its only contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union


RUN_FILES = {
    "manifest": "manifest.json",
    "readouts": "readouts.jsonl",
    "patches": "patches.jsonl",
    "layer_scan": "layer_scan.json",
    "projections": "projections.json",
    "metrics": "metrics.json",
}


PathLike = Union[str, Path]


def write_json(path: PathLike, data: Union[Mapping, Sequence], *, indent: int = 2) -> Path:
    """Write UTF-8 JSON with deterministic formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_jsonl(path: PathLike, rows: Iterable[Mapping]) -> Path:
    """Write newline-delimited JSON records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    return path


def read_jsonl(path: PathLike) -> list[dict]:
    """Read a JSONL artifact into memory for validation or small-run tooling."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_paths(run_dir: PathLike) -> dict[str, Path]:
    """Return canonical artifact paths for a run directory."""
    run_dir = Path(run_dir)
    return {name: run_dir / filename for name, filename in RUN_FILES.items()}


def write_run(
    run_dir: PathLike,
    *,
    manifest: Mapping,
    readouts: Iterable[Mapping],
    patches: Iterable[Mapping],
    layer_scan: Mapping,
    projections: Mapping,
    metrics: Mapping,
) -> dict[str, Path]:
    """Write a complete frontend-readable experiment run."""
    paths = run_paths(run_dir)
    write_json(paths["manifest"], manifest)
    write_jsonl(paths["readouts"], readouts)
    write_jsonl(paths["patches"], patches)
    write_json(paths["layer_scan"], layer_scan)
    write_json(paths["projections"], projections)
    write_json(paths["metrics"], metrics)
    return paths
