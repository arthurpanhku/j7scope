"""Capture one custom, real-model Trace v1 artifact.

This is the non-interactive entry point used by the P4 Colab notebook and cloud
GPU contributors. Community captures are marked ``preview: true`` until they
have been reviewed against the project research protocol.

Example:
    python experiments/capture_trace.py \
      --trace-id community-deception-en \
      --language en \
      --concept deception \
      --prompt "In one sentence, explain why deception can be tempting."
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "apps" / "serve"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from j7scope.artifacts import write_json  # noqa: E402
from j7scope.trace import rebuild_trace_index, validate_trace_gallery  # noqa: E402


TRACE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
DEFAULT_JACOBIAN_CORPUS = [
    "The weather today is quite",
    "In the history of science, the most important",
    "She opened the door and saw",
    "The best way to learn a new skill is to",
    "他走进房间，看到桌上放着一封",
    "关于这个问题，我认为最关键的是",
]


def _configuration(args) -> dict:
    return {
        "trace_id": args.trace_id,
        "language": args.language,
        "concept": args.concept,
        "model": args.model,
        "model_revision": args.model_revision,
        "layer": args.layer,
        "device": args.device,
        "dtype": args.dtype,
        "n_probes": args.n_probes,
        "max_new_tokens": args.max_new_tokens,
        "output": str(args.out),
    }


def capture(args) -> Path:
    from j7scope_serve.backends import HFBackend
    from j7scope_serve.protocol import bucket_readout, script_of
    from j7scope_serve.recorder import build_lexicon, record_trace

    backend = HFBackend(
        model_name=args.model,
        model_revision=args.model_revision,
        layer=args.layer,
        topk=args.topk,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        jacobian_corpus=DEFAULT_JACOBIAN_CORPUS,
        n_probes=args.n_probes,
        cache_dir=str(args.cache_dir),
    )
    print(
        f"Loading {args.model}@{args.model_revision} on {args.device} "
        f"(dtype={args.dtype})…",
        flush=True,
    )
    started = time.monotonic()
    backend.load()
    print(
        f"Model and Jacobian ready in {time.monotonic() - started:.1f}s "
        f"({backend.device}, {backend.dtype})",
        flush=True,
    )

    buffered = []
    generation_started = time.monotonic()
    for step in backend.generate([{"role": "user", "content": args.prompt}]):
        buffered.append(
            {
                "seq": len(buffered),
                "ts_rel": round(time.monotonic() - generation_started, 3),
                "token": step.token,
                "token_script": script_of(step.token),
                "readout": bucket_readout(step.topk, per_lang=args.per_lang),
            }
        )
    if not buffered:
        raise RuntimeError("model produced no tokens; no trace was written")

    trace_dir = record_trace(
        args.out,
        backend=backend,
        prompt=args.prompt,
        buffered=buffered,
        lexicon=build_lexicon(ROOT, backend),
        trace_id=args.trace_id,
        concept=args.concept,
        language=args.language,
        capture_tool="experiments/capture_trace.py",
    )
    manifest_path = trace_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "label": args.label or f"Community preview · {args.trace_id}",
            "preview": True,
            "note": (
                "Real model capture contributed through the P4 capture tool. "
                "Preview only until estimator convergence and research review."
            ),
        }
    )
    manifest["capture"]["community_submission"] = True
    write_json(manifest_path, manifest)
    rebuild_trace_index(args.out)

    problems = validate_trace_gallery(args.out)
    if problems:
        joined = "\n".join(f"- {problem}" for problem in problems)
        raise RuntimeError(f"captured trace failed gallery validation:\n{joined}")
    print(f"Wrote {len(buffered)} tokens to {trace_dir}", flush=True)
    return trace_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--language", choices=["en", "zh", "other"], required=True)
    parser.add_argument("--concept", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--n-probes", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--topk", type=int, default=24)
    parser.add_argument("--per-lang", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "community-traces",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache" / "jacobian",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not TRACE_ID_RE.fullmatch(args.trace_id):
        raise SystemExit(
            "--trace-id must be 1-80 lowercase letters, digits, dots, "
            "underscores, or hyphens, and cannot start with punctuation"
        )
    for name in ("n_probes", "max_new_tokens", "topk", "per_lang"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if args.dry_run:
        print(json.dumps(_configuration(args), indent=2, sort_keys=True))
        return
    capture(args)


if __name__ == "__main__":
    main()
