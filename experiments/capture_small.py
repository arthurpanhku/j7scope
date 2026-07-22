"""Capture REAL J-lens traces from a small model on CPU — a P3 pipeline preview.

Unlike build_demo_trace.py (synthetic), this loads an actual HuggingFace model,
fits a real Jacobian, generates, and reads out the real residual stream at each
token via j7scope.fitting.JLens — the same code path a GPU 7B run would use.

It is a **preview, not a research result**: a 0.5B model's workspace is weak and
its bilingual ability limited, so the cross-lingual numbers are not meaningful
and these traces get no DOI. Traces are marked ``preview: true``. The point is to
prove the hf pipeline end-to-end on this machine before renting a GPU (P3).

Usage (slow on CPU; keep it small):
    python experiments/capture_small.py \
        --model Qwen/Qwen2.5-0.5B-Instruct --layer 12 --n-probes 4 --max-new-tokens 24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "apps" / "serve"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Short, generic prompts about the same idea in each language, so the read-outs
# are at least loosely comparable. Free generation, so lengths differ.
PROMPTS = [
    ("preview-deception-en", "en",
     "In one sentence, what is deception and why do people do it?"),
    ("preview-deception-zh", "zh",
     "用一句话说明：什么是欺骗，人们为什么会欺骗？"),
]

# Tiny Jacobian-fit corpus (real, but small so CPU fitting stays quick).
JACOBIAN_CORPUS = [
    "The most important thing to understand is",
    "After thinking about it carefully, she decided to",
    "他仔细考虑之后，最终决定",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n-probes", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--device", default="cpu", help="cpu | mps (mps may hit unsupported ops)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "traces")
    ap.add_argument("--jacobian-cache", default=str(ROOT / ".cache" / "jacobian"))
    args = ap.parse_args()

    from j7scope_serve.backends import HFBackend
    from j7scope_serve.recorder import build_lexicon, record_trace
    from j7scope_serve.protocol import bucket_readout, script_of

    print(f"loading {args.model} on {args.device} (first run downloads ~1GB)…")
    t0 = time.time()
    be = HFBackend(model_name=args.model, layer=args.layer, topk=24,
                   max_new_tokens=args.max_new_tokens, device=args.device,
                   jacobian_corpus=JACOBIAN_CORPUS, n_probes=args.n_probes,
                   cache_dir=args.jacobian_cache)
    be.load()   # loads model + fits (or loads cached) J_l
    print(f"  model+Jacobian ready in {time.time() - t0:.0f}s")

    lexicon = build_lexicon(ROOT, be)
    written = []
    for trace_id, lang, prompt in PROMPTS:
        print(f"generating: {trace_id} …", flush=True)
        t1 = time.time()
        buffered = []
        for step in be.generate([{"role": "user", "content": prompt}]):
            readout = bucket_readout(step.topk, per_lang=8)
            buffered.append({
                "seq": len(buffered),
                "ts_rel": round(time.time() - t1, 3),
                "token": step.token,
                "token_script": script_of(step.token),
                "readout": readout,
            })
        if not buffered:
            print(f"  (no tokens produced for {trace_id})")
            continue
        path = record_trace(args.out, backend=be, prompt=prompt, buffered=buffered,
                            lexicon=lexicon, trace_id=trace_id)
        _mark_preview(path, model=args.model, layer=args.layer, language=lang)
        print(f"  wrote {path}  ({len(buffered)} tokens, {time.time() - t1:.0f}s)")
        written.append(trace_id)

    _refresh_index(args.out)
    print(f"done. real small-model traces: {written}")
    print("view: python -m j7scope_serve --backend mock --traces results/traces  ->  http://127.0.0.1:8799/")


def _mark_preview(trace_dir: Path, *, model: str, layer: int, language: str) -> None:
    """Flag the manifest as a small-model preview (real read-outs, not research)."""
    mpath = trace_dir / "manifest.json"
    m = json.loads(mpath.read_text(encoding="utf-8"))
    m["preview"] = True
    m["label"] = f"Preview · {model.split('/')[-1]} L{layer} ({language})"
    m["note"] = ("Real J-lens read-out from a small model on CPU. Pipeline preview, "
                 "not a research result; cross-lingual numbers are not meaningful at "
                 "this scale.")
    mpath.write_text(json.dumps(m, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")


def _refresh_index(out: Path) -> None:
    """Rebuild traces/index.json from whatever trace dirs exist."""
    rows = []
    for d in sorted(out.glob("*/")):
        mf = d / "manifest.json"
        if not mf.exists():
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        rows.append({
            "trace_id": m["trace_id"],
            "label": m.get("label", m["trace_id"]),
            "language": m.get("language", "?"),
            "is_demo": bool(m.get("is_demo", False)),
            "preview": bool(m.get("preview", False)),
            "n_tokens": sum(1 for _ in (d / "tokens.jsonl").open(encoding="utf-8")),
            "parallel_group": m.get("parallel_group"),
        })
    (out / "index.json").write_text(
        json.dumps({"schema_version": 1, "traces": rows}, ensure_ascii=False,
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
