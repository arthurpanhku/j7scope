"""CLI entry point: ``python -m j7scope_serve``.

Adds the repo root to sys.path so the ``hf`` backend can import the j7scope
package without an install; the ``mock`` backend needs nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# apps/serve/j7scope_serve/__main__.py -> repo root is parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from j7scope_serve.app import serve
from j7scope_serve.backends import make_backend


def main() -> None:
    p = argparse.ArgumentParser(
        prog="j7scope-serve",
        description="OpenAI-compatible sidecar that exposes a live J-Space read-out.",
    )
    p.add_argument("--backend", choices=["mock", "hf"], default="mock",
                   help="mock = no deps, synthetic read-out; hf = real model + JLens")
    p.add_argument("--model", default=None,
                   help="model name (hf: HF repo id; mock: display label)")
    p.add_argument("--layer", type=int, default=18, help="J-lens layer l")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8799)
    p.add_argument("--topk", type=int, default=24, help="J-lens top-k before bucketing")
    p.add_argument("--per-lang", type=int, default=8, help="entries shown per language")
    p.add_argument("--token-delay", type=float, default=0.04,
                   help="seconds between streamed tokens (mock pacing / demo feel)")
    p.add_argument("--device", default=None, help="hf: torch device override")
    p.add_argument("--max-new-tokens", type=int, default=256, help="hf: generation cap")
    p.add_argument("--jacobian-cache", default=None,
                   help="hf: directory to cache the fitted Jacobian")
    p.add_argument("--record", default=None, metavar="DIR",
                   help="record each session as a Trace v1 under DIR/<trace_id>/")
    p.add_argument("--traces", default=None, metavar="DIR",
                   help="serve traces from DIR at /traces (default: the --record dir)")
    args = p.parse_args()

    kw = dict(layer=args.layer, topk=args.topk, per_lang=args.per_lang)
    if args.model:
        kw["model_name"] = args.model
    if args.backend == "hf":
        kw.update(device=args.device, max_new_tokens=args.max_new_tokens,
                  cache_dir=args.jacobian_cache)
        kw.setdefault("model_name", "Qwen/Qwen2.5-7B-Instruct")

    backend = make_backend(args.backend, **kw)
    serve(backend, host=args.host, port=args.port, per_lang=args.per_lang,
          token_delay=args.token_delay, record_dir=args.record,
          traces_dir=args.traces, repo_root=_REPO_ROOT)


if __name__ == "__main__":
    main()
