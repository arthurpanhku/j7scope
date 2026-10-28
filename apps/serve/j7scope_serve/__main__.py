"""CLI entry point: ``python -m j7scope_serve``.

Adds the repo root to sys.path so the ``hf`` backend can import the j7scope
package without an install; the ``mock`` backend needs nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import logging
import math
import sys
from pathlib import Path

# apps/serve/j7scope_serve/__main__.py -> repo root is parents[3]
_REPO_ROOT = next((p for p in Path(__file__).resolve().parents
                   if (p / "pyproject.toml").is_file() and (p / "apps/site").is_dir()), None)
if _REPO_ROOT is not None and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from j7scope_serve.app import serve
from j7scope_serve.backends import make_backend, backend_names


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    config_args, _ = bootstrap.parse_known_args()
    defaults = {}
    if config_args.config:
        defaults = json.loads(Path(config_args.config).read_text(encoding="utf-8"))
        if not isinstance(defaults, dict):
            bootstrap.error("config must be a JSON object")
    p = argparse.ArgumentParser(
        prog="j7scope-serve",
        description="OpenAI-compatible sidecar that exposes a live J-Space read-out.",
    )
    p.add_argument("--backend", choices=backend_names(), default="mock",
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
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"],
                   default="auto", help="hf: model dtype (auto handles T4 vs newer GPUs)")
    p.add_argument("--model-revision", default="main",
                   help="hf: Hugging Face model branch, tag, or commit")
    p.add_argument("--max-new-tokens", type=int, default=256, help="hf: generation cap")
    p.add_argument("--jacobian-cache", default=None,
                   help="hf: directory to cache the fitted Jacobian")
    p.add_argument("--jacobian-file", default=None,
                   help="hf: precomputed Jacobian tensor from the paper fitter")
    p.add_argument("--jacobian-metadata", default=None,
                   help="hf: fitter metadata JSON (default: same stem as tensor)")
    p.add_argument("--record", default=None, metavar="DIR",
                   help="record each session as a Trace v1 under DIR/<trace_id>/")
    p.add_argument("--traces", default=None, metavar="DIR",
                   help="serve traces from DIR at /traces (default: the --record dir)")
    p.add_argument("--config", help="JSON defaults; command-line flags override them")
    p.add_argument("--version", action="version", version="%(prog)s 0.2.0rc2")
    p.add_argument("--list-backends", action="store_true")
    p.add_argument("--n-probes", type=int, default=16, help="preview Jacobian probes per prompt")
    p.add_argument("--corpus", help="JSON array of strings for preview lens fitting")
    p.add_argument("--lexicon", help="JSON surface-token to concept mapping for recorded traces")
    p.add_argument("--backend-options", help="JSON object passed to an installed backend plugin")
    p.add_argument("--allowed-origin", action="append", default=[])
    p.add_argument("--max-queue", type=int, default=4)
    p.add_argument("--queue-timeout", type=float, default=30)
    p.add_argument("--request-timeout", type=float, default=120)
    p.add_argument("--max-body", type=int, default=262144)
    p.add_argument("--max-viewers", type=int, default=16)
    p.add_argument("--session-ttl", type=int, default=86400)
    p.add_argument("--secure-cookie", action="store_true", help="enable behind an HTTPS reverse proxy")
    known = {a.dest for a in p._actions} - {"help", "version"}
    if set(defaults) - known:
        p.error(f"unknown config keys: {sorted(set(defaults) - known)}")
    p.set_defaults(**defaults)
    args = p.parse_args()
    if args.list_backends:
        print("\n".join(backend_names()))
        return
    for key in ("topk", "per_lang", "max_new_tokens", "n_probes", "queue_timeout", "request_timeout", "max_body", "max_viewers", "session_ttl"):
        value = getattr(args, key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            p.error(f"{key} must be positive")
    if args.max_queue < 0 or args.token_delay < 0:
        p.error("max_queue and token_delay cannot be negative")
    if args.backend not in backend_names():
        p.error("unknown backend")
    api_key = os.environ.get("J7SCOPE_API_KEY")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        p.error("remote binding requires J7SCOPE_API_KEY (at least 32 ASCII characters)")
    if api_key and (len(api_key) < 32 or not api_key.isascii()):
        p.error("J7SCOPE_API_KEY must contain at least 32 ASCII characters")
    logging.basicConfig(level=logging.INFO)


    kw = dict(layer=args.layer, topk=args.topk, per_lang=args.per_lang)
    if args.model:
        kw["model_name"] = args.model
    if args.backend == "hf":
        kw.update(device=args.device, max_new_tokens=args.max_new_tokens,
                  cache_dir=args.jacobian_cache, dtype=args.dtype,
                  model_revision=args.model_revision,
                  jacobian_path=args.jacobian_file,
                  jacobian_metadata_path=args.jacobian_metadata, n_probes=args.n_probes)
        if args.corpus:
            corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
            if not isinstance(corpus, list) or not corpus or not all(isinstance(x, str) and x.strip() for x in corpus):
                p.error("corpus must be a nonempty JSON array of text prompts")
            kw["jacobian_corpus"] = corpus
        kw.setdefault("model_name", "Qwen/Qwen2.5-7B-Instruct")

    if args.backend_options:
        options = json.loads(Path(args.backend_options).read_text(encoding="utf-8"))
        if not isinstance(options, dict):
            p.error("backend-options must be a JSON object")
        kw.update(options)
    backend = make_backend(args.backend, **kw)
    if hasattr(backend, "load"):
        backend.load()
    serve(backend, host=args.host, port=args.port, per_lang=args.per_lang,
          token_delay=args.token_delay, record_dir=args.record,
          traces_dir=args.traces, repo_root=_REPO_ROOT, api_key=api_key,
          allowed_origins=args.allowed_origin, max_queue=args.max_queue,
          queue_timeout=args.queue_timeout, request_timeout=args.request_timeout,
          max_body=args.max_body, max_tokens=args.max_new_tokens,
          max_viewers=args.max_viewers, secure_cookie=args.secure_cookie,
          session_ttl=args.session_ttl, lexicon_file=args.lexicon)


if __name__ == "__main__":
    main()
