"""Opt-in real pretrained-model validation; produces a bounded, auditable report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import tempfile
import secrets
import threading
import urllib.request
import time

from j7scope import __version__
from .backends import HFBackend


def validate(model, *, revision="main", layer=1, device="cpu", dtype="float32", n_probes=8):
    import torch
    from j7scope.adapters import get_adapter
    from j7scope.fitting import _Capture
    from j7scope.trace import validate_trace_dir
    from .protocol import bucket_readout, script_of
    from .recorder import record_trace, build_lexicon

    torch.manual_seed(0)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="j7scope-validation-") as work:
        backend = HFBackend(model_name=model, model_revision=revision, layer=layer,
            device=device, dtype=dtype, n_probes=n_probes, topk=8,
            jacobian_corpus=["The capital of France is", "Science helps us understand"],
            cache_dir=work, max_new_tokens=4)
        backend.load()
        load_seconds = time.monotonic() - started
        adapter = get_adapter(backend._model)
        checks = {}
        checks["pretrained_weights_loaded"] = backend._model.num_parameters() > 1_000_000
        checks["finite_jacobian"] = bool(torch.isfinite(backend._jlens.J).all())
        checks["cache_roundtrip"] = torch.equal(backend._load_cached_jacobian(), backend._jlens.J)
        messages = [{"role": "user", "content": "The capital of France is"}]
        encoded = backend._tok.apply_chat_template(messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True).to(backend._model.device)
        # Independent hook + forward validates alignment of the first generated step.
        capture = _Capture()
        handle = adapter.layers(backend._model)[layer].register_forward_hook(capture)
        try:
            with torch.no_grad():
                direct = backend._model(**encoded, use_cache=False)
        finally:
            handle.remove()
        expected_token = backend._tok.decode([direct.logits[0, -1].argmax().item()])
        expected_readout = backend._jlens.readout(capture.value[0, -1], k=8)
        steps = list(backend.generate(messages, max_tokens=4, temperature=0))
        checks["nonempty_generation"] = bool(steps)
        checks["first_token_matches_direct_forward"] = bool(steps and steps[0].text == expected_token)
        checks["first_readout_matches_independent_capture"] = bool(steps and steps[0].topk == expected_readout)
        checks["finite_readouts"] = bool(steps and all(math.isfinite(v) for step in steps for _, v in step.topk))
        # Same-source/target patch is an identity intervention; it must preserve readout.
        prompt = "The capital of France is"
        unpatched = backend._jlens.readout(backend._jlens.collect_residual(prompt), k=8)
        intervention = backend.intervene(source_prompt=prompt, target_prompt=prompt,
                                        patch_layer=0, topk=8)
        checks["identity_patch_preserves_readout"] = intervention["readout"] == unpatched
        buffered = [{"seq": i, "ts_rel": i * 0.01, "token": step.token,
                     "token_script": script_of(step.token), "readout": bucket_readout(step.topk)}
                    for i, step in enumerate(steps)]
        trace_dir = record_trace(Path(work) / "traces", backend=backend, prompt=prompt,
            buffered=buffered, lexicon=build_lexicon(None, backend), trace_id="real-model-smoke")
        checks["trace_schema_valid"] = trace_dir is not None and not validate_trace_dir(trace_dir)
        manifest = json.loads((trace_dir / "manifest.json").read_text()) if trace_dir else {}
        checks["provenance_and_preview"] = (manifest.get("preview") is True
            and manifest.get("is_demo") is False and manifest.get("jacobian", {}).get("sha1") == backend.jacobian_sha1)
        # Exercise the actual authenticated HTTP surface using the same real weights.
        from .app import SidecarServer
        key = secrets.token_urlsafe(32)
        server = SidecarServer(("127.0.0.1", 0), backend, api_key=key, token_delay=0,
                               max_tokens=4, record_dir=Path(work) / "http-traces")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
            for stream in (False, True):
                body = json.dumps({"messages": messages, "max_tokens": 4, "stream": stream}).encode()
                with urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", body, headers), timeout=60) as response:
                    data = response.read()
                if stream:
                    chunks = [json.loads(line[6:]) for line in data.decode().splitlines()
                              if line.startswith("data: ") and line != "data: [DONE]"]
                    text = "".join(chunk.get("choices", [{}])[0].get("delta", {}).get("content", "") for chunk in chunks)
                    checks["http_stream_matches_direct_generation"] = text == "".join(s.text for s in steps) and data.endswith(b"data: [DONE]\n\n")
                else:
                    checks["http_json_matches_direct_generation"] = json.loads(data)["choices"][0]["message"]["content"] == "".join(s.text for s in steps)
            with urllib.request.urlopen(urllib.request.Request(base + "/traces/index.json", headers=headers)) as response:
                checks["http_recorded_gallery"] = len(json.load(response)["traces"]) == 2
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        return {
            "schema_version": 1, "passed": all(checks.values()), "checks": checks,
            "scope": "CPU/model pipeline smoke validation; not lens convergence, semantic quality, bilingual evidence, or GPU performance",
            "created_at": datetime.now(timezone.utc).isoformat(), "j7scope": __version__,
            "model": model, "model_revision_resolved": backend.model_revision_resolved,
            "parameters": backend._model.num_parameters(), "layer": layer,
            "device": backend.device, "dtype": backend.dtype, "adapter": backend.adapter_name,
            "jacobian": {"sha1": backend.jacobian_sha1, "estimator": backend.jacobian_estimator,
                         "n_probes": n_probes, "corpus": backend.jacobian_corpus},
            "generation": {"text": "".join(s.text for s in steps), "tokens": len(steps),
                           "first_readout": steps[0].topk if steps else []},
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "torch": torch.__version__, "transformers": importlib.metadata.version("transformers")},
            "timing": {"load_and_fit_seconds": round(load_seconds, 3),
                       "total_seconds": round(time.monotonic() - started, 3)},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--revision", default="12fd25f77366fa6b3b4b768ec3050bf629380bac")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--n-probes", type=int, default=8)
    parser.add_argument("--output", default="validation-real-model.json")
    args = parser.parse_args()
    try:
        report = validate(args.model, revision=args.revision, layer=args.layer,
                          device=args.device, dtype=args.dtype, n_probes=args.n_probes)
    except Exception as exc:
        report = {"passed": False, "error": str(exc), "model": args.model, "revision": args.revision}
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"], "report": args.output}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
