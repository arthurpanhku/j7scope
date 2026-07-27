"""Reproduce the upstream pretrained Jacobian-lens walkthrough on a cloud GPU.

Defaults are pinned to Anthropic's published Qwen3.5-4B walkthrough and its
Neuronpedia-hosted n=1000 lens. The resulting JSON records resolved model/lens
revisions, GPU details, and J-lens vs logit-lens top tokens at the known
two-hop prompt. It is the final P3 math/quality gate before fitting our own 7B
lens.

Install:
    pip install -e '.[upstream]'

Cloud-GPU usage:
    python experiments/reproduce_upstream.py --preflight-only
    python experiments/reproduce_upstream.py

Configuration-only check (no GPU or downloads):
    python experiments/reproduce_upstream.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from j7scope.artifacts import write_json  # noqa: E402


UPSTREAM_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_LENS_REPO = "neuronpedia/jacobian-lens"
DEFAULT_LENS_REVISION = "qwen-n1000"
DEFAULT_LENS_FILE = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/"
    "Qwen3.5-4B_jacobian_lens_n1000.pt"
)
DEFAULT_PROMPT = "Fact: The currency used in the country shaped like a boot is"


def _configuration(args) -> dict:
    return {
        "model": args.model,
        "model_revision_requested": args.model_revision,
        "lens_repo": args.lens_repo,
        "lens_revision_requested": args.lens_revision,
        "lens_file": args.lens_file,
        "prompt": args.prompt,
        "position": -2,
        "top_k": args.top_k,
        "min_vram_gb": args.min_vram_gb,
        "requires_cuda": not args.allow_cpu,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_dependency_installed": importlib.util.find_spec("jlens") is not None,
    }


def _gpu_preflight(torch, *, min_vram_gb: float, allow_cpu: bool) -> dict:
    if not torch.cuda.is_available():
        if allow_cpu:
            return {
                "device": "cpu",
                "cuda_available": False,
                "warning": "CPU mode is diagnostic only and may be very slow.",
            }
        raise RuntimeError(
            "CUDA GPU not found. Select a cloud GPU runtime, then rerun "
            "--preflight-only before downloading model weights."
        )

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    if vram_gb < min_vram_gb:
        raise RuntimeError(
            f"GPU {props.name!r} has {vram_gb:.1f} GiB VRAM; "
            f"this run requires at least {min_vram_gb:.1f} GiB."
        )
    return {
        "device": "cuda:0",
        "cuda_available": True,
        "gpu_name": props.name,
        "vram_gb": round(vram_gb, 2),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "cuda_version": torch.version.cuda,
    }


def _snapshot_revision(path: str) -> str | None:
    parts = Path(path).parts
    try:
        return parts[parts.index("snapshots") + 1]
    except (ValueError, IndexError):
        return None


def _top_tokens(logits, tokenizer, k: int) -> list[dict]:
    top = logits.float().topk(k)
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "logit": round(float(logit), 5),
        }
        for token_id, logit in zip(
            top.indices.detach().cpu(), top.values.detach().cpu(), strict=True
        )
    ]


def _contains_euro(rows: list[dict]) -> bool:
    return any("euro" in row["token"].casefold() for row in rows)


def run(args) -> dict:
    try:
        import jlens
        import torch
        import transformers
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "upstream validation dependencies are missing; run "
            "`pip install -e '.[upstream]'` first"
        ) from exc

    hardware = _gpu_preflight(
        torch, min_vram_gb=args.min_vram_gb, allow_cpu=args.allow_cpu
    )
    if args.preflight_only:
        return {
            "status": "preflight_passed",
            "configuration": _configuration(args),
            "hardware": hardware,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }

    device = torch.device(hardware["device"])
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype=dtype,
    ).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
    )
    model = jlens.from_hf(hf_model, tokenizer)

    lens_path = hf_hub_download(
        args.lens_repo,
        filename=args.lens_file,
        revision=args.lens_revision,
    )
    lens = jlens.JacobianLens.load(lens_path)
    layers = [
        model.n_layers // 4,
        model.n_layers // 2,
        model.n_layers // 4 * 3,
        model.n_layers - 2,
    ]
    lens_logits, model_logits, _ = lens.apply(
        model,
        args.prompt,
        layers=layers,
        positions=[-2],
    )
    logit_logits, _, _ = lens.apply(
        model,
        args.prompt,
        layers=layers,
        positions=[-2],
        use_jacobian=False,
    )

    layer_rows = []
    for layer in layers:
        jacobian_tokens = _top_tokens(
            lens_logits[layer][0], tokenizer, args.top_k
        )
        logit_tokens = _top_tokens(
            logit_logits[layer][0], tokenizer, args.top_k
        )
        layer_rows.append(
            {
                "layer": layer,
                "jacobian_lens": jacobian_tokens,
                "logit_lens": logit_tokens,
                "jacobian_contains_euro": _contains_euro(jacobian_tokens),
                "logit_contains_euro": _contains_euro(logit_tokens),
            }
        )

    report = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Upstream pretrained-lens reproduction; not a J7Scope M1 result.",
        "configuration": _configuration(args),
        "hardware": hardware,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "provenance": {
            "model_revision_resolved": getattr(hf_model.config, "_commit_hash", None),
            "lens_revision_resolved": _snapshot_revision(lens_path),
            "lens_n_prompts": lens.n_prompts,
            "lens_d_model": lens.d_model,
        },
        "layers": layer_rows,
        "model_top_tokens": _top_tokens(model_logits[0], tokenizer, args.top_k),
        "quality_gate": {
            "expected_concept": "euro",
            "jacobian_layers_matching": [
                row["layer"] for row in layer_rows
                if row["jacobian_contains_euro"]
            ],
            "logit_layers_matching": [
                row["layer"] for row in layer_rows
                if row["logit_contains_euro"]
            ],
        },
    }
    write_json(args.output, report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    parser.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
    parser.add_argument("--lens-file", default=DEFAULT_LENS_FILE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-vram-gb", type=float, default=16.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "upstream-reproduction.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Diagnostic only; the pretrained reproduction is intended for CUDA.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")
    if args.min_vram_gb <= 0:
        raise SystemExit("--min-vram-gb must be > 0")
    report = (
        {
            "status": "dry_run",
            "configuration": _configuration(args),
        }
        if args.dry_run
        else run(args)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
