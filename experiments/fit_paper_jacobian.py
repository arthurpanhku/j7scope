"""Fit a production-style paper-reduction Jacobian on a CUDA GPU.

The local JSONL corpus is intentionally small and is only a smoke test. A
research run should use a pretraining-like corpus such as WikiText and retain
the checkpoint, final tensor, and metadata JSON together.

Smoke test configuration:
    python experiments/fit_paper_jacobian.py --dry-run

Cloud GPU:
    pip install -e '.[fit]'
    python experiments/fit_paper_jacobian.py --preflight-only
    python experiments/fit_paper_jacobian.py \
        --dataset Salesforce/wikitext \
        --dataset-config wikitext-103-raw-v1 \
        --max-prompts 1000 --min-chars 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from j7scope.artifacts import write_json  # noqa: E402
from j7scope.fitting import (  # noqa: E402
    JLens,
    _atomic_torch_save,
    _prompt_corpus_sha1,
    load_model,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CORPUS = ROOT / "data" / "jacobian_fit_smoke.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "jacobian-qwen2.5-7b-l18.pt"
DEFAULT_CHECKPOINT = (
    ROOT / "results" / "jacobian-qwen2.5-7b-l18.checkpoint.pt"
)


def _load_local_prompts(
    path: Path, *, text_key: str, max_prompts: int, min_chars: int
) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"corpus not found: {path}")
    prompts = []
    with path.open(encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            line = line.strip()
            if not line:
                continue
            if path.suffix.casefold() == ".jsonl":
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON"
                    ) from exc
                if isinstance(row, str):
                    text = row
                elif isinstance(row, dict) and isinstance(
                    row.get(text_key), str
                ):
                    text = row[text_key]
                else:
                    raise ValueError(
                        f"{path}:{line_number}: expected a string or "
                        f"object with string field {text_key!r}"
                    )
            else:
                text = line
            text = text.strip()
            if len(text) < min_chars:
                continue
            prompts.append(text)
            if len(prompts) >= max_prompts:
                break
    if not prompts:
        raise ValueError("corpus contains no prompts after filtering")
    return prompts


def _load_dataset_prompts(args) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face dataset support is missing; "
            "run `pip install -e '.[fit]'`"
        ) from exc

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        revision=args.dataset_revision,
        streaming=True,
    )
    prompts = []
    for row in dataset:
        text = row.get(args.text_key)
        if not isinstance(text, str):
            raise ValueError(
                f"dataset row has no string field {args.text_key!r}"
            )
        text = text.strip()
        if len(text) < args.min_chars:
            continue
        prompts.append(text)
        if len(prompts) >= args.max_prompts:
            break
    if not prompts:
        raise ValueError("dataset contains no prompts after filtering")
    if len(prompts) < args.max_prompts:
        raise ValueError(
            f"dataset ended after {len(prompts)} usable prompts; "
            f"requested {args.max_prompts}"
        )
    return prompts


def _load_prompts(args) -> list[str]:
    if args.dataset:
        return _load_dataset_prompts(args)
    return _load_local_prompts(
        args.corpus,
        text_key=args.text_key,
        max_prompts=args.max_prompts,
        min_chars=args.min_chars,
    )


def _resolve_dtype(torch, requested: str, device: str):
    if requested != "auto":
        return getattr(torch, requested)
    if device.startswith("cuda"):
        return (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    return torch.float32


def _gpu_preflight(torch, *, device: str, min_vram_gb: float,
                   allow_cpu: bool) -> dict:
    if not device.startswith("cuda"):
        if not allow_cpu:
            raise RuntimeError(
                "paper Jacobian fitting is intended for CUDA; pass "
                "--allow-cpu only for a tiny diagnostic run"
            )
        return {
            "device": device,
            "cuda_available": bool(torch.cuda.is_available()),
            "warning": "CPU mode is diagnostic only.",
        }
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. Select a cloud GPU runtime and rerun "
            "--preflight-only."
        )
    index = torch.device(device).index or 0
    properties = torch.cuda.get_device_properties(index)
    vram_gb = properties.total_memory / 1024**3
    if vram_gb < min_vram_gb:
        raise RuntimeError(
            f"GPU {properties.name!r} has {vram_gb:.1f} GiB VRAM; "
            f"this run requires at least {min_vram_gb:.1f} GiB. "
            "Lower --dim-batch and explicitly lower --min-vram-gb only "
            "after a measured smoke run."
        )
    return {
        "device": str(torch.device(device)),
        "cuda_available": True,
        "gpu_name": properties.name,
        "vram_gb": round(vram_gb, 2),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "cuda_version": torch.version.cuda,
    }


def _tensor_sha1(tensor) -> str:
    return hashlib.sha1(
        tensor.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _configuration(args) -> dict:
    source = (
        {
            "kind": "huggingface_dataset",
            "dataset": args.dataset,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "revision_requested": args.dataset_revision,
            "text_key": args.text_key,
        }
        if args.dataset
        else {
            "kind": "local",
            "path": str(args.corpus),
            "text_key": args.text_key,
            "smoke_only": args.corpus.resolve() == DEFAULT_CORPUS.resolve(),
        }
    )
    return {
        "model": args.model,
        "model_revision_requested": args.model_revision,
        "layer": args.layer,
        "device": args.device,
        "dtype_requested": args.dtype,
        "skip_first": args.skip_first,
        "dim_batch": args.dim_batch,
        "max_prompts": args.max_prompts,
        "min_chars": args.min_chars,
        "checkpoint_every": args.checkpoint_every,
        "resume": not args.no_resume,
        "corpus": source,
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "metadata": str(args.metadata),
        "min_vram_gb": args.min_vram_gb,
    }


def fit(args) -> dict:
    import torch

    hardware = _gpu_preflight(
        torch,
        device=args.device,
        min_vram_gb=args.min_vram_gb,
        allow_cpu=args.allow_cpu,
    )
    if args.preflight_only:
        return {
            "status": "preflight_passed",
            "configuration": _configuration(args),
            "hardware": hardware,
            "torch_version": torch.__version__,
        }

    prompts = _load_prompts(args)
    dtype = _resolve_dtype(torch, args.dtype, args.device)
    model, tokenizer = load_model(
        args.model,
        device=args.device,
        dtype=dtype,
        revision=args.model_revision,
    )
    lens = JLens(model, tokenizer, layer=args.layer)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    started = time.monotonic()
    jacobian = lens.estimate_jacobian_paper(
        prompts,
        skip_first=args.skip_first,
        dim_batch=args.dim_batch,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
    elapsed_seconds = time.monotonic() - started
    _atomic_torch_save(jacobian, args.output)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    report = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Smoke/benchmark only; not research evidence."
            if not args.dataset and args.corpus.resolve() == DEFAULT_CORPUS.resolve()
            else "Paper-reduction Jacobian fit."
        ),
        "configuration": _configuration(args),
        "hardware": hardware,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "provenance": {
            "model_revision_resolved": (
                getattr(model.config, "_commit_hash", None)
                or args.model_revision
            ),
            "corpus_sha1": _prompt_corpus_sha1(prompts),
            "prompts_loaded": len(prompts),
            "prompts_used": int(checkpoint["n_used"]),
            "prompts_skipped": checkpoint["skipped"],
            "estimator": "paper_replicated_batch_vjp",
        },
        "result": {
            "shape": list(jacobian.shape),
            "dtype": str(jacobian.dtype).removeprefix("torch."),
            "tensor_sha1": _tensor_sha1(jacobian),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "peak_cuda_memory_gb": (
                round(
                    torch.cuda.max_memory_allocated(
                        torch.device(args.device)
                    ) / 1024**3,
                    3,
                )
                if args.device.startswith("cuda")
                else None
            ),
            "output": str(args.output),
        },
    }
    write_json(args.metadata, report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--max-prompts", type=int, default=1000)
    parser.add_argument("--min-chars", type=int, default=0)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--min-vram-gb", type=float, default=40.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.metadata is None:
        args.metadata = args.output.with_suffix(".json")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    integer_bounds = {
        "layer": 0,
        "skip_first": 0,
        "dim_batch": 1,
        "max_prompts": 1,
        "min_chars": 0,
        "checkpoint_every": 1,
    }
    for name, minimum in integer_bounds.items():
        if getattr(args, name) < minimum:
            raise SystemExit(
                f"--{name.replace('_', '-')} must be >= {minimum}"
            )
    if args.min_vram_gb <= 0:
        raise SystemExit("--min-vram-gb must be > 0")
    report = (
        {"status": "dry_run", "configuration": _configuration(args)}
        if args.dry_run
        else fit(args)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
