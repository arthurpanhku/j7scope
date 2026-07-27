"""Validate J7Scope's position-local Jacobian math without model downloads.

The script builds a randomly initialized, tiny Qwen2 decoder and compares:

1. the tail-replay ``torch.func.jacrev`` result;
2. an independent full-model, one-output-dimension-at-a-time VJP;
3. the stochastic Gaussian-probe estimator used by the current capture path.

This checks implementation consistency, not lens quality or the paper's
corpus/position reduction. No result from this synthetic model is research
evidence.

Usage:
    python experiments/validate_jacobian.py --n-probes 4096
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from j7scope.fitting import (  # noqa: E402
    _Capture,
    _decoder_layers,
    JLens,
    exact_jacobian_for_prompt,
    jacobian_error,
    paper_jacobian_for_prompt,
)


class _Batch(dict):
    def to(self, device):
        return _Batch({key: value.to(device) for key, value in self.items()})


class _FixedTokenizer:
    def __call__(self, prompt, return_tensors):
        if return_tensors != "pt":
            raise ValueError("this validation tokenizer only returns PyTorch tensors")
        return _Batch(
            input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
        )

    def decode(self, token_ids):
        return str(token_ids[0])


def _full_graph_vjp(model, tokenizer, prompt: str, layer: int) -> torch.Tensor:
    """Independent exact Jacobian using one scalar VJP per output dimension."""
    source_capture, target_capture = _Capture(), _Capture()
    layers = _decoder_layers(model)
    handles = [
        layers[layer].register_forward_hook(source_capture),
        layers[-1].register_forward_hook(target_capture),
    ]
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.enable_grad():
            model(**inputs, use_cache=False)
            rows = []
            for output_index in range(model.config.hidden_size):
                (gradient,) = torch.autograd.grad(
                    target_capture.value[0, -1, output_index].float(),
                    source_capture.value,
                    retain_graph=output_index < model.config.hidden_size - 1,
                )
                rows.append(gradient[0, -1, :])
        return torch.stack(rows).detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def _paper_scalar_vjp(model, tokenizer, prompt: str, layer: int,
                      skip_first: int) -> torch.Tensor:
    """Slow scalar-VJP reference for the paper estimator's position reduction."""
    source_capture, target_capture = _Capture(), _Capture()
    layers = _decoder_layers(model)
    handles = [
        layers[layer].register_forward_hook(source_capture),
        layers[-1].register_forward_hook(target_capture),
    ]
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.enable_grad():
            model(**inputs, use_cache=False)
            valid_positions = torch.arange(
                skip_first,
                target_capture.value.shape[1] - 1,
                device=model.device,
            )
            rows = []
            for output_index in range(model.config.hidden_size):
                objective = target_capture.value[
                    0, valid_positions, output_index
                ].sum()
                (gradient,) = torch.autograd.grad(
                    objective,
                    source_capture.value,
                    retain_graph=output_index < model.config.hidden_size - 1,
                )
                rows.append(gradient[0, valid_positions, :].mean(dim=0))
        return torch.stack(rows).detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-probes", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = Qwen2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        use_cache=False,
    )
    model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    tokenizer = _FixedTokenizer()
    prompt = "synthetic fixed-token prompt"
    layer = 0

    exact = exact_jacobian_for_prompt(
        model, tokenizer, prompt, layer, position=-1, chunk_size=1
    )
    full_graph = _full_graph_vjp(model, tokenizer, prompt, layer)
    paper = paper_jacobian_for_prompt(
        model, tokenizer, prompt, layer, skip_first=0, dim_batch=3
    )
    paper_scalar = _paper_scalar_vjp(
        model, tokenizer, prompt, layer, skip_first=0
    )
    lens = JLens(model, tokenizer, layer)
    stochastic = lens.estimate_jacobian(
        [prompt],
        n_probes=args.n_probes,
        position=-1,
        seed=args.seed,
        show_progress=False,
    )

    report = {
        "model": "random-tiny-qwen2",
        "d_model": config.hidden_size,
        "layer": layer,
        "position": -1,
        "n_probes": args.n_probes,
        "seed": args.seed,
        "exact_vs_full_graph": jacobian_error(exact, full_graph),
        "paper_batched_vs_scalar_vjp": jacobian_error(paper, paper_scalar),
        "stochastic_vs_exact": jacobian_error(stochastic, exact),
        "scope": (
            "Implementation check only: synthetic position-local and paper-"
            "reduction Jacobians, not a research result."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
