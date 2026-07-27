"""J-lens fitting: expected input–output Jacobian readout, for Qwen2.5-style models.

Method follows "Verbalizable Representations Form a Global Workspace in
Language Models" (https://transformer-circuits.pub/2026/workspace/index.html):
the residual stream h_l at layer l is mapped into final-layer coordinates by
the expected Jacobian J_l = E_x[∂h_L/∂h_l], then decoded with the model's own
final norm + unembedding — reading out concepts the model is inclined to say
but has not said yet.

J_l is a property of the *model*, not of a probe language: it is fitted once
on a generic corpus and reused unchanged for both zh and en readouts. That is
what makes the cross-lingual comparison meaningful.

Adapted from the approach of `jlens.fitting` in anthropics/jacobian-lens
(Apache-2.0, see NOTICE), reimplemented against HuggingFace Qwen2-style dense
decoders (model.model.layers / model.model.norm / lm_head).

Estimator scope: J7Scope currently fits the position-local cloze variant
``∂h_L[p] / ∂h_l[p]``. The upstream paper implementation instead sums
cotangents over all valid current-and-future target positions and averages over
source positions. Both are Jacobian lenses, but their fitted matrices must not
be presented as numerically interchangeable.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from tqdm import tqdm


def _hidden_states(output):
    """Return a decoder layer's hidden-state tensor across HF API versions."""
    return output[0] if isinstance(output, tuple) else output


def load_model(name: str = "Qwen/Qwen2.5-7B-Instruct", device: str | None = None,
               dtype: torch.dtype = torch.bfloat16,
               revision: str | None = None):
    """Load a HF causal LM + tokenizer, frozen and in eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, revision=revision
    ).to(device)
    # Params stay frozen: autograd graphs are built from the captured residual
    # leaf only (see _Capture), which keeps Jacobian passes cheap.
    model.eval().requires_grad_(False)
    return model, tokenizer


def _decoder_layers(model):
    return model.model.layers


class _Capture:
    """Forward hook capturing a decoder layer's residual-stream output.

    When gradients are enabled, the captured tensor is swapped for a fresh
    leaf so autograd can differentiate the tail of the network against it
    (model params are frozen, so no graph exists below this point).
    """

    def __init__(self):
        self.value = None

    def __call__(self, module, args, output):
        # Decoder layers return either a bare hidden-states tensor (transformers
        # >= ~4.50 for Qwen2/Llama) or a (hidden_states, ...) tuple (older). Handle
        # both, and preserve the shape of what we return so the next layer gets
        # the same type it expected.
        is_tuple = isinstance(output, tuple)
        h = _hidden_states(output)
        if torch.is_grad_enabled() and not h.requires_grad:
            h = h.detach().requires_grad_(True)
            self.value = h
            return ((h,) + tuple(output[1:])) if is_tuple else h
        self.value = h
        return output


def capture_residual(model, tokenizer, prompt: str, layer: int) -> torch.Tensor:
    """Residual stream (1, seq, d) at the output of decoder layer `layer`."""
    cap = _Capture()
    handle = _decoder_layers(model)[layer].register_forward_hook(cap)
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        handle.remove()
    return cap.value


class _LayerInputCapture:
    """Capture one decoder invocation's args/kwargs for tail replay."""

    def __init__(self):
        self.args = None
        self.kwargs = None

    def __call__(self, module, args, kwargs):
        self.args = args
        self.kwargs = kwargs


def _replace_hidden_states(args, kwargs, hidden_states):
    """Replace a decoder call's hidden-state argument without mutating inputs."""
    if args:
        return (hidden_states, *args[1:]), kwargs
    if "hidden_states" in kwargs:
        replaced = dict(kwargs)
        replaced["hidden_states"] = hidden_states
        return args, replaced
    raise RuntimeError("decoder layer call has no hidden_states argument")


def _resolve_position(position: int, seq_len: int) -> int:
    resolved = position + seq_len if position < 0 else position
    if not 0 <= resolved < seq_len:
        raise IndexError(
            f"position {position} is out of range for sequence length {seq_len}"
        )
    return resolved


def exact_jacobian_for_prompt(model, tokenizer, prompt: str, layer: int,
                              position: int = -1,
                              chunk_size: int | None = 1) -> torch.Tensor:
    """Compute the exact position-local Jacobian for one prompt with ``jacrev``.

    The returned matrix is ``∂h_L[position] / ∂h_layer[position]`` with shape
    ``(d_out, d_in)``. A no-grad forward first captures the residual and the
    constant arguments (causal mask, rotary embeddings, positions) supplied to
    every downstream decoder layer. ``torch.func.jacrev`` then differentiates a
    pure replay of that tail, replacing only the selected source-position
    vector. This avoids differentiating the unused prefix of the network.

    ``chunk_size=1`` is deliberately conservative: exact Jacobians are a
    validation tool and a full vmap over thousands of output dimensions can
    exhaust memory. Raise it on small models when memory permits.
    """
    layers = _decoder_layers(model)
    if not 0 <= layer < len(layers) - 1:
        raise ValueError(
            f"layer must be in [0, {len(layers) - 2}] so the lens has a tail to map through"
        )
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be >= 1 or None")

    source_capture = _Capture()
    tail_captures = [_LayerInputCapture() for _ in layers[layer + 1:]]
    handles = [layers[layer].register_forward_hook(source_capture)]
    handles.extend(
        tail_layer.register_forward_pre_hook(capture, with_kwargs=True)
        for tail_layer, capture in zip(
            layers[layer + 1:], tail_captures, strict=True
        )
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if source_capture.value is None or any(c.args is None for c in tail_captures):
        raise RuntimeError("failed to capture decoder activations for exact Jacobian")

    source = source_capture.value.detach()
    position = _resolve_position(position, source.shape[1])
    position_index = torch.tensor([position], device=source.device)

    def replay_tail(source_vector):
        hidden = source.index_copy(
            1, position_index, source_vector.reshape(1, 1, -1)
        )
        for tail_layer, capture in zip(
            layers[layer + 1:], tail_captures, strict=True
        ):
            args, kwargs = _replace_hidden_states(
                capture.args, capture.kwargs, hidden
            )
            hidden = _hidden_states(tail_layer(*args, **kwargs))
        return hidden[0, position, :].float()

    source_vector = source[0, position, :]
    jacobian = torch.func.jacrev(
        replay_tail, chunk_size=chunk_size
    )(source_vector)
    return jacobian.detach().float().cpu()


def paper_jacobian_for_prompt(model, tokenizer, prompt: str, layer: int,
                              skip_first: int = 16,
                              dim_batch: int = 8) -> torch.Tensor:
    """Compute the upstream paper's exact position-reduced estimator.

    For each output dimension, a cotangent is placed at every valid target
    position. Causal attention makes the gradient at source position ``p`` the
    sum of contributions from valid targets at or after ``p``; those source
    gradients are then averaged. This matches the reduction documented by
    ``anthropics/jacobian-lens`` and intentionally differs from J7Scope's
    position-local cloze estimator.

    ``skip_first`` excludes attention-sink positions and the final position is
    always excluded because it has no next-token target. ``dim_batch`` controls
    how many output rows are evaluated together. As in the upstream fitter,
    the prompt is replicated across that batch and differentiated with an
    ordinary VJP; this avoids relying on vmap-compatible backward kernels.
    """
    layers = _decoder_layers(model)
    if not 0 <= layer < len(layers) - 1:
        raise ValueError(
            f"layer must be in [0, {len(layers) - 2}] so the lens has a tail to map through"
        )
    if skip_first < 0:
        raise ValueError("skip_first must be >= 0")
    if dim_batch < 1:
        raise ValueError("dim_batch must be >= 1")

    source_capture, target_capture = _Capture(), _Capture()
    handles = [
        layers[layer].register_forward_hook(source_capture),
        layers[-1].register_forward_hook(target_capture),
    ]
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        replicated_inputs = {
            name: (
                value.expand(dim_batch, *value.shape[1:])
                if torch.is_tensor(value)
                and value.ndim > 0
                and value.shape[0] == 1
                else value
            )
            for name, value in inputs.items()
        }
        with torch.enable_grad():
            model(**replicated_inputs, use_cache=False)
            source = source_capture.value
            target = target_capture.value
            _, seq_len, d_model = target.shape
            valid_positions = torch.arange(
                skip_first, seq_len - 1, device=target.device
            )
            if valid_positions.numel() == 0:
                raise ValueError(
                    f"prompt too short: seq_len={seq_len}, "
                    f"need > {skip_first + 1} tokens"
                )

            jacobian = torch.zeros(d_model, d_model, dtype=torch.float32)
            for dim_start in range(0, d_model, dim_batch):
                n_dims = min(dim_batch, d_model - dim_start)
                batch_indices = torch.arange(n_dims, device=target.device)
                cotangent = torch.zeros_like(target)
                cotangent[
                    batch_indices[:, None],
                    valid_positions[None, :],
                    dim_start + batch_indices[:, None],
                ] = 1
                (gradient,) = torch.autograd.grad(
                    target,
                    source,
                    grad_outputs=cotangent,
                    retain_graph=dim_start + n_dims < d_model,
                )
                source_positions = valid_positions.to(gradient.device)
                rows = gradient[
                    :n_dims, source_positions, :
                ].float().mean(dim=1)
                jacobian[dim_start:dim_start + n_dims] = rows.cpu()
    finally:
        for handle in handles:
            handle.remove()

    return jacobian


_PAPER_CHECKPOINT_FORMAT = "j7scope.paper_jacobian_checkpoint.v1"


def _prompt_corpus_sha1(prompts: list[str]) -> str:
    serialized = json.dumps(
        prompts, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha1(serialized).hexdigest()


def _atomic_torch_save(payload, path: str | Path) -> Path:
    """Atomically replace a torch artifact, keeping partial writes invisible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def jacobian_error(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    """Return scale-aware diagnostics comparing two Jacobian matrices."""
    if candidate.shape != reference.shape:
        raise ValueError(
            f"Jacobian shapes differ: {tuple(candidate.shape)} != {tuple(reference.shape)}"
        )
    candidate = candidate.detach().float().cpu()
    reference = reference.detach().float().cpu()
    delta = candidate - reference
    reference_norm = reference.norm()
    denominator = reference_norm.clamp_min(torch.finfo(torch.float32).eps)
    cosine_denominator = (
        candidate.norm() * reference_norm
    ).clamp_min(torch.finfo(torch.float32).eps)
    return {
        "relative_frobenius": float(delta.norm() / denominator),
        "cosine_similarity": float(
            (candidate.flatten() @ reference.flatten()) / cosine_denominator
        ),
        "max_absolute": float(delta.abs().max()),
    }


class JLens:
    """J-lens readout at a fixed layer.

    >>> model, tok = load_model()
    >>> jlens = JLens(model, tok, layer=18)
    >>> jlens.estimate_jacobian(corpus_prompts)   # once, language-agnostic
    >>> jlens.readout(jlens.collect_residual("他对老板做的事是一种"))
    """

    def __init__(self, model, tokenizer, layer: int):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        n_layers = len(_decoder_layers(model))
        if not 0 <= layer < n_layers - 1:
            raise ValueError(f"layer must be in [0, {n_layers - 2}] so the lens has a tail to map through")
        self.d_model = model.config.hidden_size
        self.J: torch.Tensor | None = None  # (d_out, d_in), float32, on CPU
        self._J_dev: torch.Tensor | None = None

    # ---- fitting -----------------------------------------------------------

    def estimate_jacobian(self, prompts, n_probes: int = 64, position: int = -1,
                          seed: int = 0, show_progress: bool = True) -> torch.Tensor:
        """Stochastic estimate of J = E_x[∂h_L[pos] / ∂h_l[pos]] over `prompts`.

        Uses the identity E_u[u (Jᵀu)ᵀ] = J for u ~ N(0, I): each probe costs
        one backward pass through the tail (reverse mode yields Jᵀu). Unbiased
        but noisy — scale len(prompts) × n_probes until readouts stabilize.

        The estimate is position-local (readout position onto itself), the
        variant the probe corpus is built for (cloze at the final token).
        Use :meth:`estimate_jacobian_exact` on a small prompt/model to measure
        estimator error before committing an expensive capture run.
        """
        if n_probes < 1:
            raise ValueError("n_probes must be >= 1")
        d = self.d_model
        gen = torch.Generator().manual_seed(seed)
        J = torch.zeros(d, d, dtype=torch.float32)
        n_terms = 0

        layers = _decoder_layers(self.model)
        cap_l, cap_L = _Capture(), _Capture()
        handles = [layers[self.layer].register_forward_hook(cap_l),
                   layers[-1].register_forward_hook(cap_L)]
        try:
            iterator = tqdm(prompts, desc=f"J @ layer {self.layer}") if show_progress else prompts
            for prompt in iterator:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.enable_grad():
                    self.model(**inputs, use_cache=False)
                    h_l = cap_l.value
                    out = cap_L.value[0, position, :].float()
                    for _ in range(n_probes):
                        u_cpu = torch.randn(d, generator=gen)
                        u = u_cpu.to(out.device, dtype=out.dtype)
                        (g,) = torch.autograd.grad((u * out).sum(), h_l, retain_graph=True)
                        J += torch.outer(u_cpu, g[0, position, :].float().cpu())
                        n_terms += 1
        finally:
            for h in handles:
                h.remove()

        if n_terms == 0:
            raise ValueError("prompts must contain at least one item")
        self.J = J / n_terms
        self._J_dev = None
        return self.J

    def estimate_jacobian_exact(self, prompts, position: int = -1,
                                chunk_size: int | None = 1,
                                show_progress: bool = True) -> torch.Tensor:
        """Exact mean position-local Jacobian over a small validation sample.

        This computes every Jacobian row and is therefore much more expensive
        than :meth:`estimate_jacobian`. It is intended to validate estimator
        direction and convergence on tiny models or a very small prompt sample,
        not to fit a production 7B lens.
        """
        jacobian_sum = torch.zeros(
            self.d_model, self.d_model, dtype=torch.float32
        )
        n_prompts = 0
        iterator = (
            tqdm(prompts, desc=f"exact J @ layer {self.layer}")
            if show_progress else prompts
        )
        for prompt in iterator:
            jacobian_sum += exact_jacobian_for_prompt(
                self.model,
                self.tokenizer,
                prompt,
                self.layer,
                position=position,
                chunk_size=chunk_size,
            )
            n_prompts += 1
        if n_prompts == 0:
            raise ValueError("prompts must contain at least one item")
        self.J = jacobian_sum / n_prompts
        self._J_dev = None
        return self.J

    def estimate_jacobian_paper(self, prompts, skip_first: int = 16,
                                dim_batch: int = 8,
                                show_progress: bool = True,
                                checkpoint_path: str | Path | None = None,
                                checkpoint_every: int = 1,
                                resume: bool = True) -> torch.Tensor:
        """Exact mean Jacobian using the upstream paper's position reduction.

        This is exposed separately so experiments cannot silently mix it with
        the position-local capture estimator. It is exact but expensive; use a
        pretraining-like corpus with sequences long enough to leave positions
        after ``skip_first``.

        When ``checkpoint_path`` is set, the running float32 sum is atomically
        saved every ``checkpoint_every`` visited prompts and once at completion.
        Resuming validates the full ordered prompt corpus and all estimator
        parameters before continuing, so a stale partial fit cannot be mixed
        into a new run. Too-short prompts are recorded and skipped.
        """
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")
        prompts = list(prompts)
        if not prompts:
            raise ValueError("prompts must contain at least one item")
        if not all(isinstance(prompt, str) for prompt in prompts):
            raise TypeError("every prompt must be a string")

        corpus_sha1 = _prompt_corpus_sha1(prompts)
        checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        parameters = {
            "layer": self.layer,
            "d_model": self.d_model,
            "skip_first": skip_first,
            "dim_batch": dim_batch,
        }
        model_identity = {
            "name": getattr(self.model.config, "_name_or_path", None),
            "revision": getattr(self.model.config, "_commit_hash", None),
        }
        state = {
            "format": _PAPER_CHECKPOINT_FORMAT,
            "completed": False,
            "parameters": parameters,
            "model": model_identity,
            "corpus_sha1": corpus_sha1,
            "prompts_total": len(prompts),
            "next_index": 0,
            "n_used": 0,
            "skipped": [],
            "jacobian_sum": torch.zeros(
                self.d_model, self.d_model, dtype=torch.float32
            ),
        }

        if checkpoint_path is not None and resume and checkpoint_path.exists():
            loaded = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            if not isinstance(loaded, dict):
                raise ValueError("invalid paper Jacobian checkpoint payload")
            expected = {
                "format": _PAPER_CHECKPOINT_FORMAT,
                "parameters": parameters,
                "model": model_identity,
                "corpus_sha1": corpus_sha1,
                "prompts_total": len(prompts),
            }
            mismatches = [
                key for key, value in expected.items()
                if loaded.get(key) != value
            ]
            if mismatches:
                raise ValueError(
                    "paper Jacobian checkpoint does not match this run: "
                    + ", ".join(mismatches)
                )
            jacobian_sum = loaded.get("jacobian_sum")
            if (
                not torch.is_tensor(jacobian_sum)
                or tuple(jacobian_sum.shape) != (self.d_model, self.d_model)
            ):
                raise ValueError(
                    "paper Jacobian checkpoint has an invalid jacobian_sum"
                )
            state.update(loaded)
            state["jacobian_sum"] = jacobian_sum.float().cpu()

        start_index = int(state["next_index"])
        if not 0 <= start_index <= len(prompts):
            raise ValueError("paper Jacobian checkpoint has an invalid next_index")
        iterator = range(start_index, len(prompts))
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(prompts),
                initial=start_index,
                desc=f"paper J @ layer {self.layer}",
            )

        for prompt_index in iterator:
            prompt = prompts[prompt_index]
            try:
                prompt_jacobian = paper_jacobian_for_prompt(
                    self.model,
                    self.tokenizer,
                    prompt,
                    self.layer,
                    skip_first=skip_first,
                    dim_batch=dim_batch,
                )
            except ValueError as exc:
                if not str(exc).startswith("prompt too short:"):
                    raise
                state["skipped"].append(
                    {"index": prompt_index, "reason": str(exc)}
                )
            else:
                state["jacobian_sum"] += prompt_jacobian
                state["n_used"] += 1
            state["next_index"] = prompt_index + 1
            if (
                checkpoint_path is not None
                and state["next_index"] % checkpoint_every == 0
            ):
                _atomic_torch_save(state, checkpoint_path)

        if state["n_used"] == 0:
            if checkpoint_path is not None:
                _atomic_torch_save(state, checkpoint_path)
            raise ValueError(
                "no prompts were long enough for the paper Jacobian estimator"
            )
        state["completed"] = True
        if checkpoint_path is not None:
            _atomic_torch_save(state, checkpoint_path)
        self.J = state["jacobian_sum"] / state["n_used"]
        self._J_dev = None
        return self.J

    # ---- readout -----------------------------------------------------------

    def collect_residual(self, prompt: str, position: int = -1) -> torch.Tensor:
        """h_l (d,) at `position` — the vector the lens reads."""
        return capture_residual(self.model, self.tokenizer, prompt, self.layer)[0, position, :]

    def _J_on(self, device) -> torch.Tensor:
        if self._J_dev is None or self._J_dev.device != torch.device(device):
            self._J_dev = self.J.to(device)
        return self._J_dev

    def readout(self, h_l: torch.Tensor, k: int = 20):
        """Map h_l through J into final coordinates, decode with the model's own
        final norm + unembedding. Returns top-k [(token, logit), ...]."""
        if self.J is None:
            raise RuntimeError("call estimate_jacobian() (or load a saved J) first")
        z = (self._J_on(h_l.device) @ h_l.float()).to(self.model.dtype)
        with torch.no_grad():
            logits = self.model.lm_head(self.model.model.norm(z)).float()
        top = logits.topk(k)
        return [(self.tokenizer.decode([i]), round(v, 3))
                for i, v in zip(top.indices.tolist(), top.values.tolist())]
