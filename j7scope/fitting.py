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
"""

from __future__ import annotations

import torch
from tqdm import tqdm


def load_model(name: str = "Qwen/Qwen2.5-7B-Instruct", device: str | None = None,
               dtype: torch.dtype = torch.bfloat16):
    """Load a HF causal LM + tokenizer, frozen and in eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
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
        h = output[0] if is_tuple else output
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
            model(**inputs)
    finally:
        handle.remove()
    return cap.value


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
        TODO: cross-check against exact per-position Jacobians
        (torch.func.jacrev) on a subsample, and against upstream jlens.fitting.
        """
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
                    self.model(**inputs)
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

        self.J = J / n_terms
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
