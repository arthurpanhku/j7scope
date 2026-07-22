"""M2 — cross-lingual activation patching.

Transplant the residual stream carrying a concept from a source-language
prompt into the forward pass of its syntax-aligned target-language pair, then
check whether the downstream J-lens readout follows the *concept* or the
*source language*. Readout that tracks the concept across the language switch
is causal evidence for a shared workspace (H1), stronger than M1 correlations.
"""

from __future__ import annotations

import torch

from j7scope.fitting import JLens, _decoder_layers, capture_residual


def patch_and_readout(jlens: JLens, src_prompt: str, tgt_prompt: str,
                      patch_layer: int, src_pos: int = -1, tgt_pos: int = -1,
                      k: int = 20) -> dict:
    """Patch the (patch_layer, src_pos) residual of `src_prompt` into the
    forward pass of `tgt_prompt`, then J-lens-read the patched run at
    jlens.layer. Requires patch_layer < jlens.layer, so the transplanted
    vector propagates through real computation before being read.

    Returns:
        {"readout":    top-k J-lens readout at (jlens.layer, tgt_pos),
         "next_token": top-k of the patched run's actual final-position
                       next-token distribution (sanity channel)}
    """
    if patch_layer >= jlens.layer:
        raise ValueError("patch_layer must be below jlens.layer")
    model, tokenizer = jlens.model, jlens.tokenizer

    src_vec = capture_residual(model, tokenizer, src_prompt, patch_layer)[0, src_pos, :]

    captured = {}

    def write_hook(module, args, output):
        h = output[0].clone()
        h[:, tgt_pos, :] = src_vec.to(h.dtype)
        return (h,) + tuple(output[1:])

    def read_hook(module, args, output):
        captured["h"] = output[0]
        return output

    layers = _decoder_layers(model)
    handles = [layers[patch_layer].register_forward_hook(write_hook),
               layers[jlens.layer].register_forward_hook(read_hook)]
    try:
        inputs = tokenizer(tgt_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
    finally:
        for h in handles:
            h.remove()

    readout = jlens.readout(captured["h"][0, tgt_pos, :], k=k)
    top = out.logits[0, -1, :].float().topk(k)
    next_token = [(tokenizer.decode([i]), round(v, 3))
                  for i, v in zip(top.indices.tolist(), top.values.tolist())]
    return {"readout": readout, "next_token": next_token}


def concept_hit(readout, expected) -> bool:
    """Heuristic: does any expected surface form appear among the top-k readout
    tokens? Matches by containment either way (BPE pieces vs. full words, e.g.
    "欺" ⊂ "欺骗", "decept" ⊂ "deception"). Refine into a proper zh↔en
    concept-vocabulary map as the corpus grows (see concept_taxonomy.md)."""
    tokens = [t.strip() for t, _ in readout]
    for form in expected:
        form = form.strip()
        for t in tokens:
            if t and (t in form or form in t):
                return True
    return False
