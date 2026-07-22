"""Generation backends for the sidecar.

A backend turns a chat request into a stream of ``Step``s. Each step is one
generated token plus the J-lens top-k readout at that token's position — the
raw, pre-bucketing ``[(token, score), ...]``. The server buckets it by script
and fans it out to viewers.

Two backends:

* ``MockBackend`` — zero dependencies (no torch). Streams a deterministic reply
  and synthesises plausible bilingual readouts from a small concept table, so
  the whole pipe (harness -> sidecar -> viewer) can be exercised without a GPU.
  Everything it emits is clearly marked ``is_demo``.
* ``HFBackend`` — loads a real HuggingFace dense decoder and reuses
  ``j7scope.fitting.JLens`` to read the actual residual stream at each step.
  torch/transformers are imported lazily, so this file imports fine on a box
  that only ever runs the mock.
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Iterator, List, NamedTuple, Sequence, Tuple


class Step(NamedTuple):
    text: str                       # surface text appended to the reply
    token: str                      # decoded token, for the readout event
    topk: List[Tuple[str, float]]   # raw J-lens top-k at this position


class Backend:
    model_name: str
    layer: int
    is_demo: bool = False

    def generate(self, messages: Sequence[dict], **params) -> Iterator[Step]:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Mock backend
# --------------------------------------------------------------------------

# Compact bilingual concept table (a subset mirroring data/concept_taxonomy).
# token = the surface English word the mock "says"; zh/en = the J-space columns
# it lights up while saying it.
_CONCEPTS = [
    ("deception",    ["欺骗", "撒谎", "不诚实", "谎言"],   ["deception", "lying", "dishonesty", "deceit"]),
    ("manipulation", ["操纵", "摆布", "利用", "控制"],     ["manipulation", "coercion", "exploiting", "control"]),
    ("concession",   ["让步", "妥协", "退让", "认输"],     ["concession", "compromise", "yielding", "conceding"]),
    ("evaluation",   ["评估", "考核", "测试", "审查"],     ["evaluation", "assessment", "testing", "scrutiny"]),
    ("emotion",      ["情绪", "感受", "愤怒", "焦虑"],     ["emotion", "feeling", "anger", "anxiety"]),
    ("reasoning",    ["推理", "推断", "逻辑", "因果"],     ["reasoning", "inference", "logic", "causality"]),
    ("entity",       ["巴黎", "长城", "黄河", "泰山"],     ["Paris", "GreatWall", "YellowRiver", "Everest"]),
]

_FILLER_ZH = ["概念", "意图", "判断", "行为", "结果"]
_FILLER_EN = ["concept", "intent", "judgment", "behavior", "result"]

_REPLY_TEMPLATE = (
    "This is the J7Scope J-Space sidecar running in demo mode. "
    "No real model is loaded; the concept columns you see are synthesised so you "
    "can watch the workspace read-out light up as tokens stream by."
)


def _jitter(seed: str) -> float:
    h = int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 1000) / 1000.0  # in [0, 1)


class MockBackend(Backend):
    is_demo = True

    def __init__(self, model_name: str = "j7scope-mock", layer: int = 18,
                 topk: int = 24, per_lang: int = 8):
        self.model_name = model_name
        self.layer = layer
        self.topk = topk
        self.per_lang = per_lang

    def _readout_for(self, concept_idx: int, position: int) -> List[Tuple[str, float]]:
        _, zh, en = _CONCEPTS[concept_idx % len(_CONCEPTS)]
        # Interleave the concept's zh/en words at the top, then fillers, so both
        # columns fill up. Scores descend with a little deterministic jitter.
        ordered = []
        for i in range(max(len(zh), len(en))):
            if i < len(zh):
                ordered.append(zh[i])
            if i < len(en):
                ordered.append(en[i])
        ordered += _FILLER_ZH + _FILLER_EN
        out: List[Tuple[str, float]] = []
        for rank, tok in enumerate(ordered[: self.topk]):
            score = 12.0 - rank * 0.6 - _jitter(f"{concept_idx}:{position}:{tok}") * 0.4
            out.append((tok, round(score, 3)))
        return out

    def generate(self, messages: Sequence[dict], **params) -> Iterator[Step]:
        words = _REPLY_TEMPLATE.split(" ")
        concept_cycle = itertools.cycle(range(len(_CONCEPTS)))
        for position, word in enumerate(words):
            # Advance the lit-up concept every few tokens.
            concept_idx = position // 3
            piece = (" " if position > 0 else "") + word
            yield Step(text=piece, token=word,
                       topk=self._readout_for(concept_idx, position))
        next(concept_cycle)  # keep cycle referenced; harmless


# --------------------------------------------------------------------------
# Real HuggingFace backend (lazy torch import)
# --------------------------------------------------------------------------

# Small generic corpus for the one-time Jacobian estimate. J_l is a property of
# the model, not of any probe language, so the corpus just needs to exercise the
# residual stream broadly; keep it short so warm-up is quick.
_DEFAULT_JACOBIAN_CORPUS = [
    "The weather today is quite",
    "In the history of science, the most important",
    "She opened the door and saw",
    "The best way to learn a new skill is to",
    "After a long negotiation, both sides finally",
    "他走进房间，看到桌上放着一封",
    "关于这个问题，我认为最关键的是",
    "经过长时间的讨论，双方终于达成了",
]


class HFBackend(Backend):
    """Real J-lens readout over a HuggingFace dense decoder.

    Not exercised on CPU-only / no-torch machines; intended for a GPU box with
    the model weights available. Structured to mirror MockBackend's Step stream.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", layer: int = 18,
                 topk: int = 24, max_new_tokens: int = 256, device: str = None,
                 jacobian_corpus: Sequence[str] = None, n_probes: int = 16,
                 cache_dir: str = None):
        self.model_name = model_name
        self.layer = layer
        self.topk = topk
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.jacobian_corpus = list(jacobian_corpus or _DEFAULT_JACOBIAN_CORPUS)
        self.n_probes = n_probes
        self.cache_dir = cache_dir
        self._model = None
        self._tok = None
        self._jlens = None
        self._capture = None

    # ---- warm-up ----------------------------------------------------------

    def load(self) -> None:
        import torch  # noqa: F401  (import-time check + used below)
        from j7scope.fitting import JLens, load_model, _Capture

        model, tok = load_model(self.model_name, device=self.device)
        jlens = JLens(model, tok, layer=self.layer)

        J = self._load_cached_jacobian()
        if J is not None:
            jlens.J = J
        else:
            jlens.estimate_jacobian(self.jacobian_corpus, n_probes=self.n_probes)
            self._save_cached_jacobian(jlens.J)

        # Persistent hook: capture the residual at layer l on every forward.
        cap = _Capture()
        model.model.layers[self.layer].register_forward_hook(cap)

        self._model, self._tok, self._jlens, self._capture = model, tok, jlens, cap

    def _jacobian_cache_path(self):
        if not self.cache_dir:
            return None
        import os
        key = hashlib.sha1(
            (self.model_name + f"|L{self.layer}|" + "\n".join(self.jacobian_corpus)
             + f"|p{self.n_probes}").encode("utf-8")
        ).hexdigest()[:16]
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"jacobian-{key}.pt")

    def _load_cached_jacobian(self):
        path = self._jacobian_cache_path()
        if not path:
            return None
        import os
        import torch
        if os.path.exists(path):
            return torch.load(path, map_location="cpu")
        return None

    def _save_cached_jacobian(self, J) -> None:
        path = self._jacobian_cache_path()
        if not path:
            return
        import torch
        torch.save(J, path)

    # ---- generation -------------------------------------------------------

    def generate(self, messages: Sequence[dict], **params) -> Iterator[Step]:
        import torch

        if self._model is None:
            self.load()
        model, tok, jlens, cap = self._model, self._tok, self._jlens, self._capture

        prompt_ids = tok.apply_chat_template(
            list(messages), add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        max_new = int(params.get("max_tokens") or self.max_new_tokens)
        temperature = float(params.get("temperature", 0.0) or 0.0)

        past = None
        cur = prompt_ids
        with torch.no_grad():
            for _ in range(max_new):
                out = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                h_l = cap.value[0, -1, :]
                logits = out.logits[0, -1, :]

                if temperature > 0:
                    probs = torch.softmax(logits.float() / temperature, dim=-1)
                    next_id = torch.multinomial(probs, 1)[0]
                else:
                    next_id = logits.argmax(-1)

                if next_id.item() == tok.eos_token_id:
                    break

                topk = jlens.readout(h_l, k=self.topk)
                text = tok.decode([next_id.item()])
                yield Step(text=text, token=text, topk=topk)

                cur = next_id.view(1, 1)


def make_backend(kind: str, **kw) -> Backend:
    if kind == "mock":
        return MockBackend(**{k: v for k, v in kw.items()
                              if k in ("model_name", "layer", "topk", "per_lang")})
    if kind == "hf":
        return HFBackend(**{k: v for k, v in kw.items()
                            if k in ("model_name", "layer", "topk", "max_new_tokens",
                                     "device", "jacobian_corpus", "n_probes", "cache_dir")})
    raise ValueError(f"unknown backend: {kind!r} (use 'mock' or 'hf')")
