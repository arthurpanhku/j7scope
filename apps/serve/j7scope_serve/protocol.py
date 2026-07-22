"""Wire schema shared by the sidecar server, its viewer, and harness plugins.

The sidecar publishes one *J-space event* per generated token on the
``/jspace/stream`` SSE channel. Each event carries the surface token the model
just emitted plus the J-lens readout at that position — the concepts the model
is "inclined to say but has not said yet". Because Qwen-style tokenizers are
bilingual, the readout naturally mixes Chinese and English tokens; we bucket the
top-k by script so a viewer can show a 中文 column and an English column
side by side. That split is the live analogue of J7Scope's cross-lingual
question: is the same workspace direction lighting up both languages at once?
"""

from __future__ import annotations

import time
from typing import Iterable, List, Sequence, Tuple

# Event type constants (the "type" field on every SSE payload).
EV_META = "meta"          # sent once when a viewer connects
EV_REQUEST_START = "request_start"
EV_TOKEN = "token"        # one per generated token: the interesting one
EV_REQUEST_END = "request_end"

PROTOCOL_VERSION = 1

# CJK Unified Ideographs (main block) — enough to tell zh tokens from en ones
# for display bucketing. Not meant to be a full script detector.
_CJK_RANGES: Sequence[Tuple[int, int]] = (
    (0x3400, 0x4DBF),   # Ext A
    (0x4E00, 0x9FFF),   # main
    (0xF900, 0xFAFF),   # compatibility
    (0x20000, 0x2A6DF),  # Ext B
)

_LATIN = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _has_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for lo, hi in _CJK_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def _has_latin(text: str) -> bool:
    return any(ch in _LATIN for ch in text)


def script_of(token: str) -> str:
    """Bucket a decoded token into 'zh', 'en', or 'other' for display.

    A token counts as zh if it contains any CJK character, en if it contains a
    latin letter and no CJK, else other (punctuation, digits, byte fragments).
    """
    t = token.strip()
    if not t:
        return "other"
    if _has_cjk(t):
        return "zh"
    if _has_latin(t):
        return "en"
    return "other"


def bucket_readout(
    topk: Iterable[Tuple[str, float]], *, per_lang: int = 8
) -> dict:
    """Split a single J-lens top-k list into zh / en columns.

    ``topk`` is an ordered ``[(token, score), ...]`` from most to least likely.
    Returns ``{"zh": [...], "en": [...], "other": [...]}`` where each list holds
    at most ``per_lang`` ``{"token", "score", "rank"}`` entries, preserving the
    original global ranking within each bucket.
    """
    buckets: dict = {"zh": [], "en": [], "other": []}
    for rank, (token, score) in enumerate(topk):
        b = buckets[script_of(token)]
        if len(b) < per_lang:
            b.append({"token": token, "score": round(float(score), 3), "rank": rank})
    return buckets


def token_event(
    *,
    seq: int,
    request_id: str,
    token: str,
    readout: dict,
    model: str,
    layer: int,
) -> dict:
    """Build a per-token J-space event."""
    return {
        "type": EV_TOKEN,
        "v": PROTOCOL_VERSION,
        "seq": seq,
        "request_id": request_id,
        "ts": round(time.time(), 3),
        "token": token,
        "token_script": script_of(token),
        "model": model,
        "layer": layer,
        "readout": readout,
    }


def meta_event(*, model: str, layer: int, backend: str, is_demo: bool) -> dict:
    return {
        "type": EV_META,
        "v": PROTOCOL_VERSION,
        "ts": round(time.time(), 3),
        "model": model,
        "layer": layer,
        "backend": backend,
        "is_demo": is_demo,
    }


def request_start_event(*, request_id: str, model: str) -> dict:
    return {
        "type": EV_REQUEST_START,
        "v": PROTOCOL_VERSION,
        "ts": round(time.time(), 3),
        "request_id": request_id,
        "model": model,
    }


def request_end_event(*, request_id: str, n_tokens: int) -> dict:
    return {
        "type": EV_REQUEST_END,
        "v": PROTOCOL_VERSION,
        "ts": round(time.time(), 3),
        "request_id": request_id,
        "n_tokens": n_tokens,
    }


# --- OpenAI-compatible chat-completion streaming chunks --------------------
# Just enough of the schema for AI-SDK / OpenAI clients (opencode, codex) to
# consume a streamed text response.


def chat_chunk(
    *, request_id: str, model: str, created: int, content: str = None,
    role: str = None, finish_reason: str = None,
) -> dict:
    delta: dict = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
