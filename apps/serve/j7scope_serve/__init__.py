"""j7scope-serve: a J-Space side-channel sidecar for open-model agent harnesses.

When a harness (opencode, codex, ...) points its OpenAI-compatible provider at
this sidecar, the sidecar proxies generation to a local open-weights model and
publishes the model's live J-lens read-out — the concepts it is inclined to say
next — on an SSE side-channel, split into Chinese and English columns.
"""

from .app import serve
from .backends import Backend, HFBackend, MockBackend, Step, make_backend

__all__ = ["serve", "Backend", "MockBackend", "HFBackend", "Step", "make_backend"]
