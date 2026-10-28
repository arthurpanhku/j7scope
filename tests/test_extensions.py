from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps/serve"))
from j7scope import adapters
from j7scope_serve.backends import Backend, backend_names, make_backend


def test_unknown_model_layout_rejected():
    with pytest.raises(ValueError, match="Unsupported model structure"):
        adapters.get_adapter(SimpleNamespace(model=SimpleNamespace(), config=SimpleNamespace()))


def test_known_moe_layout_rejected():
    model = SimpleNamespace(model=SimpleNamespace(layers=[], norm=object()),
                            lm_head=object(), config=SimpleNamespace(num_experts=8))
    with pytest.raises(ValueError, match="Unsupported model structure"):
        adapters.get_adapter(model)


def test_model_adapter_plugin_is_used_and_cached(monkeypatch):
    class Adapter:
        name = "research-architecture"
        def supports(self, model):
            return True
    adapter_type = Adapter
    entry = SimpleNamespace(load=lambda: adapter_type)
    monkeypatch.setattr(adapters, "entry_points", lambda **kw: [entry])
    model = SimpleNamespace()
    assert adapters.get_adapter(model).name == "research-architecture"
    monkeypatch.setattr(adapters, "entry_points", lambda **kw: pytest.fail("adapter should be cached"))
    assert adapters.get_adapter(model).name == "research-architecture"


def test_backend_entrypoint_discovery_and_contract(monkeypatch):
    import importlib.metadata
    class Plugin(Backend):
        def __init__(self, model_name, **kw):
            self.model_name = model_name
    entry = SimpleNamespace(name="lab", load=lambda: Plugin)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kw: [entry])
    assert "lab" in backend_names()
    assert make_backend("lab", model_name="my-weights").model_name == "my-weights"
    entry.load = lambda: lambda **kw: object()
    with pytest.raises(TypeError, match="subclass"):
        make_backend("lab")
