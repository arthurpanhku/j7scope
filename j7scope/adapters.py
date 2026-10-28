"""Model structure adapters. Third parties register j7scope.model_adapters entries.

Adapters expose layers(model), norm(model), unembed(model), supports(model).
Fitting still requires differentiable dense decoder blocks; structural support
is not evidence that a model's lens has been scientifically validated.
"""
from importlib.metadata import entry_points


class DenseDecoderAdapter:
    name = "hf-dense-decoder"

    def supports(self, model):
        core = getattr(model, "model", None)
        return (hasattr(core, "layers") and hasattr(core, "norm")
                and hasattr(model, "lm_head")
                and not getattr(model.config, "num_local_experts", 0)
                and not getattr(model.config, "num_experts", 0))

    def layers(self, model):
        return model.model.layers

    def norm(self, model):
        return model.model.norm

    def unembed(self, model):
        return model.lm_head


def get_adapter(model):
    cached = getattr(model, "_j7scope_adapter", None)
    if cached is not None:
        return cached
    for entry in entry_points(group="j7scope.model_adapters"):
        adapter = entry.load()()
        if adapter.supports(model):
            model._j7scope_adapter = adapter
            return adapter
    adapter = DenseDecoderAdapter()
    if not adapter.supports(model):
        raise ValueError("Unsupported model structure. Install a j7scope.model_adapters plugin; "
                         "the built-in adapter requires a dense HF decoder with layers/norm/lm_head.")
    model._j7scope_adapter = adapter
    return adapter
