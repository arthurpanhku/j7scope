# Extending J7Scope

## Backend plugins

A backend owns inference and internal-state access. Subclass `Backend`, implement
`generate(messages, **params) -> Iterator[Step]`, and expose `model_name`, `layer`,
`is_demo`, `supports_intervention`, `capabilities()`, and `ready()`. `Step` contains
`text`, `token`, and ordered `[(surface_token, score)]` read-outs. Optional `load()`
runs before the CLI accepts traffic. Optional `intervene(...)` returns the same
payload as `HFBackend.intervene`. All generation/intervention tasks are serialized
by the HTTP runtime; backend objects are not promised reentrant access.

Register the factory in your package:

```toml
[project.entry-points."j7scope.backends"]
my_backend = "my_package:ResearchBackend"
```

Then `j7scope --backend my_backend --backend-options backend.json`. The JSON object
is passed to the factory along with standard `layer`, `topk`, and `per_lang` options.
Factories should explicitly validate options. Keep provider credentials in environment
variables, not in the config or trace. Plugins are trusted installed Python code.
Built-in names `mock` and `hf` are reserved. Real backend traces should expose
`model_revision_resolved`, `device`, `dtype`, and Jacobian provenance (see recorder).
A plain text-only remote API cannot provide real J-lens top-k; do not fabricate them
or claim activation capabilities. Such a provider needs a separate explicitly labelled
behavioral-observation schema, outside this preview's internal-state contract.

## Model structure adapters

The HF backend uses `j7scope.adapters.get_adapter(model)` for decoder layers, final
normalization, and unembedding. The built-in adapter accepts dense HF decoders with
`model.model.layers`, `model.model.norm`, and `model.lm_head` (including the validated
SmolLM2 configuration). Unsupported layouts and known MoE configurations fail with
an actionable error. Other structurally compatible models still need validation.

A custom adapter provides:

```python
class ResearchAdapter:
    name = "my-model-layout"
    def supports(self, model): ...  # return bool
    def layers(self, model): ...    # ordered, hookable decoder blocks
    def norm(self, model): ...      # callable final normalization
    def unembed(self, model): ...   # callable hidden → vocabulary logits
```

Register it under `j7scope.model_adapters` using the same entry-point pattern.
Installed custom adapters are checked before the built-in fallback. The fitting
contract expects residual-shaped `[batch, sequence, hidden]` block outputs, either
bare tensors or tuples whose first element is the residual. Decoder blocks must
support differentiable tail replay; matching attribute names alone is insufficient.
Implement and test architecture-specific logic before advertising support.

## Reproducibility

Pin model revisions, preserve the Jacobian's JSON metadata with its tensor, and
record the fitting corpus and estimator. Run `j7scope-validate --model ... --revision
<commit> --layer <layer>` for your model. The default validation model is English
focused; it tests implementation, not a bilingual hypothesis. Use independent
controls, nulls, and reviewed concept vocabulary for scientific claims. The
`--lexicon` JSON mapping and preview-fitting `--corpus` can be replaced without
changing frontend code. The frontend consumes Trace v1; metric definitions remain
in the Python rigor/contrast modules.
