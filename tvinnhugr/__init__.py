"""TvinnHugr — testing cross-lingual generalization of the J-space / global workspace.

Independent third-party extension of Anthropic's global-workspace
interpretability work (anthropics/jacobian-lens, Apache-2.0; see NOTICE).
Heavy (torch/transformers) modules are imported lazily — import
`tvinnhugr.fitting` / `tvinnhugr.patching` explicitly when needed;
`tvinnhugr.metrics` and `tvinnhugr.data` are numpy/stdlib-only.
"""

__version__ = "0.1.0"
