"""J7Scope — testing cross-lingual generalization of the J-space / global workspace.

Independent third-party extension of Anthropic's global-workspace
interpretability work (anthropics/jacobian-lens, Apache-2.0; see NOTICE).
Heavy (torch/transformers) modules are imported lazily — import
`j7scope.fitting` / `j7scope.patching` explicitly when needed;
`j7scope.metrics` and `j7scope.data` are numpy/stdlib-only.
"""

__version__ = "0.1.0"
