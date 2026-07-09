"""Representation-similarity metrics for M1.

All matrix inputs are (n_samples, dim) arrays of readout directions (or
residuals), one row per prompt; zh and en matrices must be row-aligned by
prompt id. Compare the cross-lingual score against a same-language
different-prompt baseline before interpreting it.

References:
- Linear CKA: Kornblith et al. 2019, "Similarity of Neural Network
  Representations Revisited" (arXiv:1905.00414)
- SVCCA: Raghu et al. 2017, "SVCCA: Singular Vector Canonical Correlation
  Analysis for Deep Learning Dynamics and Interpretability" (arXiv:1706.05806)
"""

import numpy as np


def _center(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected (n_samples, dim), got shape {X.shape}")
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X, Y):
    """Linear CKA between (n, d1) and (n, d2) representations (same n).

    Caution: with n_samples << dim, chance-level CKA between independent
    matrices is well above 0 (e.g. ~0.7 at n=30, d=64) — never interpret a raw
    cross-lingual CKA without the shuffled-pairing null alongside it.
    """
    X, Y = _center(X), _center(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows (aligned prompts)")
    num = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    den = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(num / den)


def _svd_reduce(X, variance_kept):
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    keep = int(np.searchsorted(np.cumsum(S**2) / np.sum(S**2), variance_kept)) + 1
    return U[:, :keep] * S[:keep]


def svcca(X, Y, variance_kept=0.99):
    """Mean canonical correlation after per-side SVD reduction.

    Note: with few samples (n << dim) CCA saturates; keep n comfortably above
    the retained SVD dimensions, or prefer CKA for small corpora.
    """
    X, Y = _center(X), _center(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows (aligned prompts)")
    Qx, _ = np.linalg.qr(_svd_reduce(X, variance_kept))
    Qy, _ = np.linalg.qr(_svd_reduce(Y, variance_kept))
    corrs = np.clip(np.linalg.svd(Qx.T @ Qy, compute_uv=False), 0.0, 1.0)
    return float(corrs.mean())


def topk_overlap(tokens_a, tokens_b, k=None):
    """Overlap coefficient |A ∩ B| / k between two top-k readout token lists.

    For cross-lingual comparison, map both sides into a shared concept
    vocabulary first (raw zh/en token strings never literally match); the
    `expected` field in the probe corpus supports this. Duplicates are ignored.
    """
    a = list(tokens_a)[:k] if k else list(tokens_a)
    b = list(tokens_b)[:k] if k else list(tokens_b)
    kk = min(len(a), len(b))
    if kk == 0:
        return 0.0
    return len(set(a[:kk]) & set(b[:kk])) / kk
