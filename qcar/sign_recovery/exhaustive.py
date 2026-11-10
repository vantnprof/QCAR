"""Exact BGRQ solver by exhaustive enumeration (Algorithm 1 in the paper)."""

from __future__ import annotations

import numpy as np

from .objective import BGRQProblem

MAX_EXHAUSTIVE_DIM = 26


def solve_exhaustive(problem: BGRQProblem, chunk_size: int = 1 << 16) -> tuple[np.ndarray, float]:
    """Return a global maximiser of the BGRQ and its value.

    The objective is invariant to ``c -> -c``, so ``c[0] = +1`` is fixed and the
    remaining ``2**(D-1)`` sign patterns are enumerated in vectorised chunks.
    Coordinates with zero magnitude do not affect the reconstructed weights, so
    ties between sign patterns that differ only there are immaterial.
    """
    D = problem.dim
    if D > MAX_EXHAUSTIVE_DIM:
        raise ValueError(f"Exhaustive search over 2^{D - 1} patterns is not supported (D > {MAX_EXHAUSTIVE_DIM}).")
    if D == 1:
        c = np.ones(1)
        return c, float((c @ problem.F @ c) / (c @ problem.G @ c + problem.alpha))

    n_free = D - 1
    total = 1 << n_free
    shifts = np.arange(n_free - 1, -1, -1, dtype=np.int64)
    best_value, best_c = -np.inf, None
    for start in range(0, total, chunk_size):
        idx = np.arange(start, min(start + chunk_size, total), dtype=np.int64)
        bits = (idx[:, None] >> shifts[None, :]) & 1
        C = np.hstack([np.ones((idx.size, 1)), 2.0 * bits - 1.0])  # (chunk, D)
        numerators = (C @ problem.q) ** 2
        denominators = np.einsum("ij,jk,ik->i", C, problem.G, C) + problem.alpha
        values = numerators / denominators
        k = int(np.argmax(values))
        if values[k] > best_value:
            best_value, best_c = float(values[k]), C[k].copy()
    return best_c, best_value
