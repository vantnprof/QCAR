"""Binary generalized Rayleigh quotient (BGRQ) used by QCAR.

Notation follows the paper (Section 3). Given measured magnitudes ``v`` (unit
norm, non-negative), training data ``X`` and targets ``y``, a candidate model is

    w(a, c) = a * (c ⊙ v),   a ∈ R,  c ∈ {-1, +1}^D.

With ``B = X diag(v)``, the ridge loss is minimised over ``a`` in closed form,

    a*(c) = yᵀBc / (‖Bc‖² + ξ),

and the remaining sign problem is the BGRQ

    max_c  (cᵀ F c) / (cᵀ G c + ξ),   F = q qᵀ,  q = Bᵀy,  G = BᵀB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BGRQProblem:
    """Matrices defining one BGRQ instance."""

    B: np.ndarray  # (N, D) = X diag(v)
    q: np.ndarray  # (D,)   = Bᵀ y
    F: np.ndarray  # (D, D) = q qᵀ
    G: np.ndarray  # (D, D) = Bᵀ B
    alpha: float   # ridge parameter ξ

    @property
    def dim(self) -> int:
        return self.q.shape[0]


def build_problem(X: np.ndarray, y: np.ndarray, magnitudes: np.ndarray, alpha: float) -> BGRQProblem:
    """Form ``B, q, F, G`` from data and measured magnitudes ``v = sqrt(p̂)``."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(magnitudes, dtype=float).ravel()
    if X.shape[1] != v.shape[0]:
        raise ValueError(f"X has {X.shape[1]} columns but {v.shape[0]} magnitudes were given.")
    B = X * v[None, :]
    q = B.T @ y
    return BGRQProblem(B=B, q=q, F=np.outer(q, q), G=B.T @ B, alpha=float(alpha))


def bgrq_value(problem: BGRQProblem, c: np.ndarray) -> float:
    """Evaluate the BGRQ ratio for a sign vector ``c``."""
    c = np.asarray(c, dtype=float)
    denominator = c @ problem.G @ c + problem.alpha
    return float((c @ problem.F @ c) / denominator) if denominator != 0 else 0.0


def optimal_scale(problem: BGRQProblem, y: np.ndarray, c: np.ndarray) -> float:
    """Closed-form signed scale a*(c) (Eq. 13 of the paper)."""
    Bc = problem.B @ np.asarray(c, dtype=float)
    return float(np.asarray(y, dtype=float).ravel() @ Bc / (Bc @ Bc + problem.alpha))


def reconstruct_weights(problem: BGRQProblem, y: np.ndarray, c: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
    """Return ŵ = a*(c) (c ⊙ v)."""
    a = optimal_scale(problem, y, c)
    return a * np.asarray(c, dtype=float) * np.asarray(magnitudes, dtype=float).ravel()
