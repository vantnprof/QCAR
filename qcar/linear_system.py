"""Construction of the padded ridge-regression linear system (Section 2.3 of the paper)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RidgeSystem:
    """Padded system ``A' w' = b'`` with ``A' = blkdiag(XᵀX + ξI, I)``, ``b' = [Xᵀy; 0] / ‖Xᵀy‖``.

    ``window = (start, end)`` gives the computational-basis indices that hold the
    ``D`` model coordinates in the measured output of the quantum solver.
    """

    A: np.ndarray
    b: np.ndarray
    dim: int
    window: tuple[int, int]


def next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


def build_ridge_system(X: np.ndarray, y: np.ndarray, alpha: float, solver: str) -> RidgeSystem:
    """Build the normalised, power-of-two padded ridge system for ``solver`` in {'hhl', 'vqls'}.

    For HHL the solution is read out after post-selecting the flag qubit in ``|1⟩``,
    so the model window is shifted by the padded dimension ``M``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    D = X.shape[1]
    A = X.T @ X + alpha * np.eye(D)
    A = 0.5 * (A + A.T)  # guard against floating-point asymmetry
    b = X.T @ y

    M = next_power_of_two(D)
    if M > D:
        A_pad = np.eye(M)
        A_pad[:D, :D] = A
        A = A_pad
        b = np.concatenate([b, np.zeros(M - D)])
    b = b / np.linalg.norm(b)

    if solver == "hhl":
        window = (M, M + D)
    elif solver == "vqls":
        window = (0, D)
    else:
        raise ValueError(f"Unknown quantum solver '{solver}'. Use 'hhl' or 'vqls'.")
    return RidgeSystem(A=A, b=b, dim=D, window=window)
