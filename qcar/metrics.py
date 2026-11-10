"""Evaluation metrics used in the paper (Section 4.1.3)."""

from __future__ import annotations

import numpy as np


def nsd(w_hat: np.ndarray, w_star: np.ndarray) -> float:
    """Normalised squared deviation ‖ŵ − w*‖² / ‖w*‖²."""
    w_hat, w_star = np.asarray(w_hat, dtype=float), np.asarray(w_star, dtype=float)
    return float(np.sum((w_hat - w_star) ** 2) / np.sum(w_star**2))


def ideal_probabilities(w_star: np.ndarray) -> np.ndarray:
    """p* = (w* ⊙ w*) / ‖w*‖²."""
    w_star = np.asarray(w_star, dtype=float)
    return w_star**2 / np.sum(w_star**2)


def probability_fidelity(p: np.ndarray, q: np.ndarray) -> float:
    """Squared Bhattacharyya coefficient (Σ_j sqrt(p_j q_j))², labelled "fidelity" in the paper.

    This is a similarity between computational-basis distributions, not a quantum
    state fidelity; it is blind to coordinate signs.
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    return float(np.sum(np.sqrt(p * q)) ** 2)
