"""Greedy single-bit-flip local search for the BGRQ (the "BF" solver in the experiments).

Starting from a random sign vector, each iteration evaluates all ``D`` single-bit
flips in closed form and applies the one with the largest objective gain. The
search stops at a 1-flip local optimum or after ``max_iters`` flips. It is a
heuristic: it carries no global-optimality guarantee.
"""

from __future__ import annotations

import logging

import numpy as np

from .objective import BGRQProblem

logger = logging.getLogger(__name__)


def solve_bit_flip(
    problem: BGRQProblem,
    y: np.ndarray,
    max_iters: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    """Return ``(c, objective, objective_history)``.

    ``objective_history[0]`` is the objective of the random initial sign vector;
    each subsequent entry is the objective after one accepted flip.
    """
    rng = np.random.default_rng() if rng is None else rng
    B = problem.B
    y = np.asarray(y, dtype=float).ravel()
    xi = problem.alpha
    D = B.shape[1]

    q = B.T @ y
    col_norms_sq = np.linalg.norm(B, axis=0) ** 2
    c = rng.choice(np.array([-1, 1]), size=D)
    Bc = B @ c
    g = B.T @ Bc
    yBc = q @ c
    s = Bc @ Bc
    f_current = yBc**2 / (s + xi)
    history = [float(f_current)]

    n_iter = 0
    while n_iter < max_iters:
        best_gain, best_k = 0.0, -1
        for k in range(D):
            numerator = (yBc - 2 * c[k] * q[k]) ** 2
            denominator = s - 4 * c[k] * g[k] + 4 * col_norms_sq[k] + xi
            if denominator == 0:
                continue
            gain = numerator / denominator - f_current
            if gain > best_gain:
                best_gain, best_k = gain, k
        if best_gain <= 0:
            break
        c[best_k] *= -1
        Bc += 2 * c[best_k] * B[:, best_k]
        g = B.T @ Bc
        yBc = q @ c
        s = Bc @ Bc
        f_current += best_gain
        history.append(float(f_current))
        n_iter += 1

    if n_iter == max_iters:
        logger.info("Bit-flip search stopped at max_iters=%d before reaching a local optimum.", max_iters)
    return c.astype(float), float(f_current), history
