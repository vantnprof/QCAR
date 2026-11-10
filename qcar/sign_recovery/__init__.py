"""Sign recovery for QCAR: exact search, bit-flip local search and Dinkelbach–QAOA."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bit_flip import solve_bit_flip
from .exhaustive import solve_exhaustive
from .objective import BGRQProblem, bgrq_value, build_problem, optimal_scale, reconstruct_weights

SIGN_SOLVERS = ("exhaustive", "bit_flip", "dqaoa")


@dataclass
class SignSolution:
    signs: np.ndarray
    value: float
    history: list[float] = field(default_factory=list)
    qaoa_energy_history: dict | None = None


def solve_bgrq(
    problem: BGRQProblem,
    y: np.ndarray,
    method: str = "bit_flip",
    rng: np.random.Generator | None = None,
    max_iters: int = 1000,
    tol: float = 1e-6,
    qaoa_reps: int = 3,
    qaoa_shots: int = 100_000,
    top_k: int = 1,
    backend: str = "simulator",
    backend_name: str | None = None,
    job_dir: str | None = None,
    noise_model=None,
) -> SignSolution:
    """Solve the BGRQ with the chosen method.

    ``history`` holds the BGRQ objective along the solver's iterations: a single
    value for ``exhaustive``, the objective after each accepted flip for
    ``bit_flip``, and the Dinkelbach ratio sequence λ for ``dqaoa``.
    """
    rng = np.random.default_rng() if rng is None else rng
    if method == "exhaustive":
        c, value = solve_exhaustive(problem)
        return SignSolution(c, value, [value])
    if method == "bit_flip":
        c, value, history = solve_bit_flip(problem, y, max_iters=max_iters, rng=rng)
        return SignSolution(c, value, history)
    if method == "dqaoa":
        from .dinkelbach_qaoa import DinkelbachQAOASolver  # Qiskit Runtime import is deferred

        solver = DinkelbachQAOASolver(
            reps=qaoa_reps, shots=qaoa_shots, backend=backend, backend_name=backend_name,
            max_iters=max_iters, tolerance=tol, top_k=top_k, noise_model=noise_model,
            job_dir=job_dir, rng=rng,
        )
        c, value, history = solver.solve(problem)
        return SignSolution(c, value, history, solver.qaoa_energy_history)
    raise ValueError(f"Unknown sign solver '{method}'. Choose from {SIGN_SOLVERS}.")


__all__ = [
    "BGRQProblem", "SignSolution", "SIGN_SOLVERS", "bgrq_value", "build_problem",
    "optimal_scale", "reconstruct_weights", "solve_bgrq", "solve_bit_flip", "solve_exhaustive",
]
