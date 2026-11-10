"""QCAR estimator: quantum regression state preparation + objective-aligned sign/scale recovery."""

from __future__ import annotations

import logging
import time

import numpy as np

from .backends import AerStateSampler, IBMStateSampler
from .linear_system import build_ridge_system
from .quantum import HHL, VQLS
from .sign_recovery import solve_bgrq
from .sign_recovery.objective import build_problem, optimal_scale, reconstruct_weights

logger = logging.getLogger(__name__)


class QCAR:
    """Quantum-to-Classical Ambiguity Resolution for ridge regression.

    Pipeline (Section 3 of the paper):

    1. Build the padded ridge system and prepare its solution state with HHL or VQLS.
    2. Measure the state ``state_shots`` times in the computational basis and form
       the magnitude vector ``v̂ = sqrt(p̂)`` on the ``D`` model coordinates.
    3. Recover signs by solving the BGRQ (``sign_solver``) and the scale in closed form.

    Parameters
    ----------
    alpha : ridge parameter ξ.
    quantum_solver : ``"hhl"`` or ``"vqls"``.
    sign_solver : ``"exhaustive"`` (exact), ``"bit_flip"`` (greedy local search) or ``"dqaoa"``.
    state_shots : shots used to estimate p̂ (Stage 1).
    qaoa_shots : shots for sampling each optimised QAOA circuit (defaults to ``state_shots``).
    qaoa_reps : QAOA depth p.
    max_iters : maximum bit flips (BF) or Dinkelbach iterations (D-QAOA).
    top_k : number of most-frequent QAOA bitstrings considered per Dinkelbach iteration.
    backend : ``"simulator"`` (Aer) or ``"ibm"`` (IBM Quantum hardware for state and QAOA sampling).
    backend_name : IBM backend name, e.g. ``"ibm_torino"`` (required for ``backend="ibm"``).
    job_dir : directory where raw IBM job results are saved (hardware runs only).
    seed : seed for all classical randomness and for Aer sampling.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        quantum_solver: str = "hhl",
        sign_solver: str = "bit_flip",
        state_shots: int = 1024,
        qaoa_shots: int | None = None,
        qaoa_reps: int = 3,
        vqls_layers: int = 3,
        vqls_max_iters: int = 100,
        max_iters: int = 1000,
        tol: float = 1e-6,
        top_k: int = 1,
        backend: str = "simulator",
        backend_name: str | None = None,
        job_dir: str | None = None,
        noise_model=None,
        seed: int | None = None,
    ):
        self.alpha = float(alpha)
        self.quantum_solver = quantum_solver
        self.sign_solver = sign_solver
        self.state_shots = int(state_shots)
        self.qaoa_shots = int(qaoa_shots) if qaoa_shots is not None else int(state_shots)
        self.qaoa_reps = int(qaoa_reps)
        self.vqls_layers = int(vqls_layers)
        self.vqls_max_iters = int(vqls_max_iters)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.top_k = int(top_k)
        self.backend = backend
        self.backend_name = backend_name
        self.job_dir = job_dir
        self.noise_model = noise_model
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Fitted attributes
        self.coef_: np.ndarray | None = None
        self.prob_: np.ndarray | None = None
        self.signs_: np.ndarray | None = None
        self.scale_: float | None = None
        self.bgrq_value_: float | None = None
        self.objective_history_: list[float] | None = None
        self.qaoa_energy_history_: dict | None = None
        self.quantum_training_time_: float | None = None

    # ------------------------------------------------------------------
    def measure_state(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Stages 0–1: prepare the regression state and return renormalised probabilities p̂."""
        system = build_ridge_system(X, y, self.alpha, self.quantum_solver)
        start = time.time()
        if self.quantum_solver == "hhl":
            circuit = HHL().run(system.A, system.b)
        else:
            circuit = VQLS(layers=self.vqls_layers, maxiter=self.vqls_max_iters, rng=self._rng).run(system.A, system.b)
        self.quantum_training_time_ = time.time() - start

        if self.backend == "simulator":
            sampler = AerStateSampler(self.state_shots, seed=int(self._rng.integers(2**31 - 1)), noise_model=self.noise_model)
        elif self.backend == "ibm":
            sampler = IBMStateSampler(self.state_shots, backend_name=self.backend_name, job_dir=self.job_dir)
        else:
            raise ValueError(f"Unknown backend '{self.backend}'.")
        return sampler.run(circuit, *system.window)

    def fit(self, X: np.ndarray, y: np.ndarray, prob: np.ndarray | None = None) -> "QCAR":
        """Fit QCAR. If ``prob`` (measured probabilities p̂) is given, Stages 0–1 are skipped."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.prob_ = np.asarray(prob, dtype=float) if prob is not None else self.measure_state(X, y)

        magnitudes = np.sqrt(self.prob_)
        problem = build_problem(X, y, magnitudes, self.alpha)
        solution = solve_bgrq(
            problem,
            y,
            method=self.sign_solver,
            rng=self._rng,
            max_iters=self.max_iters,
            tol=self.tol,
            qaoa_reps=self.qaoa_reps,
            qaoa_shots=self.qaoa_shots,
            top_k=self.top_k,
            backend=self.backend,
            backend_name=self.backend_name,
            job_dir=self.job_dir,
            noise_model=self.noise_model,
        )
        self.signs_ = solution.signs
        self.bgrq_value_ = solution.value
        self.objective_history_ = solution.history
        self.qaoa_energy_history_ = solution.qaoa_energy_history
        self.scale_ = optimal_scale(problem, y, solution.signs)
        self.coef_ = reconstruct_weights(problem, y, solution.signs, magnitudes)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(X, dtype=float) @ self.coef_
