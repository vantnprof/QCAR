"""Dinkelbach–QAOA heuristic for the BGRQ (Algorithm 3 in the paper).

Each Dinkelbach iteration solves the Ising subproblem ``max_c cᵀ Q(λ) c`` with
``Q(λ) = F − λG`` (normalised by ``max|Q|``) using QAOA:

1. The QAOA parameters are optimised with COBYLA on a *local* matrix-product-state
   Aer simulator through Qiskit Runtime's local mode, for both backends. Each
   energy is a shot-based estimate with the Runtime default precision 1/64
   (4096 shots), not an exact expectation value.
2. The optimised circuit is sampled once with ``shots`` shots, either on the
   local simulator (``backend="simulator"``) or on IBM hardware
   (``backend="ibm"``).
3. Among the ``top_k`` most frequent bitstrings, the one with the largest
   ``cᵀ Q c`` updates ``λ``.

The loop stops when ``|λ⁺ − λ| < tolerance`` or after ``max_iters`` iterations
and returns the last iterate. Because QAOA is heuristic, Dinkelbach's global
convergence guarantee does not apply.
"""

from __future__ import annotations

import logging

import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2, SamplerV2, Session
from scipy.optimize import minimize

from ..backends import runtime_service, save_job_result
from .objective import BGRQProblem, bgrq_value

logger = logging.getLogger(__name__)


def ising_hamiltonian(Q: np.ndarray) -> SparsePauliOp:
    """Cost Hamiltonian whose ground states maximise ``cᵀ Q c`` (diagonal terms are constant).

    ``H = −Σ_{i<j} Q_ij Z_i Z_j``. Pauli labels are indexed by string position, and
    sampled bitstrings are decoded with the same convention, so position ``i`` of a
    label and of a bitstring both refer to variable ``c_i``.
    """
    n = Q.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.isclose(Q[i, j], 0.0):
                continue
            label = ["I"] * n
            label[i] = "Z"
            label[j] = "Z"
            terms.append(("".join(label), -Q[i, j]))
    if not terms:
        terms = [("I" * n, 0.0)]
    return SparsePauliOp.from_list(terms)


def bitstring_to_signs(bitstring: str) -> np.ndarray:
    """Map a measured bitstring to a sign vector (``'0' -> +1``, ``'1' -> -1``)."""
    return np.array([1.0 if bit == "0" else -1.0 for bit in bitstring])


class DinkelbachQAOASolver:
    def __init__(
        self,
        reps: int = 3,
        shots: int = 100_000,
        backend: str = "simulator",
        backend_name: str | None = None,
        max_iters: int = 10,
        tolerance: float = 1e-6,
        top_k: int = 1,
        noise_model=None,
        job_dir: str | None = None,
        rng: np.random.Generator | None = None,
        service=None,
    ):
        self.reps = int(reps)
        self.shots = int(shots)
        self.max_iters = int(max_iters)
        self.tolerance = float(tolerance)
        self.top_k = int(top_k)
        self.job_dir = job_dir
        self.rng = np.random.default_rng() if rng is None else rng

        self.local_sim = AerSimulator(noise_model=noise_model, method="matrix_product_state", device="CPU")
        self._pm_train = generate_preset_pass_manager(optimization_level=1, backend=self.local_sim)
        if backend == "simulator":
            self.backend = AerSimulator(noise_model=noise_model, method="matrix_product_state", device="CPU")
            self._pm_sample = generate_preset_pass_manager(optimization_level=1, backend=self.backend)
            self._sampler = SamplerV2(mode=self.backend)
        elif backend == "ibm":
            if not backend_name:
                raise ValueError("backend_name (e.g. 'ibm_torino') is required for backend='ibm'.")
            self.backend = (service or runtime_service()).backend(backend_name)
            self._pm_sample = generate_preset_pass_manager(optimization_level=3, backend=self.backend)
            self._sampler = SamplerV2(mode=self.backend)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'simulator' or 'ibm'.")
        self.backend_kind = backend

        #: QAOA training energies per Dinkelbach iteration: {iteration: [E_1, E_2, ...]}
        self.qaoa_energy_history: dict[int, list[float]] = {}

    # ------------------------------------------------------------------
    def solve(self, problem: BGRQProblem) -> tuple[np.ndarray, float, list[float]]:
        """Return ``(c, final_ratio, ratio_history)``; ``ratio_history[0]`` is the random start."""
        F, G, alpha = problem.F, problem.G, problem.alpha
        D = problem.dim
        c = 1.0 - 2.0 * self.rng.integers(2, size=D)
        lam = bgrq_value(problem, c)
        history = [float(lam)]
        params = None
        self.qaoa_energy_history = {}

        for iteration in range(self.max_iters):
            Q = F - lam * G
            Q = Q / np.max(np.abs(Q))
            if not np.allclose(Q, Q.T):
                raise ValueError("Dinkelbach matrix Q is not symmetric.")

            candidates, params, energies = self._solve_ising(Q, params)
            best_c = max(candidates, key=lambda cand: float(cand @ Q @ cand))
            lam_next = bgrq_value(problem, best_c)
            history.append(float(lam_next))
            self.qaoa_energy_history[iteration] = energies

            converged = abs(lam_next - lam) < self.tolerance
            c, lam = best_c, lam_next
            if converged:
                logger.info("Dinkelbach-QAOA converged after %d iterations.", iteration + 1)
                break

        return c, bgrq_value(problem, c), history

    def _next_seed(self) -> int:
        return int(self.rng.integers(2**31 - 1))

    # ------------------------------------------------------------------
    def _solve_ising(self, Q: np.ndarray, init_params):
        hamiltonian = ising_hamiltonian(Q)
        ansatz = self._pm_train.run(QAOAAnsatz(cost_operator=hamiltonian, reps=self.reps))
        ansatz.measure_all()

        if init_params is None:
            # Random start; later Dinkelbach iterations warm-start from the previous optimum.
            gamma = self.rng.uniform(0, 2 * np.pi)
            beta = self.rng.uniform(0, np.pi)
            init_params = [beta, gamma] * self.reps

        energies: list[float] = []
        layout_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

        def energy(theta, estimator):
            estimator.options.simulator.seed_simulator = self._next_seed()
            value = float(estimator.run([(ansatz, layout_hamiltonian, theta)]).result()[0].data.evs)
            energies.append(value)
            return value

        with Session(backend=self.local_sim) as session:
            estimator = EstimatorV2(mode=session)
            opt = minimize(energy, init_params, args=(estimator,), method="COBYLA")

        circuit = self._pm_sample.run(ansatz.assign_parameters(opt.x))
        if self.backend_kind == "simulator":
            self._sampler.options.simulator.seed_simulator = self._next_seed()
        job = self._sampler.run([(circuit,)], shots=self.shots)
        result = job.result()[0]
        if self.backend_kind == "ibm":
            logger.info("QAOA sampling job %s on %s", job.job_id(), self.backend.name)
            save_job_result(self.job_dir, job, result)

        counts = result.data.meas.get_counts()
        most_frequent = sorted(counts.items(), key=lambda item: item[1], reverse=True)[: self.top_k]
        candidates = [bitstring_to_signs(bits) for bits, _ in most_frequent]
        return candidates, opt.x, energies
