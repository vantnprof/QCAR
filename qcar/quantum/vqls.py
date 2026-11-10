"""Variational quantum linear solver (VQLS) used to prepare the regression state.

The ansatz is a hardware-efficient circuit of ``layers`` blocks, each made of
single-qubit ``R_y`` rotations followed by an even/odd ``CZ`` ladder. It produces
real amplitudes, as QCAR assumes.

Training note (reported in the paper's setup): the global cost

    C_G(θ) = 1 − |⟨b|A|x(θ)⟩|² / ⟨x(θ)|A†A|x(θ)⟩

is evaluated *exactly* from the statevector of ``|x(θ)⟩``, with ``A`` applied
classically, and minimised with COBYLA. Only the trained state-preparation
circuit is then executed (simulator or hardware) and measured in the
computational basis.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize


class VQLS:
    def __init__(self, layers: int = 3, maxiter: int = 200, optimizer: str = "COBYLA",
                 rng: np.random.Generator | None = None):
        self.layers = int(layers)
        self.maxiter = int(maxiter)
        self.optimizer = optimizer
        self.rng = np.random.default_rng() if rng is None else rng
        self.result_ = None

    def ansatz(self, theta: np.ndarray, num_qubits: int) -> QuantumCircuit:
        """Bound hardware-efficient ansatz ``|x(θ)⟩ = U(θ)|0…0⟩``."""
        qc = QuantumCircuit(num_qubits)
        k = 0
        for _ in range(self.layers):
            for q in range(num_qubits):
                qc.ry(float(theta[k]), q)
                k += 1
            for q in range(0, num_qubits - 1, 2):
                qc.cz(q, q + 1)
            for q in range(1, num_qubits - 1, 2):
                qc.cz(q, q + 1)
        return qc

    @staticmethod
    def global_cost(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
        psi = A @ x
        norm_psi = np.vdot(psi, psi).real
        if norm_psi == 0:
            return 1.0
        return float(1.0 - abs(np.vdot(b, psi)) ** 2 / norm_psi)

    def run(self, A: np.ndarray, b: np.ndarray) -> QuantumCircuit:
        """Train the ansatz for ``A x = b`` and return the measured state-preparation circuit."""
        A = np.asarray(A)
        dim = A.shape[0]
        num_qubits = int(np.log2(dim))
        if A.shape != (dim, dim) or 2**num_qubits != dim:
            raise ValueError("A must be square with a power-of-two dimension.")
        b = np.asarray(b, dtype=complex)
        if np.linalg.norm(b) == 0:
            raise ValueError("b must be non-zero.")
        b = b / np.linalg.norm(b)

        def cost(theta):
            x = Statevector.from_instruction(self.ansatz(theta, num_qubits)).data
            return self.global_cost(A, b, x)

        theta0 = self.rng.uniform(0, 2 * np.pi, self.layers * num_qubits)
        self.result_ = minimize(cost, theta0, method=self.optimizer, options={"maxiter": self.maxiter})

        circuit = QuantumCircuit(num_qubits)
        circuit.append(self.ansatz(self.result_.x, num_qubits).to_instruction(), range(num_qubits))
        circuit.measure_all()
        return circuit
