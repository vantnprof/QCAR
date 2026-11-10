"""HHL circuit for the (padded) ridge system ``A w = b``.

The construction follows Qiskit's former ``HHL`` class: amplitude-encode ``b``,
quantum phase estimation with ``exp(iAt)`` (an exact unitary, see
``HamiltonianMatrix``), exact reciprocal rotation on a flag qubit, and
uncomputation. The returned circuit measures the solution register and the
flag qubit. Post-selecting ``flag = 1`` (done by the state sampler through the
measurement window) yields the normalised solution state.
"""

from __future__ import annotations

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PhaseEstimation

from ._qiskit import ExactReciprocal, HamiltonianMatrix


def _eigenvalue_scaling(n_l: int, lambda_min: float, lambda_max: float) -> float:
    """Scaling factor so that ``lambda_min`` is represented exactly on ``n_l`` bits."""
    formatstr = "#0" + str(n_l + 2) + "b"
    lambda_min_tilde = np.abs(lambda_min * (2**n_l - 1) / lambda_max)
    if np.abs(lambda_min_tilde - 1) < 1e-7:
        lambda_min_tilde = 1
    binstr = format(int(lambda_min_tilde), formatstr)[2::]
    return sum(int(char) / (2 ** (i + 1)) for i, char in enumerate(binstr))


class HHL:
    def __init__(self, epsilon: float = 1e-2, neg_vals: bool = True):
        self.epsilon = epsilon
        self.neg_vals = neg_vals

    def run(self, A: np.ndarray, b: np.ndarray) -> QuantumCircuit:
        A = np.asarray(A)
        b = np.asarray(b, dtype=float)
        nb = int(np.log2(len(b)))
        if A.shape != (2**nb, 2**nb):
            raise ValueError("A must be square with dimension 2**n matching b.")
        if not np.allclose(A, A.conj().T):
            raise ValueError("A must be Hermitian.")

        vector_circuit = QuantumCircuit(nb)
        vector_circuit.initialize(b / np.linalg.norm(b), list(range(nb)), None)

        matrix_circuit = HamiltonianMatrix(A, evolution_time=2 * np.pi)
        matrix_circuit.tolerance = self.epsilon / 6
        kappa = matrix_circuit.condition_bounds()[1]
        neg = int(self.neg_vals)
        nl = max(nb + 1, int(np.ceil(np.log2(kappa + 1)))) + neg

        lambda_min, lambda_max = matrix_circuit.eigs_bounds()
        delta = _eigenvalue_scaling(nl - neg, lambda_min, lambda_max)
        matrix_circuit.evolution_time = 2 * np.pi * delta / lambda_min / (2**neg)
        reciprocal = ExactReciprocal(nl, delta, neg_vals=self.neg_vals)

        qb = QuantumRegister(nb)  # right-hand side / solution
        ql = QuantumRegister(nl)  # eigenvalue register
        qf = QuantumRegister(1)   # flag qubit
        qc = QuantumCircuit(qb, ql, qf)
        qc.append(vector_circuit, qb[:])
        qpe = PhaseEstimation(nl, matrix_circuit)
        qc.append(qpe, ql[:] + qb[:])
        qc.append(reciprocal, ql[::-1] + [qf[0]])
        qc.append(qpe.inverse(), ql[:] + qb[:])

        # Measure solution register (low bits) and flag (high bit): index = flag * 2**nb + x.
        creg = ClassicalRegister(nb + 1)
        qc.add_bits(creg)
        qc.measure(qb, creg[0:nb])
        qc.measure(qf, creg[nb:])
        return qc
