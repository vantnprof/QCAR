import numpy as np
import pytest

from qcar.sign_recovery import build_problem, solve_bgrq, solve_exhaustive
from qcar.sign_recovery.dinkelbach_qaoa import bitstring_to_signs, ising_hamiltonian


def test_hamiltonian_ground_state_maximises_quadratic_form():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(4, 4))
    Q = Q + Q.T
    H = ising_hamiltonian(Q).to_matrix().real.diagonal()
    # Qiskit basis index k <-> bitstring format(k, '04b'), whose string positions match Pauli label positions.
    best = int(np.argmin(H))
    c = bitstring_to_signs(format(best, "04b"))
    brute = max(np.array(s) @ Q @ np.array(s) for s in np.ndindex(*(2,) * 4) for s in [2 * np.array(s) - 1])
    assert c @ Q @ c == pytest.approx(brute)


@pytest.mark.slow
def test_dqaoa_small_instance_and_determinism():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 3.0])
    v = np.abs(rng.normal(size=4))
    problem = build_problem(X, y, v / np.linalg.norm(v), 0.01)
    _, exact = solve_exhaustive(problem)
    runs = [solve_bgrq(problem, y, method="dqaoa", rng=np.random.default_rng(3), max_iters=4,
                       qaoa_reps=1, qaoa_shots=4096, top_k=4) for _ in range(2)]
    assert runs[0].value == pytest.approx(exact, rel=1e-9)
    assert runs[0].history == runs[1].history
