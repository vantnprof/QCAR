import numpy as np
from qiskit.quantum_info import Statevector

from qcar import QCAR, ClosedFormRidge, ideal_probabilities, nsd, probability_fidelity
from qcar.linear_system import build_ridge_system
from qcar.quantum import HHL, VQLS


def small_regression(seed=0, N=40, D=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D))
    y = X @ np.array([3.0, -2.0, 1.0])[:D] + 0.05 * rng.normal(size=N)
    return X, y


def exact_window(circuit, window):
    """Exact distribution of the *measured* bits (classical-bit order), restricted to the model window."""
    measured = sorted(
        (circuit.find_bit(inst.clbits[0]).index, circuit.find_bit(inst.qubits[0]).index)
        for inst in circuit.data if inst.operation.name == "measure"
    )
    state = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
    p = state.probabilities(qargs=[qubit for _, qubit in measured])[window[0]:window[1]]
    return p / p.sum()


def test_padded_system_layout():
    X, y = small_regression()
    system = build_ridge_system(X, y, 0.01, "hhl")
    assert system.A.shape == (4, 4) and system.window == (4, 7)
    assert np.isclose(np.linalg.norm(system.b), 1.0) and system.A[3, 3] == 1.0


def test_hhl_exact_for_representable_eigenvalues():
    A, b = np.diag([1.0, 2.0, 3.0, 4.0]), np.full(4, 0.5)
    x = np.linalg.solve(A, b)
    p = exact_window(HHL().run(A, b), (4, 8))
    assert probability_fidelity(p, x**2 / np.sum(x**2)) > 1 - 1e-9


def test_hhl_state_matches_ridge_direction():
    X, y = small_regression()
    w_star = ClosedFormRidge(0.01).fit(X, y).coef_
    system = build_ridge_system(X, y, 0.01, "hhl")
    p = exact_window(HHL().run(system.A, system.b), system.window)
    assert probability_fidelity(p, ideal_probabilities(w_star)) > 0.95


def test_vqls_state_matches_ridge_direction():
    X, y = small_regression()
    w_star = ClosedFormRidge(0.01).fit(X, y).coef_
    system = build_ridge_system(X, y, 0.01, "vqls")
    circuit = VQLS(layers=3, maxiter=300, rng=np.random.default_rng(0)).run(system.A, system.b)
    p = exact_window(circuit, system.window)
    assert probability_fidelity(p, ideal_probabilities(w_star)) > 0.95


def test_end_to_end_reproducible_with_seed():
    X, y = small_regression()
    w_star = ClosedFormRidge(0.01).fit(X, y).coef_
    fits = [QCAR(alpha=0.01, quantum_solver="vqls", vqls_max_iters=200, sign_solver="exhaustive",
                 state_shots=4096, seed=7).fit(X, y) for _ in range(2)]
    assert np.array_equal(fits[0].prob_, fits[1].prob_)
    assert nsd(fits[0].coef_, w_star) < 0.1
