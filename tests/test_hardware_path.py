"""Exercise the IBM hardware code path offline with Qiskit's FakeTorino (Torino's coupling map and noise)."""

import json

import numpy as np
import pytest
from qiskit_ibm_runtime.fake_provider import FakeTorino

from qcar import QCAR, ClosedFormRidge, ideal_probabilities, probability_fidelity
from qcar.backends import IBMStateSampler, window_probabilities
from qcar.linear_system import build_ridge_system
from qcar.quantum import VQLS


class FakeService:
    """Stands in for QiskitRuntimeService: returns FakeTorino for any backend name."""

    def backend(self, name):
        return FakeTorino()


def test_window_probabilities_discards_outside_and_renormalises():
    counts = {"000": 10, "001": 30, "101": 60}  # index 5 lies outside the window [0, 4)
    p = window_probabilities(counts, 0, 4)
    assert np.allclose(p, [0.25, 0.75, 0.0, 0.0])


def test_ibm_state_sampler_on_fake_torino(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = X @ np.array([2.0, -1.0, 0.5, 1.5])
    system = build_ridge_system(X, y, 0.01, "vqls")
    circuit = VQLS(layers=3, maxiter=300, rng=np.random.default_rng(1)).run(system.A, system.b)
    sampler = IBMStateSampler(4096, backend_name="ibm_torino", job_dir=str(tmp_path), service=FakeService())
    p = sampler.run(circuit, *system.window)
    w_star = ClosedFormRidge(0.01).fit(X, y).coef_
    assert np.isclose(p.sum(), 1.0)
    assert probability_fidelity(p, ideal_probabilities(w_star)) > 0.8  # noisy device model
    saved = list(tmp_path.glob("*.json"))
    assert len(saved) == 1 and "SamplerPubResult" in saved[0].read_text()


@pytest.mark.slow
def test_dqaoa_hardware_path_on_fake_torino(tmp_path, monkeypatch):
    import qcar.sign_recovery.dinkelbach_qaoa as dq

    monkeypatch.setattr(dq, "runtime_service", lambda: FakeService())
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 3.0])
    w_star = ClosedFormRidge(0.01).fit(X, y).coef_
    model = QCAR(alpha=0.01, sign_solver="dqaoa", backend="ibm", backend_name="ibm_torino", qaoa_reps=1,
                 qaoa_shots=2048, max_iters=2, top_k=4, job_dir=str(tmp_path), seed=0)
    model.fit(X, y, prob=ideal_probabilities(w_star))
    assert len(list(tmp_path.glob("*.json"))) == len(model.objective_history_) - 1  # one sampling job per step
    assert all(json.loads(f.read_text())["__type__"] == "SamplerPubResult" for f in tmp_path.glob("*.json"))
