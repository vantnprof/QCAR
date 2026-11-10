"""Execution backends: local Aer simulation and IBM Quantum hardware.

Credentials are never stored in this repository. For hardware runs, set

    export QISKIT_IBM_TOKEN="<your API key>"
    export QISKIT_IBM_INSTANCE="<your instance CRN>"   # optional

or save an account once with ``QiskitRuntimeService.save_account(...)``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator

logger = logging.getLogger(__name__)


def runtime_service():
    """Return a ``QiskitRuntimeService`` using environment variables or a saved account."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("QISKIT_IBM_TOKEN")
    if token:
        return QiskitRuntimeService(
            channel=os.environ.get("QISKIT_IBM_CHANNEL", "ibm_quantum_platform"),
            token=token,
            instance=os.environ.get("QISKIT_IBM_INSTANCE") or None,
        )
    try:
        return QiskitRuntimeService()
    except Exception as error:  # no saved account
        raise RuntimeError(
            "No IBM Quantum credentials found. Set QISKIT_IBM_TOKEN (and optionally "
            "QISKIT_IBM_INSTANCE), or save an account with QiskitRuntimeService.save_account()."
        ) from error


def window_probabilities(counts: dict, start: int, end: int) -> np.ndarray:
    """Empirical probabilities of basis states ``start, ..., end-1``, renormalised.

    This implements Eq. (7) of the paper: outcomes outside the model window
    (padding, or failed HHL post-selection) are discarded and the remaining
    frequencies are renormalised. Returns zeros if no shot landed in the window.
    """
    window = np.zeros(end - start, dtype=float)
    for bitstring, count in counts.items():
        index = int(str(bitstring).replace(" ", ""), 2)
        if start <= index < end:
            window[index - start] += count
    total = window.sum()
    if total == 0:
        logger.warning("No measurement outcome fell inside the model window [%d, %d).", start, end)
        return window
    return window / total


def save_job_result(job_dir: str | os.PathLike | None, job, result) -> None:
    """Persist a raw IBM Runtime result (counts) as JSON for provenance."""
    if not job_dir:
        return
    from qiskit_ibm_runtime import RuntimeEncoder

    Path(job_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(job_dir) / f"{job.job_id()}.json", "w") as handle:
        json.dump(result, handle, cls=RuntimeEncoder)


class AerStateSampler:
    """Sample a prepared state in the computational basis with Aer (statevector method)."""

    def __init__(self, shots: int, seed: int | None = None, noise_model=None):
        self.shots = int(shots)
        self.seed = seed
        self.backend = AerSimulator(noise_model=noise_model, method="statevector", device="CPU")
        self._pm = generate_preset_pass_manager(optimization_level=0, backend=self.backend)

    def run(self, circuit: QuantumCircuit, start: int, end: int) -> np.ndarray:
        isa = self._pm.run(circuit)
        options = {"shots": self.shots}
        if self.seed is not None:
            options["seed_simulator"] = int(self.seed)
        counts = self.backend.run(isa, **options).result().get_counts()
        return window_probabilities(counts, start, end)


class IBMStateSampler:
    """Sample a prepared state on IBM Quantum hardware with SamplerV2 (no error mitigation)."""

    def __init__(self, shots: int, backend_name: str, job_dir: str | None = None, service=None):
        from qiskit_ibm_runtime import SamplerV2

        self.shots = int(shots)
        self.job_dir = job_dir
        self.service = service or runtime_service()
        self.backend = self.service.backend(backend_name)
        self.sampler = SamplerV2(mode=self.backend)
        self._pm = generate_preset_pass_manager(optimization_level=3, backend=self.backend)

    def run(self, circuit: QuantumCircuit, start: int, end: int) -> np.ndarray:
        isa = self._pm.run(circuit)
        job = self.sampler.run([(isa,)], shots=self.shots)
        logger.info("Submitted state-sampling job %s on %s", job.job_id(), self.backend.name)
        result = job.result()[0]
        save_job_result(self.job_dir, job, result)
        return window_probabilities(result.data.meas.get_counts(), start, end)
