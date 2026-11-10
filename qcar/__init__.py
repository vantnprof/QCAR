"""QCAR: Quantum-to-Classical Ambiguity Resolution.

Reconstructs a deployable classical ridge-regression model from computational-basis
measurements of an amplitude-encoded quantum regression state.
"""

from .baselines import ClosedFormRidge, ConjugateGradientRidge
from .metrics import ideal_probabilities, nsd, probability_fidelity
from .model import QCAR
from .sign_recovery import SIGN_SOLVERS, build_problem, solve_bgrq

__version__ = "1.0.0"

__all__ = [
    "QCAR", "ClosedFormRidge", "ConjugateGradientRidge", "SIGN_SOLVERS",
    "build_problem", "solve_bgrq", "nsd", "ideal_probabilities", "probability_fidelity",
]
