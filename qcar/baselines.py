"""Classical ridge-regression references used in the experiments."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ClosedFormRidge:
    """Normal-equation solution w* = (XᵀX + ξI)⁻¹ Xᵀy (labelled "CI" in the paper)."""

    def __init__(self, alpha: float = 0.0):
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClosedFormRidge":
        X = np.asarray(X, dtype=float)
        self.coef_ = np.linalg.inv(X.T @ X + self.alpha * np.eye(X.shape[1])) @ (X.T @ np.asarray(y, dtype=float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_


class ConjugateGradientRidge:
    """Ridge regression solved with conjugate gradients on (XᵀX + ξI) w = Xᵀy ("CGD")."""

    def __init__(self, alpha: float = 0.0, max_iter: int = 1000, tol: float = 1e-6):
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_: np.ndarray | None = None
        self.converged_: bool | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConjugateGradientRidge":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        A = X.T @ X + self.alpha * np.eye(X.shape[1])
        b = X.T @ y
        w = np.zeros(X.shape[1])
        r = b - A @ w
        p = r.copy()
        rs_old = r @ r
        self.converged_ = bool(np.sqrt(rs_old) < self.tol)
        if not self.converged_:
            for _ in range(self.max_iter):
                Ap = A @ p
                step = rs_old / (p @ Ap)
                w = w + step * p
                r = r - step * Ap
                rs_new = r @ r
                if np.sqrt(rs_new) < self.tol:
                    self.converged_ = True
                    break
                p = r + (rs_new / rs_old) * p
                rs_old = rs_new
        if not self.converged_:
            logger.warning("Conjugate gradient did not converge in %d iterations.", self.max_iter)
        self.coef_ = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_
