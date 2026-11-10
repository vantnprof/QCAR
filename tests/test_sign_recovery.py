import itertools

import numpy as np
import pytest

from qcar import QCAR, ClosedFormRidge, ideal_probabilities, nsd
from qcar.sign_recovery import bgrq_value, build_problem, optimal_scale, solve_bit_flip, solve_exhaustive


def random_problem(seed=0, N=40, D=6, alpha=0.01):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D)) @ rng.normal(size=(D, D))  # correlated features
    w = rng.normal(size=D)
    y = X @ w + 0.1 * rng.normal(size=N)
    v = np.abs(rng.normal(size=D))
    return X, y, v / np.linalg.norm(v), alpha


def test_exhaustive_matches_full_enumeration():
    X, y, v, alpha = random_problem()
    problem = build_problem(X, y, v, alpha)
    brute = max(bgrq_value(problem, np.array(c)) for c in itertools.product([-1, 1], repeat=problem.dim))
    _, value = solve_exhaustive(problem)
    assert value == pytest.approx(brute, rel=1e-12)


def test_scale_is_least_squares_optimal():
    X, y, v, alpha = random_problem(seed=1)
    problem = build_problem(X, y, v, alpha)
    c = np.sign(np.random.default_rng(2).normal(size=problem.dim))
    a_star = optimal_scale(problem, y, c)
    loss = lambda a: np.sum((X @ (a * c * v) - y) ** 2) + alpha * np.sum((a * c * v) ** 2)  # noqa: E731
    assert loss(a_star) <= min(loss(a_star + d) for d in (-1e-3, 1e-3))


def test_exact_magnitudes_recover_ridge_solution():
    X, y, _, alpha = random_problem(seed=3)
    w_star = ClosedFormRidge(alpha).fit(X, y).coef_
    model = QCAR(alpha=alpha, sign_solver="exhaustive").fit(X, y, prob=ideal_probabilities(w_star))
    assert nsd(model.coef_, w_star) < 1e-12


def test_bit_flip_never_beats_exact_search():
    for seed in range(5):
        X, y, v, alpha = random_problem(seed=seed)
        problem = build_problem(X, y, v, alpha)
        _, exact = solve_exhaustive(problem)
        c, value, history = solve_bit_flip(problem, y, max_iters=50, rng=np.random.default_rng(seed))
        assert value <= exact * (1 + 1e-12)
        assert value == pytest.approx(bgrq_value(problem, c), rel=1e-9)
        assert all(b >= a for a, b in zip(history, history[1:]))  # monotone improvement


def test_orthogonal_design_has_closed_form_signs():
    """With XᵀX diagonal the BGRQ denominator is constant, so c* = sign(q)."""
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.normal(size=(50, 6)))
    X = Q * np.sqrt(50)
    y = X @ rng.normal(size=6)
    v = np.abs(rng.normal(size=6))
    problem = build_problem(X, y, v / np.linalg.norm(v), 0.01)
    _, exact = solve_exhaustive(problem)
    assert bgrq_value(problem, np.sign(problem.q)) == pytest.approx(exact, rel=1e-10)
