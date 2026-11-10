"""Experiment 1: controlled synthetic instances (Table 2, Figs. 2-3).

For each seed: a Gaussian regression instance (scikit-learn ``make_regression``,
80% informative features, noise σ = 0.1) with an 80:20 split; classical
references (CI, CGD); HHL and VQLS state preparation measured with
``--state-shots`` shots; and sign recovery from the *same* measured
probabilities with exact search (ES), bit-flip local search (BF) and
Dinkelbach-QAOA (D-QAOA).

Paper settings (defaults): N = 50, D in {8, 16}, 50 seeds (0-49), ξ = 0.01,
VQLS 3 layers / 200 COBYLA iterations, 1024 state shots, QAOA p = 3,
1e5 QAOA sampling shots, top-k = 8, at most D outer iterations.

Example:
    python -m experiments.exp1_synthetic --dimension 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from qcar import QCAR, ClosedFormRidge, ConjugateGradientRidge, ideal_probabilities, nsd

from .common import Timer, default_output_dir, model_seed, save_rows, setup_logging, write_run_info

logger = logging.getLogger("exp1")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dimension", type=int, default=8, help="Feature dimension D.")
    p.add_argument("--n-samples", type=int, default=50, help="Number of samples N (before the 80:20 split).")
    p.add_argument("--num-runs", type=int, default=50, help="Number of seeds.")
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.01, help="Ridge parameter ξ.")
    p.add_argument("--vqls-layers", type=int, default=3)
    p.add_argument("--vqls-max-iters", type=int, default=200)
    p.add_argument("--state-shots", type=int, default=1024, help="Shots K for estimating p̂.")
    p.add_argument("--qaoa-reps", type=int, default=3, help="QAOA depth p.")
    p.add_argument("--qaoa-shots", type=int, default=100_000, help="Shots for sampling each optimised QAOA circuit.")
    p.add_argument("--top-k", type=int, default=8, help="Most frequent bitstrings considered per Dinkelbach step.")
    p.add_argument("--cgd-max-iters", type=int, default=1000)
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def run_seed(seed: int, args) -> list[dict]:
    D = args.dimension
    X, y = make_regression(n_samples=args.n_samples, n_features=D, n_informative=int(0.8 * D),
                           noise=0.1, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    rows = []

    with Timer() as t:
        ci = ClosedFormRidge(args.alpha).fit(X_train, y_train)
    w_star = ci.coef_
    rows.append(dict(model="CI", run_seed=seed, test_r2=r2_score(y_test, ci.predict(X_test)), nsd=0.0,
                     runtime=t.elapsed, coef=w_star, prob_star=ideal_probabilities(w_star)))

    with Timer() as t:
        cgd = ConjugateGradientRidge(args.alpha, max_iter=args.cgd_max_iters, tol=args.tolerance).fit(X_train, y_train)
    rows.append(dict(model="CGD", run_seed=seed, test_r2=r2_score(y_test, cgd.predict(X_test)),
                     nsd=nsd(cgd.coef_, w_star), runtime=t.elapsed, coef=cgd.coef_))

    for index, solver in enumerate(["vqls", "hhl"]):
        label = solver.upper()
        common = dict(alpha=args.alpha, quantum_solver=solver, vqls_layers=args.vqls_layers,
                      vqls_max_iters=args.vqls_max_iters, state_shots=args.state_shots, tol=args.tolerance)

        # Exact sign search (also produces the measured probabilities reused below).
        with Timer() as t:
            es = QCAR(sign_solver="exhaustive", seed=model_seed(seed, 10 * index), **common).fit(X_train, y_train)
        prob, optimal = es.prob_, es.bgrq_value_
        rows.append(dict(model=f"{label}+QCAR (ES)", run_seed=seed, test_r2=r2_score(y_test, es.predict(X_test)),
                         nsd=nsd(es.coef_, w_star), runtime=t.elapsed,
                         quantum_training_time=es.quantum_training_time_, coef=es.coef_, w_prob=prob,
                         bgrq_value=optimal, optimal_bgrq=optimal, objective_ratio=1.0, objective_history=[1.0]))

        for offset, (name, extra) in enumerate([
            ("BF", dict(sign_solver="bit_flip", max_iters=D)),
            ("D-QAOA", dict(sign_solver="dqaoa", max_iters=D, qaoa_reps=args.qaoa_reps,
                            qaoa_shots=args.qaoa_shots, top_k=args.top_k)),
        ], start=1):
            with Timer() as t:
                model = QCAR(seed=model_seed(seed, 10 * index + offset), **common, **extra).fit(X_train, y_train, prob=prob)
            rows.append(dict(model=f"{label}+QCAR ({name})", run_seed=seed,
                             test_r2=r2_score(y_test, model.predict(X_test)), nsd=nsd(model.coef_, w_star),
                             runtime=t.elapsed, coef=model.coef_, w_prob=prob, bgrq_value=model.bgrq_value_,
                             optimal_bgrq=optimal, objective_ratio=model.bgrq_value_ / optimal,
                             objective_history=[v / optimal for v in model.objective_history_],
                             qaoa_objective_dict=model.qaoa_energy_history_))
    return rows


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.verbose)
    output_dir = args.output_dir or default_output_dir("exp1")
    stem = f"exp1_D{args.dimension}"
    write_run_info(output_dir / f"{stem}_run_info.json", args, "experiments.exp1_synthetic", {"experiment": "exp1_synthetic"})
    rows = []
    for seed in range(args.first_seed, args.first_seed + args.num_runs):
        rows.extend(run_seed(seed, args))
        logger.warning("Seed %d done (%d/%d).", seed, seed - args.first_seed + 1, args.num_runs)
        save_rows(rows, output_dir / f"{stem}_detailed.csv")  # checkpoint after every seed
    print(f"Results written to {output_dir / f'{stem}_detailed.csv'}")


if __name__ == "__main__":
    main()
