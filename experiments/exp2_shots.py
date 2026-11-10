"""Experiment 2: finite-shot sensitivity on real data (Tables 3-4, Figs. 4-5).

For every run and every shot budget K: VQLS is trained, its state is sampled
with K shots, and signs are recovered with bit-flip search (BF) and with
Dinkelbach-QAOA (D-QAOA) from the same measured probabilities. D-QAOA samples
each optimised QAOA circuit with the same K shots and keeps the top-D bitstrings.

Paper settings (defaults): ξ = 0.01, VQLS 3 layers / 200 COBYLA iterations,
QAOA p = 3, K in {128, 512, 1024, 4096, 8192}, 30 runs, at most D Dinkelbach iterations.
QAOA circuits use D qubits (16 for Diabetes, 32 for Student) and are simulated
with Aer's matrix-product-state method.

Example:
    python -m experiments.exp2_shots --dataset diabetes
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sklearn.metrics import r2_score

from qcar import QCAR, ClosedFormRidge, ideal_probabilities, nsd
from qcar.datasets import load_processed

from .common import REPO_ROOT, Timer, default_output_dir, model_seed, save_rows, setup_logging, write_run_info

logger = logging.getLogger("exp2")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["diabetes", "student"], required=True)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="Directory written by prepare_data.")
    p.add_argument("--shots", type=int, nargs="+", default=[128, 512, 1024, 4096, 8192])
    p.add_argument("--num-runs", type=int, default=30)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--vqls-layers", type=int, default=3)
    p.add_argument("--vqls-max-iters", type=int, default=200)
    p.add_argument("--qaoa-reps", type=int, default=3)
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def run_once(data, seed: int, shots: int, args) -> list[dict]:
    X_train, X_test, y_train, y_test = data
    D = X_train.shape[1]
    rows = []

    with Timer() as t:
        ci = ClosedFormRidge(args.alpha).fit(X_train, y_train)
    w_star = ci.coef_
    rows.append(dict(model="CI", run_seed=seed, shots=shots, test_r2=r2_score(y_test, ci.predict(X_test)),
                     nsd=0.0, runtime=t.elapsed, w=w_star, prob=ideal_probabilities(w_star)))

    common = dict(alpha=args.alpha, quantum_solver="vqls", vqls_layers=args.vqls_layers,
                  vqls_max_iters=args.vqls_max_iters, state_shots=shots, max_iters=D, tol=args.tolerance)
    with Timer() as t:
        bf = QCAR(sign_solver="bit_flip", seed=model_seed(seed, shots), **common).fit(X_train, y_train)
    rows.append(dict(model="VQLS+QCAR (BF)", run_seed=seed, shots=shots, test_r2=r2_score(y_test, bf.predict(X_test)),
                     nsd=nsd(bf.coef_, w_star), runtime=t.elapsed, quantum_training_time=bf.quantum_training_time_,
                     w=bf.coef_, prob=bf.prob_, objective_history=bf.objective_history_))

    with Timer() as t:
        dq = QCAR(sign_solver="dqaoa", qaoa_reps=args.qaoa_reps, qaoa_shots=shots, top_k=D,
                  seed=model_seed(seed, shots + 1), **common).fit(X_train, y_train, prob=bf.prob_)
    rows.append(dict(model="VQLS+QCAR (D-QAOA)", run_seed=seed, shots=shots, test_r2=r2_score(y_test, dq.predict(X_test)),
                     nsd=nsd(dq.coef_, w_star), runtime=t.elapsed, w=dq.coef_, prob=dq.prob_,
                     objective_history=dq.objective_history_, qaoa_objective_dict=dq.qaoa_energy_history_))
    return rows


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.verbose)
    data = load_processed(args.data_dir / args.dataset)
    output_dir = args.output_dir or default_output_dir("exp2")
    write_run_info(output_dir / f"{args.dataset}_exp2_run_info.json", args, "experiments.exp2_shots",
                   {"experiment": "exp2_shots", "dimension": int(data[0].shape[1])})
    rows = []
    for seed in range(args.first_seed, args.first_seed + args.num_runs):
        for shots in args.shots:
            rows.extend(run_once(data, seed, shots, args))
        logger.warning("Run %d done (%d/%d).", seed, seed - args.first_seed + 1, args.num_runs)
        save_rows(rows, output_dir / f"{args.dataset}_exp2_results.csv")
    print(f"Results written to {output_dir / f'{args.dataset}_exp2_results.csv'}")


if __name__ == "__main__":
    main()
