"""Experiment 3: Aer simulator vs. IBM Quantum hardware on Diabetes (Table 5, Fig. 6).

Per run, two independent QCAR fits are made (BF and D-QAOA). Each trains VQLS
classically, then samples the trained 4-qubit state on the chosen backend.
D-QAOA additionally samples every optimised 16-qubit QAOA circuit on that
backend. QAOA parameters are always optimised on a local simulator; only the
final sampling runs on hardware. No readout mitigation or ZNE is applied.

Paper settings (defaults): 10 runs, 4096 shots (state and QAOA sampling),
QAOA p = 3, top-D bitstrings, at most D = 16 Dinkelbach iterations,
transpilation at optimization level 3 for hardware. The paper's hardware
run used ``ibm_torino``.

Examples:
    python -m experiments.exp3_backend --backend simulator
    QISKIT_IBM_TOKEN=... python -m experiments.exp3_backend --backend ibm --backend-name ibm_torino
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sklearn.metrics import r2_score

from qcar import QCAR, ClosedFormRidge, ideal_probabilities, nsd
from qcar.datasets import load_processed

from .common import REPO_ROOT, Timer, default_output_dir, model_seed, save_rows, setup_logging, write_run_info

logger = logging.getLogger("exp3")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["simulator", "ibm"], default="simulator")
    p.add_argument("--backend-name", default=None, help="IBM backend, e.g. ibm_torino (required for --backend ibm).")
    p.add_argument("--dataset", choices=["diabetes", "student"], default="diabetes")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--num-runs", type=int, default=10)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--vqls-layers", type=int, default=3)
    p.add_argument("--vqls-max-iters", type=int, default=200)
    p.add_argument("--qaoa-reps", type=int, default=3)
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.backend == "ibm" and not args.backend_name:
        p.error("--backend-name is required with --backend ibm")
    return args


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.verbose)
    X_train, X_test, y_train, y_test = load_processed(args.data_dir / args.dataset)
    D = X_train.shape[1]
    label = args.backend_name if args.backend == "ibm" else "simulator"
    output_dir = args.output_dir or default_output_dir("exp3")
    stem = f"{args.dataset}_{label}_exp3"
    job_dir = output_dir / f"{label}_jobs" if args.backend == "ibm" else None
    write_run_info(output_dir / f"{stem}_run_info.json", args, "experiments.exp3_backend", {"experiment": "exp3_backend", "dimension": int(D)})

    rows = []
    for seed in range(args.first_seed, args.first_seed + args.num_runs):
        with Timer() as t:
            ci = ClosedFormRidge(args.alpha).fit(X_train, y_train)
        w_star = ci.coef_
        rows.append(dict(model="CI", backend="classical", run_seed=seed, test_r2=r2_score(y_test, ci.predict(X_test)),
                         nsd=0.0, runtime=t.elapsed, w=w_star, prob=ideal_probabilities(w_star)))

        for offset, (name, method) in enumerate([("BF", "bit_flip"), ("D-QAOA", "dqaoa")]):
            model = QCAR(alpha=args.alpha, quantum_solver="vqls", sign_solver=method,
                         vqls_layers=args.vqls_layers, vqls_max_iters=args.vqls_max_iters,
                         state_shots=args.shots, qaoa_shots=args.shots, qaoa_reps=args.qaoa_reps,
                         max_iters=D, top_k=D, tol=args.tolerance, backend=args.backend,
                         backend_name=args.backend_name, job_dir=str(job_dir) if job_dir else None,
                         seed=model_seed(seed, offset))
            with Timer() as t:
                model.fit(X_train, y_train)
            rows.append(dict(model=f"VQLS+QCAR ({name})", backend=label, run_seed=seed,
                             test_r2=r2_score(y_test, model.predict(X_test)), nsd=nsd(model.coef_, w_star),
                             runtime=t.elapsed, quantum_training_time=model.quantum_training_time_,
                             w=model.coef_, prob=model.prob_, objective_history=model.objective_history_,
                             qaoa_objective_dict=model.qaoa_energy_history_))
        logger.warning("Run %d done (%d/%d).", seed, seed - args.first_seed + 1, args.num_runs)
        save_rows(rows, output_dir / f"{stem}_detailed.csv")
    print(f"Results written to {output_dir / f'{stem}_detailed.csv'}")


if __name__ == "__main__":
    main()
