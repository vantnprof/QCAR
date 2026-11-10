"""Prepare the processed Diabetes and Student Performance datasets used in Experiments 2-3.

Writes ``data/<name>/{X_train,X_test,y_train,y_test}.csv`` and checks that the
classical ridge reference (ξ = 0.01) reproduces the test R² stored in the
archived paper results.

    python -m experiments.prepare_data                      # both datasets
    python -m experiments.prepare_data --student-csv path/to/student-mat.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sklearn.metrics import r2_score

from qcar import ClosedFormRidge
from qcar import datasets

from .common import REPO_ROOT, setup_logging

# Reference values from the archived paper runs (results/exp2/*, CI rows).
EXPECTED = {"diabetes": {"dim": 16, "r2": 0.514114}, "student": {"dim": 32, "r2": 0.799214}}


def check(name, X_train, X_test, y_train, y_test) -> bool:
    r2 = r2_score(y_test, ClosedFormRidge(0.01).fit(X_train, y_train).predict(X_test))
    ok = X_train.shape[1] == EXPECTED[name]["dim"] and abs(r2 - EXPECTED[name]["r2"]) < 5e-6
    status = "OK" if ok else "MISMATCH"
    print(f"[{status}] {name}: train {X_train.shape}, test {X_test.shape}, CI test R2 = {r2:.6f} "
          f"(archived {EXPECTED[name]['r2']:.6f})")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=["diabetes", "student"], default=["diabetes", "student"])
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--student-csv", type=Path, default=None,
                   help="Local copy of student-mat.csv; downloaded from UCI if omitted.")
    args = p.parse_args(argv)
    setup_logging(verbose=True)

    all_ok = True
    for name in args.datasets:
        if name == "diabetes":
            splits = datasets.diabetes()
        else:
            csv = args.student_csv or datasets.download_student_csv(args.data_dir / "raw")
            splits = datasets.student(csv)
        datasets.save_processed(args.data_dir / name, *splits)
        all_ok &= check(name, *splits)
    if not all_ok:
        raise SystemExit("Processed data differ from the archived reference; check package versions (requirements.txt).")


if __name__ == "__main__":
    main()
