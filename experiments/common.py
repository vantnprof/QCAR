"""Shared helpers for the experiment scripts: provenance, seeding and result I/O."""

from __future__ import annotations

import datetime as dt
import json
import logging
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_PACKAGES = [
    "qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy", "scipy", "scikit-learn", "pandas", "matplotlib",
]


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("qiskit_ibm_runtime").setLevel(logging.WARNING)


def default_output_dir(experiment: str) -> Path:
    """Fresh runs go to ``runs/<experiment>/``, mirroring the archived ``results/`` layout."""
    return REPO_ROOT / "runs" / experiment


def model_seed(run_seed: int, model_index: int) -> int:
    """Independent, reproducible seed for one model within one run."""
    return int(np.random.SeedSequence([int(run_seed), int(model_index)]).generate_state(1)[0])


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def write_run_info(path: Path, args, module: str, extra: dict | None = None) -> dict:
    """Record what was run, where and with which software versions."""
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    info = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(["python", "-m", module, *sys.argv[1:]]),
        "arguments": {k: v for k, v in vars(args).items()},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": _git_commit(),
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, default=str))
    return info


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def save_rows(rows: list[dict], path: Path) -> pd.DataFrame:
    """Write result rows to CSV; arrays, lists and dicts are stored as JSON strings."""
    encoded = [
        {k: json.dumps(_jsonable(v)) if isinstance(v, (np.ndarray, list, tuple, dict)) else _jsonable(v) for k, v in row.items()}
        for row in rows
    ]
    frame = pd.DataFrame(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.start
        return False
