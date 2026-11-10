"""Load experiment results (archived ``results/`` or fresh ``runs/``) into tidy DataFrames.

Array-valued columns are stored as JSON in fresh runs and as Python/numpy reprs in
the archived 2025 runs; both formats are parsed here.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

_NP_FLOAT = re.compile(r"np\.float64\((.*?)\)")
_NP_ARRAY = re.compile(r"array\((.*?)\)")


def parse_array(entry) -> np.ndarray:
    if isinstance(entry, (list, tuple, np.ndarray)):
        return np.asarray(entry, dtype=float)
    if not isinstance(entry, str) or entry.strip() in ("", "nan", "[]"):
        return np.array([])
    text = _NP_FLOAT.sub(r"\1", entry.strip())
    try:
        return np.asarray(json.loads(text), dtype=float)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return np.asarray(ast.literal_eval(text), dtype=float)
    except (ValueError, SyntaxError):
        return np.fromstring(text.replace("\n", " ").strip("[]"), sep=" ")


def parse_history(entry) -> list[float]:
    values = parse_array(entry)
    return values.ravel().tolist()


def parse_energy_dict(entry) -> dict[int, list[float]]:
    if isinstance(entry, dict):
        return {int(k): [float(x) for x in v] for k, v in entry.items()}
    if not isinstance(entry, str) or entry.strip() in ("", "nan", "{}"):
        return {}
    text = _NP_FLOAT.sub(r"\1", _NP_ARRAY.sub(r"\1", entry))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return {}
    return {int(k): [float(x) for x in v] for k, v in parsed.items()}


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("test_r2", "nsd"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _fidelity(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.sqrt(p * q)) ** 2) if len(p) == len(q) and len(p) else float("nan")


def load_exp1(results_dir: Path, dimension: int) -> pd.DataFrame | None:
    path = Path(results_dir) / "exp1" / f"exp1_D{dimension}_detailed.csv"
    if not path.exists():
        return None
    frame = _read(path)
    ideal = {row.run_seed: parse_array(row.prob_star) for row in frame[frame.model == "CI"].itertuples()}
    measured = frame["w_prob"].apply(parse_array) if "w_prob" in frame else pd.Series([np.array([])] * len(frame))
    frame["fidelity"] = [
        _fidelity(m, ideal.get(seed, np.array([]))) if model != "CI" and model != "CGD" else np.nan
        for m, seed, model in zip(measured, frame.run_seed, frame.model)
    ]
    frame["objective_history"] = frame.get("objective_history", pd.Series(dtype=object)).apply(parse_history)
    frame["qaoa_energy"] = frame.get("qaoa_objective_dict", pd.Series(dtype=object)).apply(parse_energy_dict)
    frame["dimension"] = dimension
    return frame


def load_exp2(results_dir: Path, dataset: str) -> pd.DataFrame | None:
    path = Path(results_dir) / "exp2" / f"{dataset}_exp2_results.csv"
    if not path.exists():
        return None
    frame = _read(path)
    ideal = parse_array(frame[frame.model == "CI"]["prob"].iloc[0])
    frame["fidelity"] = [
        _fidelity(parse_array(p), ideal) if model != "CI" else np.nan for p, model in zip(frame["prob"], frame.model)
    ]
    frame["dataset"] = dataset
    return frame


def load_exp3(results_dir: Path, dataset: str = "diabetes") -> pd.DataFrame | None:
    """Concatenate all ``exp3/<dataset>_<backend>_exp3_detailed.csv`` files (backend from the file name)."""
    files = sorted((Path(results_dir) / "exp3").glob(f"{dataset}_*_exp3_detailed.csv"))
    if not files:
        return None
    frames = []
    for path in files:
        frame = _read(path)
        backend = path.name[len(dataset) + 1 : -len("_exp3_detailed.csv")]
        ideal = parse_array(frame[frame.model == "CI"]["prob"].iloc[0])
        frame["fidelity"] = [
            _fidelity(parse_array(p), ideal) if model != "CI" else np.nan for p, model in zip(frame["prob"], frame.model)
        ]
        frame["backend"] = np.where(frame.model == "CI", "classical", backend)
        frame["objective_history"] = frame["objective_history"].apply(parse_history)
        frame["qaoa_energy"] = frame["qaoa_objective_dict"].apply(parse_energy_dict)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(frame: pd.DataFrame, by: list[str], metrics=("test_r2", "nsd", "fidelity")) -> pd.DataFrame:
    """Mean, standard deviation and run count per group."""
    aggregations = {}
    for metric in metrics:
        if metric in frame:
            aggregations[f"{metric}_mean"] = (metric, "mean")
            aggregations[f"{metric}_std"] = (metric, "std")
    aggregations["n_runs"] = ("run_seed", "nunique")
    return frame.groupby(by, sort=False).agg(**aggregations).reset_index()
