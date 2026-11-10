"""Datasets used in the paper's real-data experiments (Section 4.1.2).

Both pipelines use a fixed 80:20 split (``random_state=2602``), then
degree-2 polynomial features → PCA (fitted on the training split) →
standardisation (fitted on the training split) → an intercept column of ones.
The resulting dimension is D = 16 for Diabetes (15 components + intercept) and
D = 32 for Student Performance (31 components + intercept).
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)

SPLIT_SEED = 2602
STUDENT_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
STUDENT_CATEGORICAL = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic",
]


def _poly_pca_scale(X_train, X_test, n_components: int):
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_train, X_test = poly.fit_transform(X_train), poly.transform(X_test)
    pca = PCA(n_components=n_components)
    X_train, X_test = pca.fit_transform(X_train), pca.transform(X_test)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    add_bias = lambda Z: np.c_[np.ones((Z.shape[0], 1)), Z]  # noqa: E731
    return add_bias(X_train), add_bias(X_test)


def diabetes(test_size: float = 0.2):
    """scikit-learn Diabetes (Efron et al., 2004) → D = 16."""
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=SPLIT_SEED
    )
    X_train, X_test = _poly_pca_scale(X_train, X_test, n_components=15)
    return X_train, X_test, y_train, y_test


def download_student_csv(dest: str | Path) -> Path:
    """Download ``student-mat.csv`` from the UCI repository (dataset 320) into ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "student-mat.csv"
    if target.exists():
        return target
    logger.info("Downloading %s", STUDENT_URL)
    with urllib.request.urlopen(STUDENT_URL, timeout=60) as response:
        payload = response.read()

    def find(archive_bytes: bytes):
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for name in archive.namelist():
                if name.endswith("student-mat.csv"):
                    return archive.read(name)
                if name.endswith(".zip"):
                    found = find(archive.read(name))
                    if found is not None:
                        return found
        return None

    content = find(payload)
    if content is None:
        raise RuntimeError("student-mat.csv not found in the UCI archive; download it manually (see README).")
    target.write_bytes(content)
    return target


def student(csv_path: str | Path, test_size: float = 0.2):
    """UCI Student Performance, mathematics course (Cortez, 2008); target G3 → D = 32."""
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    df = pd.read_csv(csv_path, sep=";")
    y = df["G3"].to_numpy(dtype=np.float64)
    X = df.drop(columns=["G3"])
    encoder = ColumnTransformer([("encoder", OneHotEncoder(), STUDENT_CATEGORICAL)], remainder="passthrough")
    X_encoded = encoder.fit_transform(X)
    X_encoded = X_encoded.toarray() if hasattr(X_encoded, "toarray") else np.asarray(X_encoded)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=SPLIT_SEED
    )
    X_train, X_test = _poly_pca_scale(X_train, X_test, n_components=31)
    return X_train, X_test, y_train, y_test


def save_processed(directory: str | Path, X_train, X_test, y_train, y_test) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, array in {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}.items():
        np.savetxt(directory / f"{name}.csv", array, delimiter=",")


def load_processed(directory: str | Path):
    """Load ``X_train, X_test, y_train, y_test`` written by :func:`save_processed`."""
    directory = Path(directory)
    if not (directory / "X_train.csv").exists():
        raise FileNotFoundError(
            f"No processed dataset in '{directory}'. Run `python -m experiments.prepare_data` first."
        )
    return tuple(np.loadtxt(directory / f"{name}.csv", delimiter=",") for name in ("X_train", "X_test", "y_train", "y_test"))
