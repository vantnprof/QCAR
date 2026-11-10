"""Regression tests: the archived results reproduce the reported numbers exactly."""

import hashlib
from pathlib import Path

import pytest

from analysis.io import load_exp1, load_exp2, load_exp3, summarize

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_archived_files_unchanged():
    for line in (RESULTS / "CHECKSUMS.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((RESULTS / name.strip()).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("D, model, r2, nsd", [
    (8, "HHL+QCAR (ES)", 0.9674, 0.0232), (8, "HHL+QCAR (D-QAOA)", 0.8878, 0.0998),
    (8, "VQLS+QCAR (ES)", 0.8236, 0.1064), (8, "VQLS+QCAR (D-QAOA)", 0.7741, 0.1620),
    (16, "HHL+QCAR (ES)", 0.7542, 0.1831), (16, "HHL+QCAR (D-QAOA)", 0.5495, 0.3721),
    (16, "VQLS+QCAR (ES)", 0.5393, 0.3714), (16, "VQLS+QCAR (D-QAOA)", 0.3229, 0.5221),
])
def test_exp1_means(D, model, r2, nsd):
    s = summarize(load_exp1(RESULTS, D), ["model"]).set_index("model").loc[model]
    assert s.n_runs == 50
    assert round(s.test_r2_mean, 4) == r2 and round(s.nsd_mean, 4) == nsd


@pytest.mark.parametrize("dataset, shots, r2, nsd", [
    ("diabetes", 128, 0.4088, 0.0235), ("diabetes", 512, 0.4783, 0.0095), ("diabetes", 1024, 0.4790, 0.0093),
    ("diabetes", 4096, 0.4634, 0.0120), ("diabetes", 8192, 0.4611, 0.0123),
    ("student", 128, 0.7217, 0.0183), ("student", 1024, 0.7534, 0.0092), ("student", 8192, 0.7592, 0.0076),
])
def test_exp2_dqaoa_means(dataset, shots, r2, nsd):
    frame = load_exp2(RESULTS, dataset)
    s = summarize(frame[frame.model == "VQLS+QCAR (D-QAOA)"], ["shots"]).set_index("shots").loc[shots]
    assert s.n_runs == 30
    assert round(s.test_r2_mean, 4) == r2 and round(s.nsd_mean, 4) == nsd


def test_exp3_fidelity():
    frame = load_exp3(RESULTS)
    s = summarize(frame[frame.model == "VQLS+QCAR (BF)"], ["backend"]).set_index("backend")
    assert round(s.loc["simulator", "fidelity_mean"], 3) == 0.997
    assert round(s.loc["ibm_torino", "fidelity_mean"], 3) == 0.984
    assert set(s.n_runs) == {10}
