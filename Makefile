PYTHON ?= python

.PHONY: install test reproduce tables figures data experiments smoke clean

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest

# Tables and figures from the archived paper results (no quantum resources needed).
reproduce: tables figures

tables:
	$(PYTHON) -m analysis.make_tables

figures:
	$(PYTHON) -m analysis.make_figures

data:
	$(PYTHON) -m experiments.prepare_data

# Full simulator experiments with the paper's settings (several hours). Results go to runs/.
experiments: data
	$(PYTHON) -m experiments.exp1_synthetic --dimension 8
	$(PYTHON) -m experiments.exp1_synthetic --dimension 16
	$(PYTHON) -m experiments.exp2_shots --dataset diabetes
	$(PYTHON) -m experiments.exp2_shots --dataset student
	$(PYTHON) -m experiments.exp3_backend --backend simulator
	$(PYTHON) -m analysis.make_tables --results-dir runs --out-dir runs/tables
	$(PYTHON) -m analysis.make_figures --results-dir runs --out-dir runs/figures

# A few minutes: every script end to end at reduced scale (Diabetes only).
smoke:
	$(PYTHON) -m experiments.prepare_data --datasets diabetes
	$(PYTHON) -m experiments.exp1_synthetic --dimension 8 --num-runs 2 --output-dir runs/smoke/exp1
	$(PYTHON) -m experiments.exp2_shots --dataset diabetes --num-runs 1 --shots 128 1024 --output-dir runs/smoke/exp2
	$(PYTHON) -m experiments.exp3_backend --backend simulator --num-runs 1 --output-dir runs/smoke/exp3
	$(PYTHON) -m analysis.make_tables --results-dir runs/smoke --out-dir runs/smoke/tables
	$(PYTHON) -m analysis.make_figures --results-dir runs/smoke --out-dir runs/smoke/figures

clean:
	rm -rf outputs runs/smoke .pytest_cache
