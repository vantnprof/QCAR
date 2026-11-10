# Archived results

These are the unmodified per-run result files behind the tables and figures of the paper.
Each was matched to its experiment-tracking (Weights & Biases) run: run start time plus logged
runtime equals the file's write time to the second. `CHECKSUMS.sha256` fixes their content; the
test suite verifies it.

All runs used the same software stack and the same development commit (`d6fe079`):

- Python 3.13.5;
- qiskit 2.1.0, qiskit-aer 0.17.1, qiskit-ibm-runtime 0.40.1;
- numpy 2.3.1, scipy 1.16.0, scikit-learn 1.7.1, pandas 2.3.2, matplotlib 3.10.3.

The machine was a 16-core Linux 6.8 workstation, and all simulations ran on CPU.

| File | Experiment | Started (UTC) | Runtime | Configuration |
|---|---|---|---|---|
| `exp1/exp1_D8_detailed.csv` | Exp. 1, D = 8 | 2025-09-17 19:37 | 8 min | N = 50, seeds 0–49, ξ = 0.01, VQLS 3 layers / 200 iterations, K = 1024, QAOA p = 3, S = 10⁵, top-k = 8 |
| `exp1/exp1_D16_detailed.csv` | Exp. 1, D = 16 | 2025-09-17 22:05 | 75 min | as above with D = 16 |
| `exp2/diabetes_exp2_results.csv` | Exp. 2, Diabetes (D = 16) | 2025-09-21 19:24 | 45 min | 30 runs × K ∈ {128, 512, 1024, 4096, 8192}, QAOA p = 3, S = K, top-k = D |
| `exp2/student_exp2_results.csv` | Exp. 2, Student (D = 32) | 2025-09-21 21:24 | 3.0 h | as above |
| `exp3/diabetes_simulator_exp3_detailed.csv` | Exp. 3, Aer simulator | 2025-09-23 22:13 | 13 min | 10 runs, K = S = 4096, QAOA p = 3, top-k = D = 16 |
| `exp3/diabetes_ibm_torino_exp3_detailed.csv` | Exp. 3, `ibm_torino` | 2025-09-23 22:35 | 2.3 h | as above; state and QAOA sampling on hardware, optimisation level 3, no mitigation |
| `exp3/ibm_torino_jobs/*.json` | Exp. 3 raw hardware data | same run | — | 180 SamplerV2 results (counts), named by IBM job ID |

The 180 hardware jobs are 20 four-qubit VQLS state-sampling jobs (10 runs × 2 independent fits)
and 160 sixteen-qubit QAOA sampling jobs (10 runs × 16 Dinkelbach iterations).

## Column reference

| Column | Meaning |
|---|---|
| `model` | `CI`, `CGD`, or `<HHL\|VQLS>+QCAR (<ES\|BF\|D-QAOA>)` |
| `run_seed` | run identifier. Exp. 1: data seed. Exp. 2: 0, 5, …, 145 (labels only; these runs were not seeded) |
| `shots` | state-measurement shots K (Exp. 2) |
| `backend` | `classical`, `simulator` or `ibm_hw` (= `ibm_torino`) (Exp. 3) |
| `test_r2`, `nsd` | test R² and ‖ŵ − w*‖² / ‖w*‖² |
| `prob_star` / `w_prob` (Exp. 1), `prob` (Exp. 2–3) | ideal p* and measured p̂ on the D model coordinates |
| `objective_history` | Exp. 1: BGRQ value / exact optimum per iteration (BF: accepted flips; D-QAOA: Dinkelbach steps). Exp. 3: raw BGRQ value |
| `qaoa_objective_dict` | QAOA training energies per Dinkelbach iteration, `{iteration: [E₁, E₂, …]}` |
| `runtime`, `nr`, `*_time` | wall-clock seconds on the simulator host (not hardware estimates) |

Arrays are stored as Python/numpy reprs; `analysis/io.py` parses them. Fresh runs write the same
columns as JSON.
