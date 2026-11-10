"""Regenerate the paper's figures from raw results.

    python -m analysis.make_figures                     # archived paper results
    python -m analysis.make_figures --results-dir runs  # your own re-runs

Writes PDF and PNG files to ``outputs/figures/``, named like the panels in the
manuscript's ``figures/`` folder. Points are means over runs; error bars are
95% confidence intervals of the mean (Student t); the tables report standard
deviations. Histories of different lengths are padded with their final value
before averaging (a run that stopped early keeps its final value).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import style
from .io import load_exp1, load_exp2, load_exp3

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pad_mean(sequences: list[list[float]]) -> np.ndarray | None:
    sequences = [np.asarray(s, dtype=float) for s in sequences if len(s)]
    if not sequences:
        return None
    length = max(len(s) for s in sequences)
    return np.mean([np.pad(s, (0, length - len(s)), mode="edge") for s in sequences], axis=0)


def _ci95(values) -> float:
    """Half-width of the 95% confidence interval of the mean."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def _energy_traces(frame: pd.DataFrame, which: str) -> np.ndarray | None:
    traces = []
    for energies in frame["qaoa_energy"]:
        if energies:
            key = min(energies) if which == "first" else max(energies)
            traces.append(energies[key])
    return _pad_mean(traces)


# ----------------------------------------------------------------------------- Exp. 1
def exp1_figures(results_dir: Path, out_dir: Path) -> list[str]:
    written = []
    for D in (8, 16):
        frame = load_exp1(results_dir, D)
        if frame is None:
            print(f"(skipping Exp. 1 D={D}: no results)")
            continue

        # (a) probability fidelity of the measured HHL / VQLS states
        fig, ax = plt.subplots(figsize=(2.4, 2.2))
        jitter = np.random.default_rng(0)
        for i, solver in enumerate(["HHL", "VQLS"]):
            values = frame[frame.model == f"{solver}+QCAR (ES)"].fidelity.to_numpy()
            st = style.SOLVER_STYLE[solver]
            ax.bar(i, values.mean(), width=0.6, color=st["color"], alpha=0.35, edgecolor=st["color"], hatch=st["hatch"])
            ax.scatter(i + jitter.uniform(-0.18, 0.18, len(values)), values, s=6, color=st["color"],
                       edgecolor="none", zorder=3, label=None)
            ax.text(i, 1.02, f"mean {values.mean():.3f}", ha="center", va="bottom", color=style.INK, fontsize=7)
        ax.set_xticks([0, 1], ["HHL", "VQLS"])
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(r"$\mathcal{F}_{\mathrm{prob}}$ (bar: mean; dots: runs)")
        ax.grid(axis="x", visible=False)
        written += style.save(fig, out_dir, f"fig_exp1_fidelity_d{D}")

        # (b, c) BGRQ objective relative to the exact optimum along solver iterations
        for solver in ["HHL", "VQLS"]:
            fig, ax = plt.subplots(figsize=(2.6, 2.2))
            ax.axhline(1.0, label="ES (optimum)", **{k: v for k, v in style.METHOD_STYLE["ES"].items() if k != "marker"})
            for method in ["BF", "D-QAOA"]:
                mean = _pad_mean(frame[frame.model == f"{solver}+QCAR ({method})"].objective_history.tolist())
                if mean is not None:
                    ax.plot(np.arange(len(mean)), mean, label=method, **style.METHOD_STYLE[method])
            ax.set_xlabel("Iteration (0 = random start)")
            ax.set_ylabel("BGRQ / optimum")
            ax.set_title(f"{solver}, D={D}", loc="left")
            ax.legend(loc="lower right")
            written += style.save(fig, out_dir, f"fig_exp1_{solver.lower()}_objective_d{D}")

        # (d) QAOA training energy in the first and last Dinkelbach iteration
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        for solver, color in (("HHL", style.BLUE), ("VQLS", style.ORANGE)):
            sub = frame[frame.model == f"{solver}+QCAR (D-QAOA)"]
            for which, ls in (("first", ":"), ("last", "-")):
                trace = _energy_traces(sub, which)
                if trace is not None:
                    ax.plot(np.arange(1, len(trace) + 1), trace, color=color, linestyle=ls, lw=1.2,
                            label=f"{solver}, {which} Dinkelbach iteration")
        ax.set_xlabel("COBYLA energy evaluation")
        ax.set_ylabel(r"QAOA energy $\langle H_C\rangle$ (normalised $Q$)")
        ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        written += style.save(fig, out_dir, f"fig_exp1_qaoa_energy_d{D}")
    return written


# ----------------------------------------------------------------------------- Exp. 2
def exp2_figures(results_dir: Path, out_dir: Path) -> list[str]:
    written = []
    for dataset in ("diabetes", "student"):
        frame = load_exp2(results_dir, dataset)
        if frame is None:
            print(f"(skipping Exp. 2 {dataset}: no results)")
            continue
        ci = frame[frame.model == "CI"].test_r2.mean()
        shots = sorted(frame.shots.unique())

        def axis(ax, ylabel):
            ax.set_xscale("log", base=2)
            ax.set_xticks(shots, [str(s) for s in shots])
            ax.minorticks_off()
            ax.set_xlabel("State-measurement shots $K$")
            ax.set_ylabel(ylabel)

        # Fidelity of the measured state (shared by BF and D-QAOA within a run).
        fig, ax = plt.subplots(figsize=(2.6, 2.2))
        g = frame[frame.model == "VQLS+QCAR (BF)"].groupby("shots").fidelity
        ax.errorbar(shots, g.mean().reindex(shots), yerr=g.agg(_ci95).reindex(shots), color=style.BLUE,
                    marker="o", capsize=2, label="VQLS state")
        axis(ax, r"$\mathcal{F}_{\mathrm{prob}}$")
        ax.legend(loc="lower right")
        written += style.save(fig, out_dir, f"fig_exp2_{dataset}_fidelity")

        for metric, ylabel, fname in (("test_r2", "Test $R^2$", "r2"), ("nsd", "NSD", "nsd")):
            fig, ax = plt.subplots(figsize=(2.6, 2.2))
            if metric == "test_r2":
                ax.axhline(ci, color=style.REFERENCE, linestyle="--", lw=1, label=f"CI ({ci:.3f})")
            for k, method in enumerate(["BF", "D-QAOA"]):
                g = frame[frame.model == f"VQLS+QCAR ({method})"].groupby("shots")[metric]
                offset = 1.06 ** (k - 0.5)  # small horizontal dodge so error bars do not overlap
                ax.errorbar(np.array(shots) * offset, g.mean().reindex(shots), yerr=g.agg(_ci95).reindex(shots),
                            capsize=2, label=method, **style.METHOD_STYLE[method])
            axis(ax, ylabel)
            ax.legend(loc="lower right" if metric == "test_r2" else "upper center", ncol=1 if metric == "test_r2" else 2)
            written += style.save(fig, out_dir, f"fig_exp2_{dataset}_{fname}")
    return written


# ----------------------------------------------------------------------------- Exp. 3
def exp3_figures(results_dir: Path, out_dir: Path) -> list[str]:
    frame = load_exp3(results_dir)
    if frame is None:
        print("(skipping Exp. 3: no results)")
        return []
    written = []
    backends = [b for b in frame.backend.unique() if b != "classical"]

    # (a) QAOA training energies. Parameters are optimised on a local simulator in
    # every case; the backend only determines where the final circuits are sampled.
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    for backend in backends:
        st = style.backend_style(backend)
        sub = frame[(frame.backend == backend) & (frame.model == "VQLS+QCAR (D-QAOA)")]
        for which, ls in (("first", ":"), ("last", "-")):
            trace = _energy_traces(sub, which)
            if trace is not None:
                ax.plot(np.arange(1, len(trace) + 1), trace, color=st["color"], linestyle=ls, lw=1.2,
                        label=f"{st['label']} run, {which} iteration")
    ax.set_xlabel("COBYLA energy evaluation (local simulator)")
    ax.set_ylabel(r"QAOA energy $\langle H_C\rangle$ (normalised $Q$)")
    ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    written += style.save(fig, out_dir, "fig_exp3_qaoa_energy")

    # (b) outer BGRQ objective for both sign solvers on both backends
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.3), sharey=True)
    for ax, method in zip(axes, ["BF", "D-QAOA"]):
        for backend in backends:
            st = style.backend_style(backend)
            mean = _pad_mean(frame[(frame.backend == backend) & (frame.model == f"VQLS+QCAR ({method})")].objective_history.tolist())
            if mean is not None:
                ax.plot(np.arange(len(mean)), mean, color=st["color"], linestyle=st["linestyle"],
                        marker=st["marker"], label=st["label"])
        ax.set_title(f"{method}" + (" (accepted flips)" if method == "BF" else " (Dinkelbach iterations)"), loc="left")
        ax.set_xlabel("Iteration (0 = random start)")
    axes[0].set_ylabel("BGRQ objective")
    axes[1].legend(loc="lower right")
    written += style.save(fig, out_dir, "fig_exp3_bgrq_objective")
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "figures")
    args = p.parse_args(argv)
    style.apply()
    written = exp1_figures(args.results_dir, args.out_dir)
    written += exp2_figures(args.results_dir, args.out_dir)
    written += exp3_figures(args.results_dir, args.out_dir)
    print(f"{len(written)} files written to {args.out_dir}")


if __name__ == "__main__":
    main()
