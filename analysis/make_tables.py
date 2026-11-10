"""Regenerate the paper's tables (mean ± standard deviation over runs) from raw results.

    python -m analysis.make_tables                    # archived paper results
    python -m analysis.make_tables --results-dir runs # your own re-runs

Writes CSV, Markdown and LaTeX (booktabs) versions to ``outputs/tables/``.
Every number is computed directly from the per-run CSV files; nothing is
filtered, smoothed or overridden.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .io import load_exp1, load_exp2, load_exp3, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP1_ORDER = ["CI", "CGD", "HHL+QCAR (ES)", "HHL+QCAR (BF)", "HHL+QCAR (D-QAOA)",
              "VQLS+QCAR (ES)", "VQLS+QCAR (BF)", "VQLS+QCAR (D-QAOA)"]


def pm(mean, std, digits=4) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean:.{digits}f}" if pd.isna(std) or std == 0 else f"{mean:.{digits}f} ± {std:.{digits}f}"


def pm_tex(mean, std, digits=4) -> str:
    return pm(mean, std, digits).replace("±", r"$\pm$")


def write(frame: pd.DataFrame, out_dir: Path, name: str, caption: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"{name}.csv", index=False)
    md_frame = frame.copy()
    md = "| " + " | ".join(md_frame.columns) + " |\n|" + "---|" * len(md_frame.columns) + "\n"
    md += "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in md_frame.itertuples(index=False))
    (out_dir / f"{name}.md").write_text(f"**{caption}**\n\n{md}\n")
    tex_frame = frame.map(lambda v: v.replace("±", r"$\pm$") if isinstance(v, str) else v)
    columns = "l" + "c" * (len(frame.columns) - 1)
    body = " \\\\\n".join(" & ".join(str(v) for v in row) for row in tex_frame.itertuples(index=False))
    header = " & ".join(c.replace("R2", "$R^2$").replace("F_prob", r"$\mathcal{F}_{\mathrm{prob}}$") for c in frame.columns)
    tex = (f"% {caption}\n\\begin{{tabular}}{{{columns}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
           f"{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
    (out_dir / f"{name}.tex").write_text(tex)
    print(f"\n{caption}\n{md}")


def table_exp1(results_dir: Path, out_dir: Path) -> None:
    blocks = []
    for D in (8, 16):
        frame = load_exp1(results_dir, D)
        if frame is None:
            print(f"(skipping Exp. 1 D={D}: no results)")
            continue
        s = summarize(frame, ["model"]).set_index("model").reindex([m for m in EXP1_ORDER if m in set(frame.model)])
        blocks.append(pd.DataFrame({
            "Model": s.index,
            f"D={D} test R2": [pm(r.test_r2_mean, r.test_r2_std) for r in s.itertuples()],
            f"D={D} NSD": [pm(r.nsd_mean, r.nsd_std) for r in s.itertuples()],
            f"D={D} runs": s.n_runs.values,
        }).set_index("Model"))
    if blocks:
        table = pd.concat(blocks, axis=1).reset_index()
        write(table, out_dir, "table2_exp1_synthetic",
              "Table 2 - Experiment 1, synthetic data: test R2 and NSD (mean ± std over runs).")


def table_exp2(results_dir: Path, out_dir: Path) -> None:
    for number, dataset in ((3, "diabetes"), (4, "student")):
        frame = load_exp2(results_dir, dataset)
        if frame is None:
            print(f"(skipping Exp. 2 {dataset}: no results)")
            continue
        ci_r2 = frame[frame.model == "CI"].test_r2.mean()
        s = summarize(frame[frame.model != "CI"], ["model", "shots"]).sort_values(["model", "shots"])
        rows = [{"Model": "CI", "Shots K": "--", "F_prob": "--", "Test R2": f"{ci_r2:.4f}", "NSD": "--", "Runs": "--"}]
        rows += [{"Model": r.model, "Shots K": int(r.shots), "F_prob": f"{r.fidelity_mean:.4f}",
                  "Test R2": pm(r.test_r2_mean, r.test_r2_std), "NSD": pm(r.nsd_mean, r.nsd_std), "Runs": r.n_runs}
                 for r in s.itertuples()]
        write(pd.DataFrame(rows), out_dir, f"table{number}_exp2_{dataset}",
              f"Table {number} - Experiment 2, shot sensitivity on {dataset.title()} (mean ± std over runs).")


def table_exp3(results_dir: Path, out_dir: Path) -> None:
    frame = load_exp3(results_dir)
    if frame is None:
        print("(skipping Exp. 3: no results)")
        return
    ci_r2 = frame[frame.model == "CI"].test_r2.mean()
    s = summarize(frame[frame.model != "CI"], ["backend", "model"])
    rows = [{"Backend": "classical", "Model": "CI", "F_prob": "--", "Test R2": f"{ci_r2:.4f}", "NSD": "--", "Runs": "--"}]
    rows += [{"Backend": r.backend, "Model": r.model, "F_prob": pm(r.fidelity_mean, r.fidelity_std),
              "Test R2": pm(r.test_r2_mean, r.test_r2_std), "NSD": pm(r.nsd_mean, r.nsd_std), "Runs": r.n_runs}
             for r in s.itertuples()]
    write(pd.DataFrame(rows), out_dir, "table5_exp3_backends",
          "Table 5 - Experiment 3, Diabetes on Aer vs IBM hardware: probability fidelity, test R2, NSD (mean ± std).")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "tables")
    args = p.parse_args(argv)
    np.set_printoptions(suppress=True)
    table_exp1(args.results_dir, args.out_dir)
    table_exp2(args.results_dir, args.out_dir)
    table_exp3(args.results_dir, args.out_dir)
    print(f"\nTables written to {args.out_dir}")


if __name__ == "__main__":
    main()
