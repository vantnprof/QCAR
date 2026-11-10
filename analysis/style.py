"""Shared figure style: colour-vision-safe palette with redundant marker/line encodings."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d9d8d4"
REFERENCE = "#7a7974"  # classical reference lines

# Categorical slots 1-3 (validated for CVD separation); aqua needs labels, which every figure has.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

METHOD_STYLE = {
    "ES": dict(color=INK_SECONDARY, linestyle=":", marker=None),
    "BF": dict(color=BLUE, linestyle="-", marker="o"),
    "D-QAOA": dict(color=ORANGE, linestyle="--", marker="s"),
}
SOLVER_STYLE = {"HHL": dict(color=BLUE, hatch=""), "VQLS": dict(color=ORANGE, hatch="//")}
BACKEND_STYLE = {
    "simulator": dict(color=BLUE, linestyle="-", marker="o", label="Aer simulator"),
    "ibm_torino": dict(color=ORANGE, linestyle="--", marker="s", label="ibm_torino"),
}


def backend_style(name: str) -> dict:
    return BACKEND_STYLE.get(name, dict(color=AQUA, linestyle="-.", marker="^", label=name))


def apply() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, out_dir, name: str) -> list[str]:
    """Save ``name.pdf`` and ``name.png``; returns the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        paths.append(str(path))
    plt.close(fig)
    return paths
