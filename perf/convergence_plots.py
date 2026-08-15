"""
convergence_plots.py
====================

Redraw of the bl generalisation figure from the study's own results.csv.

The original (`train_vs_test_rel_error.png`) draws ten curves on one axis --
five models x {train faint, test solid} -- in two colour families, so AE sits
on top of AEJax in blue and CAE on top of CAEJax in orange. The pairs genuinely
coincide (same model, two frameworks), which makes the overplot unreadable
rather than informative.

Split into the two questions it was trying to answer at once:

  left   how accurate is each model      -> test error vs latent, one line each
  right  is it overfitting               -> test/train ratio vs latent

Reads results/convergence/bl/results.csv; does not re-run the study.

    python perf/convergence_plots.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import argparse  # noqa: E402

SRC = Path("results/convergence/bl/results.csv")
OUTDIR = Path("results/convergence/bl")

# one distinct colour+marker per model: the original reused blue for AE/AEJax
# and orange for CAE/CAEJax, which is exactly what made it hard to read
STYLE = {
    "POD":    ("k",  "s", "-"),
    "AE":     ("C0", "o", "-"),
    "AEJax":  ("C2", "v", "--"),
    "CAE":    ("C1", "^", "-"),
    "CAEJax": ("C4", "D", "--"),
}
ORDER = ["POD", "AE", "AEJax", "CAE", "CAEJax"]


def load(src: Path):
    if not src.exists():
        sys.exit(f"{src} missing")
    rows = []
    for r in csv.DictReader(open(src)):
        try:
            r["n_latent"] = int(r["n_latent"])
            for k in ("train_rel_error", "test_rel_error", "gen_gap"):
                r[k] = float(r[k])
            for k in ("fit_time_s", "n_params", "n_epochs_run"):
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError, KeyError):
                    r[k] = float("nan")
        except (ValueError, TypeError):
            continue
        rows.append(r)
    return rows


def main(rows, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    for name in ORDER:
        sub = sorted([r for r in rows if r["model"] == name], key=lambda r: r["n_latent"])
        if not sub:
            continue
        c, m, ls = STYLE[name]
        k = [r["n_latent"] for r in sub]
        axes[0].plot(k, [r["test_rel_error"] for r in sub], color=c, marker=m,
                     ls=ls, ms=6, lw=1.8, label=name)
        # gen_gap = test / train: 1.0 means unseen snapshots reconstruct exactly
        # as well as fitted ones; above 1 is the overfitting direction
        axes[1].plot(k, [r["gen_gap"] for r in sub], color=c, marker=m,
                     ls=ls, ms=6, lw=1.8, label=name)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("test MSE / data variance")
    axes[0].set_title("Accuracy: reconstruction error on held-out snapshots")

    axes[1].axhline(1.0, color="0.3", lw=1.2)
    axes[1].text(2.1, 1.005, "no gap (test = train)", fontsize=8, color="0.3", va="bottom")
    axes[1].set_ylabel("test error / train error")
    axes[1].set_title("Generalisation: how much worse are unseen snapshots?")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
        ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128", "256"])
        ax.minorticks_off()
        ax.set_xlabel("latent dimension $k$")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)

    fig.suptitle("bl: accuracy and generalisation vs latent dimension", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.text(0.5, 0.015,
             "AE and AEJax are the same architecture in two frameworks, as are CAE and CAEJax — "
             "their curves coincide by design, which is why the original single-axis version was "
             "hard to read. POD is the linear baseline.",
             ha="center", fontsize=8, color="0.35")
    fig.savefig(outdir / "generalisation.png", dpi=140)
    print(f"wrote {outdir / 'generalisation.png'}")




def fig_cost(rows, outdir: Path, note: str) -> None:
    """Model size and wall-clock cost -- the columns the 7 Aug run got wrong."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0))
    for name in ORDER:
        sub = sorted([r for r in rows if r["model"] == name], key=lambda r: r["n_latent"])
        if not sub:
            continue
        c, m, ls = STYLE[name]
        k = [r["n_latent"] for r in sub]
        axes[0].plot(k, [r["n_params"] for r in sub], color=c, marker=m, ls=ls,
                     ms=6, lw=1.8, label=name)
        axes[1].plot(k, [r["fit_time_s"] for r in sub], color=c, marker=m, ls=ls,
                     ms=6, lw=1.8, label=name)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("trainable parameters")
    axes[0].set_title("Model size vs latent dimension")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("wall-clock fit time (s)")
    axes[1].set_title("Cost of a full fit")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
        ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128", "256"])
        ax.minorticks_off()
        ax.set_xlabel("latent dimension $k$")
        ax.grid(alpha=0.3, which="both")
    # outside the axes on the cost panel: POD runs along the bottom and an
    # inside legend sat on top of it
    axes[0].legend(fontsize=9, loc="upper left")
    axes[1].legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                   borderaxespad=0, frameon=False)
    fig.suptitle("bl: model size and fit cost vs latent dimension", fontsize=13)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.text(0.5, 0.055,
             "AE carries ~16x more parameters than POD or CAE at every latent because the study's "
             "SCALED_HIDDEN ties hidden width to k, so this axis moves bottleneck and capacity together.",
             ha="center", fontsize=8, color="0.35")
    fig.text(0.5, 0.015, note, ha="center", fontsize=8, color="0.35")
    fig.savefig(outdir / "cost.png", dpi=140)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=SRC)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--note", default="Fit time measured on GPU with JAX CUDA active.")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    rows = load(a.csv)
    main(rows, a.outdir)
    fig_cost(rows, a.outdir, a.note)
