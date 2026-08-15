"""
sweep_plots.py
==============

Figures from the JAX hyperparameter sweeps.

Every figure carries a footer spelling out the fixed configuration, because a
one-at-a-time sweep is only interpretable if you know what the other seven
knobs were set to. Axis labels name the actual quantity rather than a column
name.

    python perf/sweep_plots.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("results/sweeps")
COLORS = {"AEJax": "C1", "CAEJax": "C3"}
MARK = {"AEJax": "o", "CAEJax": "^"}

# how each axis should be drawn, and what it means in words
AXES = {
    "n_latent":      ("latent size $k$", "log", "bottleneck width"),
    "learning_rate": ("learning rate", "log", "Adam step size"),
    "weight_decay":  ("weight decay", "symlog", "L2 penalty on weights"),
    "batch_size":    ("batch size", "log", "snapshots per gradient step"),
    "width":         ("width multiplier", "log", "scales hidden units / conv channels"),
    "noise":         ("input noise / field std", "symlog", "gaussian noise added to training data"),
}
BASE_TXT = ("Fixed while each axis is swept:  latent 32  |  lr 1e-3  |  weight decay 1e-6  |  "
            "batch 32  |  width x1  |  noise 0  |  400 epochs, early stopping (patience 80), seed 0")
METRIC_TXT = ("Relative error = MSE(reconstruction, clean test snapshots) / MSE(clean test snapshots). "
              "Scored against the CLEAN signal even in the noise sweep, so reproducing noise is not rewarded.")


def load() -> list[dict]:
    p = OUT / "results.csv"
    if not p.exists():
        sys.exit(f"{p} missing -- run the sweep and `--merge` first")
    rows = []
    for r in csv.DictReader(open(p)):
        for k in ("value", "n_params", "train_loss", "val_loss",
                  "test_mse_clean", "test_rel_clean", "fit_s"):
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                r[k] = float("nan")
        r["n_epochs_run"] = int(float(r["n_epochs_run"] or 0))
        rows.append(r)
    return rows


def series(rows, sweep, model, ycol="test_rel_clean"):
    pts = [(r["value"], r[ycol]) for r in rows
           if r["sweep"] == sweep and r["model"] == model and r[ycol] == r[ycol]]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if v >= 1:
        return f"{v:g}"
    return f"{v:g}"


def _setx(ax, sweep, values=None):
    label, scale, _ = AXES[sweep]
    ax.set_xlabel(label)
    if scale == "log":
        ax.set_xscale("log")
    elif scale == "symlog":
        # weight decay and noise both include exactly 0, which log cannot show
        ax.set_xscale("symlog", linthresh=1e-8 if sweep == "weight_decay" else 1e-3)
    # only 8 points per axis, so label exactly those; matplotlib's default
    # minor decades collided badly on the width panel
    if values:
        vs = sorted(set(values))
        ax.set_xticks(vs)
        ax.set_xticklabels([_fmt(v) for v in vs], fontsize=8)
        ax.minorticks_off()


def fig_overview(rows) -> None:
    sweeps = [s for s in AXES if any(r["sweep"] == s for r in rows)]
    n = len(sweeps)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.7 * nrow), squeeze=False)
    for i, sweep in enumerate(sweeps):
        ax = axes[i // ncol][i % ncol]
        for model in ("AEJax", "CAEJax"):
            x, y = series(rows, sweep, model)
            if x:
                ax.plot(x, y, marker=MARK[model], color=COLORS[model], ms=5, label=model)
        _setx(ax, sweep, [r["value"] for r in rows if r["sweep"] == sweep])
        ax.set_yscale("log")
        ax.set_ylabel("relative error (clean test)")
        ax.set_title(f"{AXES[sweep][0]} — {AXES[sweep][2]}", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Hyperparameter sweeps on bl — JAX autoencoders, one axis varied at a time",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    fig.text(0.5, 0.042, BASE_TXT, ha="center", fontsize=8)
    fig.text(0.5, 0.012, METRIC_TXT, ha="center", fontsize=7.5, color="0.35")
    fig.savefig(OUT / "sweep_overview.png", dpi=140)


def fig_noise(rows) -> None:
    if not any(r["sweep"] == "noise" for r in rows):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for model in ("AEJax", "CAEJax"):
        x, y = series(rows, "noise", model)
        if x:
            axes[0].plot(x, y, marker=MARK[model], color=COLORS[model], ms=6, label=model)
        xg, yg = series(rows, "noise", model, "n_epochs_run")
        if xg:
            axes[1].plot(xg, yg, marker=MARK[model], color=COLORS[model], ms=6, label=model)
    axes[0].set_ylabel("relative error vs clean test signal")
    axes[0].set_title("Robustness: does added input noise degrade the ROM?")
    axes[1].set_ylabel("epochs run before early stopping")
    axes[1].set_title("Noise and training length")
    nv = [r["value"] for r in rows if r["sweep"] == "noise"]
    for ax in axes:
        _setx(ax, "noise", nv)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Input-noise sweep on bl — JAX autoencoders", fontsize=13)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.text(0.5, 0.055, BASE_TXT, ha="center", fontsize=8)
    fig.text(0.5, 0.012, METRIC_TXT, ha="center", fontsize=7.5, color="0.35")
    fig.savefig(OUT / "sweep_noise.png", dpi=140)


def fig_cost(rows) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for model in ("AEJax", "CAEJax"):
        x, y = series(rows, "n_latent", model, "fit_s")
        if x:
            axes[0].plot(x, y, marker=MARK[model], color=COLORS[model], ms=6, label=model)
        xp, yp = series(rows, "width", model, "n_params")
        if xp:
            axes[1].plot(xp, yp, marker=MARK[model], color=COLORS[model], ms=6, label=model)
    axes[0].set_xscale("log")
    kv = sorted({r["value"] for r in rows if r["sweep"] == "n_latent"})
    axes[0].set_xticks(kv); axes[0].set_xticklabels([_fmt(v) for v in kv], fontsize=8)
    axes[0].minorticks_off()
    axes[0].set_xlabel("latent size $k$")
    axes[0].set_ylabel("wall-clock fit time (s)")
    axes[0].set_title("Cost of a full fit vs latent size")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    wv = sorted({r["value"] for r in rows if r["sweep"] == "width"})
    axes[1].set_xticks(wv); axes[1].set_xticklabels([_fmt(v) for v in wv], fontsize=8)
    axes[1].minorticks_off()
    axes[1].set_xlabel("width multiplier"); axes[1].set_ylabel("parameter count")
    axes[1].set_title("Width multiplier vs model size")
    for ax in axes:
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.suptitle("Sweep cost on bl — JAX autoencoders", fontsize=13)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.text(0.5, 0.075, BASE_TXT, ha="center", fontsize=8)
    fig.text(0.5, 0.02,
             "Left panel is wall-clock for a whole fit, so it mixes per-epoch cost with how many "
             "epochs early stopping allowed. CAEJax falling from k=2 to k=16 is fewer epochs run, "
             "not a faster epoch.", ha="center", fontsize=7.5, color="0.35")
    fig.savefig(OUT / "sweep_cost.png", dpi=140)


if __name__ == "__main__":
    rows = load()
    print(f"{len(rows)} rows")
    fig_overview(rows)
    fig_noise(rows)
    fig_cost(rows)
    for p in sorted(OUT.glob("*.png")):
        print("  ", p)
