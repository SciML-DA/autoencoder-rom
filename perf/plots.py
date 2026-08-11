"""
plots.py
========

Turn perf/results/baseline.csv into the four figures that answer "what actually
drives cost".

Varying hidden width at fixed latent is what makes these readable: with only
the default width, every point on a cost-vs-params curve also moves in latent,
so the two effects are confounded. With several widths per latent the same
parameter count is reached by different routes, and if the points collapse onto
one curve then parameter count is the driver -- if they scatter by latent, it
is not.

    python perf/plots.py            # writes perf/results/*.png
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("perf/results")
DATASET = "bl"
MARKERS = {"half": "o", "default": "s", "double": "^", "fixed_sm": "v", "fixed_lg": "D",
           "narrow": "o", "wide": "^", "deep": "D"}
COLORS = {"AE": "C0", "AEJax": "C1", "CAE": "C2", "CAEJax": "C3", "POD": "k"}


def load() -> list[dict]:
    p = OUT / "baseline.csv"
    if not p.exists():
        sys.exit(f"{p} missing -- run the sweep and `--merge` first")
    rows = []
    for r in csv.DictReader(open(p)):
        for k in ("n_params", "s_per_epoch", "compile_s", "total_fit_s", "peak_mem_mb"):
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                r[k] = float("nan")
        r["latent"] = int(r["latent"])
        rows.append(r)
    return rows


def sel(rows, dataset=DATASET, model=None, variant=None):
    out = [r for r in rows if r["dataset"] == dataset]
    if model:
        out = [r for r in out if r["model"] == model]
    if variant:
        out = [r for r in out if r["variant"] == variant]
    return sorted(out, key=lambda r: r["latent"])


def fig_time_vs_latent(rows) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, fam in zip(axes, (("AE", "AEJax"), ("CAE", "CAEJax"))):
        for model in fam:
            variants = sorted({r["variant"] for r in sel(rows, model=model)})
            for v in variants:
                pts = [r for r in sel(rows, model=model, variant=v) if r["s_per_epoch"] == r["s_per_epoch"]]
                if not pts:
                    continue
                ax.plot(
                    [p["latent"] for p in pts], [p["s_per_epoch"] for p in pts],
                    marker=MARKERS.get(v, "o"), color=COLORS[model],
                    ls="-" if "Jax" not in model else "--",
                    alpha=0.85, ms=5, label=f"{model} / {v}",
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("latent size")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("s / epoch")
    axes[0].set_title(f"{DATASET}: dense AE")
    axes[1].set_title(f"{DATASET}: conv AE")
    fig.suptitle("compute time vs latent size (one line per width variant)")
    fig.tight_layout()
    fig.savefig(OUT / "time_vs_latent.png", dpi=140)


def fig_time_vs_params(rows) -> None:
    """The confound test: does cost collapse onto one curve in parameter count?"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, fam in zip(axes, (("AE", "AEJax"), ("CAE", "CAEJax"))):
        for model in fam:
            pts = [r for r in sel(rows, model=model)
                   if r["s_per_epoch"] == r["s_per_epoch"] and r["n_params"] > 0]
            if not pts:
                continue
            ax.scatter(
                [p["n_params"] for p in pts], [p["s_per_epoch"] for p in pts],
                c=[COLORS[model]] * len(pts),
                marker="o" if "Jax" not in model else "^",
                s=26, alpha=0.75, label=model,
            )
            x = np.log(np.array([p["n_params"] for p in pts]))
            y = np.log(np.array([p["s_per_epoch"] for p in pts]))
            if len(x) > 2:
                sl, ic = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 20)
                ax.plot(np.exp(xs), np.exp(ic + sl * xs), color=COLORS[model],
                        ls=":", lw=1.2, label=f"{model} slope={sl:.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("parameter count")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("s / epoch")
    axes[0].set_title(f"{DATASET}: dense AE")
    axes[1].set_title(f"{DATASET}: conv AE")
    fig.suptitle("compute time vs parameter count -- slope 1.0 = compute-bound, 0 = overhead-bound")
    fig.tight_layout()
    fig.savefig(OUT / "time_vs_params.png", dpi=140)


def fig_compile(rows) -> None:
    """Compile cost against size, pooling latent and width into one x axis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model in ("AE", "AEJax", "CAE", "CAEJax"):
        pts = [r for r in sel(rows, model=model)
               if r["compile_s"] == r["compile_s"] and r["n_params"] > 0]
        if not pts:
            continue
        axes[0].scatter([p["n_params"] for p in pts], [p["compile_s"] for p in pts],
                        c=COLORS[model], marker="o" if "Jax" not in model else "^",
                        s=26, alpha=0.75, label=model)
        axes[1].scatter([p["latent"] for p in pts], [p["compile_s"] for p in pts],
                        c=COLORS[model], marker="o" if "Jax" not in model else "^",
                        s=26, alpha=0.75, label=model)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("parameter count")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("latent size")
    for ax in axes:
        ax.set_ylabel("compile / warmup (s)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("one-time compile cost -- JAX pays XLA JIT, torch pays cuDNN autotune")
    fig.tight_layout()
    fig.savefig(OUT / "compile_cost.png", dpi=140)


def fig_jax_ratio(rows) -> None:
    """Where the JAX rewrite wins and where it loses, across both datasets."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for ds, ls in (("bl", "-"), ("circle", "--")):
        for a, b, c in (("AE", "AEJax", "C0"), ("CAE", "CAEJax", "C2")):
            xs, ys = [], []
            for r in sel(rows, dataset=ds, model=a, variant="default"):
                m = [q for q in sel(rows, dataset=ds, model=b, variant="default")
                     if q["latent"] == r["latent"]]
                if m and r["s_per_epoch"] > 0 and m[0]["s_per_epoch"] > 0:
                    xs.append(r["latent"])
                    ys.append(r["s_per_epoch"] / m[0]["s_per_epoch"])
            if xs:
                ax.plot(xs, ys, marker="o", ls=ls, color=c, label=f"{ds}: {a}/{b}")
    ax.axhline(1.0, color="k", lw=1)
    ax.text(2.2, 1.03, "JAX faster above", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("latent size")
    ax.set_ylabel("torch s/epoch  /  JAX s/epoch")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("JAX speedup over torch (>1 = JAX wins)")
    fig.tight_layout()
    fig.savefig(OUT / "jax_ratio.png", dpi=140)


if __name__ == "__main__":
    rows = load()
    fig_time_vs_latent(rows)
    fig_time_vs_params(rows)
    fig_compile(rows)
    fig_jax_ratio(rows)
    print("wrote:")
    for p in sorted(OUT.glob("*.png")):
        print("  ", p)
