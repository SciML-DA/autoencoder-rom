"""
plots.py
========

Turn a baseline CSV into the figures that answer "what actually drives cost".

Varying hidden width at fixed latent is what makes these readable: with only
the default width, every point on a cost-vs-params curve also moves in latent,
so the two effects are confounded. With several widths per latent the same
parameter count is reached by different routes.

    python perf/plots.py [--csv PATH] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATASET = "bl"

# every variant name that appears in a legend, spelled out
# "default" means a different thing for the dense and conv families, so the two
# are spelled out separately rather than sharing one entry
AE_GLOSS = {
    "half": "hidden (4k, k)",
    "default": "hidden (8k, 2k) = the study's SCALED_HIDDEN",
    "double": "hidden (16k, 4k)",
    "fixed_sm": "hidden (128, 64), fixed, independent of k",
    "fixed_lg": "hidden (1024, 256), fixed, independent of k",
}
CAE_GLOSS = {
    "narrow": "conv channels (8, 16, 32)",
    "default": "conv channels (16, 32, 64) = the study's default",
    "wide": "conv channels (32, 64, 128)",
    "deep": "conv channels (16, 32, 64, 128), one extra layer",
}
AE_VARIANTS = ("half", "default", "double", "fixed_sm", "fixed_lg")
CAE_VARIANTS = ("narrow", "default", "wide", "deep")
VMARK = {"half": "o", "default": "s", "double": "^", "fixed_sm": "v", "fixed_lg": "D",
         "narrow": "o", "wide": "^", "deep": "D"}
COLORS = {"AE": "C0", "AEJax": "C1", "CAE": "C2", "CAEJax": "C3", "POD": "k"}


def glossary_text(ae: bool = True, cae: bool = True) -> str:
    parts = ["k = latent size (n_latent)."]
    if ae:
        parts.append(
            "DENSE (AE, AEJax) variants:  "
            + "  |  ".join(f"{k} = {v}" for k, v in AE_GLOSS.items())
        )
    if cae:
        parts.append(
            "CONV (CAE, CAEJax) variants:  "
            + "  |  ".join(f"{k} = {v}" for k, v in CAE_GLOSS.items())
        )
    return "\n".join(parts)


def load(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} missing -- run the sweep and `--merge` first")
    rows = []
    for r in csv.DictReader(open(path)):
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


def piecewise(x, y):
    """Continuous two-segment fit in log-log. Returns (knee, slope_lo, slope_hi).

    A single power law is the wrong model here: cost is flat while fixed
    overhead dominates, then turns over once the model is big enough to be
    compute-bound. One slope splits the difference and describes neither
    regime -- the earlier 0.34 fit sat above every small point and below every
    large one.
    """
    o = np.argsort(x)
    x, y = np.asarray(x)[o], np.asarray(y)[o]
    best = None
    for i in range(3, len(x) - 3):
        c = x[i]
        A = np.column_stack([np.ones_like(x), np.minimum(x - c, 0), np.maximum(x - c, 0)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(np.sum((A @ coef - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, c, coef)
    if best is None:
        return None
    _, c, coef = best
    return c, float(coef[1]), float(coef[2]), coef


def fig_time_vs_params(rows, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    # --- dense: pooled, with a two-regime fit -------------------------------
    ax = axes[0]
    for model in ("AE", "AEJax"):
        pts = [r for r in sel(rows, model=model)
               if r["s_per_epoch"] == r["s_per_epoch"] and r["n_params"] > 0]
        ax.scatter([p["n_params"] for p in pts], [p["s_per_epoch"] for p in pts],
                   c=COLORS[model], marker="o" if "Jax" not in model else "^",
                   s=28, alpha=0.75, label=model)
        x = np.log10([p["n_params"] for p in pts])
        y = np.log10([p["s_per_epoch"] for p in pts])
        r = piecewise(x, y)
        if r:
            c, s_lo, s_hi, coef = r
            xs = np.linspace(x.min(), x.max(), 200)
            A = np.column_stack([np.ones_like(xs), np.minimum(xs - c, 0), np.maximum(xs - c, 0)])
            ax.plot(10 ** xs, 10 ** (A @ coef), color=COLORS[model], ls="--", lw=1.4,
                    label=f"{model}: slope {s_lo:.2f} → {s_hi:.2f}, knee {10 ** c:.1e}")
            ax.axvline(10 ** c, color=COLORS[model], ls=":", lw=0.8, alpha=0.5)
    ax.set_title(f"{DATASET}: dense AE — flat, then compute-bound above the knee")

    # --- conv: split by variant, because that is what the scatter is --------
    ax = axes[1]
    for model in ("CAE", "CAEJax"):
        for v in CAE_VARIANTS:
            pts = [r for r in sel(rows, model=model, variant=v)
                   if r["s_per_epoch"] == r["s_per_epoch"] and r["n_params"] > 0]
            if not pts:
                continue
            ax.scatter([p["n_params"] for p in pts], [p["s_per_epoch"] for p in pts],
                       c=COLORS[model], marker=VMARK[v], s=30, alpha=0.8,
                       label=f"{model} / {v}")
    ax.set_title(f"{DATASET}: conv AE — spread is architecture, not noise")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("parameter count")
        ax.set_ylabel("s / epoch")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("compute time vs parameter count")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.text(0.5, 0.012, glossary_text(), ha="center", va="bottom", fontsize=6.5)
    fig.savefig(out / "time_vs_params.png", dpi=140)


def fig_time_vs_latent(rows, out: Path) -> None:
<<<<<<< Updated upstream
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=False)
=======
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), sharey=False)
>>>>>>> Stashed changes
    for ax, fam, variants in (
        (axes[0], ("AE", "AEJax"), AE_VARIANTS),
        (axes[1], ("CAE", "CAEJax"), CAE_VARIANTS),
    ):
        for model in fam:
            for v in variants:
                pts = [r for r in sel(rows, model=model, variant=v)
                       if r["s_per_epoch"] == r["s_per_epoch"]]
                if not pts:
                    continue
                ax.plot([p["latent"] for p in pts], [p["s_per_epoch"] for p in pts],
                        marker=VMARK[v], color=COLORS[model],
                        ls="-" if "Jax" not in model else "--",
                        alpha=0.85, ms=5, label=f"{model} / {v}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("latent size k")
        ax.set_ylabel("s / epoch")
        ax.grid(alpha=0.3, which="both")
<<<<<<< Updated upstream
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_title(f"{DATASET}: dense AE — flat lines are fixed-width variants")
    axes[1].set_title(f"{DATASET}: conv AE — flat everywhere")
=======
        # outside the axes: with 10 curves per panel an inside legend covered
        # the very lines it was labelling
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0, frameon=False)
    axes[0].set_title(f"{DATASET}: dense AE — flat lines are fixed-width variants")
    axes[1].set_title(f"{DATASET}: conv AE — weakly dependent on latent")
>>>>>>> Stashed changes
    fig.suptitle("compute time vs latent size (latent is nearly free at fixed width)")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.text(0.5, 0.012, glossary_text(), ha="center", va="bottom", fontsize=6.5)
    fig.savefig(out / "time_vs_latent.png", dpi=140)


def fig_compile(rows, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for model in ("AE", "AEJax", "CAE", "CAEJax"):
        pts = [r for r in sel(rows, model=model)
               if r["compile_s"] == r["compile_s"] and r["n_params"] > 0]
        if not pts:
            continue
        kw = dict(c=COLORS[model], marker="o" if "Jax" not in model else "^",
                  s=28, alpha=0.75, label=model)
        axes[0].scatter([p["n_params"] for p in pts], [p["compile_s"] for p in pts], **kw)
        axes[1].scatter([p["latent"] for p in pts], [p["compile_s"] for p in pts], **kw)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("parameter count")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("latent size k")
    for ax in axes:
        ax.set_ylabel("compile / warmup (s)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("one-time compile cost — JAX pays XLA JIT, torch pays cuDNN autotune")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.text(
        0.5, 0.02,
        "Caveat: the highest points (mostly at k=2) are the first config run in each "
        "process and absorb one-off XLA/CUDA context init, not per-config compilation. "
        "Median per-config compile is the honest number: AEJax 3.6 s, CAEJax 2.8 s, "
        "AE 0.50 s, CAE 0.59 s.",
        ha="center", fontsize=6.5, wrap=True,
    )
    fig.savefig(out / "compile_cost.png", dpi=140)


def fig_jax_ratio(rows, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for ds, ls in ((DATASET, "-"), ("circle", "--")):
        for a, b, c in (("AE", "AEJax", "C0"), ("CAE", "CAEJax", "C2")):
            xs, ys = [], []
            for r in sel(rows, dataset=ds, model=a, variant="default"):
                m = [q for q in sel(rows, dataset=ds, model=b, variant="default")
                     if q["latent"] == r["latent"]]
                if m and r["s_per_epoch"] > 0 and m[0]["s_per_epoch"] > 0:
                    xs.append(r["latent"])
                    ys.append(r["s_per_epoch"] / m[0]["s_per_epoch"])
            if xs:
                ax.plot(xs, ys, marker="o", ls=ls, color=c, label=f"{ds}: {a} / {b}")
<<<<<<< Updated upstream
    ax.axhline(1.0, color="k", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("latent size k")
    ax.set_ylabel("torch s/epoch  /  JAX s/epoch")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="center right")
    ax.set_title("JAX speedup over torch (>1 = JAX faster), default width only")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.5, 0.02,
             "circle CAE/CAEJax sits near 0.25: JAX is ~4x SLOWER there. Its 256x64 grid "
             "makes XLA's NCHW->NHWC transposes dominate; bl's 125x37 grid does not.",
=======
    ax.axhline(1.0, color="k", lw=1.2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(0.2, 2.8)
    # placed in axes coordinates: an earlier version put this at data y=1.03,
    # where it sat under the reference line and never appeared in the render
    ax.text(0.015, 0.97, "JAX faster", transform=ax.transAxes, fontsize=9,
            va="top", color="0.25")
    # bottom-right, not bottom-left: the circle CAE/CAEJax curve enters at the
    # left edge around y=0.23 and ran straight through the label there
    ax.text(0.985, 0.03, "torch faster", transform=ax.transAxes, fontsize=9,
            va="bottom", ha="right", color="0.25")
    ax.set_xlabel("latent size k")
    ax.set_ylabel("torch s/epoch  /  JAX s/epoch")
    ax.grid(alpha=0.3, which="both")
    # lower left is the one empty quadrant once ylim is tightened
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.02, 0.10))
    ax.set_title("JAX speedup over torch (>1 = JAX faster), default width only")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.5, 0.02,
             "circle CAE/CAEJax sits near 0.25: JAX is ~4x SLOWER there, on a 256x64 grid vs bl's "
             "125x37. Cause not established — a layout (NCHW vs NHWC) test measured 1.01x, so that "
             "is ruled out; the cost is in the transposed-conv gradient.",
>>>>>>> Stashed changes
             ha="center", fontsize=6.5, wrap=True)
    fig.savefig(out / "jax_ratio.png", dpi=140)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("perf/results/baseline.csv"))
    p.add_argument("--outdir", type=Path, default=Path("perf/results"))
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    rows = load(a.csv)
    fig_time_vs_params(rows, a.outdir)
    fig_time_vs_latent(rows, a.outdir)
    fig_compile(rows, a.outdir)
    fig_jax_ratio(rows, a.outdir)
    print("wrote:")
    for f in sorted(a.outdir.glob("*.png")):
        print("  ", f)
