"""
Plots for the sparse-sensor field-estimation task: spectra, observability,
extended POD modes, method comparisons, and the reconstruction animation that
is the whole point of the exercise.

Everything takes plain arrays rather than model objects, so a POD-LSE result, an
extended-POD result and a two-branch AE result all go through the same
functions and are therefore drawn on the same colour scale. Comparing two
reconstructions plotted with independent limits is a way to make a bad model
look fine.

Headless by default -- ``matplotlib.use("Agg")`` is set by the caller, not here,
so importing this module does not hijack an interactive session.

Lives beside the estimators rather than in ``plotting/`` so that
``field_estimation`` is self-contained: it is one package, taken or left whole.
``plotting/`` is now purely the ROM-side figures. Not imported by
``field_estimation/__init__.py`` -- nothing in the estimators needs matplotlib,
and a headless fit should not pay for it.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation, PillowWriter

__all__ = [
    "extent_from_meanfield",
    "plot_spectrum",
    "plot_observability",
    "plot_extended_modes",
    "plot_comparison",
    "plot_error_history",
    "plot_horizon",
    "animate_reconstruction",
    "pick_writer",
]


def extent_from_meanfield(mf) -> Optional[tuple]:
    """(x0, x1, y0, y1) in mm for ``imshow``, or None if there is no mean field."""
    if mf is None:
        return None
    return (float(mf.x.min()), float(mf.x.max()), float(mf.y.min()), float(mf.y.max()))


def _field_panel(ax, F, lim, extent, title, cmap="RdBu_r"):
    # .T because the arrays are (Nx, Ny) and imshow wants row=y
    im = ax.imshow(F.T, origin="lower", cmap=cmap, vmin=-lim, vmax=lim,
                   extent=extent, aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_spectrum(Sigma: np.ndarray, out: str, n_show: int = 200, floors=None):
    """POD energy spectrum and cumulative energy.

    ``floors`` optionally overlays ``{label: r}`` markers, e.g. the truncation
    each model used, so the reader can see where on the spectrum a choice sits
    rather than being told a number.
    """
    energy = Sigma**2 / np.sum(Sigma**2)
    cum = np.cumsum(energy)
    k = np.arange(1, min(len(Sigma), n_show) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(k, energy[: len(k)], ".-", lw=1, ms=4)
    ax[0].set(xlabel="mode $k$", ylabel=r"$\sigma_k^2/\sum\sigma^2$", title="POD spectrum")
    ax[0].grid(alpha=0.3)

    ax[1].plot(k, cum[: len(k)], ".-", lw=1, ms=4)
    for frac in (0.9, 0.95, 0.99):
        r = int(np.searchsorted(cum, frac)) + 1
        ax[1].axhline(frac, ls=":", c="k", lw=0.8)
        ax[1].annotate(f"{frac:.0%} at r={r}", (0.45, frac),
                       xycoords=("axes fraction", "data"), fontsize=8, va="bottom")
    for label, r in (floors or {}).items():
        for a in ax:
            a.axvline(r, ls="--", lw=0.9, alpha=0.6)
        ax[1].annotate(label, (r, 0.05), fontsize=7, rotation=90, ha="right")
    ax[1].set(xlabel="mode $k$", ylabel="cumulative energy", title="cumulative",
              ylim=(0, 1.02))
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return cum


def plot_observability(obs: dict, out: str, n_show: int = 60):
    """Per-mode linear observability from ``epod.mode_observability``.

    Left: ``rho2``, the fraction of each field mode explained by the linear span
    of the sensor coefficients -- the ceiling for any *linear* estimator, mode by
    mode. Right: the correlation matrix, which shows *which* sensor mode carries
    each field mode rather than only how much.

    Read this before believing any reconstruction score. If ``rho2`` falls off a
    cliff at mode 6, a model that appears to reconstruct mode 20 is either
    exploiting nonlinearity the linear analysis cannot see -- the interesting
    case -- or leaking. The two look identical in an aggregate NMSE.
    """
    rho2, R, cum = obs["rho2"], obs["R"], obs["cum"]
    n = min(n_show, len(rho2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(np.arange(1, n + 1), rho2[:n], color="C0", alpha=0.85)
    ax[0].plot(np.arange(1, n + 1), cum[:n], "k.-", lw=1, ms=4,
               label="best linear NMSE using modes $1..k$")
    ax[0].set(xlabel="field mode $k$", ylabel=r"explained fraction $\rho_k^2$",
              title="linear observability from the sensors", ylim=(0, 1.02))
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")

    m = min(n, R.shape[0])
    j = min(R.shape[1], 40)
    im = ax[1].imshow(np.abs(R[:m, :j]), cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax[1].set(xlabel="sensor mode $j$", ylabel="field mode $k$",
              title=r"$|corr(b_k, c_j)|$")
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_extended_modes(Psi_ext, unflat, out: str, n_modes: int = 6, mf=None):
    """The extended POD modes: what the flow does when each sensor mode fires.

    These are more useful than the field POD modes for this problem. A field POD
    mode is what the flow does; an extended mode is the part of that the load
    cells can actually see, so the set of them *is* the reachable subspace of any
    linear estimator drawn in physical space.
    """
    n = min(n_modes, Psi_ext.shape[1])
    G = unflat(Psi_ext[:, :n])  # (Nu, n, Nx, Ny)
    extent = extent_from_meanfield(mf)

    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.0), squeeze=False)
    for j in range(n):
        for row, comp in enumerate(("u", "v")):
            F = G[row, j]
            lim = np.nanmax(np.abs(F)) or 1.0
            _field_panel(axes[row][j], F, lim, extent,
                         f"$\\psi^{{ext}}_{{{j + 1}}}$ ({comp})")
    fig.suptitle("extended POD modes -- the flow correlated with each sensor mode")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_comparison(rows: Sequence[dict], out: str, floor: Optional[float] = None):
    """Horizontal bar chart of test NMSE per method, sorted best first.

    The mean-predictor line at 1.0 and the projection floor are both drawn,
    because a bare bar chart of NMSE answers neither "did it learn anything"
    nor "is the basis or the sensor the limit".
    """
    rows = sorted(rows, key=lambda r: r["nmse_test"])
    labels = [r["label"] for r in rows]
    vals = [r["nmse_test"] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 2.0))
    ax.barh(y, vals, color=["C2" if v < 1 else "C3" for v in vals], alpha=0.85)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(1.0, c="k", ls="--", lw=1)
    ax.annotate("predicting the mean", (1.0, len(rows) - 0.4), rotation=90,
                fontsize=7, ha="right", va="bottom")
    if floor is not None:
        ax.axvline(floor, c="C0", ls=":", lw=1.2)
        ax.annotate(f"basis floor {floor:.3g}", (floor, -0.4), rotation=90,
                    fontsize=7, ha="right", va="bottom", color="C0")
    ax.set(xlabel="test NMSE (0 perfect, 1 = the temporal mean)", xscale="log")
    ax.grid(alpha=0.3, axis="x")
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.4f}", (v, yi), fontsize=7, va="center",
                    xytext=(4, 0), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_error_history(per_snapshot: dict, out: str, dt: float):
    """Per-snapshot NMSE against time, one line per method.

    A good mean score hiding a handful of catastrophic frames is a completely
    different failure from a uniformly mediocre one, and only the first is
    likely to be a synchronisation glitch or a sensor dropout. The aggregate
    number cannot tell them apart.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, e in per_snapshot.items():
        ax.plot(np.arange(len(e)) * dt, e, lw=0.9, label=label, alpha=0.85)
    ax.axhline(1.0, c="k", ls="--", lw=1)
    ax.set(xlabel="time in the test block [s]", ylabel="per-snapshot NMSE",
           yscale="log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_horizon(curves: dict, out: str, dt: float):
    """Closed-loop forecast error against horizon, one line per forecaster.

    Report this, not the one-step error. Every forecaster ever built looks
    excellent one step ahead at 250 Hz, because the state barely moves.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, (h, e) in curves.items():
        ax.plot(np.asarray(h) * dt, e, ".-", lw=1.2, ms=4, label=label)
    ax.axhline(1.0, c="k", ls="--", lw=1)
    ax.set(xlabel="forecast horizon [s]", ylabel="closed-loop NMSE", yscale="log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def pick_writer(out: str, fps: int = 20):
    """An animation writer for ``out``, falling back when ffmpeg is missing.

    Returns ``(writer, path)`` -- the path may have a different extension from
    the one asked for, which is the point. ffmpeg is on almost every laptop and
    on almost no compute node unless a module is loaded, so a job that hard-codes
    ``.mp4`` dies eight hours in, at the last line, having thrown away the run.
    Producing a GIF and saying so is strictly better than that.

    mp4 is worth having where it exists: a 300-frame GIF of a 159x99 field is
    ~15 MB against ~1 MB for H.264, and GIF's 256-colour palette visibly bands a
    smooth divergent colourmap.
    """
    root, ext = os.path.splitext(out)
    want_video = ext.lower() in (".mp4", ".m4v", ".mov", ".webm")

    if want_video and shutil.which("ffmpeg") and animation.writers.is_available("ffmpeg"):
        return animation.FFMpegWriter(fps=fps, bitrate=2400,
                                      extra_args=["-pix_fmt", "yuv420p"]), out
    if want_video:
        print(f"    (no ffmpeg; writing {root}.gif instead of {ext})")
        return PillowWriter(fps=fps), root + ".gif"
    return PillowWriter(fps=fps), root + ".gif"


def animate_reconstruction(
    G_true: np.ndarray,
    G_pred: np.ndarray,
    out: str,
    mf=None,
    component: int = 0,
    n_frames: int = 200,
    fps: int = 20,
    title: str = "",
    dpi: int = 100,
) -> str:
    """True / predicted / error, animated. ``G_*`` are (Nu, Nt, Nx, Ny).

    The middle panel is what the model infers from twelve force channels and
    nothing else. Any structure that appears there was inferred from the load
    cells -- which is the claim the whole project is making, so it is worth
    being able to watch it.

    All three panels share one colour scale, taken from the truth. Autoscaling
    the prediction independently is how an under-energetic reconstruction gets
    to look correct.

    ``out`` may be ``.mp4`` or ``.gif``; see ``pick_writer`` for what happens
    when ffmpeg is not installed. Returns the path actually written.
    """
    n = min(n_frames, G_true.shape[1])
    T = G_true[component, :n]
    P = G_pred[component, :n]
    E = P - T
    lim = float(np.nanpercentile(np.abs(T), 99.5)) or 1.0
    extent = extent_from_meanfield(mf)
    err = float(np.nanmean(E**2) / np.nanvar(T)) if np.nanvar(T) > 0 else float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    ims = [
        _field_panel(axes[0], T[0], lim, extent, "PIV (truth)"),
        _field_panel(axes[1], P[0], lim, extent, "reconstructed from forces"),
        _field_panel(axes[2], E[0], lim, extent, f"error  (NMSE {err:.3f})"),
    ]
    fig.colorbar(ims[0], ax=axes, fraction=0.02, pad=0.01)
    sup = fig.suptitle(title or "frame 0", fontsize=10)

    def update(i):
        ims[0].set_data(T[i].T)
        ims[1].set_data(P[i].T)
        ims[2].set_data(E[i].T)
        sup.set_text(f"{title}  frame {i + 1}/{n}")
        return ims

    writer, path = pick_writer(out, fps)
    anim = FuncAnimation(fig, update, frames=n, blit=False)
    anim.save(path, writer=writer, dpi=dpi)
    plt.close(fig)
    return path
