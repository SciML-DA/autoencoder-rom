"""Plots for estimating a flow field from sparse sensors.

Draws POD spectra, linear observability, extended POD modes, method comparisons,
per-snapshot error, forecast error against horizon, and an animation of the
reconstruction. Every function takes plain arrays rather than estimator objects,
so results from `PODLSE` and `BranchedAE` pass through the same
functions and share one color scale.

This module doesn't select a matplotlib backend; set one, such as `Agg`, before
importing it. `field_estimation/__init__.py` doesn't import matplotlib

Typical usage example:

  import matplotlib
  matplotlib.use("Agg")
  from field_estimation import plots

  plots.plot_spectrum(Sigma, "spectrum.png", floors={"r=64": 64})
  plots.plot_observability(model.observability(), "observability.png")
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import animation
from matplotlib.animation import AbstractMovieWriter, FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from .epod import FloatArray, Observability

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

#: The `(x0, x1, y0, y1)` bounds `imshow` takes for its `extent` argument.
Extent = tuple[float, float, float, float]


class _Grid(Protocol):
    """A measurement grid with streamwise and cross-stream coordinates."""

    @property
    def x(self) -> npt.NDArray[Any]:
        """Streamwise coordinates, shape `(Nx,)`."""
        ...

    @property
    def y(self) -> npt.NDArray[Any]:
        """Cross-stream coordinates, shape `(Ny,)`."""
        ...


def extent_from_meanfield(mf: _Grid | None) -> Extent | None:
    """Computes the `imshow` extent of a measurement grid.

    Args:
      mf: A grid with `x` and `y` coordinates, such as a mean field, or `None`.

    Returns:
      The grid bounds as `(x0, x1, y0, y1)`, or `None` when `mf` is `None`.
    """
    if mf is None:
        return None
    return (float(mf.x.min()), float(mf.x.max()), float(mf.y.min()), float(mf.y.max()))


def _field_panel(
    ax: Axes,
    F: FloatArray,
    lim: float,
    extent: Extent | None,
    title: str,
    cmap: str = "RdBu_r",
) -> AxesImage:
    """Draws one field on a symmetric color scale.

    Args:
      ax: Axes to draw on.
      F: Field, shape `(Nx, Ny)`.
      lim: Magnitude of the color scale limits, `[-lim, lim]`.
      extent: Grid bounds, or `None` to use pixel coordinates.
      title: Panel title.
      cmap: Colormap name.

    Returns:
      The image, for animation updates and color bars.
    """
    # `imshow` puts rows on the y-axis, and the field is `(Nx, Ny)`.
    im = ax.imshow(
        F.T,
        origin="lower",
        cmap=cmap,
        vmin=-lim,
        vmax=lim,
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_spectrum(
    Sigma: FloatArray,
    out: str,
    n_show: int = 200,
    floors: Mapping[str, int] | None = None,
) -> FloatArray:
    """Plots the POD energy spectrum and its cumulative sum.

    Args:
      Sigma: Singular values, shape `(r,)`, in descending order.
      out: Path to save the figure to.
      n_show: Maximum number of modes to plot.
      floors: Labeled mode counts to mark on both panels, such as the truncation
        each model uses.

    Returns:
      The cumulative energy fraction, shape `(r,)`.

    Raises:
      ValueError: If `Sigma` is empty or holds no energy.
    """
    total = float(np.sum(Sigma**2))
    if Sigma.size == 0 or total == 0.0:
        raise ValueError("Sigma must hold at least one nonzero singular value")
    energy = Sigma**2 / total
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
        ax[1].annotate(
            f"{frac:.0%} at r={r}",
            (0.45, frac),
            xycoords=("axes fraction", "data"),
            fontsize=8,
            va="bottom",
        )
    for label, r in (floors or {}).items():
        for a in ax:
            a.axvline(r, ls="--", lw=0.9, alpha=0.6)
        ax[1].annotate(label, (r, 0.05), fontsize=7, rotation=90, ha="right")
    ax[1].set(xlabel="mode $k$", ylabel="cumulative energy", title="cumulative", ylim=(0, 1.02))
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return cum


def plot_observability(obs: Observability, out: str, n_show: int = 60) -> None:
    """Plots how much of each field mode the sensors resolve linearly.

    The left panel shows `rho2`, the explained fraction per field mode, with the
    best linear NMSE using modes 1 to k overlaid. The right panel shows the
    magnitude of the correlation between each field mode and each sensor mode.

    Args:
      obs: The result of `epod.mode_observability`.
      out: Path to save the figure to.
      n_show: Maximum number of field modes to plot.
    """
    rho2, R, cum = obs["rho2"], obs["R"], obs["cum"]
    n = min(n_show, len(rho2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(np.arange(1, n + 1), rho2[:n], color="C0", alpha=0.85)
    ax[0].plot(np.arange(1, n + 1), cum[:n], "k.-", lw=1, ms=4, label="best linear NMSE using modes $1..k$")
    ax[0].set(
        xlabel="field mode $k$",
        ylabel=r"explained fraction $\rho_k^2$",
        title="linear observability from the sensors",
        ylim=(0, 1.02),
    )
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")

    m = min(n, R.shape[0])
    j = min(R.shape[1], 40)
    im = ax[1].imshow(np.abs(R[:m, :j]), cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax[1].set(xlabel="sensor mode $j$", ylabel="field mode $k$", title=r"$|corr(b_k, c_j)|$")
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_extended_modes(
    Psi_ext: FloatArray,
    unflat: Callable[[FloatArray], FloatArray],
    out: str,
    n_modes: int = 6,
    mf: _Grid | None = None,
) -> None:
    """Plots the leading extended POD modes, one column per mode.

    Each column shows the streamwise and cross-stream components of the flow
    associated with one sensor mode.

    Args:
      Psi_ext: Extended modes in the flat layout, shape `(N_x, r_s)`.
      unflat: Maps flat fields `(N_x, n)` to the grid `(Nu, n, Nx, Ny)`.
      out: Path to save the figure to.
      n_modes: Maximum number of modes to plot.
      mf: Grid to take the axis bounds from, or `None` for pixel coordinates.

    Raises:
      ValueError: If `unflat` returns fewer than two velocity components.
    """
    n = min(n_modes, Psi_ext.shape[1])
    G = unflat(Psi_ext[:, :n])
    if G.ndim != 4 or G.shape[0] < 2:
        raise ValueError(f"unflat must return (Nu >= 2, n, Nx, Ny), got shape {G.shape}")
    extent = extent_from_meanfield(mf)

    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.0), squeeze=False)
    for j in range(n):
        for row, comp in enumerate(("u", "v")):
            F = G[row, j]
            lim = np.nanmax(np.abs(F)) or 1.0
            _field_panel(axes[row][j], F, lim, extent, f"$\\psi^{{ext}}_{{{j + 1}}}$ ({comp})")
    fig.suptitle("extended POD modes -- the flow correlated with each sensor mode")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_comparison(
    rows: Sequence[Mapping[str, Any]],
    out: str,
    floor: float | None = None,
) -> None:
    """Plots test NMSE per method as horizontal bars, best first.

    Marks NMSE = 1.0, and the projection
    floor when given.

    Args:
      rows: One result per method. Each must hold `"label"`, the method name,
        and `"nmse_test"`, its test NMSE.
      out: Path to save the figure to.
      floor: Projection floor of the field basis, or `None` to omit it.
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
    ax.annotate("predicting the mean", (1.0, len(rows) - 0.4), rotation=90, fontsize=7, ha="right", va="bottom")
    if floor is not None:
        ax.axvline(floor, c="C0", ls=":", lw=1.2)
        ax.annotate(
            f"basis floor {floor:.3g}",
            (floor, -0.4),
            rotation=90,
            fontsize=7,
            ha="right",
            va="bottom",
            color="C0",
        )
    ax.set(xlabel="test NMSE (0 perfect, 1 = the temporal mean)", xscale="log")
    ax.grid(alpha=0.3, axis="x")
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.4f}", (v, yi), fontsize=7, va="center", xytext=(4, 0), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_error_history(per_snapshot: Mapping[str, FloatArray], out: str, dt: float) -> None:
    """Plots per-snapshot NMSE against time, one line per method.

    Args:
      per_snapshot: Per-snapshot NMSE for each method, keyed by label.
      out: Path to save the figure to.
      dt: Time between consecutive snapshots, in seconds.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, e in per_snapshot.items():
        ax.plot(np.arange(len(e)) * dt, e, lw=0.9, label=label, alpha=0.85)
    ax.axhline(1.0, c="k", ls="--", lw=1)
    ax.set(xlabel="time in the test block [s]", ylabel="per-snapshot NMSE", yscale="log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_horizon(
    curves: Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike]],
    out: str,
    dt: float,
) -> None:
    """Plots closed-loop forecast error against horizon, one line per forecaster.

    Args:
      curves: For each forecaster, keyed by label, the horizons in steps and the
        closed-loop NMSE at each.
      out: Path to save the figure to.
      dt: Time between consecutive snapshots, in seconds.
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


def pick_writer(out: str, fps: int = 20) -> tuple[AbstractMovieWriter, str]:
    """Chooses an animation writer for an output path.

    Writes video through ffmpeg when `out` has a video extension and ffmpeg is
    installed. Otherwise writes a GIF, and replaces the extension to match.

    Args:
      out: Requested output path.
      fps: Frames per second.

    Returns:
      The writer and the path it writes to, which can differ from `out` in its
      extension.

    Raises:
      ValueError: If `fps` is below 1.
    """
    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}")
    root, ext = os.path.splitext(out)
    want_video = ext.lower() in (".mp4", ".m4v", ".mov", ".webm")

    if want_video and shutil.which("ffmpeg") and animation.writers.is_available("ffmpeg"):
        return animation.FFMpegWriter(fps=fps, bitrate=2400, extra_args=["-pix_fmt", "yuv420p"]), out
    if want_video:
        print(f"    (no ffmpeg; writing {root}.gif instead of {ext})")
        return PillowWriter(fps=fps), root + ".gif"
    return PillowWriter(fps=fps), root + ".gif"


def animate_reconstruction(
    G_true: FloatArray,
    G_pred: FloatArray,
    out: str,
    mf: _Grid | None = None,
    component: int = 0,
    n_frames: int = 200,
    fps: int = 20,
    title: str = "",
    dpi: int = 100,
) -> str:
    """Animates the true field, the reconstruction, and their difference.

    All three panels share one color scale, set from the true field.

    Args:
      G_true: True fields, shape `(Nu, Nt, Nx, Ny)`.
      G_pred: Reconstructed fields, with the same `Nu`, `Nx`, and `Ny`, and at
        least as many snapshots as are animated.
      out: Requested output path, such as `.mp4` or `.gif`. See `pick_writer`.
      mf: Grid to take the axis bounds from, or `None` for pixel coordinates.
      component: Velocity component to animate.
      n_frames: Maximum number of frames.
      fps: Frames per second.
      title: Title shown above the panels.
      dpi: Resolution of each frame.

    Returns:
      The path written, which can differ from `out` in its extension.

    Raises:
      ValueError: If the arrays are not four-dimensional or disagree in shape,
        if `component` is out of range, or if `n_frames` or `fps` is below 1.
    """
    if G_true.ndim != 4 or G_pred.ndim != 4:
        raise ValueError(f"G_true and G_pred must be (Nu, Nt, Nx, Ny), got {G_true.shape} and {G_pred.shape}")
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    n = min(n_frames, G_true.shape[1])
    true_space = (G_true.shape[0], *G_true.shape[2:])
    pred_space = (G_pred.shape[0], *G_pred.shape[2:])
    if true_space != pred_space or G_pred.shape[1] < n:
        raise ValueError(
            f"G_pred {G_pred.shape} must match G_true {G_true.shape} in Nu, Nx, and Ny and hold at least {n} snapshots"
        )
    if not 0 <= component < G_true.shape[0]:
        raise ValueError(f"component must be in [0, {G_true.shape[0]}), got {component}")

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

    def update(i: int) -> list[AxesImage]:
        """Draws frame `i` into the three panels.

        Args:
          i: Frame index.

        Returns:
          The updated images.
        """
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
