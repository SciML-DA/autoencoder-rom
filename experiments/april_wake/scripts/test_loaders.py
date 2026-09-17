#!/usr/bin/env python
"""
test_loaders.py
===============

End-to-end smoke test: load the April experiment, fit POD-LSE, reconstruct the
velocity field from the load cells alone, and animate the result.

    python experiments/april_wake/scripts/test_loaders.py                      # defaults
    python experiments/april_wake/scripts/test_loaders.py --n 2000 --r-field 40
    python experiments/april_wake/scripts/test_loaders.py --run 4p5d_10ms_yaw_30_15_0

Outputs into ``results/loaders/``:

    spectrum.png      POD energy spectrum and cumulative energy
    reconstruction.gif  true / predicted / error, animated over the test block
    modes.png         the leading spatial POD modes

The reconstruction animation is the point. The left panel is PIV; the middle
panel is what POD-LSE infers from twelve force channels and nothing else. Any
structure that appears in the middle panel was inferred from the load cells.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless: cx3 login/compute nodes have no display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

# this file is experiments/april_wake/scripts/<name>.py, so three levels
# up is the repo root -- which is what makes `experiments` importable.
# `src` needs no insert: the editable install puts it on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from datasets import split_indices  # noqa: E402
from experiments.april_wake.case_reader import (  # noqa: E402
    F_PIV_HZ,
    find_force_files,
    load_meanfield,
    load_run,
    pair_indices,
    read_dat,
    sync_forces,
)
from field_estimation.epod import PODLSE, nmse, pod  # noqa: E402

# ── data ──────────────────────────────────────────────────────────────────────


def yaw_from_run(run: str) -> tuple:
    """('4p5d_10ms_yaw_30_15_0') -> (30, 15). The force files name only the
    first two discs, which is also all that is instrumented."""
    tail = run.split("yaw_")[-1].split("_")
    return (int(tail[0]), int(tail[1]))


def load_case(run: str, n_snapshots: int, method: str = "block"):
    """Return (Q, S, grid_shape, meanfield).

    Q is (N_x, N_t) with N_x = 2 * Nx * Ny -- u stacked on top of v, columns as
    snapshots, which is what `field_estimation.epod` expects throughout.
    """
    print(f"loading {run} ({n_snapshots} snapshots) ...")
    X = load_run(run, max_snapshots=n_snapshots)  # (2, Nt, Nx, Ny)
    pairs = pair_indices(run, max_snapshots=n_snapshots)
    _, n_t, n_x, n_y = X.shape

    # NaN -> 0. This matches Roman's reference notebook, but it is a real
    # modelling choice, not a no-op: those points then read as "zero velocity"
    # rather than "no data", and POD will happily spend modes describing the
    # hole. It is defensible here only because the invalid fraction is ~0.7%
    # and the mean is subtracted before the decomposition.
    n_bad = int(np.isnan(X).sum())
    Q = np.nan_to_num(X.reshape(2, n_t, n_x * n_y)).transpose(0, 2, 1)
    Q = np.concatenate([Q[0], Q[1]], axis=0)  # (2*Nx*Ny, Nt)
    print(f"  field    Q {Q.shape}   ({n_bad} NaN zeroed, {100 * n_bad / X.size:.2f}%)")

    recs = find_force_files()
    yaw = yaw_from_run(run)
    match = [r for r in recs if r.is_sync and r.yaw_key == yaw]
    if not match:
        raise FileNotFoundError(
            f"no sync force file for yaw {yaw}; have {sorted({r.yaw_key for r in recs if r.is_sync})}"
        )
    F = read_dat(match[0].path)
    S = sync_forces(F, pairs, method=method)  # (12, Nt)
    print(f"  sensors  S {S.shape}   from {os.path.basename(match[0].path)}")

    try:
        mf = load_meanfield()
    except Exception as e:  # the mean field is only for axes labels
        print(f"  (no mean field: {e})")
        mf = None
    return Q, S, (n_x, n_y), mf


# ── plots ─────────────────────────────────────────────────────────────────────


def plot_spectrum(Q, out):
    _, Sigma, _, _ = pod(Q, subtract_mean=True)
    energy = Sigma**2 / np.sum(Sigma**2)
    cum = np.cumsum(energy)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    k = np.arange(1, min(len(Sigma), 200) + 1)
    ax[0].semilogy(k, energy[: len(k)], ".-", lw=1, ms=4)
    ax[0].set(xlabel="mode $k$", ylabel="energy fraction $\\sigma_k^2/\\sum\\sigma^2$", title="POD spectrum")
    ax[0].grid(alpha=0.3)

    ax[1].plot(k, cum[: len(k)], ".-", lw=1, ms=4)
    for frac in (0.9, 0.95, 0.99):
        r = int(np.searchsorted(cum, frac)) + 1
        ax[1].axhline(frac, ls=":", c="k", lw=0.8)
        ax[1].annotate(f"{frac:.0%} at r={r}", (0.4, frac), xycoords=("axes fraction", "data"), fontsize=8, va="bottom")
    ax[1].set(xlabel="mode $k$", ylabel="cumulative energy", title="cumulative", ylim=(0, 1.02))
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")
    return Sigma, cum


def plot_modes(Q, grid, mf, out, n_modes=6):
    n_x, n_y = grid
    Psi, _, _, _ = pod(Q, r=n_modes, subtract_mean=True)
    ext = _extent(mf)

    fig, axes = plt.subplots(2, 3, figsize=(13, 5.5))
    for k, ax in enumerate(axes.ravel()):
        mode = Psi[: n_x * n_y, k].reshape(n_x, n_y)
        lim = np.abs(mode).max()
        ax.imshow(mode.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext, aspect="equal")
        ax.set_title(f"mode {k + 1} ($u$)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("leading POD spatial modes, streamwise component")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def _extent(mf):
    """imshow extent in mm, or None to fall back to pixel indices."""
    if mf is None:
        return None
    return [float(mf.x.min()), float(mf.x.max()), float(mf.y.min()), float(mf.y.max())]


def animate(Q_true, Q_pred, grid, mf, out, n_frames=200, fps=20):
    """True / predicted / error, animated. Streamwise component only."""
    n_x, n_y = grid
    n_pix = n_x * n_y
    n_frames = min(n_frames, Q_true.shape[1])
    step = max(1, Q_true.shape[1] // n_frames)
    frames = range(0, n_frames * step, step)

    def grid_of(Qm, t):
        return Qm[:n_pix, t].reshape(n_x, n_y).T  # -> (Ny, Nx) for imshow

    ext = _extent(mf)
    # Fixed colour limits across the whole animation. Per-frame autoscaling
    # makes a good reconstruction and a bad one look identical, because each
    # panel silently renormalises to its own range.
    vmin, vmax = np.percentile(Q_true[:n_pix], [1, 99])
    err = Q_true[:n_pix] - Q_pred[:n_pix]
    elim = np.percentile(np.abs(err), 99)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    ims = []
    for ax, title, cmap, lo, hi in (
        (axes[0], "PIV (truth)", "Blues_r", vmin, vmax),
        (axes[1], "POD-LSE from 12 force channels", "Blues_r", vmin, vmax),
        (axes[2], "error", "RdBu_r", -elim, elim),
    ):
        im = ax.imshow(
            np.zeros((n_y, n_x)),
            origin="lower",
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            extent=ext,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ims.append(im)
    txt = fig.suptitle("")
    fig.tight_layout()

    def update(t):
        a, b = grid_of(Q_true, t), grid_of(Q_pred, t)
        ims[0].set_data(a)
        ims[1].set_data(b)
        ims[2].set_data(a - b)
        txt.set_text(f"t = {t / F_PIV_HZ:.3f} s   (frame {t})")
        return ims

    print(f"  rendering {len(list(frames))} frames ...")
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {out}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="POD-LSE smoke test on the April experiment")
    p.add_argument("--run", default="4p5d_10ms_yaw_0_0_0")
    p.add_argument("--n", type=int, default=1500, help="snapshots to load")
    p.add_argument("--r-field", type=int, default=30)
    p.add_argument("--r-sensor", type=int, default=10)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--sync", default="block", choices=["block", "decimate", "nearest"])
    p.add_argument("--gap", type=int, default=100, help="guard band between splits")
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--out", default="results/loaders")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    Q, S, grid, mf = load_case(args.run, args.n, args.sync)

    print("\nPOD spectrum ...")
    _, cum = plot_spectrum(Q, os.path.join(args.out, "spectrum.png"))
    print(f"  90% energy at r={int(np.searchsorted(cum, 0.90)) + 1}, 99% at r={int(np.searchsorted(cum, 0.99)) + 1}")
    plot_modes(Q, grid, mf, os.path.join(args.out, "modes.png"))

    # Contiguous split with a guard band. A random split leaks badly here: at
    # 250 Hz consecutive snapshots are nearly identical, so a randomly held-out
    # frame has near-copies of itself in the training set and the score is
    # meaningless.
    tr, _, te = split_indices(Q.shape[1], val_frac=0, test_frac=0.25, gap=args.gap)
    print(f"\nsplit: train {tr[0]}..{tr[-1]} ({len(tr)}), test {te[0]}..{te[-1]} ({len(te)}), gap {args.gap}")

    print(f"\nfitting POD-LSE (r_field={args.r_field}, r_sensor={args.r_sensor}, ridge={args.ridge:g}) ...")
    model = PODLSE(r_field=args.r_field, r_sensor=args.r_sensor, ridge=args.ridge).fit(Q[:, tr], S[:, tr])

    train_err = model.score(Q[:, tr], S[:, tr])
    test_err = model.score(Q[:, te], S[:, te])
    baseline = nmse(Q[:, te], np.repeat(Q[:, tr].mean(1, keepdims=True), len(te), axis=1))

    print(f"\n  nmse train        : {train_err:.4f}")
    print(f"  nmse test         : {test_err:.4f}")
    print(f"  nmse mean-predictor: {baseline:.4f}   <- must beat this")
    if test_err >= baseline:
        print("  !! not beating the temporal mean. The sensors are contributing")
        print("     nothing: check the PIV/force pairing first, then the drift.")
    else:
        print(f"  -> {100 * (1 - test_err / baseline):.1f}% better than the mean")

    print("\nanimating test block ...")
    Q_pred = model.predict(S[:, te])
    animate(Q[:, te], Q_pred, grid, mf, os.path.join(args.out, "reconstruction.gif"), n_frames=args.frames)

    print(f"\ndone -> {args.out}/\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
