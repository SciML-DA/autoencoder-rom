#!/usr/bin/env python
"""
make_reconstruction_video.py
============================

Render the reconstruction video from a ``video_pack.npz`` written by
``sparse_sensor_sweep.py`` or ``sparse_sensor_study.py``.

    python scripts/make_reconstruction_video.py results/sparse_sweep/main/video_pack.npz
    python scripts/make_reconstruction_video.py <pack> --side-by-side
    python scripts/make_reconstruction_video.py <pack> --component v --fps 30 --format gif

Why this is a separate script
-----------------------------
The sweep already renders on the compute node, but a node almost never has
ffmpeg, so what you get there is a GIF: 256 colours, which visibly bands a
smooth divergent colourmap, and roughly fifteen times the file size of the same
footage in H.264. Your laptop does have ffmpeg.

So the pack is the thing to sync. It is a few tens of megabytes -- the truth is
stored once and shared by every prediction -- against gigabytes for the raw
snapshots, and rendering from it is seconds. Re-render as often as you like with
different framerates, components and colour limits without touching the cluster.

**This script deliberately imports nothing from ``src/``.** numpy and matplotlib
and nothing else, so it runs anywhere you happen to have the pack, including a
machine with no torch and no JAX.

What is in a pack
-----------------
``truth``      (Nu, n_frames, Nx, Ny) float32, NaN at masked points
``pred_<i>``   the same shape, one per model kept
``x``, ``y``   grid coordinates in mm
``meta``       JSON: per-prediction label and NMSE, the timestep, the run

    python scripts/make_reconstruction_video.py <pack> --list
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

COMPONENTS = {"u": 0, "v": 1}


def load_pack(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {"labels": {}}
    preds = {k: z[k] for k in z.files if k.startswith("pred_")}
    return z["truth"], preds, meta, z["x"], z["y"]


def pick_writer(out, fps):
    """mp4 where ffmpeg exists, GIF otherwise. Returns (writer, actual_path)."""
    root, ext = os.path.splitext(out)
    if ext.lower() == ".gif":
        return PillowWriter(fps=fps), out
    if shutil.which("ffmpeg") and animation.writers.is_available("ffmpeg"):
        return animation.FFMpegWriter(fps=fps, bitrate=3000,
                                      extra_args=["-pix_fmt", "yuv420p"]), out
    print(f"  no ffmpeg on PATH -- writing {root}.gif instead")
    return PillowWriter(fps=fps), root + ".gif"


def _panel(ax, F, vmin, vmax, extent, title, cmap="RdBu_r"):
    # .T because the arrays are (Nx, Ny) and imshow wants row = y
    im = ax.imshow(F.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def prepare(truth, preds, view, comp, clim):
    """Apply the view and settle the colour scale. Returns (T, {k: P}, scales).

    ``view="fluctuation"`` (the default) subtracts the temporal mean of the
    *truth* from everything. This is not cosmetic: the streamwise component has
    a ~10 m/s mean, so on a divergent scale centred at zero the total field is a
    uniform block of colour and the wake -- the entire content of the video --
    is invisible. It is also the honest view, because the mean is subtracted
    before the decomposition and is not something the model reconstructs; NMSE
    is measured on exactly what this shows.

    ``view="total"`` puts the mean back and switches to a sequential map with
    limits from the data. Use it for a figure that has to look like a wake to
    someone who has not read the methods.

    The same scale is used for every panel, always. Autoscaling a prediction
    independently of the truth is how an under-energetic reconstruction gets to
    look correct.
    """
    T = truth[comp].astype(np.float64)
    P = {k: v[comp].astype(np.float64) for k, v in preds.items()}
    if view == "fluctuation":
        mean = np.nanmean(T, axis=0, keepdims=True)
        T = T - mean
        P = {k: v - mean for k, v in P.items()}
        lim = clim or float(np.nanpercentile(np.abs(T), 99.5)) or 1.0
        return T, P, dict(vmin=-lim, vmax=lim, cmap="RdBu_r", err_cmap="RdBu_r",
                          err_vmin=-lim, err_vmax=lim, unit="fluctuation")
    lo, hi = np.nanpercentile(T, [0.5, 99.5])
    elim = clim or float(np.nanpercentile(np.abs(T - np.nanmean(T, 0, keepdims=True)),
                                          99.5)) or 1.0
    return T, P, dict(vmin=float(lo), vmax=float(hi), cmap="viridis",
                      err_cmap="RdBu_r", err_vmin=-elim, err_vmax=elim, unit="total")


def render_triptych(T, P, out, extent, label, nmse, comp, fps, dpi, dt, sc):
    """truth | reconstruction | error, one model. The default, and usually the one you want.

    Truth and prediction share one scale; the error gets its own, centred at
    zero, because in the total-field view the error is a small signed quantity
    on top of a large positive one and plotting it on the field's scale renders
    it as a blank panel.
    """
    E = P - T
    n = T.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7))
    ims = [_panel(axes[0], T[0], sc["vmin"], sc["vmax"], extent, "PIV (truth)", sc["cmap"]),
           _panel(axes[1], P[0], sc["vmin"], sc["vmax"], extent,
                  "reconstructed from forces", sc["cmap"]),
           _panel(axes[2], E[0], sc["err_vmin"], sc["err_vmax"], extent,
                  "error", sc["err_cmap"])]
    cb = fig.colorbar(ims[0], ax=axes[:2], fraction=0.02, pad=0.01)
    cb.set_label(f"${'uv'[comp]}$ [m/s], {sc['unit']}", fontsize=8)
    cbe = fig.colorbar(ims[2], ax=axes[2], fraction=0.04, pad=0.02)
    cbe.set_label("error [m/s]", fontsize=8)
    sup = fig.suptitle("", fontsize=10)

    def update(i):
        ims[0].set_data(T[i].T)
        ims[1].set_data(P[i].T)
        ims[2].set_data(E[i].T)
        sup.set_text(f"{label}   NMSE {nmse:.3f}   t = {i * dt:.3f} s   "
                     f"({i + 1}/{n})")
        return ims

    _save(fig, update, n, out, fps, dpi)


def render_side_by_side(T, items, out, extent, comp, fps, dpi, dt, sc):
    """Truth on the left, every model beside it, on one shared colour scale.

    This is the figure that actually makes an argument: two reconstructions
    plotted separately can always be made to look similar; side by side at a
    fixed scale they cannot.
    """
    n = T.shape[0]
    items = sorted(items, key=lambda it: it[1])  # by NMSE, best first

    ncol = 1 + len(items)
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 4.0), squeeze=False)
    axes = axes[0]
    ims = [_panel(axes[0], T[0], sc["vmin"], sc["vmax"], extent, "PIV (truth)", sc["cmap"])]
    for ax, (P, err, label) in zip(axes[1:], items):
        ims.append(_panel(ax, P[0], sc["vmin"], sc["vmax"], extent,
                          f"{label}\nNMSE {err:.3f}", sc["cmap"]))
    cb = fig.colorbar(ims[0], ax=axes, fraction=0.015, pad=0.01)
    cb.set_label(f"${'uv'[comp]}$ [m/s], {sc['unit']}", fontsize=8)
    sup = fig.suptitle("", fontsize=10)

    def update(i):
        ims[0].set_data(T[i].T)
        for im, (P, _, _) in zip(ims[1:], items):
            im.set_data(P[i].T)
        sup.set_text(f"t = {i * dt:.3f} s   ({i + 1}/{n})")
        return ims

    _save(fig, update, n, out, fps, dpi)


def _save(fig, update, n, out, fps, dpi):
    writer, path = pick_writer(out, fps)
    FuncAnimation(fig, update, frames=n, blit=False).save(path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB, {n} frames)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pack", help="path to video_pack.npz")
    p.add_argument("--out", default=None, help="output dir (default: next to the pack)")
    p.add_argument("--list", action="store_true", help="list what is in the pack and exit")
    p.add_argument("--models", nargs="*", default=None,
                   help="pred_ keys to render (default: all)")
    p.add_argument("--side-by-side", action="store_true",
                   help="one video with truth and every model on a shared scale")
    p.add_argument("--component", default="u", choices=list(COMPONENTS))
    p.add_argument("--view", default="fluctuation", choices=["fluctuation", "total"],
                   help="fluctuation (default) subtracts the truth's temporal mean; "
                        "total keeps it and uses a sequential colourmap")
    p.add_argument("--format", default="mp4", choices=["mp4", "gif"])
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--frames", type=int, default=None, help="cap the frame count")
    p.add_argument("--clim", type=float, default=None,
                   help="fixed colour limit [m/s]; default is the truth's 99.5th pct")
    args = p.parse_args()

    truth, preds, meta, x, y = load_pack(args.pack)
    labels = meta.get("labels", {})
    dt = float(meta.get("dt", 1 / 250.0))
    out_dir = args.out or os.path.dirname(os.path.abspath(args.pack))
    os.makedirs(out_dir, exist_ok=True)

    print(f"pack   : {args.pack}")
    print(f"truth  : {truth.shape}  (Nu, frames, Nx, Ny)   dt={dt * 1e3:.1f} ms")
    for k in sorted(preds):
        info = labels.get(k, {})
        print(f"  {k:10s} {info.get('label', '?'):40s} NMSE {info.get('nmse', float('nan')):.4f}"
              f"{'  (pinned)' if info.get('pinned') else ''}")
    if args.list:
        return 0

    keys = args.models or sorted(preds)
    missing = [k for k in keys if k not in preds]
    if missing:
        print(f"\nnot in the pack: {missing}. Available: {sorted(preds)}")
        return 2

    if args.frames:
        truth = truth[:, : args.frames]
        preds = {k: v[:, : args.frames] for k, v in preds.items()}

    comp = COMPONENTS[args.component]
    extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    T, P, sc = prepare(truth, {k: preds[k] for k in keys}, args.view, comp, args.clim)
    print(f"\nview {args.view}, colour scale [{sc['vmin']:.2f}, {sc['vmax']:.2f}] m/s")

    if args.side_by_side:
        items = [(P[k], float(labels.get(k, {}).get("nmse", np.nan)),
                  labels.get(k, {}).get("label", k)) for k in keys]
        render_side_by_side(
            T, items,
            os.path.join(out_dir, f"comparison_{args.component}_{args.view}.{args.format}"),
            extent, comp, args.fps, args.dpi, dt, sc)
    else:
        for k in keys:
            info = labels.get(k, {})
            render_triptych(
                T, P[k],
                os.path.join(out_dir, f"{k}_{args.component}_{args.view}.{args.format}"),
                extent, info.get("label", k), float(info.get("nmse", np.nan)),
                comp, args.fps, args.dpi, dt, sc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
