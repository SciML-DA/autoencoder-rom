#!/usr/bin/env python
"""
loss_curves.py
==============

Per-epoch training and validation loss for every sensor-branch model, in the
style of `convergence_study.py`'s `loss_curves_latent*.png`.

    qsub experiments/april_wake/hpc/loss_curves.pbs
    python experiments/april_wake/scripts/loss_curves.py --tag noise_band --sensor-noise 0.1 --band-hz 40

The sweeps report one score per fit and throw the trajectory away, which is
enough to rank models and not enough to say why one lost. These are the curves:
whether a model was still descending when it stopped, whether its validation
loss turned up while its training loss kept falling, and how far apart the two
ran.

Three things the score column cannot tell you and these can:

  * A model that early-stopped on a plateau and one that stopped because the
    validation loss started rising are the same number and opposite problems.
  * A training curve still descending at the stop says the budget bound the
    fit; a flat one says the budget was ample and something else bound it.
  * The train/validation separation is the overfitting, drawn rather than
    inferred from two numbers.

Models are plotted a few to a panel, grouped by latent space, because eight
overlapping curves on one axis is not a plot anybody reads.

The closed-form PODLSE has no epochs. It appears as a horizontal reference on
the validation panel: the score a linear solve reaches without any of this.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# this file is experiments/april_wake/scripts/<name>.py, so three levels
# up is the repo root -- which is what makes `experiments` importable.
# `src` needs no insert: the editable install puts it on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.april_wake.data_preprocessing import (  # noqa: E402
    add_data_args,
    band_limit,
    load_data,
    make_split,
)
from field_estimation.branched_ae import BranchedAE, LinearLatent, TorchLatent  # noqa: E402
from field_estimation.epod import PODLSE, delay_embed, nmse, pod  # noqa: E402

# one colour per branch, one line style per latent space, so a panel that mixes
# them stays readable and a reader can tell the two axes apart at a glance
COLOR = {"linear": "C0", "mlp": "C1", "cnn": "C2", "gru": "C3"}
LS = {"pod": "-", "ae": "--"}


def rule(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}", flush=True)


def build_latents(Q, tr, unflat, args):
    """POD and autoencoder latent spaces, both fitted on the training block."""
    out = {}
    if "pod" in args.latents:
        Psi, _, _, qm = pod(Q[:, tr], r=args.r_field, subtract_mean=True, method=args.pod_method)
        out["pod"] = LinearLatent(Psi, qm, device=args.device)
        print(f"  pod latent r={args.r_field}")
    if "ae" in args.latents:
        from models.data_driven.autoencoders import AE

        dims = tuple(max(int(f * args.r_field), args.r_field + 1) for f in args.ae_hidden_scale)
        t0 = time.time()
        p = AE(
            n_latent=args.r_field,
            hidden=dims,
            epochs=args.ae_epochs,
            batch_size=args.ae_batch,
            learning_rate=args.ae_lr,
            patience=args.ae_patience,
            device=args.device,
            seed=args.seed,
        )
        p.fit(unflat(Q[:, tr], dtype=np.float32))
        out["ae"] = TorchLatent(p, device=args.device)
        print(f"  ae latent r={args.r_field} dims={dims}: {time.time() - t0:.0f}s")
    return out


def fit_one(Q, S, tr, te, lat, args, latent_name, branch):
    """One branched fit, keeping the whole loss trajectory.

    `track_sets` scores the model on the training block and on the held-out
    test block in field NMSE after every `track_every` epochs. That pair is the
    overfitting diagnostic: the optimiser's own loss curves are latent-space
    objectives on a holdout carved out of training, and they say whether the
    fit is still descending, not whether it generalises.
    """
    t0 = time.time()
    m = BranchedAE(
        lat,
        branch=branch,
        n_delays=args.n_delays,
        delay_stride=args.delay_stride,
        hidden=tuple(args.hidden),
        gru_hidden=args.gru_hidden,
        cnn_channels=tuple(args.cnn_channels),
        lambda_field=args.lambda_field,
        latent_weight=args.latent_weight,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        sensor_noise=args.sensor_noise,
        epochs=args.epochs,
        batch_size=args.batch,
        val_fraction=args.val_fraction,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        track_sets={"train": (Q, S, tr), "test": (Q, S, te)},
        track_every=args.track_every,
        track_max_cols=args.track_cols,
    ).fit(Q, S, tr)

    hist = list(m.loss_history or [])
    vhist = list(m.val_loss_history or [])
    best_ep = int(np.argmin(vhist)) + 1 if vhist else float("nan")
    row = dict(
        latent=latent_name,
        branch=branch,
        label=f"{latent_name}+{branch}",
        nmse_train=m.score(Q, S, tr),
        nmse_test=nmse(Q[:, te], m.predict(S, te)),
        epochs_run=len(hist),
        best_epoch=best_ep,
        stopped_early=int(bool(hist) and len(hist) < args.epochs),
        train_loss=hist[-1] if hist else float("nan"),
        val_loss=min(vhist) if vhist else float("nan"),
        n_params=m.n_params,
        seconds=time.time() - t0,
        history_train=hist,
        history_val=vhist,
        track=m.track_history,
    )
    print(
        f"  {row['label']:<14} test {row['nmse_test']:.4f}  "
        f"train {row['nmse_train']:.4f}  {len(hist):>5} epochs"
        f"{'*' if row['stopped_early'] else ' '}  best@{best_ep}  "
        f"{row['seconds']:.0f}s",
        flush=True,
    )
    return row


def save_generalisation(rows, outpath, title, ref=None):
    """Train and test NMSE on ONE axis, per epoch. The overfitting picture.

    Two curves on separate panels cannot show a gap; the gap is the diagnostic.
    Solid is training, dashed is the held-out test block, both in field NMSE.

      * both falling together      -> still fitting, budget is the limit
      * train falls, test flattens -> the useful capacity is spent
      * train falls, test RISES    -> overfitting from that epoch onwards
      * both flat and high         -> underfitting, or nothing left to learn
    """
    rows = [r for r in rows if r.get("track")]
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), sharey=True, squeeze=False)
    for ax, r in zip(axes[0], rows):
        for name, ls in (("train", "-"), ("test", "--")):
            pts = r["track"].get(name, [])
            if not pts:
                continue
            ep = [p[0] for p in pts]
            v = [p[1] for p in pts]
            ax.plot(ep, v, ls=ls, lw=1.5, color=COLOR.get(r["branch"], "C4"), label=name)
        if ref is not None and np.isfinite(ref):
            ax.axhline(ref, color="k", ls="-.", lw=1.1, label="PODLSE")
        if np.isfinite(r["best_epoch"]):
            ax.axvline(r["best_epoch"] - 1, color="0.4", ls=":", lw=1.0)
        ax.set(title=r["label"], xlabel="epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0][0].set_ylabel("field NMSE")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")


def save_panel(rows, outpath, title, ref=None):
    """Train and validation loss for the rows given, two panels."""
    rows = [r for r in rows if r["history_train"]]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for r in rows:
        c, ls = COLOR.get(r["branch"], "C4"), LS.get(r["latent"], "-")
        axes[0].plot(r["history_train"], color=c, ls=ls, lw=1.3, label=r["label"])
        if r["history_val"]:
            axes[1].plot(r["history_val"], color=c, ls=ls, lw=1.3, label=r["label"])
            if np.isfinite(r["best_epoch"]):
                # where early stopping actually took the weights from
                axes[1].axvline(r["best_epoch"] - 1, color=c, ls=":", lw=0.9, alpha=0.7)
    for ax, lab in zip(axes, ("training loss", "validation loss")):
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, which="both")
    if ref is not None and np.isfinite(ref):
        axes[1].axhline(ref, color="k", ls="-.", lw=1.2, label="PODLSE (closed form)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")


def save_summary(rows, ref, outpath, title):
    """Final train vs test per model, with the closed-form line."""
    if not rows:
        return
    rs = sorted(rows, key=lambda r: r["nmse_test"], reverse=True)
    y = np.arange(len(rs))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8, 0.52 * len(rs) + 2))
    ax.barh(y + h / 2, [r["nmse_test"] for r in rs], height=h, color="C0", label="test")
    ax.barh(y - h / 2, [r["nmse_train"] for r in rs], height=h, color="C1", label="train")
    if np.isfinite(ref):
        ax.axvline(ref, color="k", ls="--", lw=1.4, label="PODLSE (closed form)")
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rs], fontsize=8)
    ax.set(xlabel="NMSE", title=title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    p.add_argument("--delays", type=int, nargs="+", default=[25])
    p.add_argument("--delay-stride", type=int, default=1)

    m = p.add_argument_group("model")
    m.add_argument("--r-field", type=int, default=16)
    m.add_argument("--r-sensor", type=int, default=None)
    m.add_argument("--n-delays", type=int, default=25)
    m.add_argument("--ridge", type=float, default=1e-3)
    m.add_argument("--pod-method", default="randomized", choices=["svd", "snapshot", "randomized", "auto"])
    m.add_argument("--latents", nargs="+", default=["pod", "ae"], choices=["pod", "ae"])
    m.add_argument(
        "--branches", nargs="+", default=["linear", "mlp", "cnn", "gru"], choices=["linear", "mlp", "cnn", "gru"]
    )
    m.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    m.add_argument("--gru-hidden", type=int, default=64)
    m.add_argument("--cnn-channels", type=int, nargs="+", default=[32, 64])
    m.add_argument("--ae-hidden-scale", type=float, nargs="+", default=[8, 2])

    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=2000)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--patience", type=int, default=200)
    t.add_argument("--val-fraction", type=float, default=0.2)
    t.add_argument("--weight-decay", type=float, default=0.0)
    t.add_argument("--sensor-noise", type=float, default=0.0)
    t.add_argument("--lambda-field", type=float, default=0.0)
    t.add_argument("--latent-weight", default="energy", choices=["energy", "unit"])
    t.add_argument("--ae-epochs", type=int, default=400)
    t.add_argument("--ae-batch", type=int, default=64)
    t.add_argument("--ae-lr", type=float, default=1e-3)
    t.add_argument("--ae-patience", type=int, default=40)
    t.add_argument("--track-every", type=int, default=5, help="epochs between train/test NMSE evaluations")
    t.add_argument("--track-cols", type=int, default=400, help="snapshots sampled from each block when tracking")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None)

    o = p.add_argument_group("output")
    o.add_argument("--out", default="results/loss_curves")
    o.add_argument("--tag", default="default")
    args = p.parse_args()

    out = os.path.join(args.out, args.tag)
    os.makedirs(out, exist_ok=True)

    rule("0. data")
    Q, S, unflat, case0, run_id, cases = load_data(args)
    Q_full = None
    if args.band_hz:
        Q_full = Q
        Q = band_limit(Q, args.band_hz, 250.0, run_id)
    tr, te = make_split(args, Q.shape[1], run_id, cases)[:2]

    rule("1. closed-form reference")
    Sd = delay_embed(S, args.n_delays, args.delay_stride, args.delay_ahead)
    r_s = min(args.r_sensor or Sd.shape[0], Sd.shape[0], len(tr) - 1)
    lin = PODLSE(r_field=args.r_field, r_sensor=r_s, ridge=args.ridge, pod_method=args.pod_method).fit(
        Q[:, tr], Sd[:, tr]
    )
    ref = nmse(Q[:, te], lin.predict(Sd[:, te]))
    print(f"  podlse         test {ref:.4f}  train {lin.score(Q[:, tr], Sd[:, tr]):.4f}")

    rule("2. latent spaces")
    latents = build_latents(Q, tr, unflat, args)

    rule("3. fits")
    rows = []
    for lk, lat in latents.items():
        for br in args.branches:
            rows.append(fit_one(Q, S, tr, te, lat, args, lk, br))

    rule("4. plots")
    what = args.tag
    # one figure per latent space rather than eight curves on one axis
    for lk in latents:
        save_generalisation(
            [r for r in rows if r["latent"] == lk],
            os.path.join(out, f"generalisation_{lk}.png"),
            f"train vs test, {lk} latent  |  {what}",
            ref=ref,
        )
        save_panel(
            [r for r in rows if r["latent"] == lk],
            os.path.join(out, f"loss_curves_{lk}.png"),
            f"{lk} latent, r={args.r_field}, L={args.n_delays}  |  {what}",
            ref=None,
        )
    # and one per branch, so a branch can be compared across latent spaces
    for br in args.branches:
        save_panel(
            [r for r in rows if r["branch"] == br],
            os.path.join(out, f"loss_curves_branch_{br}.png"),
            f"{br} branch, r={args.r_field}, L={args.n_delays}  |  {what}",
            ref=None,
        )
    save_summary(rows, ref, os.path.join(out, "summary.png"), f"final NMSE  |  {what}")

    import csv

    keys = [k for k in rows[0] if not k.startswith("history_")] if rows else []
    with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    np.savez_compressed(
        os.path.join(out, "histories.npz"),
        **{f"{r['label']}_track_{k}": np.asarray(v) for r in rows for k, v in (r.get("track") or {}).items()},
        **{f"{r['label']}_train": np.asarray(r["history_train"]) for r in rows},
        **{f"{r['label']}_val": np.asarray(r["history_val"]) for r in rows},
        podlse_test=np.asarray([ref]),
    )
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2, default=str)

    rule("done")
    print(f"  {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
