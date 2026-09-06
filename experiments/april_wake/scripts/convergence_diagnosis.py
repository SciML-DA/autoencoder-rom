#!/usr/bin/env python
"""
convergence_diagnosis.py
========================

Is the branched network under-fitting, or is it at the information limit?

    qsub experiments/april_wake/hpc/convergence_diagnosis.pbs
    python experiments/april_wake/scripts/convergence_diagnosis.py --run 4p5d_10ms_yaw_0_0_0 --quick

Every sweep so far has produced the same flat picture: no latent size, no
branch architecture and no choice of POD-versus-autoencoder moves test NMSE by
more than a couple of percent. The tempting reading is that the sensors are the
limit and architecture cannot matter. That reading is only safe if the networks
were actually fitted, and one number says they were not:

    podlse      (closed form)  0.7507
    pod+linear  (400 epochs)   0.7722      +0.0215

Those two are *the same model class*. Both map the same delay-embedded sensors
linearly onto the same POD coefficients. One is solved exactly; the other by
Adam. The gap between them is not physics -- it is the error floor of the
training procedure, and it is larger than the difference between any two
architectures in the sweep. Until it is explained, no architecture comparison
means anything.

This script explains it, by separating three candidate causes:

  H1  data budget    the branched trainer holds out `val_fraction` of the
                     training block for early stopping, so it fits on 80% of
                     what the closed-form solver gets. That is not a modelling
                     difference, and it is invisible in the score column.
  H2  optimisation   too few epochs, or a learning rate that stalls. Diagnosed
                     against a target that is known exactly.
  H3  genuine        the class really is at its limit, and the gap is noise.

The design is a known-answer test. For the linear branch the optimum is
computable in closed form, so "how close did training get" is measurable rather
than a matter of opinion -- which is not true for the MLP, CNN or GRU, and is
why the linear branch is the right probe even though nobody wants to deploy it.

Sections:

  1  references        closed form on the full block, and on the same subset
                       the branched trainer actually sees
  2  epoch convergence pod+linear against training budget, with and without
                       the validation holdout
  3  learning rate     the other knob, at the best budget
  4  capacity          train NMSE per family: can each fit its own data at all?
  5  performance       default configs, side by side, train and test
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

from experiments.april_wake.data_preprocessing import add_data_args, load_data, make_split  # noqa: E402
from field_estimation.branched_ae import BranchedAE, LinearLatent, TorchLatent  # noqa: E402
from field_estimation.epod import (  # noqa: E402
    PODLSE,
    cosine,
    delay_embed,
    energy_ratio,
    nmse,
)
from field_estimation.epod import pod as _pod  # noqa: E402


def rule(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


# ── 1. the references ─────────────────────────────────────────────────────────


def closed_form(Q, Sd, tr, te, args, n_sub=None, label=""):
    """PODLSE fitted in closed form. The number training has to reach.

    `n_sub` fits on only the first `n_sub` training snapshots, which is how the
    branched trainer's validation holdout is reproduced: it carves its holdout
    off the training block, so it sees fewer snapshots than the closed-form
    solver does, and comparing the two without accounting for that attributes a
    data-budget difference to the architecture.
    """
    idx = tr if n_sub is None else tr[:n_sub]
    t0 = time.time()
    m = PODLSE(r_field=args.r_field, r_sensor=args.r_sensor, ridge=args.ridge, pod_method=args.pod_method).fit(
        Q[:, idx], Sd[:, idx]
    )
    pred = m.predict(Sd[:, te])
    out = dict(
        label=label,
        n_fit=len(idx),
        nmse_test=nmse(Q[:, te], pred),
        nmse_train=m.score(Q[:, idx], Sd[:, idx]),
        seconds=time.time() - t0,
    )
    print(
        f"  {label:<42} n_fit {len(idx):>5}   train {out['nmse_train']:.4f}"
        f"   test {out['nmse_test']:.4f}   {out['seconds']:.1f}s"
    )
    return out


# ── the branched fit, wrapped ─────────────────────────────────────────────────


def fit_branch(Q, S, tr, te, lat, args, *, branch, epochs, lr, val_fraction, seed=0, label=""):
    """One branched fit, returning scores and what the training actually did."""
    t0 = time.time()
    m = BranchedAE(
        lat,
        branch=branch,
        n_delays=args.n_delays,
        delay_stride=args.delay_stride,
        hidden=tuple(args.hidden),
        gru_hidden=args.gru_hidden,
        cnn_channels=tuple(args.cnn_channels),
        latent_weight=args.latent_weight,
        learning_rate=lr,
        n_epochs=epochs,
        batch_size=args.batch,
        val_fraction=val_fraction,
        patience=args.patience,
        seed=seed,
        device=args.device,
    ).fit(Q, S, tr)
    pred = m.predict(S, te)
    hist = list(m.loss_history or [])
    out = dict(
        label=label,
        branch=branch,
        epochs=epochs,
        lr=lr,
        val_fraction=val_fraction,
        seed=seed,
        nmse_test=nmse(Q[:, te], pred),
        nmse_train=m.score(Q, S, tr),
        cos_test=cosine(Q[:, te] - Q[:, te].mean(1, keepdims=True), pred - pred.mean(1, keepdims=True)),
        energy_test=energy_ratio(Q[:, te] - Q[:, te].mean(1, keepdims=True), pred - pred.mean(1, keepdims=True)),
        epochs_run=len(hist),
        train_loss=(hist[-1] if hist else float("nan")),
        stopped_early=int(bool(hist) and len(hist) < epochs),
        n_fit=int(round(len(tr) * (1.0 - val_fraction))),
        n_params=m.n_params,
        seconds=time.time() - t0,
    )
    return out, hist


# ── 2. epochs ─────────────────────────────────────────────────────────────────


def section_epochs(Q, S, tr, te, lat, args, ref_full, ref_sub):
    """pod+linear against training budget, against a target known exactly."""
    rule("2. optimisation convergence -- pod+linear against a known optimum")
    print("  The linear branch spans exactly the same functions as PODLSE, so")
    print("  the closed-form score is the target. Distance to it is optimisation")
    print("  error and nothing else.\n")
    print(f"  target, full training block          : {ref_full['nmse_test']:.4f}")
    print(f"  target, same subset the branch sees  : {ref_sub['nmse_test']:.4f}\n")

    rows, curves = [], {}
    print(f"  {'epochs':>7} {'val_frac':>9} {'run':>6} {'stop':>5} {'train':>8} {'test':>8} {'gap to CF':>10} {'s':>6}")
    for vf in args.val_fractions:
        for ep in args.epoch_grid:
            r, hist = fit_branch(
                Q,
                S,
                tr,
                te,
                lat,
                args,
                branch="linear",
                epochs=ep,
                lr=args.lr,
                val_fraction=vf,
                label=f"linear ep{ep} vf{vf}",
            )
            ref = ref_sub if vf > 0 else ref_full
            r["gap"] = r["nmse_test"] - ref["nmse_test"]
            r["section"] = "epochs"
            rows.append(r)
            curves[(vf, ep)] = hist
            print(
                f"  {ep:>7} {vf:>9.2f} {r['epochs_run']:>6} "
                f"{'yes' if r['stopped_early'] else 'no':>5} "
                f"{r['nmse_train']:>8.4f} {r['nmse_test']:>8.4f} "
                f"{r['gap']:>+10.4f} {r['seconds']:>6.0f}"
            )

    _verdict_epochs(rows, ref_full, ref_sub)
    return rows, curves


def _verdict_epochs(rows, ref_full, ref_sub):
    best = min(rows, key=lambda r: r["nmse_test"])
    ref = ref_sub if best["val_fraction"] > 0 else ref_full
    gap = best["nmse_test"] - ref["nmse_test"]

    # did more epochs still help at the top of the grid?
    by_vf = {}
    for r in rows:
        by_vf.setdefault(r["val_fraction"], []).append(r)
    still = False
    for vf, rs in by_vf.items():
        rs = sorted(rs, key=lambda r: r["epochs"])
        if len(rs) >= 2 and rs[-1]["nmse_test"] < rs[-2]["nmse_test"] - 1e-3:
            still = True

    print(
        f"\n  best branched fit : {best['nmse_test']:.4f} at "
        f"{best['epochs']} epochs, val_fraction {best['val_fraction']:.2f}"
    )
    print(f"  its closed-form target : {ref['nmse_test']:.4f}   gap {gap:+.4f}")

    vf0 = [r for r in rows if r["val_fraction"] == 0.0]
    if vf0:
        b0 = min(vf0, key=lambda r: r["nmse_test"])
        d = b0["nmse_test"] - ref_full["nmse_test"]
        print(f"\n  with NO validation holdout (val_fraction 0), the branch fits")
        print(f"  the same {ref_full['n_fit']} snapshots as the closed form:")
        print(f"    branched {b0['nmse_test']:.4f}  vs  closed form {ref_full['nmse_test']:.4f}   gap {d:+.4f}")

    print()
    if still:
        print("  VERDICT: test error was still falling at the largest budget.")
        print("  The networks are UNDER-TRAINED. Every architecture comparison")
        print("  in the sweeps is invalid -- extend the budget and rerun.")
    elif abs(gap) < 0.005:
        print("  VERDICT: training reaches the closed-form optimum. The fit is")
        print("  not the problem, so the flat architecture comparison is a real")
        print("  result about the sensors rather than an artefact.")
    elif vf0 and abs(b0["nmse_test"] - ref_full["nmse_test"]) < 0.005 < abs(gap):
        print("  VERDICT: the gap is the VALIDATION HOLDOUT, not optimisation.")
        print("  Training converges once it is given the same data. The sweep's")
        print("  branched models are handicapped by val_fraction, and that is a")
        print("  data-budget artefact, not an architecture result.")
    else:
        print(f"  VERDICT: a residual gap of {gap:+.4f} survives both more epochs")
        print("  and the data-budget correction. That points at the optimiser or")
        print("  the loss (latent-space MSE is not field-space MSE), not capacity.")


# ── 3. learning rate ──────────────────────────────────────────────────────────


def section_lr(Q, S, tr, te, lat, args, ref_full):
    rule("3. learning rate, at the largest budget and no holdout")
    rows = []
    print(f"  {'lr':>9} {'run':>6} {'train':>8} {'test':>8} {'gap to CF':>10}")
    for lr in args.lr_grid:
        r, _ = fit_branch(
            Q,
            S,
            tr,
            te,
            lat,
            args,
            branch="linear",
            epochs=max(args.epoch_grid),
            lr=lr,
            val_fraction=0.0,
            label=f"linear lr{lr}",
        )
        r["gap"] = r["nmse_test"] - ref_full["nmse_test"]
        r["section"] = "lr"
        rows.append(r)
        print(f"  {lr:>9.1e} {r['epochs_run']:>6} {r['nmse_train']:>8.4f} {r['nmse_test']:>8.4f} {r['gap']:>+10.4f}")
    b = min(rows, key=lambda r: r["nmse_test"])
    print(f"\n  best lr {b['lr']:.1e} -> {b['nmse_test']:.4f} (gap {b['gap']:+.4f})")
    if b["lr"] in (min(args.lr_grid), max(args.lr_grid)):
        print("  !! the optimum is at the edge of the grid -- widen it")
    return rows


# ── 4 & 5. capacity and performance ───────────────────────────────────────────


def section_performance(Q, S, tr, te, latents, args, ref_full):
    """Every default config, side by side, with train error kept in view.

    Train error is the capacity check. A model that cannot fit its own training
    data is not short of test data and not overfitting -- it is either too small
    or not being optimised, and no amount of held-out evaluation will tell you
    which. All of these sit near 0.7 in training, which is the signature of an
    information limit rather than a capacity one.
    """
    rule("4 & 5. capacity and performance -- default configs")
    rows = []
    print(f"  {'model':<16}{'train':>8}{'test':>8}{'cos':>7}{'E':>6}{'epochs':>8}{'stop':>6}{'params':>11}{'s':>6}")
    print(
        f"  {'podlse (CF)':<16}{ref_full['nmse_train']:>8.4f}"
        f"{ref_full['nmse_test']:>8.4f}{'-':>7}{'-':>6}{'-':>8}{'-':>6}"
        f"{'-':>11}{ref_full['seconds']:>6.0f}"
    )
    for lk, lat in latents.items():
        for br in args.branches:
            r, _ = fit_branch(
                Q,
                S,
                tr,
                te,
                lat,
                args,
                branch=br,
                epochs=args.perf_epochs,
                lr=args.lr,
                val_fraction=args.perf_val_fraction,
                label=f"{lk}+{br}",
            )
            r["section"] = "performance"
            rows.append(r)
            print(
                f"  {lk + '+' + br:<16}{r['nmse_train']:>8.4f}"
                f"{r['nmse_test']:>8.4f}{r['cos_test']:>7.3f}"
                f"{r['energy_test']:>6.2f}{r['epochs_run']:>8}"
                f"{'yes' if r['stopped_early'] else 'no':>6}"
                f"{r['n_params']:>11,d}{r['seconds']:>6.0f}"
            )
    if rows:
        b = min(rows, key=lambda r: r["nmse_test"])
        d = b["nmse_test"] - ref_full["nmse_test"]
        print(
            f"\n  best network {b['label']} {b['nmse_test']:.4f} vs closed form {ref_full['nmse_test']:.4f}  ({d:+.4f})"
        )
        print("  " + ("a network wins" if d < -0.005 else "the closed-form linear estimator is still the best model"))
    return rows


# ── 6. learning curve ─────────────────────────────────────────────────────────


def section_learning_curve(Q, S, tr, te, args):
    """Test error against the number of training snapshots.

    The claim this settles is that a single run does not carry enough data --
    which the multi-run experiment did NOT test. That experiment pooled five
    different yaws under one shared linear map, and its per-run scores degrade
    monotonically with yaw magnitude (0.777 at 0/0 to 0.897 at -30/-15). That is
    a heterogeneity effect: one map compromising across five different flows.
    Whether *more snapshots of the same flow* would help is a separate question,
    and there is no more of that data to collect -- so it has to be answered by
    extrapolation from what there is.

    Fit on a growing subset, score on the fixed test block, and read the slope:

      still falling at 100%  ->  data-limited. More snapshots at one condition
                                 would help, and that is a recording
                                 recommendation rather than a modelling one.
      flat by 50%            ->  saturated. The estimator has all the data it
                                 can use and the limit is elsewhere.

    Two details that decide whether the curve means anything.

    The subset is taken from the END of the training block, so every fit sits
    the same distance in time from the test block. Taking it from the start
    would confound volume with temporal distance, and a wake decorrelates, so
    the smaller fits would look worse for the wrong reason.

    The curve is swept at several window lengths because the parameter count
    scales with L: at L=100 the map has 76,800 parameters against 4,365
    snapshots, and at L=10 it has 7,680. If the long-window curves are still
    falling while the short ones are flat, the shortage is relative to model
    size rather than absolute -- which is a different recommendation again.
    """
    rule("6. learning curve -- is one run enough data?")
    rows = []
    n_tr = len(tr)
    print(f"  training block: {n_tr} snapshots; subsets taken from its end\n")
    print(f"  {'L':>5} {'n_par':>9} {'n_train':>8} {'par/snap':>9} {'train':>8} {'test':>8}")
    for L in args.lc_delays:
        Sd = delay_embed(S, L, args.delay_stride, args.delay_ahead)
        for frac in args.train_fractions:
            n_sub = max(int(round(frac * n_tr)), args.r_field + 10)
            if n_sub > n_tr:
                continue
            idx = tr[-n_sub:]
            r_s = min(args.r_sensor or Sd.shape[0], Sd.shape[0], len(idx) - 1)
            m = PODLSE(r_field=args.r_field, r_sensor=r_s, ridge=args.ridge, pod_method=args.pod_method).fit(
                Q[:, idx], Sd[:, idx]
            )
            r = dict(
                section="learning_curve",
                n_delays=L,
                frac=frac,
                n_train=len(idx),
                n_params=m.n_params,
                nmse_train=m.score(Q[:, idx], Sd[:, idx]),
                nmse_test=nmse(Q[:, te], m.predict(Sd[:, te])),
            )
            rows.append(r)
            print(
                f"  {L:>5} {m.n_params:>9,d} {len(idx):>8} "
                f"{m.n_params / len(idx):>9.1f} "
                f"{r['nmse_train']:>8.4f} {r['nmse_test']:>8.4f}"
            )
        print()

    _verdict_learning_curve(rows, args)
    return rows


def _verdict_learning_curve(rows, args):
    print("  slope over the last doubling of the training set:")
    verdicts = []
    for L in args.lc_delays:
        rs = sorted([r for r in rows if r["n_delays"] == L], key=lambda r: r["n_train"])
        if len(rs) < 2:
            continue
        a, b = rs[-2], rs[-1]
        drop = a["nmse_test"] - b["nmse_test"]
        per_double = drop / max(np.log2(b["n_train"] / a["n_train"]), 1e-9)
        verdicts.append((L, per_double, b))
        print(
            f"    L={L:<4} {a['n_train']:>5} -> {b['n_train']:>5} snapshots: "
            f"{a['nmse_test']:.4f} -> {b['nmse_test']:.4f}  "
            f"({per_double:+.4f} per doubling)"
        )

    if not verdicts:
        return
    best = min(verdicts, key=lambda v: v[2]["nmse_test"])
    L, slope, r = best
    print()
    if slope > 0.01:
        need = r["nmse_test"] - 2 * slope
        print(f"  VERDICT: still falling at the full training block ({slope:+.4f} per doubling at L={L}).")
        print(f"  The estimator is DATA-LIMITED. Doubling the record twice would")
        print(f"  reach roughly {need:.3f} if the trend held -- so recording more")
        print(f"  snapshots at one yaw is a real experimental recommendation.")
    elif slope > 0.002:
        print(f"  VERDICT: still falling, but slowly ({slope:+.4f} per doubling).")
        print("  More data would help marginally. It is not the main limit.")
    else:
        print(f"  VERDICT: the curve is flat ({slope:+.4f} per doubling).")
        print("  The estimator has all the data it can use; one run is enough")
        print("  for this model, and the limit is the sensors, not the record.")

    heavy = [v for v in verdicts if v[1] > 0.01]
    if heavy and len(verdicts) > 1 and heavy != verdicts:
        print(
            f"\n  Note: only the longer windows are still improving "
            f"(L={[v[0] for v in heavy]}). The shortage is relative to model"
        )
        print("  size, not absolute -- a shorter window is the cheaper fix.")


# ── plots ─────────────────────────────────────────────────────────────────────


def make_plots(ep_rows, curves, lr_rows, perf_rows, lc_rows, ref_full, ref_sub, out):
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))

    a = ax[0]
    for vf in sorted({r["val_fraction"] for r in ep_rows}):
        rs = sorted([r for r in ep_rows if r["val_fraction"] == vf], key=lambda r: r["epochs"])
        a.semilogx(
            [r["epochs"] for r in rs], [r["nmse_test"] for r in rs], "o-", label=f"pod+linear, val_fraction {vf:g}"
        )
    a.axhline(ref_full["nmse_test"], c="k", ls="--", lw=1.4, label="closed form, full block")
    a.axhline(ref_sub["nmse_test"], c="C3", ls=":", lw=1.4, label="closed form, same subset")
    a.set(xlabel="training epochs", ylabel="test NMSE", title="optimisation convergence against a known optimum")
    a.legend(fontsize=7)
    a.grid(alpha=0.3, which="both")

    a = ax[1]
    for (vf, ep), h in sorted(curves.items()):
        if h and ep == max(r["epochs"] for r in ep_rows):
            a.loglog(np.arange(1, len(h) + 1), h, lw=1.2, label=f"val_fraction {vf:g}")
    a.set(xlabel="epoch", ylabel="training loss", title="training loss -- is it still descending?")
    a.legend(fontsize=7)
    a.grid(alpha=0.3, which="both")

    a = ax[2]
    if lc_rows:
        for L in sorted({r["n_delays"] for r in lc_rows}):
            rs = sorted([r for r in lc_rows if r["n_delays"] == L], key=lambda r: r["n_train"])
            a.semilogx([r["n_train"] for r in rs], [r["nmse_test"] for r in rs], "o-", label=f"L={L}")
        a.axhline(ref_full["nmse_test"], c="k", ls="--", lw=1.2, label="closed form, full block")
        a.set(xlabel="training snapshots", ylabel="test NMSE", title="learning curve -- is one run enough data?")
        a.legend(fontsize=7)
        a.grid(alpha=0.3, which="both")

    a = ax[3]
    if perf_rows:
        rs = sorted(perf_rows, key=lambda r: r["nmse_test"])
        y = np.arange(len(rs))
        a.barh(y, [r["nmse_test"] for r in rs], color="C0", label="test")
        a.barh(y, [r["nmse_train"] for r in rs], color="C1", height=0.45, label="train")
        a.axvline(ref_full["nmse_test"], c="k", ls="--", lw=1.4, label="closed form")
        a.set_yticks(y)
        a.set_yticklabels([r["label"] for r in rs], fontsize=7)
        a.set(xlabel="NMSE", title="performance, default configs")
        a.legend(fontsize=7)
        a.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    p = os.path.join(out, "convergence_diagnosis.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"\n  wrote {p}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    # make_split reads both of these; they are declared per-script rather than
    # in add_data_args, so every entry point has to supply them.
    p.add_argument("--delays", type=int, nargs="+", default=[25])
    p.add_argument("--delay-stride", type=int, default=1)

    d = p.add_argument_group("diagnosis")
    d.add_argument("--epoch-grid", type=int, nargs="+", default=[50, 100, 200, 400, 800, 1600])
    d.add_argument(
        "--val-fractions",
        type=float,
        nargs="+",
        default=[0.2, 0.0],
        help="0.2 is the trainer's default; 0.0 removes the holdout "
        "so the branch sees the same data as the closed form",
    )
    d.add_argument("--lr-grid", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3, 1e-2])
    d.add_argument("--branches", nargs="+", default=["linear", "mlp", "cnn", "gru"])
    d.add_argument("--latents", nargs="+", default=["pod", "ae"], choices=["pod", "ae"])
    d.add_argument("--perf-epochs", type=int, default=800)
    d.add_argument("--perf-val-fraction", type=float, default=0.0)
    d.add_argument(
        "--train-fractions",
        type=float,
        nargs="+",
        default=[0.125, 0.25, 0.5, 0.75, 1.0],
        help="learning curve: fractions of the training block, taken "
        "from its end so temporal distance to the test block is "
        "held constant",
    )
    d.add_argument(
        "--lc-delays",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100],
        help="learning curve: window lengths, since the parameter count scales with L",
    )
    d.add_argument("--skip", nargs="*", default=[], choices=["epochs", "lr", "performance", "learning_curve"])

    b = p.add_argument_group("model")
    b.add_argument("--r-field", type=int, default=16)
    b.add_argument("--r-sensor", type=int, default=None)
    b.add_argument("--n-delays", type=int, default=25)
    b.add_argument("--ridge", type=float, default=1e-3)
    b.add_argument("--pod-method", default="randomized", choices=["svd", "snapshot", "randomized", "auto"])
    b.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    b.add_argument("--gru-hidden", type=int, default=64)
    b.add_argument("--cnn-channels", type=int, nargs="+", default=[32, 64])
    b.add_argument("--latent-weight", default="energy", choices=["energy", "unit"])
    b.add_argument("--lr", type=float, default=1e-3)
    b.add_argument("--batch", type=int, default=128)
    b.add_argument("--patience", type=int, default=200)
    b.add_argument("--ae-epochs", type=int, default=400)
    b.add_argument("--ae-batch", type=int, default=64)
    b.add_argument("--ae-lr", type=float, default=1e-3)
    b.add_argument("--ae-patience", type=int, default=40)
    b.add_argument("--ae-hidden-scale", type=float, nargs="+", default=[8, 2])
    b.add_argument("--device", default=None)

    o = p.add_argument_group("output")
    o.add_argument("--out", default="results/convergence_diagnosis")
    o.add_argument("--tag", default="default")
    o.add_argument("--quick", action="store_true", help="tiny grids, for checking the script runs at all")
    args = p.parse_args()

    if args.quick:
        args.epoch_grid = [10, 20]
        args.lr_grid = [1e-3]
        args.branches = ["linear"]
        args.latents = ["pod"]
        args.perf_epochs = 20
        args.ae_epochs = 20
        args.train_fractions = [0.5, 1.0]
        args.lc_delays = [5]
        args.n = args.n or 400
        args.r_field = 4

    out = os.path.join(args.out, args.tag)
    os.makedirs(out, exist_ok=True)

    rule("0. data")
    Q, S, unflat, case0, run_id, cases = load_data(args)
    tr, te = make_split(args, Q.shape[1], run_id, cases)[:2]
    Sd = delay_embed(S, args.n_delays, args.delay_stride, args.delay_ahead)

    rule("1. the references -- what training has to reach")
    ref_full = closed_form(Q, Sd, tr, te, args, label="podlse, full training block")
    n_sub = int(round(len(tr) * (1.0 - max(args.val_fractions))))
    ref_sub = closed_form(Q, Sd, tr, te, args, n_sub=n_sub, label=f"podlse, first {n_sub} only (branch's budget)")
    print(f"\n  the holdout alone is worth {ref_sub['nmse_test'] - ref_full['nmse_test']:+.4f} NMSE")

    # Latent spaces, both fitted on the training block only. Built the same way
    # sparse_sensor_sweep.py builds them, so a number here is comparable with a
    # number there.
    latents = {}
    if "pod" in args.latents:
        Psi, _, _, qm = _pod(Q[:, tr], r=args.r_field, subtract_mean=True, method=args.pod_method)
        latents["pod"] = LinearLatent(Psi, qm, device=args.device)
        print(f"  POD latent r={args.r_field}")
    if "ae" in args.latents:
        from models.data_driven.autoencoders import AE

        # Hidden widths scale with the latent, as the sweep does: fixed widths
        # make the parameter count independent of r and the curve flat by
        # construction.
        dims = tuple(max(int(f * args.r_field), args.r_field + 1) for f in args.ae_hidden_scale)
        t0 = time.time()
        grid = unflat(Q[:, tr], dtype=np.float32)
        p_ae = AE(
            n_latent=args.r_field,
            layer_dims=dims,
            n_epochs=args.ae_epochs,
            batch_size=args.ae_batch,
            learning_rate=args.ae_lr,
            patience=args.ae_patience,
            device=args.device,
            seed=0,
        ).fit(grid)
        latents["ae"] = TorchLatent(p_ae, device=args.device)
        print(f"  AE latent r={args.r_field} dims={dims}: {time.time() - t0:.1f}s, {p_ae.n_params / 1e6:.2f}M params")

    rows, curves = [], {}
    if "epochs" not in args.skip:
        r, curves = section_epochs(Q, S, tr, te, latents["pod"], args, ref_full, ref_sub)
        rows += r
    lr_rows = []
    if "lr" not in args.skip:
        lr_rows = section_lr(Q, S, tr, te, latents["pod"], args, ref_full)
        rows += lr_rows
    lc_rows = []
    if "learning_curve" not in args.skip:
        lc_rows = section_learning_curve(Q, S, tr, te, args)
        rows += lc_rows

    perf_rows = []
    if "performance" not in args.skip:
        perf_rows = section_performance(Q, S, tr, te, latents, args, ref_full)
        rows += perf_rows

    make_plots(
        [r for r in rows if r.get("section") == "epochs"], curves, lr_rows, perf_rows, lc_rows, ref_full, ref_sub, out
    )

    import csv

    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2, default=str)

    rule("done")
    print(f"  {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
