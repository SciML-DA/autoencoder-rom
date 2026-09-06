#!/usr/bin/env python
"""
hyper_search.py
===============

The staged hyperparameter search over the branched-AE architecture.

    qsub experiments/april_wake/hpc/hyper_search.pbs
    python experiments/april_wake/scripts/hyper_search.py --stages screen          # then pair, random

Everything that defines the search lives in the CONFIG block below, not on the
command line. There are seventeen axes; passing them as flags would produce a
qsub line nobody can read or reproduce, and the point of a search is that the
design is a recorded artefact rather than an argument string in someone's shell
history. Edit the block, commit it, and the run is reproducible.

Why this exists
---------------
Every sweep so far moved one axis with the rest pinned at defaults nobody
chose, and the diagnosis says that is now the binding problem: the nonlinear
branches reach their best validation loss at epoch 10-25 and then overfit for
another two hundred, because an MLP branch carries 57,104 parameters against
3,492 training snapshots. Sixteen parameters per snapshot. Model *size* has
never been varied.

Trials run smallest-first
-------------------------
Within every stage, configurations are ordered by an estimate of parameter
count, ascending. The prior is that small models win here, so ordering this way
puts the most informative trials at the top of the log: tail it half way and
you can already tell whether the search is worth finishing, rather than
discovering it after the queue has eaten the whole walltime.

Selection discipline
--------------------
Train / validation / test, contiguous with gaps. Trials are scored on
*validation*. The test block is not even computed unless the stage is
``confirm``, which re-runs the best handful with more seeds. A few hundred
random trials scored against one test block will find something that looks good
by chance -- at this budget that is a near-certainty, not a risk -- and the
cheapest way to not do it is to make the number unavailable.
"""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import os
import sys
import time

import numpy as np

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
from field_estimation.epod import PODLSE, cosine, delay_embed, nmse, pod  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG -- edit this, not the command line
# ══════════════════════════════════════════════════════════════════════════════

# The centre point. Stage `screen` varies one axis at a time away from this.
# Treat it as arbitrary: it is today's defaults, and whether they are anywhere
# near optimal is precisely what is in question.
CENTRE = dict(
    latent="pod",
    branch="mlp",
    r_field=16,
    n_delays=25,
    hidden=(32, 32),
    cnn_channels=(16, 32),
    kernel_size=5,
    gru_hidden=32,
    gru_layers=1,
    activation="tanh",
    dropout=0.0,
    weight_decay=0.0,
    sensor_noise=0.0,
    lr=3e-3,
    band_hz=None,
    ensemble=1,
    seed=0,
    lambda_field=0.0,
)

# Stage 1. One axis at a time, per architecture. Ranges deliberately extend
# both below and above the current defaults -- "smaller is better" is the
# hypothesis under test, not an assumption to build in.
SCREEN = {
    "r_field": [2, 4, 8, 16, 32, 64, 128],
    "n_delays": [1, 5, 10, 25, 50, 100],
    "hidden": [(8,), (16,), (32,), (16, 16), (32, 32), (128, 128), (256, 256)],
    "cnn_channels": [(4,), (8,), (16,), (8, 16), (16, 32), (32, 64), (64, 128)],
    "kernel_size": [3, 5, 9, 15],
    "gru_hidden": [8, 16, 32, 64, 128],
    "gru_layers": [1, 2],
    "activation": ["tanh", "relu", "gelu"],
    "dropout": [0.0, 0.1, 0.25, 0.5],
    # Top of this range used to be 1e-1, which is exactly the "standard
    # practice" value that Kim et al. (2025) find is ~30x too small for
    # over-parameterised models in the data-constrained regime. Stopping there
    # makes "small and undecayed wins" unfalsifiable: it is the same shape as
    # never having tried enough decay -- hence the range running out to 3.0.
    "weight_decay": [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0],
    "sensor_noise": [0.0, 0.05, 0.1, 0.25, 0.5, 1.0],
    "lr": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "band_hz": [None, 20.0, 30.0, 40.0, 60.0],
}

# Which axes apply to which branch. Screening `gru_hidden` on an MLP would burn
# a fit to re-measure the centre point.
AXES_FOR = {
    "linear": ["r_field", "n_delays", "weight_decay", "sensor_noise", "lr", "band_hz"],
    "mlp": ["r_field", "n_delays", "hidden", "activation", "dropout", "weight_decay", "sensor_noise", "lr", "band_hz"],
    "cnn": [
        "r_field",
        "n_delays",
        "cnn_channels",
        "kernel_size",
        "activation",
        "dropout",
        "weight_decay",
        "sensor_noise",
        "lr",
        "band_hz",
    ],
    "gru": [
        "r_field",
        "n_delays",
        "gru_hidden",
        "gru_layers",
        "dropout",
        "weight_decay",
        "sensor_noise",
        "lr",
        "band_hz",
    ],
}

SCREEN_MODELS = [("pod", "linear"), ("pod", "mlp"), ("pod", "cnn"), ("pod", "gru"), ("ae", "linear"), ("ae", "mlp")]

# Stage 2. Full factorial on pairs where an interaction is expected. Each entry
# is (axis_a, values_a, axis_b, values_b, [(latent, branch), ...]).
PAIRS = [
    # does explicit regularisation substitute for a smaller network? this pair
    # *is* the shrink-vs-penalise question, so its decay levels have to reach
    # the regime where penalising could plausibly win -- the old top of 1e-2
    # could only ever have answered "shrink"
    ("hidden", [(8,), (32,), (128, 128)], "weight_decay", [0.0, 1e-3, 1e-1, 3.0], [("pod", "mlp")]),
    # both consume the same scarce data
    ("hidden", [(8,), (32,), (128, 128)], "n_delays", [5, 25, 100], [("pod", "mlp")]),
    # the stage-N noise null result was measured at one (large) width
    ("hidden", [(8,), (32,), (128, 128)], "sensor_noise", [0.0, 0.1, 0.5], [("pod", "mlp")]),
    # a narrower target may need fewer modes
    ("band_hz", [None, 20.0, 40.0], "r_field", [4, 16, 64], [("pod", "linear"), ("pod", "mlp")]),
    # AE models are trained on the latent term but *scored* in field space, and
    # for a non-orthonormal AE decoder those are not the same objective (they
    # are for POD, which is why this is an AE-only pair -- scanning it on POD
    # would only rescale the loss and buy a decoder pass for nothing). Crossed
    # with r_field because the latent/field discrepancy should grow with latent
    # dimension: if lambda_field matters anywhere, it matters most at large r.
    ("lambda_field", [0.0, 0.1, 1.0], "r_field", [4, 16, 64], [("ae", "linear"), ("ae", "mlp")]),
]

# Stage 3. Monte Carlo over everything at once. ("log", lo, hi) samples
# log-uniformly; a list is sampled uniformly.
RANDOM_SPACE = {
    "latent": ["pod", "ae"],
    "branch": ["linear", "mlp", "cnn", "gru"],
    "r_field": ("logint", 2, 128),
    "n_delays": ("logint", 2, 100),
    "hidden": [(8,), (16,), (32,), (64,), (16, 16), (32, 32), (64, 64), (128, 128), (32, 32, 32)],
    "cnn_channels": [(4,), (8,), (16,), (8, 16), (16, 32), (32, 64)],
    "kernel_size": [3, 5, 9, 15],
    "gru_hidden": [8, 16, 32, 64, 128],
    "gru_layers": [1, 2],
    "activation": ["tanh", "relu", "gelu"],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
    # bottom raised from 1e-7: the screen found 0 through 1e-3 indistinguishable,
    # so the lowest decades only spend trials re-measuring "no decay". Top raised
    # to 3.0 for the reason in SCREEN above.
    "weight_decay": ("log", 1e-6, 3.0),
    "sensor_noise": [0.0, 0.05, 0.1, 0.25, 0.5],
    "lr": ("log", 3e-4, 3e-2),
    "band_hz": [None, 20.0, 30.0, 40.0, 60.0],
    # a list rather than ("log", ...) so 0.0 -- the value every trial so far has
    # used, and the one the others must be judged against -- stays reachable
    "lambda_field": [0.0, 0.01, 0.1, 0.3, 1.0],
}
N_RANDOM = 400  # trials; the driver stops on walltime, not on this
CONFIRM_TOP = 20  # kept for reference; superseded by CONFIRM_PER_BAND
CONFIRM_PER_BAND = 4  # best-on-validation per band -- 4 x 5 bands = 20 configs,
# the same budget as the old global top-20 but spread so
# every band gets a test number instead of only 20 Hz
CONFIRM_SEEDS = 5

TRAIN_EPOCHS = 2000
PATIENCE = 200
BATCH = 128
VAL_FRACTION = 0.2  # inside the training block, for early stopping
SELECT_GAP = 100  # guard band between the train and validation blocks
SELECT_FRACTION = 0.2  # tail of the training block reserved for selection

# ══════════════════════════════════════════════════════════════════════════════

FIELDS = [
    "stage",
    "trial",
    "latent",
    "branch",
    "r_field",
    "n_delays",
    "hidden",
    "cnn_channels",
    "kernel_size",
    "gru_hidden",
    "gru_layers",
    "activation",
    "dropout",
    "weight_decay",
    "sensor_noise",
    "lr",
    "band_hz",
    "ensemble",
    "lambda_field",
    "seed",
    "axis",
    "n_params",
    "size_proxy",
    "nmse_train",
    "nmse_val",
    "cos_val",
    "nmse_test",
    "nmse_fullband",
    "epochs_run",
    "best_epoch",
    "stopped_early",
    "seconds",
]

KEY = (
    "stage",
    "latent",
    "branch",
    "r_field",
    "n_delays",
    "hidden",
    "cnn_channels",
    "kernel_size",
    "gru_hidden",
    "gru_layers",
    "activation",
    "dropout",
    "weight_decay",
    "sensor_noise",
    "lr",
    "band_hz",
    "ensemble",
    "lambda_field",
    "seed",
)


def rule(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def _keyfield(v):
    """Canonical text for one key field.

    csv writes None as an empty field and reads it back as '', so a live config
    holding None ("None") and the row it wrote ("") do not compare equal unless
    both are normalised here. Without this the resume check misses every trial
    with a null field -- in practice every trial at band_hz=None, i.e. most of
    the search -- and re-runs the lot on each link of a chain.
    """
    return "" if v in (None, "", "None") else str(v)


def key_of(cfg):
    return tuple(_keyfield(cfg.get(k)) for k in KEY)


def size_proxy(cfg) -> int:
    """Cheap monotone stand-in for parameter count, used only for ordering.

    The real count needs a built network, and building every candidate just to
    sort them would cost more than the ordering saves. This tracks it closely
    enough: input width times the branch's own widths.
    """
    n_in = 12 * int(cfg["n_delays"])
    r = int(cfg["r_field"])
    b = cfg["branch"]
    if b == "linear":
        return n_in * r
    if b == "mlp":
        w = list(cfg["hidden"])
        dims = [n_in, *w, r]
        return sum(a * b_ for a, b_ in zip(dims, dims[1:]))
    if b == "cnn":
        c = list(cfg["cnn_channels"])
        k = int(cfg["kernel_size"])
        dims = [12, *c]
        conv = sum(a * b_ * k for a, b_ in zip(dims, dims[1:]))
        return conv + c[-1] * int(cfg["n_delays"]) * r
    if b == "gru":
        h = int(cfg["gru_hidden"])
        return int(cfg["gru_layers"]) * 3 * (12 + h) * h + h * r
    return n_in * r


# ── data ──────────────────────────────────────────────────────────────────────


def three_way(args, Q, run_id, cases):
    """Train / validation / test, contiguous, gap-separated.

    The validation block is the tail of the training block, so the test block is
    untouched by selection and stays where `make_split` put it.
    """
    tr, te = make_split(args, Q.shape[1], run_id, cases)[:2]
    n_val = int(round(len(tr) * SELECT_FRACTION))
    va = tr[-n_val:]
    tr_in = tr[: len(tr) - n_val - SELECT_GAP]
    # A short record plus a long delay window plus this second guard band can
    # consume the training block entirely, and the failure downstream is an
    # IndexError inside the ridge solve rather than anything that names the
    # cause. Say it here instead.
    if len(tr_in) < 50:
        raise ValueError(
            f"only {len(tr_in)} training snapshots survive the split: "
            f"{Q.shape[1]} total, {len(tr)} after make_split's warm-up and gap, "
            f"then {n_val} to validation and {SELECT_GAP} to the guard band. "
            "Use a longer record, a shorter --delays, or lower SELECT_GAP."
        )
    print(f"  select: train {len(tr_in)}, val {len(va)}, test {len(te)} (gap {SELECT_GAP})")
    return tr_in, va, te


class Latents:
    """POD and AE latent spaces, built once per (kind, r) and reused."""

    def __init__(self, Q, tr, unflat, args):
        self.Q, self.tr, self.unflat, self.args = Q, tr, unflat, args
        self._c = {}

    def get(self, kind, r):
        if (kind, r) in self._c:
            return self._c[(kind, r)]
        if kind == "pod":
            Psi, _, _, qm = pod(self.Q[:, self.tr], r=r, subtract_mean=True, method="randomized")
            lat = LinearLatent(Psi, qm, device=self.args.device)
        else:
            from models.data_driven.autoencoders import AE

            dims = tuple(max(int(f * r), r + 1) for f in (8, 2))
            p = AE(
                n_latent=r,
                layer_dims=dims,
                n_epochs=400,
                batch_size=64,
                learning_rate=1e-3,
                patience=40,
                device=self.args.device,
                seed=0,
            )
            p.fit(self.unflat(self.Q[:, self.tr], dtype=np.float32))
            lat = TorchLatent(p, device=self.args.device)
        self._c[(kind, r)] = lat
        print(f"    [built {kind} latent r={r}]", flush=True)
        return lat


def band_of(cfg, args):
    """The band this trial is run at.

    ``--band-hz`` pins the whole run to one level, which is how a band sweep
    should be driven from outside the search; with it unset, the per-trial
    value from the search space is used.
    """
    if args.band_hz:
        return float(args.band_hz)
    b = cfg.get("band_hz")
    return None if b in (None, "", "None") else float(b)


# ── one trial ─────────────────────────────────────────────────────────────────


def run_trial(cfg, data, latents, args, with_test=False):
    Q, Q_full, S, tr, va, te = data
    t0 = time.time()
    lat = latents.get(cfg["latent"], int(cfg["r_field"]))

    n_ens = int(cfg.get("ensemble", 1) or 1)
    preds_va, preds_te, trs, eps, bests, stops = [], [], [], [], [], []
    for j in range(n_ens):
        m = BranchedAE(
            lat,
            branch=cfg["branch"],
            n_delays=int(cfg["n_delays"]),
            hidden=tuple(cfg["hidden"]),
            cnn_channels=tuple(cfg["cnn_channels"]),
            kernel_size=int(cfg["kernel_size"]),
            gru_hidden=int(cfg["gru_hidden"]),
            gru_layers=int(cfg["gru_layers"]),
            activation=cfg["activation"],
            dropout=float(cfg["dropout"]),
            weight_decay=float(cfg["weight_decay"]),
            sensor_noise=float(cfg["sensor_noise"]),
            lambda_field=float(cfg.get("lambda_field", 0.0) or 0.0),
            learning_rate=float(cfg["lr"]),
            n_epochs=TRAIN_EPOCHS,
            batch_size=BATCH,
            val_fraction=VAL_FRACTION,
            patience=PATIENCE,
            seed=int(cfg["seed"]) * 1000 + j,
            device=args.device,
        ).fit(Q, S, tr)
        preds_va.append(m.predict(S, va))
        if with_test:
            preds_te.append(m.predict(S, te))
        trs.append(m.score(Q, S, tr))
        h, vh = list(m.loss_history or []), list(m.val_loss_history or [])
        eps.append(len(h))
        bests.append(int(np.argmin(vh)) + 1 if vh else -1)
        stops.append(int(bool(h) and len(h) < TRAIN_EPOCHS))
        n_par = m.n_params

    pv = np.mean(preds_va, axis=0)
    row = dict(cfg)
    row.update(
        n_params=n_par * n_ens,
        size_proxy=size_proxy(cfg),
        nmse_train=float(np.mean(trs)),
        nmse_val=nmse(Q[:, va], pv),
        cos_val=cosine(Q[:, va] - Q[:, va].mean(1, keepdims=True), pv - pv.mean(1, keepdims=True)),
        nmse_test=(nmse(Q[:, te], np.mean(preds_te, axis=0)) if with_test else float("nan")),
        nmse_fullband=(
            nmse(Q_full[:, te], np.mean(preds_te, axis=0)) if with_test and Q_full is not None else float("nan")
        ),
        epochs_run=int(np.mean(eps)),
        best_epoch=int(np.mean(bests)),
        stopped_early=int(np.mean(stops) > 0.5),
        seconds=time.time() - t0,
    )
    return row


def reference(Q, S, tr, va, te, args, with_test=False):
    """Closed-form PODLSE. In every stage, because it still wins."""
    Sd = delay_embed(S, int(CENTRE["n_delays"]), 1, args.delay_ahead)
    m = PODLSE(r_field=int(CENTRE["r_field"]), r_sensor=None, ridge=1e-3, pod_method="randomized").fit(
        Q[:, tr], Sd[:, tr]
    )
    v = nmse(Q[:, va], m.predict(Sd[:, va]))
    t = nmse(Q[:, te], m.predict(Sd[:, te])) if with_test else float("nan")
    print(f"  PODLSE reference: val {v:.4f}" + (f"  test {t:.4f}" if with_test else ""))
    return v


# ── the stages ────────────────────────────────────────────────────────────────


def stage_screen():
    out = []
    for lk, br in SCREEN_MODELS:
        for axis in AXES_FOR[br]:
            for v in SCREEN[axis]:
                c = dict(CENTRE, latent=lk, branch=br, axis=axis)
                c[axis] = v
                out.append(c)
    return out


def stage_pair():
    out = []
    for a, av, b, bv, models in PAIRS:
        for lk, br in models:
            for x, y in itertools.product(av, bv):
                c = dict(CENTRE, latent=lk, branch=br, axis=f"{a}x{b}")
                c[a], c[b] = x, y
                out.append(c)
    return out


def stage_random(seed=0, n=None):
    """Monte Carlo over the whole space.

    `seed` shards the stage: separate seeds draw disjoint samples of the same
    distribution, so N jobs at seeds 0..N-1 can run *in parallel* under separate
    tags and their union is still a valid random sample -- there is nothing
    sequential about a Monte Carlo stage, and chaining it just multiplies queue
    waits. Merge the shard CSVs before running `confirm`, which does depend on
    seeing every prior trial.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(N_RANDOM if n is None else n):
        c = dict(CENTRE, axis="random")
        for k, spec in RANDOM_SPACE.items():
            if isinstance(spec, tuple) and spec[0] == "log":
                c[k] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
            elif isinstance(spec, tuple) and spec[0] == "logint":
                c[k] = int(round(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2])))))
            else:
                c[k] = spec[rng.integers(len(spec))]
        c["seed"] = i
        out.append(c)
    return out


def stage_confirm(out_dir):
    """Re-run the best-on-validation with more seeds, and touch the test block."""
    path = os.path.join(out_dir, "results.csv")
    if not os.path.exists(path):
        print("  nothing to confirm: no results.csv yet")
        return []
    rows = [r for r in csv.DictReader(open(path)) if r["stage"] != "confirm" and r["nmse_val"] not in ("", "nan")]
    rows.sort(key=lambda r: float(r["nmse_val"]))

    # Stratify by band. nmse_val is not comparable across bands -- a narrower
    # target is a smaller target -- so a global top-N ranks band width, not
    # model quality: taking the plain top 20 here selected 17 configurations at
    # 20 Hz, 3 at 30 Hz, and nothing at fullband. Since confirm is the only
    # stage that touches the test block, that would have spent the single shot
    # on one band and left the headline fullband case with no test number.
    per_band = collections.defaultdict(list)
    for r in rows:
        per_band[r["band_hz"] if r["band_hz"] not in ("", "None") else None].append(r)
    picked = []
    for b in sorted(per_band, key=lambda k: (k is not None, k)):
        picked += per_band[b][:CONFIRM_PER_BAND]
    rule(
        f"confirm: {CONFIRM_PER_BAND} per band over {len(per_band)} bands "
        f"= {len(picked)} configs x {CONFIRM_SEEDS} seeds"
    )

    out = []
    for r in picked:
        c = dict(CENTRE, axis="confirm")
        for k in ("latent", "branch", "activation"):
            c[k] = r[k]
        for k in ("r_field", "n_delays", "kernel_size", "gru_hidden", "gru_layers"):
            c[k] = int(float(r[k]))
        for k in ("dropout", "weight_decay", "sensor_noise", "lr"):
            c[k] = float(r[k])
        c["band_hz"] = None if r["band_hz"] in ("", "None") else float(r["band_hz"])
        for k in ("hidden", "cnn_channels"):
            c[k] = tuple(int(x) for x in r[k].strip("()").rstrip(",").split(",") if x)
        # One row per seed, not one ensemble-averaged row. Two reasons, both
        # from the protocol: the spread across seeds is the yardstick for
        # significance, and an ensemble mean has no spread -- it collapses to a
        # single number exactly where the winner is being chosen. And averaging
        # five predictions lowers NMSE by itself, so an ensembled confirm score
        # is not comparable to the single-model random-stage score it is meant
        # to confirm; the configuration would appear to improve on confirmation
        # for reasons that have nothing to do with the configuration.
        c["ensemble"] = 1
        for s in range(CONFIRM_SEEDS):
            out.append(dict(c, seed=s))
    return out


STAGES = {"screen": stage_screen, "pair": stage_pair, "random": stage_random}


# ── driver ────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    p.add_argument("--delays", type=int, nargs="+", default=[100])
    p.add_argument("--delay-stride", type=int, default=1)
    p.add_argument("--stages", nargs="+", default=["screen"], choices=["screen", "pair", "random", "confirm"])
    p.add_argument("--out", default="results/hyper_search")
    p.add_argument("--tag", default="default")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--max-hours", type=float, default=0.0, help="stop launching new trials after this long (0 = no limit)"
    )
    p.add_argument("--limit", type=int, default=0, help="cap trials, for testing")
    p.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="shard the random stage; separate seeds draw disjoint "
        "samples that can run in parallel under separate tags",
    )
    p.add_argument("--n-random", type=int, default=0, help="trials in this random shard (0 = the full N_RANDOM)")
    args = p.parse_args()

    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.csv")

    rule("0. data")
    Q_raw, S, unflat, case0, run_id, cases = load_data(args)

    # band_hz changes the *target*, not the model, so everything derived from Q
    # -- the splits, the POD/AE latents, and the PODLSE reference -- has to be
    # rebuilt when it changes. That is why it cannot be handed to run_trial as a
    # keyword the way the model arguments are; treating it as one was the bug
    # that made the axis inert. Trials are ordered by band below so this runs
    # once per level rather than once per trial, and only the current level is
    # kept, because the AE latents are not small.
    ctx: dict = {}
    refs: dict = {}  # band -> PODLSE reference, for the stage summary

    def band_data(band):
        if ctx and ctx["band"] == band:
            return ctx
        Q, Q_full = Q_raw, None
        if band:
            Q_full = Q_raw
            Q = band_limit(Q_raw, band, 250.0, run_id)
            print(f"  [target band-limited to {band:g} Hz]", flush=True)
        tr, va, te = three_way(args, Q, run_id, cases)
        ctx.clear()
        ctx.update(
            band=band,
            Q=Q,
            Q_full=Q_full,
            tr=tr,
            va=va,
            te=te,
            latents=Latents(Q, tr, unflat, args),
            ref=reference(Q, S, tr, va, te, args),
        )
        refs[band] = ctx["ref"]
        return ctx

    done = set()
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            done.add(key_of(r))  # same normalisation as the configs
        print(f"  resuming: {len(done)} trial(s) already in {path}")
    else:
        with open(path, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    t_start = time.time()
    for st in args.stages:
        if st == "confirm":
            cfgs = stage_confirm(out_dir)
        elif st == "random":
            cfgs = stage_random(args.random_seed, args.n_random or None)
        else:
            cfgs = STAGES[st]()
        for c in cfgs:
            c["stage"] = st
        # band first so the rebuild above happens once per level rather than
        # once per trial; smallest-model-first within a level, so `tail -f`
        # half way through a level is still informative
        cfgs.sort(key=lambda c: (band_of(c, args) or 0.0, size_proxy(c)))
        todo = [c for c in cfgs if key_of(c) not in done]
        if args.limit:
            todo = todo[: args.limit]
        rule(f"stage {st}: {len(cfgs)} configs, {len(todo)} to run (smallest model first)")

        best = (float("inf"), None)
        for i, c in enumerate(todo):
            if args.max_hours and (time.time() - t_start) / 3600 > args.max_hours:
                print("  walltime budget reached; stopping cleanly")
                break
            c["trial"] = i
            d = band_data(band_of(c, args))
            row = run_trial(
                c, (d["Q"], d["Q_full"], S, d["tr"], d["va"], d["te"]), d["latents"], args, with_test=(st == "confirm")
            )
            with open(path, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=FIELDS).writerow({k: row.get(k) for k in FIELDS})
            done.add(key_of(c))
            if row["nmse_val"] < best[0]:
                best = (row["nmse_val"], c)
            flag = "  <-- best so far" if row["nmse_val"] == best[0] else ""
            print(
                f"  [{i + 1:>4}/{len(todo)}] {c['latent']}+{c['branch']:<6} "
                f"{c.get('axis', ''):<14} par {row['n_params']:>8,d}  "
                f"val {row['nmse_val']:.4f}  train {row['nmse_train']:.4f}  "
                f"best@{row['best_epoch']:<4} {row['seconds']:.0f}s{flag}",
                flush=True,
            )
        if best[1]:
            # the reference for the winner's own band -- a banded target is a
            # smaller target, so comparing across bands would flatter whichever
            # band happened to be narrowest
            r0 = refs.get(band_of(best[1], args))
            print(f"\n  stage {st} best: val {best[0]:.4f}" + (f" vs PODLSE {r0:.4f}" if r0 is not None else ""))
            print(f"    {json.dumps({k: str(v) for k, v in best[1].items() if k in KEY})}")

    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(
            dict(
                centre={k: str(v) for k, v in CENTRE.items()},
                screen={k: str(v) for k, v in SCREEN.items()},
                n_random=N_RANDOM,
                epochs=TRAIN_EPOCHS,
                args=vars(args),
            ),
            fh,
            indent=2,
            default=str,
        )
    rule("done")
    print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
