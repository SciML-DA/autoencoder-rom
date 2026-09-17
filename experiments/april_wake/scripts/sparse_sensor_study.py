#!/usr/bin/env python
"""
sparse_sensor_study.py
======================

The full sparse-sensor reconstruction study: reconstruct the PIV velocity field
around disc 2 from the twelve load-cell channels alone, with every method in the
repo, on one contiguous held-out block, at matched latent dimension.

    python experiments/april_wake/scripts/sparse_sensor_study.py --quick             # ~2 min, sanity
    python experiments/april_wake/scripts/sparse_sensor_study.py                     # the real thing
    python experiments/april_wake/scripts/sparse_sensor_study.py --cross-yaw         # held-out yaw
    python experiments/april_wake/scripts/sparse_sensor_study.py --forecast          # + task 3

What it runs, and why the grid is shaped this way
-------------------------------------------------
Every method here is the same three operators ``phi -> y -> phi`` with a sensor
map ``G`` bolted on, differing only in which of them is linear:

    E/D      G        what it is
    ------   ------   ----------------------------------------------
    POD      linear   POD-LSE / extended POD -- the baseline
    POD      MLP/GRU  nonlinear sensor map, linear manifold
    AE/CAE   linear   nonlinear manifold, linear sensor map
    AE/CAE   MLP/GRU  the full two-branch autoencoder

Running the whole 2x2 is the point. It *attributes* any improvement instead of
just observing one: a win in the last row that row 2 already had means the
manifold was never the limitation, and a win that appears only in the last row is
a real interaction between the two.

Crossed with ``--delays``, which is the other axis that matters. A linear map
from twelve instantaneous channels can only address a twelve-dimensional
subspace of the flow whatever ``r_field`` says; twenty-five lags raise that to
three hundred. See the rank-ceiling note in ``field_estimation/epod.py``.

Reported against two reference lines, both of which are printed next to every
score because neither alone is enough:

* **1.0** -- predicting the temporal mean. Below this the method learned
  something; at or above it, it did not, whatever the training loss said.
* **the projection floor** -- the best any method could do inside the field
  basis it was given. A score at the floor means the sensors are doing all they
  can and the basis is now the limit; a score far above it means the sensors or
  the fit are the limit. These call for opposite next moves.

Outputs, into ``results/sparse_sensors/<tag>/``::

    results.csv          one row per fitted model, appended across runs
    spectrum.png         POD spectrum with each truncation marked
    observability.png    per-mode linear observability from the sensors
    extended_modes.png   the reachable subspace, drawn in physical space
    comparison.png       test NMSE per method against both reference lines
    error_history.png    per-snapshot NMSE through the test block
    reconstruction.gif   truth / prediction / error for the best model
    horizon.png          closed-loop forecast error   (--forecast only)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # cx3 compute nodes have no display

import numpy as np  # noqa: E402

# this file is experiments/april_wake/scripts/<name>.py, so three levels
# up is the repo root -- which is what makes `experiments` importable.
# `src` needs no insert: the editable install puts it on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.april_wake.case_reader import (  # noqa: E402
    F_PIV_HZ,
    RUNS,
    build_case,
    concat_cases,
)
from experiments.april_wake.data_preprocessing import (  # noqa: E402
    add_data_args,
    band_limit,
    load_data,
    make_split,
)
from field_estimation import plots as rp  # noqa: E402
from field_estimation.branched_ae import (  # noqa: E402
    BranchedAE,
    LatentForecaster,
    LinearLatent,
    TorchLatent,
    default_device,
)
from field_estimation.epod import (  # noqa: E402
    PODLSE,
    cosine,
    delay_embed,
    energy_ratio,
    fluctuation_variance,
    mode_observability,
    nmse,
    nmse_per_snapshot,
    pod,
    projection_floor,
    ridge_cv,
)

CSV_FIELDS = [
    "tag",
    "runs",
    "test_run",
    "model",
    "latent",
    "branch",
    "r_field",
    "r_sensor",
    "n_delays",
    "ridge",
    "n_params",
    "nmse_train",
    "nmse_test",
    "nmse_latent",
    "cos_test",
    "energy_test",
    "nmse_by_run",
    "nmse_fullband",
    "floor_test",
    "fit_seconds",
    "seed",
    "n_train",
    "n_test",
]


def nmse_by_run(Q, pred, te, run_id, cases) -> str:
    """Test NMSE restricted to each source run, normalised by that run alone.

    A pooled multi-run NMSE is not comparable with a single-run one, and the
    difference is not small. Pooling five yaws puts the between-yaw variation
    into the denominator -- a large, low-rank, trivially predictable component,
    since a different yaw means a different mean force and a different mean wake
    position -- so the score improves for a reason that has nothing to do with
    reconstructing the wake any better. The projection floor moving 0.3966 ->
    0.3398 on the same basis is that effect with the sensors taken out of it.

    `nmse` normalises by the fluctuation variance of whatever block it is given,
    so scoring one run's test columns on their own restores the like-for-like
    comparison: this column against a single-run job's headline number.

    Returns:
        `"run=value;run=value"`, empty when there is only one run.
    """
    keys = np.unique(run_id[te])
    if len(keys) < 2:
        return ""
    out = []
    for k in keys:
        sel = run_id[te] == k
        name = cases[int(k)].run if int(k) < len(cases) else f"run{k}"
        out.append(f"{name}={nmse(Q[:, te][:, sel], pred[:, sel]):.4f}")
    return ";".join(out)


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}", flush=True)


# ── model runners ─────────────────────────────────────────────────────────────


def run_linear(Q, S, tr, te, args, Psi, q_mean, results, per_snap, preds, run_id=None, cases=None, Q_full=None):
    """POD-LSE and extended POD, at each delay length."""
    floor_te = projection_floor(Q[:, te], Psi, q_mean)

    for L in args.delays:
        Sd = delay_embed(S, L, args.delay_stride, args.delay_ahead)
        r_s = args.r_sensor if args.r_sensor else min(Sd.shape[0], len(tr) - 1)
        r_s = min(r_s, Sd.shape[0])

        ridge = args.ridge
        if args.ridge_cv:
            t0 = time.time()
            ridge, grid = ridge_cv(Q, Sd, tr, k=args.cv_folds, gap=args.gap, r_sensor=r_s, standardise_sensors=True)
            print(
                f"    ridge CV (L={L}): {ridge:g}  "
                f"[{', '.join(f'{k:g}:{v:.4f}' for k, v in grid.items())}]"
                f"  {time.time() - t0:.0f}s"
            )

        for name in ("podlse", "epod"):
            t0 = time.time()
            if name == "podlse":
                m = PODLSE(
                    r_field=args.r_field,
                    r_sensor=r_s,
                    ridge=ridge,
                    sensor_basis=args.sensor_basis,
                    pod_method=args.pod_method,
                ).fit(Q[:, tr], Sd[:, tr])
                lat = nmse(m.project(Q[:, te]), m.encode(Sd[:, te]))
            else:
                m = PODLSE(r_sensor=r_s, ridge=ridge).fit(Q[:, tr], Sd[:, tr])
                lat = float("nan")
            dt = time.time() - t0
            pred = m.predict(Sd[:, te])
            label = f"{name} L={L}"
            results.append(
                dict(
                    model=name,
                    latent="pod",
                    branch="closed-form",
                    r_field=args.r_field if name == "podlse" else -1,
                    r_sensor=r_s,
                    n_delays=L,
                    ridge=ridge,
                    n_params=m.n_params,
                    nmse_train=m.score(Q[:, tr], Sd[:, tr]),
                    nmse_test=nmse(Q[:, te], pred),
                    nmse_latent=lat,
                    # On the FLUCTUATION, as the reference notebook does and as
                    # sparse_sensor_sweep.py already did. Computed on the raw field
                    # both vectors are dominated by the ~10 m/s mean flow and the
                    # cosine is ~0.97 whatever the reconstruction does.
                    cos_test=cosine(Q[:, te] - Q[:, te].mean(1, keepdims=True), pred - pred.mean(1, keepdims=True)),
                    energy_test=energy_ratio(
                        Q[:, te] - Q[:, te].mean(1, keepdims=True), pred - pred.mean(1, keepdims=True)
                    ),
                    nmse_by_run=(nmse_by_run(Q, pred, te, run_id, cases) if run_id is not None else ""),
                    nmse_fullband=(nmse(Q_full[:, te], pred) if Q_full is not None else float("nan")),
                    floor_test=floor_te,
                    fit_seconds=dt,
                    label=label,
                )
            )
            per_snap[label] = nmse_per_snapshot(Q[:, te], pred)
            preds[label] = pred[:, : args.frames].astype(np.float32)
            _say(results[-1])
            if name == "epod" and L == args.delays[0]:
                _STASH["psi_ext"] = m.Psi_ext[:, : args.n_ext_modes].astype(np.float32)
    return floor_te


def build_latents(Q, tr, args, case0, unflat):
    """Fit each E/D pair once, on the training block, and reuse it everywhere.

    Fitting the autoencoder inside the branch loop is the obvious way to write
    this and it costs an hour per redundant fit. It also makes the comparison
    across branches noisier than it should be, because two branches would then
    be measured against two different decoders.
    """
    out = {}
    if "pod" in args.latents:
        t0 = time.time()
        Psi, Sigma, _, q_mean = pod(
            Q[:, tr], r=args.r_field, subtract_mean=True, method=args.pod_method, seed=args.seed
        )
        print(f"  POD basis r={args.r_field}: {time.time() - t0:.1f}s")
        out["pod"] = (LinearLatent(Psi, q_mean, device=args.device), Psi, Sigma, q_mean)

    grid_tr = None
    for kind in ("ae", "cae"):
        if kind not in args.latents:
            continue
        from models.data_driven.autoencoders import AE, CAE  # torch import deferred

        if grid_tr is None:
            # rebuilt from Q, not taken from case0.X. Two reasons: with several
            # runs concatenated, case0.X is one run's snapshots and would be
            # silently misaligned with the training indices; and going through Q
            # guarantees the AE sees exactly the rows the POD saw, so the
            # comparison is about capacity and not about masking.
            grid_tr = unflat(Q[:, tr], dtype=np.float32)  # (Nu, n_tr, Nx, Ny)
        cls = AE if kind == "ae" else CAE
        t0 = time.time()
        p = cls(
            n_latent=args.r_field,
            n_epochs=args.ae_epochs,
            batch_size=args.ae_batch,
            learning_rate=args.ae_lr,
            patience=args.ae_patience,
            device=args.device,
            seed=args.seed,
        )
        p.fit(grid_tr)
        print(
            f"  {kind.upper()} latent={args.r_field}: {time.time() - t0:.1f}s, "
            f"{p.n_params / 1e6:.1f}M params, "
            f"{p.training_history.n_epochs_run} epochs, "
            f"train MSE {p.training_history.train[-1]:.3e}"
        )
        out[kind] = (TorchLatent(p, device=args.device), None, None, None)
    return out


def run_branched(Q, S, tr, te, args, latents, results, per_snap, preds, floor_te, run_id=None, cases=None, Q_full=None):
    """The two-branch models: every (latent, branch, delay) combination asked for."""
    for lname, (lat, Psi, _, q_mean) in latents.items():
        # each latent space has its own floor -- an AE bottleneck of the same
        # size is a different subspace (and not a subspace at all), so quoting
        # the POD floor next to an AE score would be comparing to the wrong bound
        floor = floor_te if lname == "pod" else _latent_floor(lat, Q, te)
        for branch in args.branches:
            for L in args.delays:
                if branch == "linear" and L == 1 and lname == "pod":
                    continue  # exactly POD-LSE, already run in closed form
                t0 = time.time()
                m = BranchedAE(
                    lat,
                    branch=branch,
                    n_delays=L,
                    delay_stride=args.delay_stride,
                    hidden=tuple(args.hidden),
                    gru_hidden=args.gru_hidden,
                    cnn_channels=tuple(args.cnn_channels),
                    lambda_field=args.lambda_field,
                    latent_weight=args.latent_weight,
                    n_epochs=args.epochs,
                    batch_size=args.batch,
                    learning_rate=args.lr,
                    patience=args.patience,
                    device=args.device,
                    seed=args.seed,
                    verbose=args.verbose,
                ).fit(Q, S, tr)
                dt = time.time() - t0
                pred = m.predict(S, te)
                label = f"{lname}+{branch} L={L}"
                results.append(
                    dict(
                        model="branched",
                        latent=lname,
                        branch=branch,
                        r_field=args.r_field,
                        r_sensor=-1,
                        n_delays=L,
                        ridge=float("nan"),
                        n_params=m.n_params,
                        nmse_train=m.score(Q, S, tr),
                        nmse_test=nmse(Q[:, te], pred),
                        nmse_by_run=(nmse_by_run(Q, pred, te, run_id, cases) if run_id is not None else ""),
                        nmse_fullband=(nmse(Q_full[:, te], pred) if Q_full is not None else float("nan")),
                        nmse_latent=m.latent_score(Q, S, te),
                        floor_test=floor,
                        fit_seconds=dt,
                        label=label,
                    )
                )
                per_snap[label] = nmse_per_snapshot(Q[:, te], pred)
                preds[label] = pred[:, : args.frames].astype(np.float32)
                _say(results[-1])
                _stash(args, f"pred_{lname}_{branch}_L{L}", pred[:, : args.n_save])


def _latent_floor(lat, Q, te):
    """Round-trip error of the E/D pair alone -- the bound for its own sensors."""
    return nmse(Q[:, te], lat.decode(lat.encode(Q[:, te])))


def run_forecast(Q, S, tr, te, args, latents, results):
    """Task 3: nowcast with G, then roll the latent forward and decode.

    The composition ``s -> G -> y -> F -> y' -> D -> phi'`` is the whole
    pipeline. Reported closed-loop, because the one-step error at 250 Hz is
    dominated by the fact that the state barely moves between samples and
    flatters everything.
    """
    lname = "pod" if "pod" in latents else next(iter(latents))
    lat = latents[lname][0]
    L = max(args.delays)

    Z_tr = lat.encode(Q[:, tr])
    print(f"  fitting the latent forecaster on {Z_tr.shape} ...")
    f = LatentForecaster(
        n_latent=lat.n_latent,
        n_delays=L,
        n_unroll=args.n_unroll,
        hidden=args.gru_hidden,
        n_epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose,
    ).fit(Z_tr, np.arange(len(tr)))

    Z_te_true = lat.encode(Q[:, te])
    nowcast = BranchedAE(
        lat,
        branch="gru",
        n_delays=L,
        n_epochs=args.epochs,
        gru_hidden=args.gru_hidden,
        device=args.device,
        seed=args.seed,
    ).fit(Q, S, tr)
    Z_te_now = nowcast.encode(S, te)

    # One fixed normaliser for every horizon. Normalising each horizon by its
    # own window variance measures how far the flow happens to move over that
    # window as much as it measures the model, and it makes h=1 divide by zero.
    var = fluctuation_variance(Q[:, te])
    horizons = [h for h in (1, 5, 10, 25, 50, 100, 200) if h < len(te) // 4]
    starts = list(range(L, len(te) - max(horizons), max(1, (len(te) - max(horizons) - L) // 20)))

    curves = {}
    sources = [("truth-initialised", Z_te_true), ("sensor-initialised", Z_te_now)]
    for src, Z0 in sources:
        errs = []
        for h in horizons:
            e = [nmse(Q[:, te[t0 : t0 + h]], lat.decode(f.rollout(Z0[:, t0 - L : t0], h)), var) for t0 in starts]
            errs.append(float(np.mean(e)))
        curves[src] = (horizons, errs)
        _record_forecast(results, args, lname, f"gru/{src}", horizons, errs)
        print(f"    {src}: " + "  ".join(f"h={h}:{v:.3f}" for h, v in zip(horizons, errs)))

    # Persistence is the honest null model: hold the last known latent state.
    # Anything that fails to beat it has learned the identity map and nothing
    # more, and at 250 Hz that is an easy trap -- the state barely moves.
    errs = []
    for h in horizons:
        e = [
            nmse(Q[:, te[t0 : t0 + h]], lat.decode(np.repeat(Z_te_true[:, t0 - 1 : t0], h, axis=1)), var)
            for t0 in starts
        ]
        errs.append(float(np.mean(e)))
    curves["persistence"] = (horizons, errs)
    _record_forecast(results, args, lname, "persistence", horizons, errs)
    print("    persistence:      " + "  ".join(f"h={h}:{v:.3f}" for h, v in zip(horizons, errs)))
    return curves


def _record_forecast(results, args, lname, branch, horizons, errs):
    for h, v in zip(horizons, errs):
        results.append(
            dict(
                model="forecast",
                latent=lname,
                branch=branch,
                r_field=args.r_field,
                r_sensor=-1,
                n_delays=h,
                ridge=float("nan"),
                n_params=0,
                nmse_train=float("nan"),
                nmse_test=v,
                nmse_latent=float("nan"),
                floor_test=float("nan"),
                fit_seconds=0.0,
                label=f"forecast {branch} h={h}",
            )
        )


# ── reporting ─────────────────────────────────────────────────────────────────


def _say(r: dict) -> None:
    print(
        f"    {r['label']:28s} test {r['nmse_test']:.4f}  "
        f"train {r['nmse_train']:.4f}  floor {r['floor_test']:.4f}  "
        f"{r['n_params']:>9,d} par  {r['fit_seconds']:.0f}s",
        flush=True,
    )


_STASH: dict = {}


def _stash(args, key, arr):
    if args.save_arrays:
        _STASH[key] = np.asarray(arr, np.float32)


def write_csv(path, rows, args, tr, te, held):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(
                {
                    **r,
                    "tag": args.tag,
                    "runs": "|".join(args.runs or [args.run]),
                    "test_run": held,
                    "seed": args.seed,
                    "n_train": len(tr),
                    "n_test": len(te),
                }
            )
    print(f"  wrote {path}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)

    m = p.add_argument_group("models")
    m.add_argument("--r-field", type=int, default=64, help="latent / POD truncation")
    m.add_argument("--r-sensor", type=int, default=None)
    m.add_argument("--delays", type=int, nargs="+", default=[1, 25])
    m.add_argument("--delay-stride", type=int, default=1)
    m.add_argument("--ridge", type=float, default=1e-4)
    m.add_argument("--ridge-cv", action="store_true")
    m.add_argument("--cv-folds", type=int, default=5)
    m.add_argument("--latents", nargs="+", default=["pod"], choices=["pod", "ae", "cae"])
    # nargs="*" so `--branches` on its own means "linear baselines only" --
    # that is a real configuration (experiments/april_wake/hpc/sensors.pbs), not a mistake
    m.add_argument("--branches", nargs="*", default=["linear", "mlp", "gru"], choices=["linear", "mlp", "cnn", "gru"])
    m.add_argument("--pod-method", default="randomized", choices=["svd", "snapshot", "randomized", "auto"])
    m.add_argument("--forecast", action="store_true", help="also run task 3")

    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=400)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--patience", type=int, default=40)
    t.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    t.add_argument("--gru-hidden", type=int, default=64)
    t.add_argument("--cnn-channels", type=int, nargs="+", default=[32, 64])
    t.add_argument("--lambda-field", type=float, default=0.0)
    t.add_argument("--latent-weight", default="energy", choices=["energy", "unit"])
    t.add_argument("--n-unroll", type=int, default=5)
    t.add_argument("--ae-epochs", type=int, default=300)
    t.add_argument("--ae-batch", type=int, default=64)
    t.add_argument("--ae-lr", type=float, default=1e-3)
    t.add_argument("--ae-patience", type=int, default=40)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None)

    o = p.add_argument_group("output")
    o.add_argument("--out", default="results/sparse_sensors")
    o.add_argument("--tag", default="default")
    o.add_argument("--frames", type=int, default=200)
    o.add_argument("--n-save", type=int, default=200)
    o.add_argument("--n-ext-modes", type=int, default=6)
    o.add_argument("--save-arrays", action="store_true")
    o.add_argument("--no-plots", action="store_true")
    o.add_argument("--verbose", action="store_true")
    o.add_argument("--quick", action="store_true", help="tiny run, for a smoke test")

    args = p.parse_args()
    if args.quick:
        args.n = args.n or 800
        args.r_field, args.epochs, args.ae_epochs = 16, 60, 40
        args.delays, args.branches, args.frames = [1, 10], ["linear", "mlp"], 40
    args.device = default_device(args.device)
    out = os.path.join(args.out, args.tag)
    os.makedirs(out, exist_ok=True)
    print(f"device: {args.device}   output: {out}")

    rule("1. data")
    Q, S, unflat, case0, run_id, cases = load_data(args)
    Q_full = None
    if args.band_hz:
        Q_full = Q
        Q = band_limit(Q, args.band_hz, 250.0, run_id)
    tr, te, held = make_split(args, Q.shape[1], run_id, cases)

    rule("2. latent spaces")
    latents = build_latents(Q, tr, args, case0, unflat)

    results, per_snap, preds = [], {}, {}

    rule("3. linear baselines")
    Psi, Sigma, q_mean = _basis(latents, Q, tr, args)
    floor_te = run_linear(Q, S, tr, te, args, Psi, q_mean, results, per_snap, preds, run_id, cases, Q_full)

    rule("4. two-branch models")
    run_branched(Q, S, tr, te, args, latents, results, per_snap, preds, floor_te, run_id, cases, Q_full)

    curves = {}
    if args.forecast:
        rule("5. forecasting (task 3)")
        curves = run_forecast(Q, S, tr, te, args, latents, results)

    rule("6. results")
    # forecast rows measure a different quantity (h-step-ahead, not nowcast),
    # so they must not compete for "best" -- a 1-step forecast would win every
    # time and it is not solving the task the other rows are solving
    nowcast = [r for r in results if r["model"] != "forecast"]
    best = min(nowcast, key=lambda r: r["nmse_test"])
    print(
        f"  best: {best['label']}  test NMSE {best['nmse_test']:.4f} "
        f"(floor {best['floor_test']:.4f}, mean-predictor 1.0)"
    )
    if best.get("nmse_by_run"):
        print("\n  per run, each normalised by its own test block (comparable with a single-run job):")
        for part in best["nmse_by_run"].split(";"):
            name, val = part.split("=")
            print(f"    {name:<28} {val}")
    write_csv(os.path.join(out, "results.csv"), results, args, tr, te, held)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2, default=str)

    if not args.no_plots:
        rule("7. plots")
        obs = mode_observability(Psi.T @ (Q[:, tr] - q_mean), _sensor_coeffs(S, tr, args))
        rp.plot_spectrum(Sigma, os.path.join(out, "spectrum.png"), floors={f"r={args.r_field}": args.r_field})
        rp.plot_observability(obs, os.path.join(out, "observability.png"))
        if "psi_ext" in _STASH:
            rp.plot_extended_modes(
                _STASH["psi_ext"],
                unflat,
                os.path.join(out, "extended_modes.png"),
                n_modes=args.n_ext_modes,
                mf=case0.mf,
            )
        # nowcast rows only: a forecast row measures h-step-ahead error, which
        # is a different quantity on the same axis, and at h=200 it dwarfs the
        # chart it is not competing in. horizon.png is where those belong.
        rp.plot_comparison(nowcast, os.path.join(out, "comparison.png"), floor=floor_te)
        rp.plot_error_history(per_snap, os.path.join(out, "error_history.png"), dt=args.stride / F_PIV_HZ)
        if curves:
            rp.plot_horizon(curves, os.path.join(out, "horizon.png"), dt=args.stride / F_PIV_HZ)
        _animate(best, Q, te, preds, args, unflat, case0, out)
        print(f"  plots -> {out}/")

    if args.save_arrays and _STASH:
        np.savez_compressed(os.path.join(out, "arrays.npz"), **_STASH)
        print(f"  arrays -> {out}/arrays.npz")
    print()
    return 0


def _basis(latents, Q, tr, args):
    """The POD basis used for the floors and the observability diagnostics."""
    if "pod" in latents:
        _, Psi, Sigma, q_mean = latents["pod"]
        return Psi, Sigma, q_mean
    Psi, Sigma, _, q_mean = pod(Q[:, tr], r=args.r_field, subtract_mean=True, method=args.pod_method, seed=args.seed)
    return Psi, Sigma, q_mean


def _sensor_coeffs(S, tr, args):
    Sd = delay_embed(S, max(args.delays), args.delay_stride, args.delay_ahead)[:, tr]
    Sc = Sd - Sd.mean(1, keepdims=True)
    sd = Sc.std(1, keepdims=True)
    _, _, C, _ = pod(Sc / np.where(sd > 0, sd, 1.0), args.r_sensor)
    return C


def _animate(best, Q, te, preds, args, unflat, case0, out):
    """Animate the winning model's test-block reconstruction.

    Uses the prediction cached when the model was fitted rather than refitting
    it: a refit is minutes of GPU for a picture, and with a different RNG draw
    it would not be the model whose number is in the CSV.
    """
    pred = preds.get(best["label"])
    if pred is None:
        print("  (no cached prediction for the best model; skipping the animation)")
        return
    n = pred.shape[1]
    rp.animate_reconstruction(
        unflat(Q[:, te[:n]]),
        unflat(pred.astype(np.float64)),
        os.path.join(out, "reconstruction.gif"),
        mf=case0.mf,
        n_frames=n,
        title=f"{best['label']}  NMSE {best['nmse_test']:.3f}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
