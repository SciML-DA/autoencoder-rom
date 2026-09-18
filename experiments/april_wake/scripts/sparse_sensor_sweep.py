#!/usr/bin/env python
"""
sparse_sensor_sweep.py
======================

Convergence study for sparse-sensor reconstruction. Sweeps the axes that matter
one at a time, holding the rest at a baseline, and writes a resumable CSV plus
the convergence plots and a video pack.

    python experiments/april_wake/scripts/sparse_sensor_sweep.py --quick            # ~3 min, smoke test
    python experiments/april_wake/scripts/sparse_sensor_sweep.py                    # the real thing
    python experiments/april_wake/scripts/sparse_sensor_sweep.py --stages A B       # just those two
    qsub experiments/april_wake/hpc/sparse_sensor_sweep.pbs                         # on cx3

Why staged, and not one big grid
--------------------------------
The full cross product of {latent size} x {delay length} x {E/D family} x
{branch} x {seed} is thousands of fits and tells you almost nothing, because
every curve in it is confounded with every other axis. Sweeping one axis at a
time against a fixed baseline gives you curves you can actually read, at a small
fraction of the cost.

    A  latent   r_field in {4..128}, delays fixed   -> the convergence curve
    B  delays   n_delays in {1..100}, r_field fixed -> where history stops paying
    C  sensors  channel ablation                    -> how few sensors you need
    D  seeds    the best few configs, repeated      -> is the gap real or noise

Stage A is the direct analogue of ``convergence_study.py`` for this task, and
it is the one to run first.

What makes it affordable
------------------------
Three things, all of which matter at the real problem size:

* **POD modes are nested.** The rank-16 basis is the first sixteen columns of the
  rank-128 basis, so stage A computes *one* decomposition at the largest
  ``r_field`` and slices it. That turns six SVDs into one.
* **Autoencoders are cached to disk**, keyed by a hash of the hyperparameters
  that determine the fit. A resumed job, or a second stage using the same
  ``r_field``, reloads instead of retraining. Deleting ``--cache-dir`` forces a
  refit -- and you must do that if you change the model code, because the hash
  covers hyperparameters, not source.
* **The CSV is appended row by row and read back on startup.** A job that hits
  walltime loses the row it was on, not the run. Resubmit and it continues.

Outputs, into ``results/sparse_sweep/<tag>/``::

    results.csv              one row per fit, appended, resumable
    convergence_latent.png   NMSE vs latent size, with the projection floor
    convergence_delays.png   NMSE vs window length
    ablation_sensors.png     NMSE vs which channels were kept
    seed_spread.png          the spread that any claimed gap has to clear
    spectrum.png             POD spectrum, with every sweep point marked
    observability.png        per-mode linear observability
    video_pack.npz           truth + the best predictions, for the video script
    reconstruction_*.gif     rendered on the node (mp4 if ffmpeg is present)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from config.model_config import load_model_from_config, save_model_to_config  # noqa: E402
from experiments.april_wake.case_reader import F_PIV_HZ, RUNS  # noqa: E402
from experiments.april_wake.data_preprocessing import (  # noqa: E402
    add_data_args,
    band_limit,
    load_data,
    make_split,
)
from field_estimation import plots as rp  # noqa: E402
from field_estimation.branched_ae import TorchLatent, default_device  # noqa: E402
from field_estimation.epod import (  # noqa: E402
    cosine,
    delay_embed,
    energy_ratio,
    mode_observability,
    nmse,
    pod,
    projection_floor,
)

CSV_FIELDS = [
    "stage",
    "model",
    "latent",
    "branch",
    "r_field",
    "r_sensor",
    "r_sensor_used",
    "ridge",
    "n_delays",
    "channels",
    "seed",
    "n_params",
    "nmse_train",
    "nmse_test",
    "nmse_latent",
    "cos_test",
    "energy_test",
    "floor_test",
    # Training diagnostics. Without these a flat architecture comparison is
    # uninterpretable: a model that stopped at the epoch cap and one that
    # early-stopped on a plateau look identical in the score column and mean
    # opposite things. `n_fit` is the snapshots the fit actually saw, which for
    # a branched model is (1 - val_fraction) of the training block and so is
    # *smaller* than the closed-form estimators' -- a difference that reads as
    # a modelling gap if it is not recorded.
    "sensor_noise",
    "weight_decay",
    "lr",
    "ensemble",
    # With --band-hz, `nmse_test` scores against the BANDED target the model was
    # fitted to and `nmse_fullband` scores the same prediction against the
    # unfiltered field. Reporting only the first would let band-limiting flatter
    # itself: shrinking the target shrinks the error without the reconstruction
    # improving. The pair is the honest statement.
    "nmse_fullband",
    "epochs_run",
    "train_loss",
    "val_loss",
    "stopped_early",
    "n_fit",
    "fit_seconds",
    "n_train",
    "n_test",
    "run",
    "tag",
]

# The row's identity: the configuration as *requested*, so a planned config and
# the row it eventually writes hash to the same thing. That is what makes the
# CSV resumable and what stops a restart doubling every row.
#
# `r_sensor` here is the requested value (often None, meaning "all of them");
# the value actually used after clamping to the delay-embedded width is recorded
# separately as `r_sensor_used`. Keying on the resolved value instead looks
# equivalent and silently breaks resumption, because None != 120.
KEY = (
    "stage",
    "model",
    "latent",
    "branch",
    "r_field",
    "r_sensor",
    "ridge",
    "n_delays",
    "channels",
    "seed",
    "sensor_noise",
    "weight_decay",
    "lr",
    "ensemble",
)


def rule(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}", flush=True)


def _norm(v) -> str:
    """Canonical string for a key field, in memory and after a CSV round trip.

    ``csv.DictWriter`` writes ``None`` as an empty string, so an in-memory
    ``r_sensor=None`` comes back as ``""`` and the planned config stops matching
    the row it wrote. Both spellings collapse to "None" here. Without this,
    resumption silently does nothing and every restart doubles the CSV.
    """
    t = "" if v is None else str(v)
    return "None" if t == "" else t


def key_of(row) -> tuple:
    return tuple(_norm(row[k]) for k in KEY)


def _apply_unless_given(args, **defaults):
    """Set ``defaults`` on ``args``, skipping anything named on the command line.

    argparse cannot distinguish "the user typed the default" from "the user typed
    nothing", so this reads ``sys.argv`` directly. Crude, but the alternative --
    giving every flag a ``None`` default and resolving each one by hand -- spreads
    the same logic over thirty arguments.
    """
    typed = {a.split("=")[0].lstrip("-").replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
    for k, v in defaults.items():
        if k not in typed:
            setattr(args, k, v)


# ── the autoencoder cache ─────────────────────────────────────────────────────


def _ae_hash(kind, r, seed, args, n_train) -> str:
    """Content hash of everything that determines the fit.

    Deliberately *not* including the source of ``models/data_driven/autoencoders/ae.py``: hashing
    source would invalidate the cache on a comment change, and hashing nothing
    would serve a stale model after a real change. The compromise is that this
    covers hyperparameters and you clear ``--cache-dir`` by hand when you touch
    the model. Same trade-off ``config/model_config.py`` makes for the ESN.
    """
    blob = json.dumps(
        {
            "kind": kind,
            "flat_layout": "component-first",
            "r": r,
            "seed": seed,
            "n_train": int(n_train),
            "epochs": args.ae_epochs,
            "batch": args.ae_batch,
            "lr": args.ae_lr,
            "patience": args.ae_patience,
            "run": "|".join(args.runs or [args.run]),
            "n": args.n,
            "stride": args.stride,
            "test_fraction": args.test_fraction,
            "force_smooth": args.force_smooth,
            "probes": args.probes,
            "notch": args.notch,
            "notch_q": args.notch_q,
            "hidden_scale": list(args.ae_hidden_scale),
            "cae_channels": list(args.cae_channels),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Latents:
    """Fitted E/D pairs, built on demand and reused across every stage.

    POD is the interesting case: the modes are nested, so one decomposition at
    the largest ``r_field`` in the sweep serves every smaller one by slicing. On
    the real problem that is the difference between one two-minute randomized SVD
    and six of them.
    """

    def __init__(self, Q, tr, unflat, args):
        self.Q, self.tr, self.unflat, self.args = Q, tr, unflat, args
        self.dev = args.device
        self._pod = None
        self._cache = {}
        os.makedirs(args.cache_dir, exist_ok=True)

    def pod_basis(self, r):
        """(Psi[:, :r], Sigma, q_mean) from one decomposition at r_max."""
        if self._pod is None:
            t0 = time.time()
            r_max = max(self.args.latents_sweep + [self.args.r_field])
            Psi, Sigma, _, qm = pod(
                self.Q[:, self.tr], r=r_max, subtract_mean=True, method=self.args.pod_method, seed=self.args.seed
            )
            print(f"  POD basis r={r_max} (nested, sliced for smaller): {time.time() - t0:.1f}s", flush=True)
            self._pod = (Psi, Sigma, qm)
        Psi, Sigma, qm = self._pod
        return Psi[:, :r], Sigma, qm

    def get(self, kind, r, seed):
        ck = (kind, r, seed)
        if ck in self._cache:
            return self._cache[ck]

        if kind == "pod":
            Psi, _, qm = self.pod_basis(r)
            Latent = backend_classes(self.args.backend)[2]
            lat = Latent(Psi, qm) if self.args.backend == "jax" else Latent(Psi, qm, device=self.dev)
        else:
            lat = TorchLatent(self._autoencoder(kind, r, seed), device=self.dev)
        self._cache[ck] = lat
        return lat

    def _autoencoder(self, kind, r, seed):
        a = self.args
        name = f"{kind}_r{r}_s{seed}_{_ae_hash(kind, r, seed, a, len(self.tr))}"
        from models.data_driven.autoencoders import AE, CAE

        if not a.no_cache:
            cached = load_model_from_config(q=name, load_dir=a.cache_dir, device=self.dev)
            if isinstance(cached, AE | CAE):
                print(f"  {kind.upper()} r={r} seed={seed}: cached", flush=True)
                return cached

        grid = self.unflat(self.Q[:, self.tr], dtype=np.float32)
        t0 = time.time()
        if kind == "ae":
            # Hidden widths scale with the latent, as convergence_study.py does,
            # for two reasons. Fixed widths make the dense AE's parameter count
            # essentially independent of r, so its curve is flat by construction
            # and says nothing about latent size -- and at r=128 the default
            # (512, 128) puts a hidden layer at exactly the latent width, so the
            # bottleneck is no longer uniquely the bottleneck.
            dims = tuple(max(int(f * r), r + 1) for f in a.ae_hidden_scale)
            cls, kw = AE, {"hidden": dims}
        else:
            cls, kw = CAE, {"channels": tuple(a.cae_channels)}
        p = cls(
            n_latent=r,
            epochs=a.ae_epochs,
            batch_size=a.ae_batch,
            learning_rate=a.ae_lr,
            patience=a.ae_patience,
            device=self.dev,
            seed=seed,
            **kw,
        ).fit(grid)
        print(
            f"  {kind.upper()} r={r} seed={seed}: {time.time() - t0:.1f}s, "
            f"{p.n_params / 1e6:.1f}M params, "
            f"{p.training_history.n_epochs_run} epochs",
            flush=True,
        )
        save_model_to_config(p, save_dir=a.cache_dir, name=name)
        return p


# ── one fit ───────────────────────────────────────────────────────────────────


BACKENDS = {
    "torch": ("PODLSE", "BranchedAE", "LinearLatent"),
    "jax": ("PODLSEJax", "BranchedAEJax", "LinearLatentJax"),
}


def backend_classes(backend: str):
    """Returns (PODLSE, BranchedAE, LinearLatent) for a backend."""
    import field_estimation

    return tuple(getattr(field_estimation, n) for n in BACKENDS[backend])


def fit_one(cfg, Q, S, tr, te, latents, args, videos, Q_full=None):
    """Fit and score one configuration. Returns a CSV row."""
    Lin, Branch, _ = backend_classes(args.backend)
    ch = _channels(S, cfg["channels"])
    Sd = delay_embed(S[ch], cfg["n_delays"], args.delay_stride, args.delay_ahead)
    t0 = time.time()

    if cfg["model"] in ("podlse", "epod"):
        r_s = min(cfg["r_sensor"] or Sd.shape[0], Sd.shape[0])
        if cfg["model"] == "podlse":
            m = Lin(
                r_field=cfg["r_field"],
                r_sensor=r_s,
                ridge=cfg["ridge"],
                sensor_basis=args.sensor_basis,
                pod_method=args.pod_method,
            ).fit(Q[:, tr], Sd[:, tr])
            lat_err = nmse(m.project(Q[:, te]), m.encode(Sd[:, te]))
            floor = m.floor(Q[:, te])
        else:
            m = Lin(r_field=None, r_sensor=r_s, ridge=cfg["ridge"]).fit(Q[:, tr], Sd[:, tr])
            lat_err = float("nan")
            Psi, _, qm = latents.pod_basis(cfg["r_field"])
            floor = projection_floor(Q[:, te], Psi, qm)
        pred = m.predict(Sd[:, te])
        train_err = m.score(Q[:, tr], Sd[:, tr])
        n_par = m.n_params
        r_s_out = r_s
    else:
        lat = latents.get(cfg["latent"], cfg["r_field"], cfg["seed"])
        kw = dict(
            branch=cfg["branch"],
            n_delays=cfg["n_delays"],
            delay_stride=args.delay_stride,
            hidden=tuple(args.hidden),
            gru_hidden=args.gru_hidden,
            cnn_channels=tuple(args.cnn_channels),
            lambda_field=args.lambda_field,
            latent_weight=args.latent_weight,
            sensor_noise=cfg["sensor_noise"],
            weight_decay=cfg["weight_decay"],
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            epochs=args.epochs,
            batch_size=args.batch,
            learning_rate=cfg["lr"],
            patience=args.patience,
            seed=cfg["seed"],
        )
        if args.backend == "torch":
            kw["device"] = args.device  # jax picks its device from XLA
        n_ens = int(cfg.get("ensemble", 1) or 1)
        if n_ens == 1:
            m = Branch(lat, **kw).fit(Q, S[ch], tr)
            pred = m.predict(S[ch], te)
            train_err = m.score(Q, S[ch], tr)
            lat_err = m.latent_score(Q, S[ch], te)
        else:
            # Average the *predictions*, not the weights. Two networks that
            # reach equally good but different minima have no meaningful
            # average in parameter space -- the mean of their weights is
            # generally worse than either -- while the mean of their outputs
            # is at least as good as the average member and usually better,
            # because their errors are partly independent.
            #
            # This is variance reduction and nothing else: it cannot represent
            # anything a single member could not. A gain here measures
            # run-to-run spread, so it should be read next to stage D.
            preds, trs, lats = [], [], []
            for j in range(n_ens):
                kw_j = dict(kw, seed=cfg["seed"] * 1000 + j)
                mj = Branch(lat, **kw_j).fit(Q, S[ch], tr)
                preds.append(mj.predict(S[ch], te))
                trs.append(mj.score(Q, S[ch], tr))
                lats.append(mj.latent_score(Q, S[ch], te))
                if j == 0:
                    m = mj
            pred = np.mean(preds, axis=0)
            train_err = float(np.mean(trs))
            lat_err = float(np.mean(lats))
        floor = nmse(Q[:, te], lat.decode(lat.encode(Q[:, te])))
        n_par = m.n_params * n_ens
        r_s_out = -1

    # training diagnostics; the closed-form estimators have no epochs and see
    # the whole training block, which is exactly the contrast worth recording
    hist = list(getattr(m, "loss_history", []) or [])
    vhist = list(getattr(m, "val_loss_history", []) or [])
    n_ep = len(hist)
    stopped = bool(n_ep and n_ep < int(getattr(m, "epochs", 0) or 0))
    n_fit = len(tr)
    if cfg["model"] not in ("podlse", "epod"):
        n_fit = int(round(len(tr) * (1.0 - float(getattr(m, "val_fraction", 0.0)))))

    test_err = nmse(Q[:, te], pred)
    full_err = nmse(Q_full[:, te], pred) if Q_full is not None else float("nan")
    # cosine and the energy ratio alongside NMSE: together they say *why* an
    # NMSE is what it is. The reference notebook reports cosine only, so this is
    # also what makes these numbers comparable with it.
    cos_te = cosine(Q[:, te] - Q[:, te].mean(1, keepdims=True), pred - pred.mean(1, keepdims=True))
    en_te = energy_ratio(Q[:, te] - Q[:, te].mean(1, keepdims=True), pred - pred.mean(1, keepdims=True))
    row = dict(
        cfg,
        r_sensor_used=r_s_out,
        n_params=n_par,
        nmse_train=train_err,
        nmse_test=test_err,
        nmse_latent=lat_err,
        cos_test=cos_te,
        energy_test=en_te,
        nmse_fullband=full_err,
        floor_test=floor,
        epochs_run=n_ep,
        train_loss=(hist[-1] if hist else float("nan")),
        val_loss=(vhist[-1] if vhist else float("nan")),
        stopped_early=int(stopped),
        n_fit=n_fit,
        fit_seconds=time.time() - t0,
        n_train=len(tr),
        n_test=len(te),
        run="|".join(args.runs or [args.run]),
        tag=args.tag,
    )
    videos.offer(label_of(cfg), test_err, pred[:, : args.video_frames])
    ep = f"  {n_ep}ep{'*' if stopped else ''}" if n_ep else ""
    print(
        f"    {label_of(cfg):38s} test {test_err:.4f}  train {train_err:.4f}  "
        f"cos {cos_te:.3f}  E {en_te:.2f}  floor {floor:.4f}  "
        f"{n_par:>8,d} par  {row['fit_seconds']:.0f}s{ep}",
        flush=True,
    )
    return row


def _channels(S, spec):
    """Index array for a channel subset. ``spec`` is 'all' | 'disc2' | 'disc3' | 'forces'.

    The 12 channels are two 6-component balances, discs 2 and 3, in that order --
    Fx Fy Fz Mx My Mz each. Confirm this against the rig before quoting an
    ablation: the channel order is a property of the DAQ wiring and is not
    recoverable from the .dat file.
    """
    n = S.shape[0]
    if spec == "all":
        return np.arange(n)
    if spec == "disc2":
        return np.arange(min(6, n))
    if spec == "disc3":
        return np.arange(6, n)
    if spec == "forces":  # Fx Fy Fz of each balance, moments dropped
        return np.array([i for i in range(n) if i % 6 < 3])
    if spec == "moments":
        return np.array([i for i in range(n) if i % 6 >= 3])
    raise ValueError(f"unknown channel spec {spec!r}")


def label_of(cfg) -> str:
    base = cfg["model"] if cfg["model"] in ("podlse", "epod") else f"{cfg['latent']}+{cfg['branch']}"
    out = f"{base} r={cfg['r_field']} L={cfg['n_delays']}"
    if cfg["stage"] == "R" and cfg["model"] in ("podlse", "epod"):
        out += f" rs={cfg['r_sensor']} lam={cfg['ridge']:g}"
    if cfg["channels"] != "all":
        out += f" [{cfg['channels']}]"
    if cfg["seed"]:
        out += f" s{cfg['seed']}"
    return out


class VideoPack:
    """The ``k`` best predictions seen, plus the best one per pinned pattern.

    Pins are the point: a video of only the winner is a demo, not a comparison.
    ``pins=["podlse"]`` reserves a slot for the best linear baseline, so the
    pack always contains something to hold the nonlinear result against — even
    when the winner is itself linear, in which case the two slots collapse and
    you get one video, which is the honest outcome.

    A pin is one *slot*, not a filter: only the best row matching each pattern
    is kept, not every row that matches it.

    Each entry is (N_x, n_frames) float32 — about 37 MB at the real problem
    size and 300 frames, so k=2 plus a pin is ~110 MB resident. Lower
    ``--video-frames`` before lowering ``--video-top``: the comparison is worth
    more than the extra seconds of footage.
    """

    def __init__(self, k, pins):
        self.k, self.pins = k, list(pins)
        self._pinned = {}  # pattern -> (err, label, pred)
        self._top = {}  # label   -> (err, pred)

    def offer(self, label, err, pred):
        if not np.isfinite(err):
            return
        pred = np.asarray(pred, np.float32)
        for pat in self.pins:
            if pat in label and (pat not in self._pinned or err < self._pinned[pat][0]):
                self._pinned[pat] = (err, label, pred)

        if label in self._top:
            if err < self._top[label][0]:
                self._top[label] = (err, pred)
            return
        if len(self._top) < self.k:
            self._top[label] = (err, pred)
            return
        worst = max(self._top, key=lambda k_: self._top[k_][0])
        if err < self._top[worst][0]:
            del self._top[worst]
            self._top[label] = (err, pred)

    @property
    def items(self) -> dict:
        """{label: (nmse, pred, pinned)}, pinned slots taking precedence."""
        out = {lab: (err, pred, False) for lab, (err, pred) in self._top.items()}
        for err, lab, pred in self._pinned.values():
            out[lab] = (err, pred, True)
        return out


# ── stages ────────────────────────────────────────────────────────────────────


def base_cfg(args, **kw):
    cfg = dict(
        stage="",
        model="branched",
        latent="pod",
        branch="mlp",
        r_field=args.r_field,
        r_sensor=args.r_sensor,
        ridge=args.ridge,
        n_delays=args.n_delays,
        channels="all",
        seed=args.seed,
        sensor_noise=args.sensor_noise,
        weight_decay=args.weight_decay,
        lr=args.lr,
        ensemble=args.ensemble,
    )
    cfg.update(kw)
    return cfg


def stage_A(args):
    """Latent-size convergence. The curve this whole script exists for."""
    out = []
    for r in args.latents_sweep:
        for model in ("podlse", "epod"):
            out.append(base_cfg(args, stage="A", model=model, latent="pod", branch="closed-form", r_field=r))
        for lk in args.latents:
            for br in args.branches:
                out.append(base_cfg(args, stage="A", model="branched", latent=lk, branch=br, r_field=r))
    return out


def stage_B(args):
    """Window-length convergence at fixed latent size."""
    out = []
    for L in args.delays_sweep:
        out.append(base_cfg(args, stage="B", model="podlse", latent="pod", branch="closed-form", n_delays=L))
        for lk in args.latents:
            for br in args.branches:
                out.append(base_cfg(args, stage="B", model="branched", latent=lk, branch=br, n_delays=L))
    return out


def stage_C(args):
    """Sensor ablation. Drop the disc-3 balance first -- it is the informative one."""
    out = []
    for ch in args.channel_sets:
        out.append(base_cfg(args, stage="C", model="podlse", latent="pod", branch="closed-form", channels=ch))
        for lk in args.latents:
            for br in args.branches:
                out.append(base_cfg(args, stage="C", model="branched", latent=lk, branch=br, channels=ch))
    return out


def stage_D(args):
    """Seed spread. Any claimed gap has to clear this, or it is not a result."""
    out = []
    for seed in range(args.n_seeds):
        for lk in args.latents:
            for br in args.branches:
                out.append(base_cfg(args, stage="D", model="branched", latent=lk, branch=br, seed=seed))
    return out


def stage_N(args):
    """Input-noise regularisation, at fixed latent size and window.

    The convergence diagnosis found the branched models at a training NMSE of
    ~0.50 against a test NMSE of ~0.94: capacity to spare, not enough data to
    constrain it. That is overfitting, and it is the opposite of the
    under-fitting the same diagnosis found for the *linear* branch, so the fix
    is a regulariser rather than a bigger budget.

    Gaussian noise on the sensor window is the natural one here. For a linear
    model, input noise of variance s^2 is exactly equivalent to ridge with
    lambda = s^2 * n -- so this gives the networks the same regularisation the
    closed-form estimator already gets from its ridge, by a route that
    generalises to a nonlinear branch. It costs nothing at prediction time.

    The closed-form rows are included at every noise level even though noise
    does not touch them: they are a flat reference line across the plot, and a
    stage whose reference moves is a stage with a bug.
    """
    out = []
    for sd in args.noise_sweep:
        out.append(base_cfg(args, stage="N", model="podlse", latent="pod", branch="closed-form", sensor_noise=sd))
        for lk in args.latents:
            for br in args.branches:
                out.append(base_cfg(args, stage="N", model="branched", latent=lk, branch=br, sensor_noise=sd))
    return out


def stage_W(args):
    """Adam weight decay, at fixed latent size and window."""
    out = []
    for wd in args.wd_sweep:
        out.append(base_cfg(args, stage="W", model="podlse", latent="pod", branch="closed-form", weight_decay=wd))
        for lk in args.latents:
            for br in args.branches:
                out.append(base_cfg(args, stage="W", model="branched", latent=lk, branch=br, weight_decay=wd))
    return out


def stage_L(args):
    """Learning rate, with the plateau schedule left on.

    The convergence diagnosis found the optimum pinned at the top of its grid
    (1e-2) on the linear branch, which is a grid that was too narrow rather
    than an answer. This sweeps it per architecture, because a rate that suits
    a linear map is not obviously the rate that suits a GRU.
    """
    out = []
    for lr in args.lr_sweep:
        out.append(base_cfg(args, stage="L", model="podlse", latent="pod", branch="closed-form", lr=lr))
        for lk in args.latents:
            for br in args.branches:
                out.append(base_cfg(args, stage="L", model="branched", latent=lk, branch=br, lr=lr))
    return out


def stage_E(args):
    """Seed ensembling: average the predictions of N independently fitted models."""
    out = []
    for n in args.ensemble_sweep:
        for lk in args.latents:
            for br in args.branches:
                out.append(base_cfg(args, stage="E", model="branched", latent=lk, branch=br, ensemble=n))
    return out


def stage_R(args):
    """
    Low-rank tuning, at the ranks the sensors actually resolve.

    The diagnostic put the linear ceiling at NMSE 0.757 against 0.828 achieved,
    with only 3 of 64 field modes observable (mean rho^2 on test -0.035). Stage A
    swept r_field to 128, where everything above ~4 fits noise; this sweeps the
    ranks that carry signal, against the two knobs that decide how much of it
    survives the solve.
    """
    out = []
    for r in args.rank_sweep:
        for r_s in args.r_sensor_sweep:
            for ridge in args.ridge_sweep:
                out.append(
                    base_cfg(
                        args,
                        stage="R",
                        model="podlse",
                        latent="pod",
                        branch="closed-form",
                        r_field=r,
                        r_sensor=r_s,
                        ridge=ridge,
                    )
                )
        for lk in args.latents:
            for br in args.branches:
                out.append(base_cfg(args, stage="R", model="branched", latent=lk, branch=br, r_field=r))
    return out


STAGES = {
    "A": stage_A,
    "B": stage_B,
    "C": stage_C,
    "D": stage_D,
    "N": stage_N,
    "W": stage_W,
    "L": stage_L,
    "E": stage_E,
    "R": stage_R,
}
STAGE_NAME = {
    "A": "latent-size convergence",
    "B": "window-length convergence",
    "N": "input-noise regularisation",
    "W": "weight decay",
    "L": "learning rate",
    "E": "seed ensembling",
    "C": "sensor ablation",
    "D": "seed spread",
    "R": "low-rank tuning",
}


# ── plots ─────────────────────────────────────────────────────────────────────


def _series(rows, stage, x):
    """{label: (xs, ys)} for one stage, averaged over seeds, sorted by x."""
    out = {}
    for r in rows:
        if r["stage"] != stage:
            continue
        lab = r["model"] if r["model"] in ("podlse", "epod") else f"{r['latent']}+{r['branch']}"
        out.setdefault(lab, {}).setdefault(float(r[x]), []).append(float(r["nmse_test"]))
    return {lab: (np.array(sorted(d)), np.array([np.mean(d[k]) for k in sorted(d)])) for lab, d in out.items()}


def plot_curve(rows, stage, x, xlabel, out, floors=None, logx=True):
    s = _series(rows, stage, x)
    if not s:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab, (xs, ys) in sorted(s.items()):
        ax.plot(xs, ys, ".-", lw=1.4, ms=7, label=lab)
    if floors:
        fx, fy = floors
        ax.plot(fx, fy, "k--", lw=1.2, label="projection floor (best possible)")
    ax.axhline(1.0, c="k", ls=":", lw=1)
    ax.annotate("predicting the mean", (ax.get_xlim()[0], 1.0), fontsize=7, va="bottom", ha="left")
    ax.set(xlabel=xlabel, ylabel="test NMSE", yscale="log")
    if logx:
        ax.set_xscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_ablation(rows, out):
    d = {}
    for r in rows:
        if r["stage"] != "C":
            continue
        lab = r["model"] if r["model"] in ("podlse", "epod") else f"{r['latent']}+{r['branch']}"
        d.setdefault(lab, {})[r["channels"]] = float(r["nmse_test"])
    if not d:
        return
    specs = sorted({c for v in d.values() for c in v})
    x = np.arange(len(specs))
    w = 0.8 / max(len(d), 1)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (lab, v) in enumerate(sorted(d.items())):
        ax.bar(x + i * w, [v.get(c, np.nan) for c in specs], w, label=lab, alpha=0.9)
    ax.set_xticks(x + 0.4 - w / 2, specs)
    ax.axhline(1.0, c="k", ls=":", lw=1)
    ax.set(ylabel="test NMSE", yscale="log", xlabel="channels kept")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_seeds(rows, out):
    d = {}
    for r in rows:
        if r["stage"] != "D":
            continue
        d.setdefault(f"{r['latent']}+{r['branch']}", []).append(float(r["nmse_test"]))
    if not d:
        return
    labs = sorted(d)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labs)), 4.6))
    ax.boxplot([d[k] for k in labs], tick_labels=labs)
    for i, k in enumerate(labs, 1):
        ax.plot(np.full(len(d[k]), i), d[k], "k.", ms=6, alpha=0.6)
    ax.set(ylabel="test NMSE", yscale="log", title="seed spread -- a claimed gap has to clear this")
    ax.grid(alpha=0.3, axis="y", which="both")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)

    s = p.add_argument_group("sweep")
    s.add_argument("--stages", nargs="+", default=["A", "B", "C", "D"], choices=list(STAGES))
    s.add_argument(
        "--latents-sweep", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128], help="stage A: r_field values"
    )
    s.add_argument(
        "--delays-sweep", type=int, nargs="+", default=[1, 2, 5, 10, 25, 50, 100], help="stage B: window lengths"
    )
    s.add_argument(
        "--channel-sets",
        nargs="+",
        default=["all", "disc2", "disc3", "forces", "moments"],
        help="stage C: channel subsets",
    )
    s.add_argument("--n-seeds", type=int, default=5, help="stage D")
    s.add_argument(
        "--wd-sweep",
        type=float,
        nargs="+",
        default=[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        help="stage W: Adam weight decay",
    )
    s.add_argument(
        "--lr-sweep",
        type=float,
        nargs="+",
        default=[3e-4, 1e-3, 3e-3, 1e-2],
        help="stage L: learning rate, with the plateau schedule on",
    )
    s.add_argument(
        "--ensemble",
        type=int,
        default=1,
        help="stage E: average the predictions of this many seeds. "
        "Variance reduction only -- it changes nothing about "
        "what a single model can represent, so a gain here is "
        "a statement about run-to-run spread, not capacity.",
    )
    s.add_argument("--ensemble-sweep", type=int, nargs="+", default=[1, 2, 3, 5, 8], help="stage E: ensemble sizes")
    s.add_argument(
        "--noise-sweep",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0],
        help="stage N: sensor-noise standard deviations, in units of the standardised channel",
    )
    s.add_argument(
        "--rank-sweep",
        type=int,
        nargs="+",
        default=[2, 3, 4, 6, 8],
        help="stage R: r_field values, at the observable ranks",
    )
    s.add_argument(
        "--r-sensor-sweep",
        type=int,
        nargs="+",
        default=[12, 25, 50, 100, 200, 300],
        help="stage R: sensor-POD truncations. The top of the range is "
        "the number of available channels (12 sensors x 25 delays "
        "= 300); the previous default stopped at 200, where test "
        "NMSE was still falling monotonically, so the sweep was "
        "reporting the edge of the grid rather than a minimum.",
    )
    s.add_argument(
        "--ridge-sweep", type=float, nargs="+", default=[1e-6, 1e-4, 1e-3, 1e-2, 1e-1], help="stage R: ridge values"
    )
    s.add_argument("--latents", nargs="+", default=["pod", "ae"], choices=["pod", "ae", "cae"])
    s.add_argument("--branches", nargs="+", default=["linear", "mlp", "gru"], choices=["linear", "mlp", "cnn", "gru"])

    b = p.add_argument_group("baseline (held fixed while another axis sweeps)")
    b.add_argument("--r-field", type=int, default=64)
    b.add_argument("--r-sensor", type=int, default=None)
    b.add_argument("--n-delays", type=int, default=25)
    b.add_argument("--delay-stride", type=int, default=1)
    b.add_argument("--ridge", type=float, default=1e-4)
    b.add_argument("--weight-decay", type=float, default=0.0, help="L2 penalty in Adam. Held fixed outside stage W.")
    b.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau decay factor; 1.0 disables the schedule and trains at a fixed rate",
    )
    b.add_argument("--lr-patience", type=int, default=10, help="plateau epochs before the learning rate decays")
    b.add_argument(
        "--sensor-noise",
        type=float,
        default=0.0,
        help="Gaussian noise on the sensor window during training, held fixed outside stage N",
    )
    b.add_argument(
        "--backend",
        default="torch",
        choices=["torch", "jax"],
        help="torch is the default; jax runs the decompositions on device and supports only --latents pod",
    )
    b.add_argument("--pod-method", default="randomized", choices=["svd", "snapshot", "randomized", "auto"])

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
    t.add_argument(
        "--ae-hidden-scale",
        type=float,
        nargs="+",
        default=[8, 2],
        help="dense AE hidden widths as multiples of the latent size",
    )
    t.add_argument("--cae-channels", type=int, nargs="+", default=[16, 32, 64])
    t.add_argument("--ae-epochs", type=int, default=400)
    t.add_argument("--ae-batch", type=int, default=64)
    t.add_argument("--ae-lr", type=float, default=1e-3)
    t.add_argument("--ae-patience", type=int, default=40)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None)

    o = p.add_argument_group("output")
    o.add_argument("--out", default="results/sparse_sweep")
    o.add_argument("--tag", default="default")
    o.add_argument("--cache-dir", default=None, help="AE cache (default <out>/<tag>/cache)")
    o.add_argument("--no-cache", action="store_true", help="always refit autoencoders")
    o.add_argument("--video-frames", type=int, default=300)
    o.add_argument("--video-top", type=int, default=2)
    o.add_argument("--no-video", action="store_true")
    o.add_argument("--no-render", action="store_true", help="write video_pack.npz but do not render on the node")
    o.add_argument("--fresh", action="store_true", help="ignore an existing results.csv")
    o.add_argument("--quick", action="store_true")

    args = p.parse_args()
    if args.quick:
        # Only fill in what was not asked for explicitly. A preset that silently
        # overrides the flag you just typed is a preset you cannot debug with:
        # `--quick --latents-sweep 4 16` should sweep 4 and 16, not the preset's.
        _apply_unless_given(
            args,
            n=900,
            latents_sweep=[4, 8, 16],
            delays_sweep=[1, 5, 15],
            channel_sets=["all", "disc2"],
            n_seeds=2,
            r_field=8,
            n_delays=10,
            epochs=60,
            ae_epochs=50,
            branches=["linear", "mlp"],
            video_frames=40,
        )
    if args.backend == "jax" and set(args.latents) - {"pod"}:
        p.error("--backend jax supports only --latents pod; the AE/CAE encoders are torch-only")
    args.device = default_device(args.device)
    out = os.path.join(args.out, args.tag)
    os.makedirs(out, exist_ok=True)
    args.cache_dir = args.cache_dir or os.path.join(out, "cache")
    csv_path = os.path.join(out, "results.csv")

    print(f"device: {args.device}   output: {out}   cache: {args.cache_dir}")

    rule("1. data")
    Q, S, unflat, case0, run_id, cases = load_data(args)
    # Keep the unfiltered field so every row can be scored both ways. The POD
    # basis, the projection floors and the fits all use the banded target; only
    # `nmse_fullband` looks at the original.
    Q_full = None
    if args.band_hz:
        Q_full = Q
        Q = band_limit(Q, args.band_hz, 250.0, run_id)
    args.delays = args.delays_sweep + [args.n_delays]  # make_split needs the longest
    tr, te = make_split(args, Q.shape[1], run_id, cases)[:2]

    rule("2. planning")
    plan = []
    for st in args.stages:
        plan += [dict(c, stage=st) for c in STAGES[st](args)]
    # de-duplicate: the stage baselines overlap by construction (stage A at the
    # baseline r_field is the same fit as stage B at the baseline n_delays)
    seen, uniq = set(), []
    for c in plan:
        k = key_of(c)
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    plan = uniq

    done = {}
    if os.path.exists(csv_path) and not args.fresh:
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                done[key_of(r)] = r
        print(f"  resuming: {len(done)} rows already in {csv_path}")
    todo = [c for c in plan if key_of(c) not in done]
    print(f"  {len(plan)} configurations, {len(todo)} to run ({len(plan) - len(todo)} cached)")
    for st in args.stages:
        n = sum(1 for c in todo if c["stage"] == st)
        print(f"    stage {st}  {STAGE_NAME[st]:28s} {n:>4d} fits")

    latents = Latents(Q, tr, unflat, args)
    videos = VideoPack(args.video_top, pins=["podlse"])
    rows = list(done.values())

    new = not os.path.exists(csv_path) or args.fresh
    mode = "w" if new else "a"
    with open(csv_path, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
            fh.flush()
        for st in args.stages:
            batch = [c for c in todo if c["stage"] == st]
            if not batch:
                continue
            rule(f"stage {st} -- {STAGE_NAME[st]}  ({len(batch)} fits)")
            for cfg in batch:
                row = fit_one(cfg, Q, S, tr, te, latents, args, videos, Q_full)
                rows.append(row)
                w.writerow(row)
                fh.flush()  # a walltime kill then costs one row, not the run

    rule("3. plots")
    Psi, Sigma, qm = latents.pod_basis(max(args.latents_sweep + [args.r_field]))
    floors = ([r for r in args.latents_sweep], [projection_floor(Q[:, te], Psi[:, :r], qm) for r in args.latents_sweep])
    plot_curve(rows, "A", "r_field", "latent size $r$", os.path.join(out, "convergence_latent.png"), floors=floors)
    plot_curve(rows, "B", "n_delays", "window length $L$ [samples]", os.path.join(out, "convergence_delays.png"))
    plot_ablation(rows, os.path.join(out, "ablation_sensors.png"))
    plot_seeds(rows, os.path.join(out, "seed_spread.png"))
    rp.plot_spectrum(Sigma, os.path.join(out, "spectrum.png"), floors={f"r={r}": r for r in args.latents_sweep})
    Sd = delay_embed(S, args.n_delays, args.delay_stride, args.delay_ahead)[:, tr]
    Sc = Sd - Sd.mean(1, keepdims=True)
    sd = Sc.std(1, keepdims=True)
    _, _, C, _ = pod(Sc / np.where(sd > 0, sd, 1.0), args.r_sensor)
    rp.plot_observability(mode_observability(Psi.T @ (Q[:, tr] - qm), C), os.path.join(out, "observability.png"))

    if not args.no_video:
        rule("4. video pack")
        pack_path = os.path.join(out, "video_pack.npz")
        if not videos.items and os.path.exists(pack_path):
            # a fully-resumed job ran no fits, so nothing was offered -- but the
            # pack from the run that did the fits is already sitting there
            print(f"  nothing refitted; keeping the existing {pack_path}")
        elif not videos.items:
            # ...and if it is not, refit just the rows worth filming. Cheap: the
            # linear ones are a solve and the autoencoders come out of the cache.
            print("  nothing refitted this run; rebuilding from the best rows")
            for r in _video_rows(rows):
                fit_one(_cfg_from_row(r), Q, S, tr, te, latents, args, videos, Q_full)
            _write_videos(videos, Q, te, unflat, case0, out, args)
        else:
            _write_videos(videos, Q, te, unflat, case0, out, args)

    rule("5. summary")
    best = min(rows, key=lambda r: float(r["nmse_test"]))
    lin = [r for r in rows if r["model"] in ("podlse", "epod")]
    print(f"  best overall  : {_lab(best)}  NMSE {float(best['nmse_test']):.4f}")
    if lin:
        bl = min(lin, key=lambda r: float(r["nmse_test"]))
        print(f"  best linear   : {_lab(bl)}  NMSE {float(bl['nmse_test']):.4f}")
        gain = 1 - float(best["nmse_test"]) / float(bl["nmse_test"])
        print(f"  nonlinear gain: {gain:+.1%}")
    print(f"\n  {len(rows)} rows -> {csv_path}")
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2, default=str)
    print()
    return 0


def _cfg_from_row(r) -> dict:
    """A CSV row back into a config dict, with the types restored.

    Everything comes off a CSV as a string, and ``fit_one`` indexes arrays with
    these -- so a missed cast is a TypeError at best and a silently different
    experiment at worst.
    """
    rs = _norm(r["r_sensor"])
    return dict(
        stage=r["stage"],
        model=r["model"],
        latent=r["latent"],
        branch=r["branch"],
        r_field=int(r["r_field"]),
        r_sensor=None if rs == "None" else int(rs),
        n_delays=int(r["n_delays"]),
        channels=r["channels"],
        seed=int(r["seed"]),
    )


def _video_rows(rows, k=2):
    """The rows worth filming: the best overall, plus the best linear baseline.

    Deduplicated, because when the linear model *is* the best the two are the
    same row and refitting it twice would just film it twice.
    """
    ok = [r for r in rows if np.isfinite(float(r["nmse_test"]))]
    if not ok:
        return []
    ranked = sorted(ok, key=lambda r: float(r["nmse_test"]))
    picks = ranked[:k]
    lin = [r for r in ranked if r["model"] in ("podlse", "epod")]
    if lin and lin[0] not in picks:
        picks.append(lin[0])
    seen, out = set(), []
    for r in picks:
        if key_of(r) not in seen:
            seen.add(key_of(r))
            out.append(r)
    return out


def _lab(r):
    base = r["model"] if r["model"] in ("podlse", "epod") else f"{r['latent']}+{r['branch']}"
    return f"{base} r={r['r_field']} L={r['n_delays']} [{r['channels']}]"


def _write_videos(videos, Q, te, unflat, case0, out, args):
    """One npz with the truth stored once and every kept prediction beside it.

    Stored on the grid, not flat, so the video script needs nothing from this
    repo but numpy and matplotlib -- which matters, because you will want to
    render it on your laptop where ffmpeg exists.
    """
    n = min(args.video_frames, len(te))
    truth = unflat(Q[:, te[:n]], dtype=np.float32)
    pack = {
        "truth": truth,
        "x": case0.mf.x if case0.mf else np.arange(truth.shape[2], dtype=float),
        "y": case0.mf.y if case0.mf else np.arange(truth.shape[3], dtype=float),
    }
    meta = {"dt": args.stride / F_PIV_HZ, "run": args.run, "labels": {}}

    for i, (label, (err, pred, pinned)) in enumerate(sorted(videos.items.items(), key=lambda kv: kv[1][0])):
        k = f"pred_{i}"
        pack[k] = unflat(pred[:, :n].astype(np.float64), dtype=np.float32)
        meta["labels"][k] = {"label": label, "nmse": float(err), "pinned": bool(pinned)}
        print(f"  {label:40s} NMSE {err:.4f}{'  (pinned)' if pinned else ''}")

    pack["meta"] = json.dumps(meta)
    path = os.path.join(out, "video_pack.npz")
    np.savez_compressed(path, **pack)
    print(f"  wrote {path}  ({os.path.getsize(path) / 1e6:.0f} MB, {n} frames)")

    if args.no_render:
        print("  --no-render: sync the pack and use experiments/april_wake/scripts/make_reconstruction_video.py")
        return
    for k, info in meta["labels"].items():
        p = rp.animate_reconstruction(
            pack["truth"],
            pack[k],
            os.path.join(out, f"reconstruction_{k}.mp4"),
            mf=case0.mf,
            n_frames=n,
            title=f"{info['label']}  NMSE {info['nmse']:.3f}",
        )
        print(f"  wrote {p}")


if __name__ == "__main__":
    raise SystemExit(main())
