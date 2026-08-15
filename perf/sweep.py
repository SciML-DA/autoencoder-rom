"""
sweep.py
========

Hyperparameter sweeps for the JAX autoencoders on bl, using the optimised
implementations in `tools.autoencoders_jax_fast`.

One-at-a-time, not factorial: each hyperparameter is swept over 8 values with
the rest held at the study's defaults. Eight parameters crossed would be 8^8
configs; OAT is 8 per parameter and answers "what does this knob do" without
pretending to map the joint space.

The noise sweep is the exception worth explaining: noise is added to the data
the model trains on, but reconstruction error is scored against the *clean*
signal. Scoring against the noisy data would reward a model for reproducing the
noise, which is the opposite of what a ROM should do.

    python perf/sweep.py --shard 0 --nshards 3
    python perf/sweep.py --merge
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

OUT = Path("results/sweeps")
DATASET = "bl"
MODELS = ("AEJax", "CAEJax")
N_EPOCHS = 400          # the user's "300/400 for good accuracy"; early stopping still applies
SEED = 0

# defaults every sweep holds fixed except its own axis
BASE = {
    "n_latent": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-6,
    "batch_size": 32,
    "width": 1.0,        # multiplier on the default hidden / conv widths
    "noise": 0.0,        # fraction of the per-field std added as gaussian noise
}

SWEEPS = {
    "n_latent":      (2, 4, 8, 16, 32, 64, 128, 256),
    "learning_rate": (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    "weight_decay":  (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2),
    "batch_size":    (8, 16, 32, 64, 128, 256, 512, 1024),
    "width":         (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0),
    "noise":         (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
    # latent at FIXED width. The n_latent axis above uses the study's
    # SCALED_HIDDEN = (8k, 2k), so moving k from 2 to 256 also multiplies the
    # dense AE's capacity by 129x -- accuracy gains there cannot be attributed
    # to the bottleneck rather than to the bigger network. Holding width
    # constant separates the two. (CAEJax is barely affected either way: its
    # conv channels are already independent of k, only the bottleneck dense
    # layer scales.)
    "latent_fixed_w": (2, 4, 8, 16, 32, 64, 128, 256),
}
# the width the fixed-width latent sweep pins to: SCALED_HIDDEN(64), mid-range
FIXED_HIDDEN = (512, 128)

FIELDS = [
    "model", "sweep", "value", "n_latent", "learning_rate", "weight_decay",
    "batch_size", "width", "noise", "n_params", "train_loss", "val_loss",
    "test_mse_clean", "test_rel_clean", "n_epochs_run", "fit_s",
]


def configs() -> list[tuple[str, str, float]]:
    return [(m, s, v) for s in SWEEPS for v in SWEEPS[s] for m in MODELS]


def _scaled(base: tuple, mult: float) -> tuple:
    """Widen or narrow a layer spec, keeping every layer at least 1 unit."""
    return tuple(max(1, int(round(c * mult))) for c in base)


def run(model: str, sweep: str, value, cache: dict) -> dict:
    from convergence_study import CONV_DEFAULT, SCALED_HIDDEN, TRAIN_DEFAULTS, fit_data
    from datasets import SPECS, load_snapshots, prepare_split
    from tools import autoencoders_jax_fast as J

    cfg = dict(BASE)
    if sweep == "latent_fixed_w":
        cfg["n_latent"] = value          # width stays at FIXED_HIDDEN below
    else:
        cfg[sweep] = value

    if "clean" not in cache:
        cache["clean"] = load_snapshots(SPECS[DATASET])
    X = cache["clean"]

    # noise is scaled per field by that field's own std, so one setting means
    # the same relative perturbation for u, v and w
    if cfg["noise"] > 0:
        rng = np.random.default_rng(SEED)
        sd = np.nanstd(X, axis=(1, 2, 3), keepdims=True)
        X_noisy = X + rng.normal(0.0, 1.0, X.shape).astype(X.dtype) * sd * cfg["noise"]
        X_noisy[np.isnan(X)] = np.nan          # keep the solid-body mask intact
    else:
        X_noisy = X

    Xtr_n, Xv_n, Xte_n, _ = prepare_split(X_noisy)
    _, _, Xte_clean, _ = prepare_split(X)
    X_fit, val_fraction = fit_data(Xtr_n, Xv_n)

    k = int(cfg["n_latent"])
    train = {
        **TRAIN_DEFAULTS,
        "n_epochs": N_EPOCHS,
        "batch_size": int(cfg["batch_size"]),
        "seed": SEED,
        "val_fraction": val_fraction,
        "learning_rate": float(cfg["learning_rate"]),
        "weight_decay": float(cfg["weight_decay"]),
        "threshold": 1e-4,
    }
    base_hidden = FIXED_HIDDEN if sweep == "latent_fixed_w" else tuple(SCALED_HIDDEN(k))
    if model == "AEJax":
        m = J.AEJax(n_latent=k, hidden=_scaled(base_hidden, cfg["width"]),
                    activation="tanh", **train)
    else:
        conv = dict(CONV_DEFAULT)
        conv["channels"] = _scaled(conv["channels"], cfg["width"])
        m = J.CAEJax(n_latent=k, activation="tanh", **conv, **train)

    t0 = time.perf_counter()
    m.fit(X_fit)
    fit_s = time.perf_counter() - t0

    # scored against the CLEAN test signal, so reproducing noise is not rewarded
    Q = m.preprocess_snapshot(Xte_clean)
    recon = m.reconstruct(Xte_clean)
    mse = float(np.mean((Q + m.Q_mean - recon) ** 2))
    rel = mse / float(np.mean(Q**2))
    hist_t = list(getattr(m, "loss_history", []))
    hist_v = list(getattr(m, "val_loss_history", []))
    return dict(
        model=model, sweep=sweep, value=value, **{k2: cfg[k2] for k2 in BASE},
        n_params=float(getattr(m, "n_params", float("nan"))),
        train_loss=hist_t[-1] if hist_t else float("nan"),
        val_loss=min(hist_v) if hist_v else float("nan"),
        test_mse_clean=mse, test_rel_clean=rel,
        n_epochs_run=int(getattr(m, "n_epochs_run", 0) or 0), fit_s=fit_s,
    )


def cmd_run(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mine = configs()[a.shard :: a.nshards]
    # tag the filename when only some axes are run, so adding an axis later
    # cannot clobber the shard files from the original full sweep
    tag = ("_" + a.only.replace(",", "-")) if getattr(a, "only", None) else ""
    path = OUT / f"sweep_shard{a.shard}{tag}.csv"
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} configs -> {path}", flush=True)
    cache: dict = {}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, (m, s, v) in enumerate(mine, 1):
            try:
                row = run(m, s, v, cache)
            except Exception as e:
                print(f"  [{i}/{len(mine)}] {m} {s}={v} FAILED {type(e).__name__}: {e}"[:160], flush=True)
                continue
            w.writerow(row)
            f.flush()
            print(f"  [{i}/{len(mine)}] {m:7s} {s:14s}={str(v):8s} "
                  f"rel={row['test_rel_clean']:.4e} epochs={row['n_epochs_run']:4d} "
                  f"{row['fit_s']:6.1f}s", flush=True)


def cmd_merge(_a) -> None:
    rows = []
    for p in sorted(OUT.glob("sweep_shard*.csv")):
        rows += list(csv.DictReader(open(p)))
    if not rows:
        sys.exit(f"no shard CSVs in {OUT}")
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"merged {len(rows)} rows -> {OUT / 'results.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    p.add_argument("--merge", action="store_true")
    p.add_argument("--only", default=None,
                   help="comma-separated sweep names, to add an axis without rerunning all")
    a = p.parse_args()
    if a.only:
        keep = set(a.only.split(","))
        _all = configs
        configs = lambda: [c for c in _all() if c[1] in keep]  # noqa: E731
    (cmd_merge if a.merge else cmd_run)(a)
