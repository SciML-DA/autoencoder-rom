"""
regress.py
==========

Guard rail for the perf work: prove a speedup did not move the numbers.

Every optimisation here is supposed to be mathematically neutral, but "the loss
curve looks the same" is not a check. This records the full train/val loss
history for each model at a fixed seed, and compares a later run against it.

The subtlety is that GPU training is not automatically bit-reproducible --
cuDNN picks algorithms by heuristic and some conv backwards accumulate with
atomics, so two identical runs can differ in the last few ULP. Claiming a
change is safe therefore needs a *noise floor* first: `capture` runs everything
twice and records how much the untouched code disagrees with itself. `compare`
then flags a change only when it exceeds that floor by a margin.

    python perf/regress.py capture              # on main, before any change
    python perf/regress.py compare              # on the branch, after

Deliberately small (latent 8/64, 40 epochs) -- this is a correctness check that
should run in a couple of minutes, not a benchmark. Use perf/bench.py for timing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

GOLDEN = Path("perf/golden.json")
LATENTS = (8, 64)
EPOCHS = 40
SEED = 0
MODELS = ("AE", "CAE", "AEJax", "CAEJax")


def run_one(name: str, n_latent: int, dataset: str) -> dict:
    """Fit one model and return everything that must not silently change."""
    from convergence_study import DATASETS, build_models, fit_data, pick_device
    from datasets import SPECS, load_snapshots, prepare_split

    spec = DATASETS[dataset]
    X = load_snapshots(SPECS[dataset])
    X_train, X_val, X_test, _ = prepare_split(X)
    X_fit, val_fraction = fit_data(X_train, X_val)

    m = build_models(
        n_latent, SEED, spec, pick_device(), val_fraction, n_epochs=EPOCHS
    )[name]
    m.fit(X_fit)

    # loss curves catch a changed update rule; reconstruction catches a changed
    # forward pass that happens to leave the loss alone
    Q = m.preprocess_snapshot(X_test)
    recon = m.reconstruct(X_test)
    return {
        "train": [float(v) for v in getattr(m, "loss_history", [])],
        "val": [float(v) for v in getattr(m, "val_loss_history", [])],
        "test_mse": float(np.mean((Q + m.Q_mean - recon) ** 2)),
        "n_epochs_run": int(getattr(m, "n_epochs_run", 0) or 0),
        "n_params": float(getattr(m, "n_params", float("nan"))),
    }


def rel_diff(a: list[float], b: list[float]) -> float:
    """Largest relative disagreement over the common prefix of two curves."""
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    x, y = np.array(a[:n]), np.array(b[:n])
    denom = np.maximum(np.abs(x), 1e-30)
    return float(np.max(np.abs(x - y) / denom))


def summarise(r1: dict, r2: dict) -> dict:
    return {
        "train": rel_diff(r1["train"], r2["train"]),
        "val": rel_diff(r1["val"], r2["val"]),
        "test_mse": abs(r1["test_mse"] - r2["test_mse"]) / max(abs(r1["test_mse"]), 1e-30),
        "len_train": (len(r1["train"]), len(r2["train"])),
    }


def cmd_capture(args) -> None:
    out: dict = {"_meta": {"epochs": EPOCHS, "seed": SEED, "dataset": args.dataset}}
    for n_latent in LATENTS:
        for name in MODELS:
            key = f"{name}@{n_latent}"
            a = run_one(name, n_latent, args.dataset)
            b = run_one(name, n_latent, args.dataset)  # noise floor
            noise = summarise(a, b)
            out[key] = {"golden": a, "noise": noise}
            print(
                f"  {key:14s} train_noise={noise['train']:.3e} "
                f"val_noise={noise['val']:.3e} mse_noise={noise['test_mse']:.3e}",
                flush=True,
            )
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {GOLDEN}")


def cmd_compare(args) -> None:
    if not GOLDEN.exists():
        sys.exit(f"{GOLDEN} missing -- run `capture` on the baseline first")
    ref = json.loads(GOLDEN.read_text())
    print(f"{'model':14s} {'train':>10s} {'val':>10s} {'test_mse':>10s} {'floor':>10s}  verdict")
    print("-" * 68)
    worst = 0.0
    bad = []
    for n_latent in LATENTS:
        for name in MODELS:
            key = f"{name}@{n_latent}"
            if key not in ref:
                continue
            now = run_one(name, n_latent, args.dataset)
            d = summarise(ref[key]["golden"], now)
            floor = max(ref[key]["noise"]["train"], ref[key]["noise"]["val"], 1e-12)
            # a change is suspicious once it is well clear of the noise the
            # untouched code already produces
            limit = max(floor * args.margin, args.atol)
            ok = d["train"] <= limit and d["val"] <= limit
            worst = max(worst, d["train"], d["val"])
            if not ok:
                bad.append(key)
            print(
                f"{key:14s} {d['train']:10.2e} {d['val']:10.2e} "
                f"{d['test_mse']:10.2e} {floor:10.2e}  {'ok' if ok else 'CHANGED'}"
            )
    print(f"\nworst relative drift: {worst:.3e}")
    if bad:
        sys.exit(f"FAIL: {', '.join(bad)} moved beyond the noise floor")
    print("PASS: all models within the noise floor")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["capture", "compare"])
<<<<<<< Updated upstream
    p.add_argument("--dataset", default=os.environ.get("REGRESS_DATASET", "circle"))
=======
    # bl is the target dataset -- a golden captured on circle would not cover
    # the grid shape and step count the real work runs at
    p.add_argument("--dataset", default=os.environ.get("REGRESS_DATASET", "bl"))
>>>>>>> Stashed changes
    p.add_argument("--margin", type=float, default=10.0, help="x noise floor")
    p.add_argument("--atol", type=float, default=1e-9)
    a = p.parse_args()
    (cmd_capture if a.cmd == "capture" else cmd_compare)(a)
