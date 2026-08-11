"""
baseline.py
===========

Uncontended timing baseline for every model across the full latent sweep, plus
the parameter-count scaling that follows from it. Run before any optimisation
so later work has something to be measured against.

Two measurement rules, both learned the hard way:

* JAX caches JIT compilation process-wide, so a fit that pays for compilation
  and one that inherits it are not comparable -- a 30-epoch run once came out
  faster in wall time than a 10-epoch one. Every config therefore gets a
  throwaway warmup fit at its own shape, and compile cost is reported
  separately rather than smeared into s/epoch.
* One process per GPU. Sharing a card between timed runs scattered identical
  configs by 3.5x. Shards are split across *separate* GPUs by the array job,
  never packed onto one.

    python perf/baseline.py --shard 0 --nshards 3      # one GPU's worth
    python perf/baseline.py --merge                    # combine + analyse

Peak memory is recorded per config so the later throughput work knows how many
runs actually fit on one card.
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

OUT = Path("perf/results")
LATENTS = (2, 4, 8, 16, 32, 64, 128, 256)
WARM, MEASURE = 2, 15

# bl is the target dataset; circle runs default-width only, as a cross-check
PRIMARY = "bl"
SECONDARY = "circle"

# Varying width at fixed latent decouples parameter count from latent size --
# without it, every point on a cost-vs-params curve also moves in latent, and
# the two effects cannot be told apart.
AE_HIDDEN = {
    "half": lambda k: (4 * k, k),
    "default": lambda k: (8 * k, 2 * k),      # SCALED_HIDDEN in the study
    "double": lambda k: (16 * k, 4 * k),
    "fixed_sm": lambda k: (128, 64),          # width independent of latent
    "fixed_lg": lambda k: (1024, 256),
}
CAE_CONV = {
    "narrow": {"channels": (8, 16, 32), "kernel_size": 3, "stride": 2, "pad": 1},
    "default": {"channels": (16, 32, 64), "kernel_size": 3, "stride": 2, "pad": 1},
    "wide": {"channels": (32, 64, 128), "kernel_size": 3, "stride": 2, "pad": 1},
    "deep": {"channels": (16, 32, 64, 128), "kernel_size": 3, "stride": 2, "pad": 1},
}

FIELDS = [
    "dataset", "model", "latent", "variant", "n_params", "s_per_epoch",
    "epochs_measured", "compile_s", "total_fit_s", "peak_mem_mb",
    "n_train", "n_x", "gpu",
]


def configs() -> list[tuple[str, str, int, str]]:
    """(dataset, model, latent, variant) over the full grid."""
    out = []
    for k in LATENTS:
        out.append((PRIMARY, "POD", k, "default"))
        for v in AE_HIDDEN:
            out += [(PRIMARY, "AE", k, v), (PRIMARY, "AEJax", k, v)]
        for v in CAE_CONV:
            out += [(PRIMARY, "CAE", k, v), (PRIMARY, "CAEJax", k, v)]
        # secondary dataset: default width only, to keep the grid affordable
        for m in ("POD", "AE", "AEJax", "CAE", "CAEJax"):
            out.append((SECONDARY, m, k, "default"))
    return out


def variant_spec(base: dict, model: str, variant: str) -> dict:
    """A copy of the study's spec with only the width overridden."""
    spec = dict(base)
    if model in ("AE", "AEJax"):
        spec["hidden"] = AE_HIDDEN[variant]
    elif model in ("CAE", "CAEJax"):
        spec["conv"] = CAE_CONV[variant]
    return spec


def run(dataset: str, model: str, n_latent: int, variant: str, cache: dict) -> dict:
    import torch

    from convergence_study import DATASETS, build_models, fit_data, pick_device
    from datasets import SPECS, load_snapshots, prepare_split

    if dataset not in cache:
        X = load_snapshots(SPECS[dataset])
        X_train, X_val, X_test, meta = prepare_split(X)
        X_fit, val_fraction = fit_data(X_train, X_val)
        n_x = int((~np.isnan(X[0, 0])).sum() * X.shape[0])
        cache[dataset] = (X_fit, val_fraction, meta, n_x)
    X_fit, val_fraction, meta, n_x = cache[dataset]

    spec = variant_spec(DATASETS[dataset], model, variant)
    device = pick_device()
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else device

    def fit(n_epochs: int):
        kw = {} if model == "POD" else {"n_epochs": n_epochs}
        m = build_models(n_latent, 0, spec, device, val_fraction, **kw)[model]
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        m.fit(X_fit)
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else float("nan")
        ran = int(getattr(m, "n_epochs_run", n_epochs) or n_epochs)
        npar = float(getattr(m, "n_params", float("nan")))
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        return dt, ran, npar, peak

    if model == "POD":
        # a single truncated SVD, not an epoch loop: s/epoch is meaningless
        t, _, npar, peak = fit(0)
        return dict(
            dataset=dataset, model=model, latent=n_latent, variant=variant,
            n_params=npar, s_per_epoch=float("nan"), epochs_measured=0,
            compile_s=float("nan"), total_fit_s=t, peak_mem_mb=peak,
            n_train=meta["n_train"], n_x=n_x, gpu=gpu,
        )

    t_warm, r_warm, npar, _ = fit(WARM)          # absorbs JIT / autotune
    t, ran, _, peak = fit(MEASURE)               # steady state only
    s = t / max(ran, 1)
    return dict(
        dataset=dataset, model=model, latent=n_latent, variant=variant,
        n_params=npar, s_per_epoch=s, epochs_measured=ran,
        compile_s=t_warm - s * max(r_warm, 1),
        total_fit_s=t, peak_mem_mb=peak, n_train=meta["n_train"], n_x=n_x, gpu=gpu,
    )


def cmd_run(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mine = configs()[a.shard :: a.nshards]
    path = OUT / f"baseline_shard{a.shard}.csv"
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} configs -> {path}", flush=True)
    cache: dict = {}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, (d, m, k, v) in enumerate(mine, 1):
            try:
                row = run(d, m, k, v, cache)
            except Exception as e:
                print(f"  [{i}/{len(mine)}] {d}/{m}/{v}@{k} FAILED {type(e).__name__}: {e}"[:150], flush=True)
                continue
            w.writerow(row)
            f.flush()
            sp = row["s_per_epoch"]
            print(
                f"  [{i}/{len(mine)}] {d:7s} {m:7s} {v:8s} k={k:<4d} "
                f"params={row['n_params']:>12,.0f} "
                + (f"s/epoch={sp:7.3f}" if sp == sp else f"fit={row['total_fit_s']:7.3f}s")
                + f" cmp={row['compile_s']:6.2f}",
                flush=True,
            )


def cmd_merge(_a) -> None:
    rows = []
    for p in sorted(OUT.glob("baseline_shard*.csv")):
        with open(p) as f:
            rows += list(csv.DictReader(f))
    if not rows:
        sys.exit(f"no shard CSVs in {OUT}")
    merged = OUT / "baseline.csv"
    with open(merged, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"merged {len(rows)} rows -> {merged}\n")

    def num(r, k):
        try:
            return float(r[k])
        except (ValueError, KeyError, TypeError):
            return float("nan")

    for dataset in sorted({r["dataset"] for r in rows}):
        print(f"=== {dataset} ===")
        print(f"{'model':8s} {'latent':>6s} {'params':>13s} {'s/epoch':>9s} {'compile':>8s} {'mem MB':>8s}")
        sub = [r for r in rows if r["dataset"] == dataset]
        for r in sorted(sub, key=lambda r: (r["model"], int(r["latent"]))):
            sp = num(r, "s_per_epoch")
            print(
                f"{r['model']:8s} {int(r['latent']):6d} {num(r, 'n_params'):13,.0f} "
                + (f"{sp:9.3f}" if sp == sp else f"{num(r, 'total_fit_s'):8.3f}*")
                + f" {num(r, 'compile_s'):8.2f} {num(r, 'peak_mem_mb'):8.0f}"
            )
        # cost vs size: slope of log(s/epoch) against log(params). 1.0 means
        # time tracks parameter count; ~0 means fixed overhead dominates and the
        # model is too small to matter
        print(f"\n  scaling  log-log slope of s/epoch vs n_params")
        for model in sorted({r["model"] for r in sub}):
            pts = [
                (num(r, "n_params"), num(r, "s_per_epoch"))
                for r in sub
                if r["model"] == model
            ]
            pts = [(p, s) for p, s in pts if p == p and s == s and p > 0 and s > 0]
            if len(pts) < 3:
                continue
            x = np.log(np.array([p for p, _ in pts]))
            y = np.log(np.array([s for _, s in pts]))
            slope, intercept = np.polyfit(x, y, 1)
            # what fraction of the smallest config's time is size-independent
            floor = float(np.exp(intercept) * min(p for p, _ in pts) ** slope)
            smallest = min(s for _, s in pts)
            print(
                f"    {model:8s} slope={slope:5.2f}   "
                f"(1.0 = compute-bound, 0.0 = overhead-bound)"
                + (f"   ~{100 * (1 - floor / smallest):.0f}% overhead at smallest" if smallest > 0 else "")
            )
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    p.add_argument("--merge", action="store_true")
    a = p.parse_args()
    (cmd_merge if a.merge else cmd_run)(a)
